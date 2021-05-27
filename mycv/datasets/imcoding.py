import os
from tqdm import tqdm
import json
from math import log10
import cv2
import torch

import mycv.utils.image as imgUtils
import mycv.utils.aug as augUtils
from mycv.utils.coding import MS_SSIM, cal_bpp
from mycv.datasets.imagenet import ImageNetCls


def _get_imgpaths(datasets: list, verbose=True):
    """ get image paths

    Args:
        datasets (list): dataset names
        verbose (bool, optional): . Defaults to True.
    """
    img_paths = []
    if 'COCO' in datasets:
        if verbose:
            print('Loading COCO train2017 dataset...')
        from mycv.paths import COCO_DIR
        ann_path = COCO_DIR / 'annotations/images_train2017.json'
        assert ann_path.exists(), f'{ann_path} does not exist'
        images = json.load(open(ann_path, 'r'))['images']
        for imgInfo in images:
            impath = str(COCO_DIR / 'train2017' / imgInfo['file_name'])
            img_paths.append(impath)
    if 'CLIC' in datasets:
        raise DeprecationWarning()
        if verbose:
            print('Loading CLIC mobile and professional training set...')
        from mycv.paths import CLIC_DIR
        img_names = os.listdir(CLIC_DIR / 'train')
        img_names = img_names * 10
        for imname in img_names:
            impath = str(CLIC_DIR / 'train' / imname)
            img_paths.append(impath)
    if 'CLIC400' in datasets:
        if verbose:
            print('Loading CLIC 400x400 dataset...')
        from mycv.paths import CLIC_DIR
        img_dir = CLIC_DIR / 'train400'
        assert img_dir.exists()
        img_names = os.listdir(img_dir)
        for imname in img_names:
            impath = str(img_dir / imname)
            img_paths.append(impath)
    if 'NIC' in datasets:
        if verbose:
            print('Loading NIC training set...')
        NIC_DIR = 'D:/Datasets/NIC/train'
        for s in ['Animal', 'Building', 'Mountain', 'Street']:
            img_dir = NIC_DIR + '/' + s
            img_names = os.listdir(img_dir)
            for imname in img_names:
                impath = img_dir + '/' + imname
                img_paths.append(impath)
    if 'imagenet200' in datasets:
        if verbose:
            print('Loading imagenet mini200 dataset...')
        from mycv.paths import IMAGENET_DIR
        list_path = IMAGENET_DIR / 'annotations/train200_600.txt'
        img_dir = 'train'
        assert list_path.exists(), f'Error: {list_path} does not exist.'
        lines = open(list_path, 'r').read().strip().split('\n')
        for l in lines:
            impath = str(IMAGENET_DIR / img_dir / l.split()[0])
            img_paths.append(impath)
    assert len(img_paths) > 0, 'No image path loaded'
    return img_paths


class LoadImages(torch.utils.data.Dataset):
    """ Image loading dataset for image coding
    """
    def __init__(self, datasets=['COCO','CLIC'], img_size=256, input_norm=False,
                 verbose=True):
        assert isinstance(img_size, int)
        self.img_paths = _get_imgpaths(datasets, verbose)
        self.img_size  = img_size
        self._input_norm = input_norm
        self._input_mean = ImageNetCls.input_mean
        self._input_std  = ImageNetCls.input_std

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int):
        impath = self.img_paths[index]
        # image
        im = cv2.imread(impath)
        assert im is not None, f'Error loading image {impath}'
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # data augmentation
        if min(im.shape[:2]) < self.img_size:
            im = imgUtils.scale(im, size=self.img_size, side='shorter')
        assert min(im.shape[:2]) >= self.img_size, f'{impath}, {im.shape}'
        im = augUtils.random_crop(im, crop_hw=(self.img_size,self.img_size))
        im = self._random_aug(im) # random augmentation
        assert imgUtils.is_image(im)

        # to tensor
        im = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        if self._input_norm:
            # normalize such that mean = 0 and std = 1
            im = im.sub_(ImageNetCls.input_mean).div_(ImageNetCls.input_std)

        assert im.shape == (3, self.img_size, self.img_size)
        return im

    def _random_aug(self, im):
        r = lambda: torch.rand(1).item()
        if r() > 0.5:
            im = cv2.flip(im, 1) # horizontal flip
        if r() > 0.5:
            im = cv2.flip(im, 0) # vertical flip
        if r() > 0.5:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        else:
            im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return im


def _imread(img_path):
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    input_ = imgUtils.pad_divisible(im, div=64)
    input_ = torch.from_numpy(input_).permute(2, 0, 1).float() / 255.0 # C,H,W
    # 0~1, float32, RGB, HWC
    input_: torch.Tensor
    return input_, im


def nic_evaluate(model: torch.nn.Module, input_norm=False, verbose=True, dataset='kodak'):
    """ Test on Kodak dataset

    Args:
        model (torch.nn.Module): torch model
        input_norm (bool, optional): normalize input or not. Defaults to None.
    """
    if verbose:
        print(f'Evaluating {type(model)} on {dataset} dataset...')
    model.eval()
    device = next(model.parameters()).device
    forward_nic = getattr(model, 'forward_nic', model.forward)

    if dataset == 'kodak':
        # kodak dataset
        from mycv.paths import KODAK_DIR
        img_dir = KODAK_DIR
    elif dataset == 'clic':
        from mycv.paths import CLIC_DIR
        img_dir = CLIC_DIR / 'valid'
    elif dataset == 'imagenet':
        from mycv.paths import IMAGENET_DIR
        img_dir = IMAGENET_DIR / 'val'
    elif dataset == 'cityscapes':
        from mycv.paths import CITYSCAPES_DIR
        img_dir = CITYSCAPES_DIR / 'leftImg8bit/val'
    else:
        img_dir = dataset
    img_paths = [str(p) for p in img_dir.glob('**/*.*')]
    img_paths.sort()

    # traverse the dataset
    msssim_func = MS_SSIM(max_val=1.0)
    psnr_sum, msssim_sum, bpp_sum = 0, 0, 0
    pbar = tqdm(img_paths) if verbose else img_paths
    for bi, impath in enumerate(pbar):
        # load image
        input_, im = _imread(impath)
        imh, imw = im.shape[:2]
        # debugging
        # if True:
        #     import matplotlib.pyplot as plt
        #     # im = imgs[0] * testset._input_std + testset._input_mean
        #     im = input_.permute(1,2,0).numpy()
        #     plt.imshow(im); plt.show()
        if input_norm:
            raise NotImplementedError()

        # forward pass
        input_ = input_.unsqueeze(0).to(device=device)
        with torch.no_grad():
            output = forward_nic(input_)
            if isinstance(output, dict):
                fake, probs = output['x_hat'], output['likelihoods']
            else:
                fake, probs = output
        fake: torch.Tensor # should be between 0~1
        fake = fake.clamp_(min=0, max=1)
        assert fake.shape == input_.shape and fake.dtype == input_.dtype

        # MS-SSIM
        fake = fake[:, :, :imh, :imw] # 0~1, float32
        real = input_[:, :, :imh, :imw]
        ms = msssim_func(fake, real).item()
        # PSNR
        fake = fake.cpu().squeeze_(0).permute(1, 2, 0)
        fake = fake.mul_(255).round_().to(dtype=torch.uint8).numpy()
        ps = imgUtils.psnr_dB(im, fake)
        # Bpp
        if probs is not None:
            if isinstance(probs, dict):
                p1, p2 = probs['y'], probs['z']
            else:
                p1, p2 = probs
            bpp = cal_bpp(p1, imh*imw) + cal_bpp(p2, imh*imw)
            bpp = bpp.item()
        else:
            bpp = -1024
        # if True: # debugging
        #     import matplotlib.pyplot as plt
        #     plt.figure(); plt.imshow(im)
        #     plt.figure(); plt.imshow(fake); plt.show()
        # recording
        msssim_sum += ms
        psnr_sum += ps
        bpp_sum += bpp
        if verbose:
            _num = bi + 1
            msssim = msssim_sum / _num
            psnr   = psnr_sum   / _num
            bpp    = bpp_sum    / _num
            msg = f'MS: {msssim:.4g}, PSNR: {psnr:.4g}, bpp: {bpp:.4g}'
            pbar.set_description(msg)
    # average over all images
    msssim = msssim_sum / len(img_paths)
    psnr   = psnr_sum   / len(img_paths)
    bpp    = bpp_sum    / len(img_paths)
    results = {
        'psnr': psnr,
        'msssim': msssim,
        'msssim_db': -10*log10(1-msssim),
        'bpp': bpp
    }
    return results


if __name__ == "__main__":
    # from tqdm import tqdm
    # import matplotlib.pyplot as plt
    # dataset = LoadImages(datasets=['NIC'])
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    # for imgs in tqdm(dataloader):
    #     for im in imgs:
    #         # im = im * dataset._input_std + dataset._input_mean
    #         im = im.permute(1,2,0).numpy()
    #         plt.imshow(im); plt.show()

    from mycv.models.nic.mini import MiniNIC
    from mycv.paths import MYCV_DIR
    model = MiniNIC(enable_bpp=True)
    checkpoint = torch.load(MYCV_DIR / 'runs/imcoding/mini_6/last.pt')
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    model.eval()
    results = nic_evaluate(model, input_norm=False)
    print(results)
