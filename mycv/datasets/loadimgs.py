import os
from pathlib import Path
import json
from tqdm import tqdm
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import torch

import mycv.utils.image as imgUtils
import mycv.utils.aug as augUtils
from mycv.utils.coding import psnr_dB, MS_SSIM, cal_bpp
from mycv.datasets.imagenet import RGB_MEAN, RGB_STD


def _get_imgpaths(datasets: list, verbose=True):
    """ get image paths

    Args:
        datasets (list):
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
    if 'Kodak' in datasets:
        raise NotImplementedError()
    if 'imagenet' in datasets:
        raise NotImplementedError()
        from mycv.paths import ILSVRC_DIR
        list_path = ILSVRC_DIR / 'ImageSets/CLS-LOC/train_cls.txt'
        img_dir = 'Data/CLS-LOC/train'
        assert list_path.exists(), f'Error: {list_path} does not exist.'
        lines = open(list_path, 'r').read().strip().split('\n')
        for l in lines:
            impath = str(ILSVRC_DIR / img_dir / l.split()[0]) + '.JPEG'
            img_paths.append(impath)
    return img_paths


class LoadImages(torch.utils.data.Dataset):
    ''' Image loading dataset for image coding
    '''
    def __init__(self, datasets=['COCO','CLIC'], img_size=256, input_norm=False,
                 verbose=True):
        assert isinstance(img_size, int)

        self.img_paths = _get_imgpaths(datasets, verbose)
        self.img_size  = img_size
        # self.transform = album.Compose([
        #     album.RandomCrop(img_size, img_size, p=1.0),
        #     album.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.6, hue=0.04, p=1)
        # ])
        self._input_norm = input_norm
        self._input_mean = torch.FloatTensor(RGB_MEAN).view(3, 1, 1)
        self._input_std  = torch.FloatTensor(RGB_STD).view(3, 1, 1)

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
        if torch.rand(1).item() > 0.5:
            im = cv2.flip(im, 1) # horizontal flip
        assert imgUtils.is_image(im)

        # to tensor
        im = torch.from_numpy(im).permute(2, 0, 1).float() / 255
        if self._input_norm:
            # normalize such that mean = 0 and std = 1
            im = (im - self._input_mean) / self._input_std

        assert im.shape == (3, self.img_size, self.img_size)
        return im


def _imread(img_path):
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    input_ = imgUtils.pad_divisible(im, div=16)
    input_ = torch.from_numpy(input_).permute(2, 0, 1).float() / 255.0 # C,H,W
    # 0~1, float32, RGB, HWC
    input_: torch.Tensor
    return input_, im


def kodak_val(model: torch.nn.Module, input_norm=None):
    ''' Test on Kodak dataset

    Args:
        model: torch model
    '''
    model.eval()
    device = next(model.parameters()).device

    # kodak dataset
    from mycv.paths import KODAK_DIR
    img_names = os.listdir(KODAK_DIR)
    img_paths = [str(KODAK_DIR / imname) for imname in img_names]

    # traverse the dataset
    msssim_func = MS_SSIM(max_val=1.0)
    psnr_all, msssim_all = [], []
    for impath in tqdm(img_paths):
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
            feat = model.encode(input_)
            output = model.decode(feat)
        output: torch.Tensor # should be between 0~1
        assert output.shape == input_.shape and output.dtype == input_.dtype

        # MSSSIM
        rec = output[:, :, :imh, :imw] # 0~1, float32
        tgt = input_[:, :, :imh, :imw]
        ms = msssim_func(rec, tgt).item()
        # PSNR
        rec = rec.cpu().squeeze(0).permute(1, 2, 0)
        rec = (rec * 255).to(dtype=torch.uint8).numpy()
        ps = psnr_dB(im, rec)
        # if True: # debugging
        #     import matplotlib.pyplot as plt
        #     plt.figure(); plt.imshow(im)
        #     plt.figure(); plt.imshow(rec); plt.show()
        # recording
        msssim_all.append(ms)
        psnr_all.append(ps)
    # average over all images
    msssim = sum(msssim_all) / len(msssim_all)
    psnr = sum(psnr_all) / len(psnr_all)
    results = {
        'psnr': psnr,
        'msssim': msssim,
        # 'msssim_db': None
    }
    return results


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # dataset = LoadImages(datasets=['COCO', 'CLIC400'])
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    # for imgs in tqdm(dataloader):
    #     for im in imgs:
    #         # im = im * dataset._input_std + dataset._input_mean
    #         im = im.permute(1,2,0).numpy()
    #         plt.imshow(im); plt.show()

    from mycv.models.nic.mini import IMCoding
    from mycv.paths import WEIGHTS_DIR
    model = IMCoding()
    checkpoint = torch.load(WEIGHTS_DIR / 'miniMSE.pt')
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    model.eval()
    results = kodak_val(model, input_norm=False)
    print(results)
