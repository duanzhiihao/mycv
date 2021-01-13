import os
from pathlib import Path
import json
from tqdm import tqdm
import cv2
import albumentations as album
import torch

import mycv.utils.image as imgUtils
import mycv.utils.aug as augUtils
from mycv.paths import ILSVRC_DIR

def get_classes():
    root = Path(ILSVRC_DIR)
    wnids = os.listdir(root / f'Data/CLS-LOC/train')
    assert len(wnids) == 1000
    wnids.sort()
    wnid_to_idx = {s:i for i,s in enumerate(wnids)}
    return wnids, wnid_to_idx

WNIDS, WNID_TO_IDX = get_classes()
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD  = (0.229, 0.224, 0.225)


class ImageNetCls(torch.utils.data.Dataset):
    '''
    ImageNet Classification dataset
    '''
    def __init__(self, split='train', img_size=224, input_norm=True):
        assert os.path.exists(ILSVRC_DIR)
        assert isinstance(img_size, int)

        root = Path(ILSVRC_DIR)
        if split == 'train':
            list_path = root / 'ImageSets/CLS-LOC/train_cls.txt'
            img_dir = 'Data/CLS-LOC/train'
        elif split == 'val':
            list_path = root / 'ImageSets/CLS-LOC/val.txt'
            img_dir = 'Data/CLS-LOC/val'
        else:
            list_path = root / f'ImageSets/CLS-LOC/{split}.txt'
            img_dir = 'Data/CLS-LOC/train'
        assert list_path.exists(), f'Error: {list_path} does not exist.'
        lines = open(list_path, 'r').read().strip().split('\n')
        self.img_paths = []
        for l in lines:
            impath = str(root / img_dir / l.split()[0]) + '.JPEG'
            self.img_paths.append(impath)
        # assert len(self.img_paths) in {1281167, 50000}

        self.split     = split
        self.img_size  = img_size
        self.transform = album.Compose([
            album.RandomCrop(img_size, img_size, p=1.0),
            album.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.6, hue=0.04, p=1)
        ])
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
        # label
        if self.split == 'val':
            label = -1
        else:
            wnid = Path(impath).parts[-2]
            label = WNID_TO_IDX[wnid]

        # data augmentation
        # im = self.transform(im)
        if self.split.startswith('train'):
            low, high = int(self.img_size/224*256), int(self.img_size/224*384)
            im = augUtils.random_scale(im, low=low, high=high)
            # im = augUtils.random_crop(im, crop_hw=(self.img_size,self.img_size))
            im = self.transform(image=im)['image']
            if torch.rand(1).item() > 0.5:
                im = cv2.flip(im, 1) # horizontal flip
        else:
            assert self.split.startswith('val')
            # resize and center crop
            im = imgUtils.scale(im, int(self.img_size/224*256), side='shorter')
            im = imgUtils.center_crop(im, crop_hw=(self.img_size,self.img_size))
        assert imgUtils.is_image(im)

        # to tensor
        im = torch.from_numpy(im).permute(2, 0, 1).float() / 255
        # normalize such that mean = 0 and std = 1
        if self._input_norm:
            im = (im - self._input_mean) / self._input_std

        assert im.shape[1:] == (self.img_size, self.img_size)
        return im, label


def imagenet_val(model: torch.nn.Module, split='val', testloader=None,
                 img_size=None, batch_size=None, workers=None, input_norm=None):
    """ Imagenet validation

    Args:
        model (torch.nn.Module): pytorch model
        split (str, optional): see ImageNetCls. Defaults to 'val'.
        testloader (optional): if not provided, a new dataloader will be created, \
            and the following arguments must be provided:
            img_size (int)
            batch_size (int)
            workers (int)
            input_norm (bool)
    """
    assert split.startswith('val')
    model.eval()
    device = next(model.parameters()).device

    # test set
    if testloader is None:
        assert all([v is not None for v in (img_size, batch_size, workers, input_norm)])
        testset = ImageNetCls(split=split, img_size=img_size, input_norm=input_norm)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=workers,
            pin_memory=True, drop_last=False
        )
    # annotations
    if split == 'val':
        # original imagenet validation labels
        fpath = ILSVRC_DIR / 'Annotations' / 'cls_val.txt'
        labels_all = open(fpath).read().strip().split('\n')
        labels_all = [int(line.split()[1]) for line in labels_all]
        # 'real' labels https://github.com/google-research/reassessed-imagenet
        fpath = ILSVRC_DIR / 'Annotations' / 'cls_val_real.json'
        labels_real_all = json.load(open(fpath, 'r'))
        assert len(testloader.dataset) == len(labels_all) == len(labels_real_all)
    else:
        labels_all = []
        labels_real_all = None

    # traverse the dataset
    preds = []
    for imgs, labels in tqdm(testloader):
        # debugging
        # if True:
        #     import matplotlib.pyplot as plt
        #     im = imgs[0] * testset._input_std + testset._input_mean
        #     im = im.permute(1,2,0).numpy()
        #     plt.imshow(im); plt.show()
        imgs = imgs.to(device=device)
        with torch.no_grad():
            p = model(imgs)
        assert p.dim() == 2
        _, p = torch.max(p.cpu(), dim=1)
        preds.append(p)
        if split != 'val':
            # custom val set
            labels_all.append(labels)
    preds = torch.cat(preds, dim=0)
    assert preds.dim() == 1

    # compare the predictions and labels
    if split == 'val':
        # original labels
        tps = [p == t for p,t in zip(preds,labels_all)]
        acc = sum(tps).item() / len(tps)
        # 'real' labels
        tps = [p in t for p,t in zip(preds,labels_real_all) if len(t) > 0]
        acc_real = sum(tps) / len(tps)
    else:
        labels_all = torch.cat(labels_all, dim=0)
        assert preds.shape == labels_all.shape
        tps = (preds == labels_all)
        acc = tps.sum().item() / len(tps)
        acc_real = 0
    results = {'top1_old': acc, 'top1_real': acc_real}
    return results


if __name__ == "__main__":
    # dataset = ImageNetCls(split='train')
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)

    # import matplotlib.pyplot as plt
    # for imgs, labels in tqdm(dataloader):
    #     # continue
    #     for im, lbl in zip(imgs, labels):
    #         im = im * dataset._input_std + dataset._input_mean
    #         im = im.permute(1,2,0).numpy()
    #         plt.imshow(im); plt.show()

    from mycv.models.cls.resnet import resnet50
    from mycv.paths import WEIGHTS_DIR
    model = resnet50(num_classes=1000)
    model.load_state_dict(torch.load(WEIGHTS_DIR / 'resnet50-19c8e357.pth'))
    model = model.cuda()
    model.eval()
    results = imagenet_val(model, split='val',
                img_size=224, batch_size=64, workers=4, input_norm=True)
    print(results)
