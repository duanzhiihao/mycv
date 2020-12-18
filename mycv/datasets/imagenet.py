import os
from pathlib import Path
import json
from tqdm import tqdm
import random
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
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
    def __init__(self, split='train', img_size=224):
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
        im = (im - self._input_mean) / self._input_std

        assert im.shape[1:] == (self.img_size, self.img_size)
        return im, label


def imagenet_val(model, img_size, batch_size, workers, split='val'):
    '''
    Test on ImageNet validation set

    Args:
        model: torch model
    '''
    assert split.startswith('val')
    model: torch.nn.Module
    model.eval()
    device = next(model.parameters()).device

    # test set
    testset = ImageNetCls(split=split, img_size=img_size)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, drop_last=False
    )
    # annotations
    # https://github.com/google-research/reassessed-imagenet
    if split == 'val':
        labels_path = Path(ILSVRC_DIR) / 'Annotations' / 'cls_val_real.json'
        labels_all = json.load(open(labels_path, 'r'))
        assert len(testset) == len(labels_all)
    else:
        labels_all = []

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
        tps = [p in t for p,t in zip(preds,labels_all) if len(t) > 0]
        acc = sum(tps) / len(tps)
    else:
        labels_all = torch.cat(labels_all, dim=0)
        assert preds.shape == labels_all.shape
        tps = (preds == labels_all)
        acc = tps.sum().item() / len(tps)
    results = {'top1': acc}
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
    results = imagenet_val(model, img_size=224, batch_size=64, workers=4, split='val_600_100')
    print(results['top1'])
