import os
from pathlib import Path
import json
from tqdm import tqdm
import cv2
import torch

from mycv.utils.image_ops import letterbox
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
    def __init__(self, split='train', img_size=256, augment=True, to_square=True):
        assert os.path.exists(ILSVRC_DIR)
        assert isinstance(img_size, int)

        root = Path(ILSVRC_DIR)
        if split == 'train':
            list_path = root / 'ImageSets/CLS-LOC/train_cls.txt'
        elif split == 'val':
            list_path = root / 'ImageSets/CLS-LOC/val.txt'
        else:
            raise NotImplementedError()
        print(f'Loading imagenet {split} list...')
        lines = open(list_path, 'r').read().strip().split('\n')
        self.img_paths = []
        for l in lines:
            impath = str(root / f'Data/CLS-LOC/{split}/{l.split()[0]}') + '.JPEG'
            self.img_paths.append(impath)
        assert len(self.img_paths) in {1281167, 50000}

        self.split     = split
        self.img_size  = img_size
        self.to_square = to_square
        self.augment   = augment
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
        wnid = impath.split('\\')[-2]
        if self.split == 'train':
            label = WNID_TO_IDX[wnid]
        else:
            label = -1

        # data augmentation
        if self.augment:
            if torch.rand(1).item() > 0.5:
                im = cv2.flip(im, 1) # horizontal flip
        # resize, pad to square
        im, ratio, pads = letterbox(im, self.img_size, side='longer',
                                    to_square=self.to_square, div=32)
        # to tensor
        im = torch.from_numpy(im).permute(2, 0, 1).float() / 255
        # normalize such that mean = 0 and std = 1
        im = (im - self._input_mean) / self._input_std

        assert im.shape[1:] == (self.img_size, self.img_size)
        return im, label


def imagenet_val(model, img_size, batch_size, workers):
    '''
    Test on ImageNet validation set

    Args:
        model: torch model
    '''
    model: torch.nn.Module
    model.eval()
    device = next(model.parameters()).device

    # test set
    to_square = False if batch_size == 1 else True
    testset = ImageNetCls('val', img_size=img_size, augment=False, to_square=to_square)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, drop_last=False
    )
    # annotations
    # https://github.com/google-research/reassessed-imagenet
    labels_path = Path(ILSVRC_DIR) / 'Annotations' / 'cls_val_real.json'
    labels = json.load(open(labels_path, 'r'))
    assert len(testset) == len(labels)

    # traverse the dataset
    preds = []
    for imgs, _ in tqdm(testloader):
        # debugging
        # if True:
        #     import matplotlib.pyplot as plt
        #     im = imgs[0].permute(1,2,0).numpy()
        #     plt.imshow(im); plt.show()
        imgs = imgs.to(device=device)
        with torch.no_grad():
            p = model(imgs)
        assert p.dim() == 2 and p.shape[1] == len(WNIDS)
        _, p = torch.max(p.cpu(), dim=1)
        preds.append(p)
    preds = torch.cat(preds, dim=0)
    # compare the predictions and labels
    tps = [p in t for p,t in zip(preds,labels) if len(t) > 0]
    acc = sum(tps) / len(tps)
    results = {'top1': acc}
    return results


if __name__ == "__main__":
    dataset = ImageNetCls(split='val')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    data = next(iter(dataloader))

    debug = 1
