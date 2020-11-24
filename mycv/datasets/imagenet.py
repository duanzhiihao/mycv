import os
from pathlib import Path
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


class ImageNetCls(torch.utils.data.Dataset):
    '''
    ImageNet Classification dataset
    '''
    def __init__(self, split='train', img_size=256, augment=True):
        assert os.path.exists(ILSVRC_DIR)
        root = Path(ILSVRC_DIR)
        print('Loading imagenet training list...')

        lines = open(root / 'ImageSets/CLS-LOC/train_cls.txt', 'r').read().strip().split('\n')
        self.img_paths = []
        for l in lines:
            impath = str(root / f'Data/CLS-LOC/{split}/{l.split()[0]}') + '.JPEG'
            self.img_paths.append(impath)
        assert len(self.img_paths) == 1281167

        assert isinstance(img_size, int)
        self.img_size = img_size
        self.augment = augment

        self.sanity_check()
    
    def sanity_check(self):
        pass

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
        label = WNID_TO_IDX[wnid]

        # data augmentation
        if self.augment:
            if torch.rand(1).item() > 0.5:
                im = cv2.flip(im, 1) # horizontal flip
        # resize, pad to square
        im, ratio, pads = letterbox(im, tgt_size=self.img_size)
        # to tensor
        im = torch.from_numpy(im).permute(2, 0, 1).float() / 255
        assert im.shape[1:] == (self.img_size, self.img_size)
        return im, label


if __name__ == "__main__":
    dataset = ImageNetCls(split='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    data = next(iter(dataloader))

    debug = 1
