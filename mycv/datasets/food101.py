import os
from pathlib import Path
import json
from tqdm import tqdm

import cv2
import torch

from mycv.utils.image_ops import letterbox
from mycv.utils.aug import rand_aug_cls
from mycv.paths import FOOD101_DIR

def get_classes():
    root = Path(FOOD101_DIR)
    cls_names = [s for s in os.listdir(root / 'images')]
    cls_names.sort()
    name_to_idx = {name:i for i,name in enumerate(cls_names)}
    assert len(cls_names) == 101
    return cls_names, name_to_idx

CLASS_NAMES, CLS_NAME_TO_IDX = get_classes()
RGB_MEAN = (0.485, 0.456, 0.406) # imagenet
RGB_STD  = (0.229, 0.224, 0.225) # imagenet


class Food101(torch.utils.data.Dataset):
    '''
    Food-101 dataset https://www.kaggle.com/dansbecker/food-101
    '''
    def __init__(self, split, img_size=320, augment=True, to_square=True):
        assert split in {'train', 'test'}
        root = Path(FOOD101_DIR)
        img_dir = root / 'images'
        json_path = root / 'meta' / f'{split}.json'
        json_data = json.load(open(json_path, 'r'))

        # print(f'Loading Food-101 {split} set...')
        self.img_paths = []
        self.labels = []
        for cname in CLASS_NAMES:
            img_names = json_data[cname]
            self.img_paths += [str(img_dir/imname)+'.jpg' for imname in img_names]
            self.labels += [CLS_NAME_TO_IDX[cname]] * len(img_names)

        self.img_size = img_size
        self.augment  = augment
        assert to_square == True # todo
        self._input_mean = torch.FloatTensor(RGB_MEAN).view(3, 1, 1)
        self._input_std  = torch.FloatTensor(RGB_STD).view(3, 1, 1)
        
        # self.sanity_check()

    # def sanity_check(self):
    #     print('Checking images...')
    #     assert len(self.img_paths) == len(self.labels)
    #     for i, impath in enumerate(tqdm(self.img_paths)):
    #         assert isinstance(impath, str) and os.path.exists(impath), impath
    #         cid = self.labels[i]
    #         assert isinstance(cid, int) and 0<=cid<101, cid

    def __len__(self):
        assert len(self.img_paths) == len(self.labels)
        return len(self.img_paths)

    def __getitem__(self, index):
        im = cv2.imread(self.img_paths[index])
        assert im is not None

        # resize, pad to square
        im, ratio, pads = letterbox(im, tgt_size=self.img_size)
        if self.augment:
            im = rand_aug_cls(im)

        # convert cv2 image to tensor
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        assert im.shape == (3, self.img_size, self.img_size)

        im = (im - self._input_mean) / self._input_std
        label = self.labels[index]
        return im, label


def food101_val(model, img_size, batch_size, workers):
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
    testset = Food101('test', img_size=img_size, augment=False, to_square=to_square)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, drop_last=False
    )

    # traverse the dataset
    tp = 0
    total = 0
    for imgs, labels in tqdm(testloader):
        # debugging
        # if True:
        #     import matplotlib.pyplot as plt
        #     im = imgs[0].permute(1,2,0).numpy()
        #     plt.imshow(im); plt.show()
        imgs = imgs.to(device=device)
        with torch.no_grad():
            p = model(imgs)
        assert p.dim() == 2 and p.shape[1] == len(CLASS_NAMES)
        _, p = torch.max(p.cpu(), dim=1)
        # compare the predictions and labels
        tp += (p == labels).sum()
        total += len(imgs)
    # compute overall accuracy
    assert total == len(testset)
    acc = tp / total
    results = {'top1': acc}
    return results


if __name__ == "__main__":
    # dataset = Food101('train')
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=8)

    dataset = Food101(split='train', img_size=256, augment=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=8,
        pin_memory=True, drop_last=False
    )
    for imgs, labels in tqdm(dataloader):
        pass
    # imgs, labels = next(iter(dataloader))

    # im = imgs[0] * dataset._input_std + dataset._input_mean
    # im = im.permute(1,2,0).numpy()
    # print(CLASS_NAMES[labels[0].item()])
    # import matplotlib.pyplot as plt
    # plt.imshow(im); plt.show()
