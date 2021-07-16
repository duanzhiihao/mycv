from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf

import mycv.utils.aug as augUtils
from mycv.paths import IMAGENET_DIR

def _get_classes():
    fpath = IMAGENET_DIR / 'annotations/classes.txt'
    with open(fpath, 'r') as f:
        wnids = f.read().strip().split('\n')
    assert len(wnids) == 1000
    wnid_to_idx = {s:i for i,s in enumerate(wnids)}
    return wnids, wnid_to_idx


class ImageNetCls(torch.utils.data.Dataset):
    '''
    ImageNet Classification dataset
    '''
    WNIDS, WNID_TO_IDX = _get_classes()
    RGB_MEAN = (0.485, 0.456, 0.406)
    RGB_STD  = (0.229, 0.224, 0.225)
    input_mean = torch.FloatTensor(RGB_MEAN).view(3, 1, 1)
    input_std  = torch.FloatTensor(RGB_STD).view(3, 1, 1)
    def __init__(self, split='train', img_size=224, input_norm=True):
        assert IMAGENET_DIR.is_dir() and isinstance(img_size, int)

        if split == 'val':
            img_dir = IMAGENET_DIR / 'val'
        elif split.startswith('train_') or split.startswith('val_'):
            img_dir = IMAGENET_DIR / split
            split = split.split('_')[0]
        else:
            img_dir = IMAGENET_DIR / 'train'

        self._img_paths = []
        self._labels = []
        pairs = self.get_image_label_pairs(split)
        for imname, label in pairs:
            self._img_paths.append(str(img_dir / imname))
            # self._img_paths.append(str(img_dir / lines[0].split()[0])) # debugging
            self._labels.append(label)

        self.split     = split
        self.img_size  = img_size
        self.num_class = max(self._labels) + 1
        self.transform = tvt.transforms.Compose([
            tvt.transforms.RandomResizedCrop(img_size),
            tvt.transforms.RandomHorizontalFlip(),
            tvt.transforms.ToTensor()
        ])
        self._input_norm = input_norm
        # if color_aug:
        #     self.transform = tvt.Compose([
        #         tvt.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.6, hue=0.04),
        #         tvt.ToTensor()
        #     ])
        # else:
        #     self.transform = tvt.ToTensor()

    def __len__(self):
        assert len(self._img_paths) == len(self._labels)
        return len(self._img_paths)

    def __getitem__(self, index: int):
        # image
        impath = self._img_paths[index]
        img = Image.open(impath).convert('RGB')
        # label
        label = self._labels[index]
        if self.split == 'train': # sanity check
            wnid = Path(impath).parts[-2]
            assert label == self.WNID_TO_IDX[wnid]

        img_size = self.img_size
        if self.split.startswith('train'):
            # data augmentation
            # low, high = int(img_size/224*256), int(img_size/224*384)
            # img = tvf.resize(img, size=random.randint(low, high))
            # img = augUtils.random_crop(img, crop_hw=(img_size,img_size))
            # img = augUtils.random_hflip(img)
            im = self.transform(img)
        elif self.split.startswith('val'):
            # resize and center crop
            img = tvf.resize(img, size=int(img_size/224*256))
            img = tvf.center_crop(img, (img_size,img_size))
            im = tvf.to_tensor(img)
        else:
            raise ValueError(f'{self.split} is not supported.')

        assert im.shape == (3, img_size, img_size), f'{im.shape}, {impath}'

        # normalize such that mean = 0 and std = 1
        if self._input_norm:
            im = im.sub_(ImageNetCls.input_mean).div_(ImageNetCls.input_std)

        return im, label

    @staticmethod
    def get_image_label_pairs(split):
        ann_path = IMAGENET_DIR / f'annotations/{split}.txt'
        assert ann_path.is_file(), f'Error: {ann_path} does not exist.'
        with open(ann_path, 'r') as f:
            lines = f.read().strip().split('\n')
        pairs = []
        for s in lines:
            imname, label = s.split()
            pairs.append((imname, int(label)))
        return pairs


def imagenet_val(model: torch.nn.Module, testloader=None,
                 split=None, img_size=None, batch_size=None, workers=None, input_norm=None):
    """ Imagenet validation. Either specify testloader, or specify other arguments

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
    # test set
    if testloader is None:
        assert all([v is not None for v in (split, img_size, batch_size, workers, input_norm)])
        assert split.startswith('val')
        testset = ImageNetCls(split=split, img_size=img_size, input_norm=input_norm)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=workers,
            pin_memory=True, drop_last=False
        )
    else:
        assert all([v is None for v in (split, img_size, batch_size, workers, input_norm)])
    split = testloader.dataset.split
    nC = testloader.dataset.num_class

    model.eval()
    device = next(model.parameters()).device
    forward_ = getattr(model, 'forward_cls', model.forward)

    predictions = []
    top1_tpsum, top5_tpsum, _num = 0, 0, 0
    # traverse the dataset
    pbar = tqdm(testloader)
    with torch.no_grad():
        for imgs, labels in pbar:
            # _debug(imgs)
            nB = imgs.shape[0]
            # forward pass
            imgs = imgs.to(device=device)
            p = forward_(imgs)
            assert p.shape == (nB, nC) and labels.shape == (nB,)
            # reduce to top5
            _, pred = torch.topk(p, k=5, dim=1, largest=True)
            pred = pred.cpu()
            # record for real labels
            predictions.append(pred[:, 0])
            # compute top5 match
            correct = pred.eq(labels.view(nB, 1).expand_as(pred))
            _top1 = correct[:, 0].sum().item()
            _top5 = correct.any(dim=1).sum().item()
            top1_tpsum += _top1
            top5_tpsum += _top5
            # verbose
            _num += nB
            msg = f'top1: {top1_tpsum/_num:.4g}, top5: {top5_tpsum/_num:.4g}'
            pbar.set_description(msg)

    predictions = torch.cat(predictions, dim=0).cpu()
    total_num = len(testloader.dataset)
    assert len(predictions) == total_num

    # compare the predictions and labels
    acc_top1 = top1_tpsum / total_num
    acc_top5 = top5_tpsum / total_num

    # 'real' labels https://github.com/google-research/reassessed-imagenet
    if split == 'val':
        fpath = IMAGENET_DIR / 'annotations' / 'val_real.json'
        labels_real_all = json.load(open(fpath, 'r'))
        assert len(predictions) == len(labels_real_all)
        # compute accuracy for 'real' labels
        tps = [p.item() in t for p,t in zip(predictions,labels_real_all) if len(t) > 0]
        acc_real = sum(tps) / len(tps)
    else:
        acc_real = acc_top1

    results = {'top1': acc_top1, 'top5': acc_top5, 'top1_real': acc_real}
    return results

def _debug(imgs):    
    import matplotlib.pyplot as plt
    im = imgs[0] * ImageNetCls.input_std + ImageNetCls.input_mean
    im = im.permute(1,2,0).numpy()
    plt.imshow(im); plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # dataset = ImageNetCls(split='train')
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)
    # for imgs, labels in tqdm(dataloader):
    #     # continue
    #     for im, lbl in zip(imgs, labels):
    #         im = im * dataset.input_std + dataset.input_mean
    #         im = im.permute(1,2,0).numpy()
    #         plt.imshow(im); plt.show()
    #     imgs = imgs

    from mycv.models.cls.resnet import resnet50
    from mycv.paths import MYCV_DIR
    model = resnet50(num_classes=1000)
    model.load_state_dict(torch.load(MYCV_DIR / 'weights/resnet/res50_nonorm.pt')['model'])
    # model.load_state_dict(torch.load(MYCV_DIR / 'runs/imagenet/res50_1/best.pt')['model'])
    model = model.cuda()
    model.eval()
    results = imagenet_val(model, split='val',
                img_size=224, batch_size=64, workers=4, input_norm=False)
    print(results)
