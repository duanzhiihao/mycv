from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as tnf
import torchvision.transforms.functional as tvf

from mycv.paths import IMAGENET_DIR

def get_classes():
    fpath = IMAGENET_DIR / 'annotations/classes.txt'
    with open(fpath, 'r') as f:
        wnids = f.read().strip().split('\n')
    assert len(wnids) == 1000
    wnid_to_idx = {s:i for i,s in enumerate(wnids)}
    return wnids, wnid_to_idx

WNIDS, WNID_TO_IDX = get_classes()


class ImageNetCls(torch.utils.data.Dataset):
    def __init__(self, split='val_ms64', img_size=14, input_norm=False):
        assert IMAGENET_DIR.is_dir() and isinstance(img_size, int)

        if split.startswith('val'):
            ann_path = IMAGENET_DIR / 'annotations/val.txt'
            img_dir = IMAGENET_DIR / split
        else:
            raise NotImplementedError()
            img_dir = IMAGENET_DIR / 'train'
        assert ann_path.is_file(), f'Error: {ann_path} does not exist.'
        with open(ann_path, 'r') as f:
            lines = f.read().strip().split('\n')

        self._img_paths = []
        self._labels = []
        for l in lines:
            imname, label = l.split()
            self._img_paths.append(str(img_dir / imname.replace('.JPEG', '.pt')))
            # self._img_paths.append(str(img_dir / lines[0].split()[0])) # debugging
            self._labels.append(int(label))

        self.split     = split
        self.img_size  = img_size
        self.num_class = max(self._labels) + 1
        assert input_norm == False
        self._input_norm = input_norm

    def __len__(self):
        assert len(self._img_paths) == len(self._labels)
        return len(self._img_paths)

    def __getitem__(self, index: int):
        # image
        impath = self._img_paths[index]
        X: torch.Tensor = torch.load(impath, map_location='cpu')
        nH, nW = X.shape[2:4]
        assert X.shape == (1, 192, nH, nW)
        # label
        label = self._labels[index]
        if self.split == 'train': # sanity check
            raise NotImplementedError()
            wnid = Path(impath).parts[-2]
            assert label == WNID_TO_IDX[wnid]

        img_size = self.img_size
        if self.split.startswith('train'):
            # data augmentation
            raise NotImplementedError()
        elif self.split.startswith('val'):
            # resize and center crop
            _f = (img_size/14*16) / min(nH, nW)
            _size = (round(nH*_f), round(nW*_f))
            X = tnf.interpolate(X, size=_size, mode='bilinear', align_corners=False)
            # center crop 14x14
            nH, nW = X.shape[2:4]
            _top = (nH - img_size) // 2
            _left = (nW - img_size) // 2
            X = X[:, :, _top:_top+img_size, _left:_left+img_size]
        else:
            raise ValueError(f'{self.split} is not supported.')

        X = X.squeeze_(0)
        assert X.shape == (192, img_size, img_size), f'{X.shape}, {impath}'
        return X, label


def imagenet_val(model: torch.nn.Module, split='val_ms64', testloader=None,
                 img_size=None, batch_size=None, workers=None):
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
    forward_ = getattr(model, 'forward_cls', model.forward)

    # test set
    if testloader is None:
        assert all([v is not None for v in (img_size, batch_size, workers)])
        testset = ImageNetCls(split=split, img_size=img_size)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=workers,
            pin_memory=True, drop_last=False
        )
    nC = testloader.dataset.num_class

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

    # from mycv.models.cls.resnet import resnet50
    # from mycv.paths import MYCV_DIR
    # model = resnet50(num_classes=1000)
    # model.load_state_dict(torch.load(MYCV_DIR / 'weights/resnet50-19c8e357.pth'))
    # # model.load_state_dict(torch.load(MYCV_DIR / 'runs/imagenet/res50_1/best.pt')['model'])
    # model = model.cuda()
    # model.eval()
    # results = imagenet_val(model, split='val',
    #             img_size=224, batch_size=64, workers=4, input_norm=True)
    # print(results)
