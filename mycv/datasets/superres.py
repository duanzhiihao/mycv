import os
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch

from mycv.utils.image import psnr_dB, crop_divisible


def get_dir(dataset, scale):
    """ Get dataset HR folder and LR folder

    Args:
        dataset (str): dataset name
        scale (int): super resolution factor
    """
    if dataset == 'div2k_train':
        from mycv.paths import DIV2K_DIR
        assert scale in (2, 3, 4)
        hr_dir = DIV2K_DIR / 'train_hr'
        lr_dir = DIV2K_DIR / f'train_lr_bicubic/x{scale}'
    elif dataset == 'div2k_val':
        from mycv.paths import DIV2K_DIR
        assert scale in (2, 3, 4)
        hr_dir = DIV2K_DIR / 'val_hr'
        lr_dir = DIV2K_DIR / f'val_lr_bicubic/x{scale}'
    elif dataset in ['set5', 'set14', 'urban100']:
        from mycv.paths import SR_DIR
        hr_dir = SR_DIR / dataset
        lr_dir = None
    elif dataset == 'kodak':
        from mycv.paths import KODAK_DIR
        hr_dir = KODAK_DIR
        lr_dir = None
    else:
        assert os.path.isdir(dataset)
        hr_dir = Path(dataset)
        lr_dir = None
    return hr_dir, lr_dir


class SRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, scale, lr_size=48, one_mean_std=(False,False,False),
                 verbose=True):
        hr_dir, lr_dir = get_dir(dataset, scale)
        hr_names = os.listdir(hr_dir)
        self.hr_paths = [str(hr_dir / hname) for hname in hr_names]
        self.lr_paths = [str(lr_dir / hname.replace('.png', f'x{scale}.png')) \
                         for hname in hr_names]
        self.lr_size = lr_size
        self.one_mean_std = one_mean_std
        if verbose:
            print('Checking LR images...')
            for lpath in tqdm(self.lr_paths):
                assert os.path.exists(lpath)

    def __len__(self):
        assert len(self.hr_paths) == len(self.lr_paths)
        return len(self.hr_paths) * 10

    def __getitem__(self, index):
        # load image
        _idx = index % len(self.hr_paths)
        hrpath = self.hr_paths[_idx]
        lrpath = self.lr_paths[_idx]
        lr = cv2.cvtColor(cv2.imread(lrpath), cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(cv2.imread(hrpath), cv2.COLOR_BGR2RGB)
        # random crop
        lr, hr = random_patch(lr, hr, lr_patch_size=self.lr_size)
        lr, hr = _random_aug(lr, hr)
        # to tensor
        lr = torch.from_numpy(lr).permute(2, 0, 1).float()
        hr = torch.from_numpy(hr).permute(2, 0, 1).float()
        # formatting
        if self.one_mean_std[0]:
            lr.div_(255)
            hr.div_(255)
        if self.one_mean_std[1]:
            raise NotImplementedError()
        if self.one_mean_std[2]:
            raise NotImplementedError()
        return lr, hr


def random_patch(lr, hr, lr_patch_size=48):
    """ Get random patches from the lr and hr images

    Args:
        lr (np.ndarray): low-res image
        hr (np.ndarray): high-res image
        lr_patch_size (int, optional): low-res patch size. Defaults to 48.
    """
    # sanity check
    lps = lr_patch_size
    lrh, lrw = lr.shape[:2]
    hrh, hrw = hr.shape[:2]
    assert hrh % lrh == 0 and hrw % lrw == 0
    scale = hrh // lrh

    # top-left point in the low-res image
    ly = random.randint(0, lrh - lps)
    lx = random.randint(0, lrw - lps)
    # top-left point in the high-res image
    hy, hx = scale * ly, scale * lx
    # get windows
    lrwindow = lr[ly:ly+lps, lx:lx+lps, :]
    hrwindow = hr[hy:hy+lps*scale, hx:hx+lps*scale, :]

    return lrwindow, hrwindow


def _random_aug(lr, hr):
    if random.random() < 0.5: # vertical flip
        lr = cv2.flip(lr, 0)
        hr = cv2.flip(hr, 0)
    if random.random() < 0.5: # horizontal flip
        lr = cv2.flip(lr, 1)
        hr = cv2.flip(hr, 1)
    if random.random() < 0.5:
        if random.random() < 0.5:
            lr = cv2.rotate(lr, cv2.ROTATE_90_CLOCKWISE)
            hr = cv2.rotate(hr, cv2.ROTATE_90_CLOCKWISE)
        else:
            lr = cv2.rotate(lr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            hr = cv2.rotate(hr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return lr, hr


def sr_evaluate(model, dataset, scale, verbose=True):
    """ Super resolution evalation function

    Args:
        model (torch.nn.Module): pytorch model
        dataset (str): dataset name
        scale (int): super resolution factor
        verbose (bool, optional): Defaults to True.
    """
    hr_dir, lr_dir = get_dir(dataset, scale)
    hr_names = os.listdir(hr_dir)
    hr_names.sort()

    psnr_sum = 0
    pbar = enumerate(hr_names)
    if verbose:
        pbar = tqdm(pbar)
    for it, hrname in pbar:
        # get high-resolution image
        assert hrname.endswith('.png')
        hrpath = hr_dir / hrname
        hr = cv2.imread(str(hrpath))
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        hr = crop_divisible(hr, div=scale)
        hrh, hrw = hr.shape[:2]
        assert hrh % scale == 0 and hrw % scale == 0

        # get low-resolution image
        if lr_dir is not None:
            lrpath = lr_dir / hrname.replace('.png', f'x{scale}.png')
            lr = cv2.imread(str(lrpath))
            lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        else:
            # lr = cv2.resize(hr, (hrw//scale,hrh//scale), interpolation=cv2.INTER_AREA)
            lr = Image.fromarray(hr)
            lr = lr.resize((hrw//scale,hrh//scale), Image.ANTIALIAS)
            lr = np.array(lr)
        assert lr.shape[:2] == (hrh//scale, hrw//scale)

        # predict high-res image
        with torch.no_grad():
            hr_p = model.sr_numpy(lr)

        _psnr = psnr_dB(hr_p, hr)
        psnr_sum += _psnr

        if verbose:
            _imn = hrname.split('.')[0]
            _pmean = psnr_sum / (it + 1)
            msg = f'image: {_imn}, psnr: {_psnr:.2f}, avg: {_pmean:.2f}'
            pbar.set_description(msg)
        # if True:
        #     from mycv.utils.visualization import zoom_in
        #     hr = zoom_in(hr, (hrw//2,hrh//2), 32, 4)
        #     hr_p = zoom_in(hr_p, (hrw//2,hrh//2), 32, 4)
        #     combine = np.concatenate([hr,hr_p], axis=1)
        #     plt.imshow(combine); plt.show()

    results = {
        'psnr': psnr_sum / (it + 1)
    }
    return results


if __name__ == '__main__':
    scale = 2
    dataset = SRDataset('div2k_train', scale=scale, lr_size=48)
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True,
        num_workers=0, pin_memory=True
    )
    for lrs, hrs in trainloader:
        for lr,hr in zip(lrs, hrs):
            lr = lr.permute(1,2,0).to(dtype=torch.uint8).numpy()
            hr = hr.permute(1,2,0).to(dtype=torch.uint8).numpy()
            lrh,lrw = lr.shape[:2]
            lr = cv2.resize(lr, (lrw*scale,lrh*scale), interpolation=cv2.INTER_NEAREST)
            combine = np.concatenate([lr,hr], axis=1)
            plt.imshow(combine); plt.show()
        debug = 1

    class test():
        def sr_numpy(self, lr):
            lrh, lrw = lr.shape[:2]
            hr = cv2.resize(lr, (lrw*2,lrh*2), interpolation=cv2.INTER_CUBIC)
            return hr
    
    model = test()
    results = sr_evaluate(model, 'set14', scale=2)
    print(results)
