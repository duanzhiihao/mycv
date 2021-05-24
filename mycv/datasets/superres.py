import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from mycv.utils.image import psnr_dB, crop_divisible


def evaluate_folder(model, dataset, scale=2, verbose=True):
    assert scale in (2, 4, 8)

    if dataset in ['set5', 'set14']:
        from mycv.paths import SR_DIR
        img_dir = SR_DIR / dataset
    else:
        assert os.path.isdir(dataset)
        img_dir = dataset

    img_dir = Path(img_dir)
    img_names = os.listdir(img_dir)
    img_names.sort()

    psnr_sum = 0
    pbar = enumerate(img_names)
    if verbose:
        pbar = tqdm(pbar)
    for it, imname in pbar:
        assert imname.endswith('.png')
        impath = img_dir / imname
        hr = cv2.imread(str(impath))
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        hr = crop_divisible(hr, div=scale)
        hrh, hrw = hr.shape[:2]

        lr = cv2.resize(hr, (hrw//scale,hrh//scale))
        hr_p = model.sr_numpy(lr)

        _psnr = psnr_dB(hr_p, hr)
        psnr_sum += _psnr

        if verbose:
            _imn = imname.split('.')[0]
            _pmean = psnr_sum / (it + 1)
            msg = f'image: {_imn}, psnr: {_psnr:.2f}, avg: {_pmean:.2f}'
            pbar.set_description(msg)
        if False:
            combine = np.concatenate([hr,hr_p], axis=1)
            plt.imshow(combine); plt.show()

    results = {
        'psnr': psnr_sum / (it + 1)
    }
    return results


def evaluate_div2k(model, dataset, scale=2, verbose=True):
    assert scale in (2, 3, 4)
    from mycv.paths import DIV2K_DIR

    if dataset == 'div2k_bicubic':
        hr_dir = DIV2K_DIR / 'val_hr'
        lr_dir = DIV2K_DIR / f'val_lr_bicubic/x{scale}'
    elif dataset in ['set5', 'set14']:
        from mycv.paths import SR_DIR
        hr_dir = SR_DIR / dataset
        lr_dir = None
    else:
        assert os.path.isdir(dataset)
        hr_dir = Path(dataset)
        lr_dir = None

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
            lr = cv2.resize(hr, (hrw//scale,hrh//scale))
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
        #     combine = np.concatenate([hr,hr_p], axis=1)
        #     plt.imshow(combine); plt.show()

    results = {
        'psnr': psnr_sum / (it + 1)
    }
    return results


if __name__ == '__main__':
    class test():
        def sr_numpy(self, lr):
            lrh, lrw = lr.shape[:2]
            hr = cv2.resize(lr, (lrw*2,lrh*2), interpolation=cv2.INTER_CUBIC)
            return hr
    
    model = test()
    results = sr_evaluate(model, 'set14', scale=2)
    print(results)
