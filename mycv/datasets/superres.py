import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from mycv.utils.image import psnr_dB, crop_divisible


def sr_evaluate(model, dataset, rate=2, verbose=True):
    assert rate in (2, 4, 8)

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
        hr = crop_divisible(hr, div=rate)
        hrh, hrw = hr.shape[:2]

        lr = cv2.resize(hr, (hrw//rate,hrh//rate))
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


if __name__ == '__main__':
    class test():
        def sr_numpy(self, lr):
            lrh, lrw = lr.shape[:2]
            hr = cv2.resize(lr, (lrw*2,lrh*2), interpolation=cv2.INTER_CUBIC)
            return hr
    
    model = test()
    results = sr_evaluate(model, 'set14', rate=2)
    print(results)
