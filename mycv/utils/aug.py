import random
import numpy as np
import cv2
import torch

import mycv.utils.image as imgUtils


def rand_aug_cls(im: np.ndarray):
    ''' Data augmentation for image classification

    Args:
        im: BGR
    '''
    assert imgUtils.is_image(im)
    # horizontal flip
    if torch.rand(1) > 0.5:
        im = cv2.flip(im, 1)
    # color
    im = augment_hsv(im, hgain=0.004, sgain=0.4, vgain=0.2)
    # Additive Gaussian
    # im = im.astype(np.float32)
    # im = im + np.random.randn(im.shape)
    return im


def random_scale(im: np.ndarray, low: int, high: int):
    ''' random scale
    '''
    assert imgUtils.is_image(im)
    size = random.randint(low, high)
    im = imgUtils.scale(im, size, side='shorter')
    return im


def random_crop(im: np.ndarray, crop_hw: tuple):
    ''' random crop
    '''
    assert imgUtils.is_image(im)
    if im.shape[:2] == crop_hw:
        return im

    height, width = im.shape[:2]
    ch, cw = crop_hw
    y1 = random.randint(0, height-ch)
    x1 = random.randint(0, width-cw)
    im = im[y1:y1+ch, x1:x1+cw, :]
    return im


def augment_hsv(im, hgain=0.1, sgain=0.5, vgain=0.5):
    '''
    HSV space augmentation
    '''
    raise DeprecationWarning()

    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1 # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
    assert im.dtype == np.uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(np.uint8)
    im = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return im


def hflip(im: np.ndarray):
    '''
    Horizontal flip

    Args:
        im: image
    '''
    im = cv2.flip(im, 1) # horizontal flip
    return im


if __name__ == "__main__":
    from mycv.paths import MYCV_DIR
    img_path = MYCV_DIR / 'images/bus.jpg'
    assert img_path.exists()

    im = cv2.imread(str(img_path))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    import albumentations as album
    import matplotlib.pyplot as plt
    plt.figure(); plt.axis('off'); plt.imshow(im)
    
    transform = album.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.06, p=1)
    for _ in range(8):
        # imaug = augment_hsv(im, hgain=0.1, sgain=0, vgain=0)
        imaug = transform(image=im)['image']
        plt.figure(); plt.imshow(imaug)

    plt.show()
