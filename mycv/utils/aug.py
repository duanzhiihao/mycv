from typing import Union
import random
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms.functional as tvf

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
    """ random crop

    Args:
        im (np.ndarray): [description]
        crop_hw (tuple): [description]
    """
    if imgUtils.is_image(im, cv2_ok=True, pil_ok=False):
        im = _random_crop_cv2(im, crop_hw)
    elif imgUtils.is_image(im, cv2_ok=False, pil_ok=True):
        im = _random_crop_pil(im, crop_hw)
    else:
        raise ValueError('Input is neither a valid cv2 or PIL image.')
    return im


def _random_crop_cv2(im: np.ndarray, crop_hw: tuple):
    """ random crop for cv2
    """
    height, width = im.shape[:2]
    if (height, width) == crop_hw:
        return im
    ch, cw = crop_hw
    assert height >= ch and width >= cw
    y1 = random.randint(0, height-ch)
    x1 = random.randint(0, width-cw)
    im = im[y1:y1+ch, x1:x1+cw, :]
    return im


def _random_crop_pil(img: Image.Image, crop_hw: tuple):
    """ random crop for cv2
    """
    height, width = img.height, img.width
    if (height, width) == crop_hw:
        return img
    ch, cw = crop_hw
    assert height >= ch and width >= cw
    y1 = random.randint(0, height-ch)
    x1 = random.randint(0, width-cw)
    img = img.crop(box=(x1, y1, x1+cw, y1+ch))
    return img


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


def random_hflip(im: Union[np.ndarray, Image.Image]):
    '''
    Random horizontal flip with probability 0.5

    Args:
        im: cv2 or PIL image
    '''
    flag = (torch.rand(1) > 0.5).item()
    if flag:
        return im

    if isinstance(im, np.ndarray):
        im = cv2.flip(im, 1) # horizontal flip
    elif isinstance(im, Image.Image):
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        raise ValueError()

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
