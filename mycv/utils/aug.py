import numpy as np
import cv2
import torch


def rand_aug_cls(im: np.ndarray):
    '''
    Data augmentation for image classification

    Args:
        im: BGR
    '''
    assert isinstance(im, np.ndarray) and im.dtype == np.uint8
    # horizontal flip
    if torch.rand(1) > 0.5:
        im = cv2.flip(im, 1)
    # color
    im = augment_hsv(im, hgain=0.1, sgain=0.5, vgain=0.5)
    # Additive Gaussian
    # im = im.astype(np.float32)
    # im = im + np.random.randn(im.shape)
    return im


def augment_hsv(im, hgain=0.1, sgain=0.5, vgain=0.5):
    '''
    HSV space augmentation
    '''
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
    pass