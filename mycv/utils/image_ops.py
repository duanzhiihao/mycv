from typing import Tuple
import numpy as np
import cv2
import torch


def get_img(img_path, out_type='tensor', div=16, color='RGB'):
    '''
    Read image
    '''
    im = cv2.imread(img_path)
    hw_org: tuple = im.shape[0:2]
    if color.upper() == 'RGB':
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        assert color.upper == 'BGR'
    im = pad_divisible(im, div=div)
    im = im.astype(np.float32) / 255.0
    if out_type == 'array':
        # 0~1, float32, RGB, HWC
        return im, hw_org
    else:
        assert out_type == 'tensor'
    im = torch.from_numpy(im.transpose(2, 0, 1)) # C,H,W
    im: torch.Tensor
    return im, hw_org


def pad_divisible(im: np.ndarray, div):
    '''
    zero-pad the bottom and right border such that \
        the image is divisible by div
    '''
    H_ORG, W_ORG, ch = im.shape
    h_pad = round(div * np.ceil(H_ORG / div))
    w_pad = round(div * np.ceil(W_ORG / div))
    padded = np.zeros([h_pad, w_pad, ch], dtype=im.dtype)
    padded[:H_ORG, :W_ORG, :] = im
    return padded


def letterbox(img: np.ndarray, tgt_size:int=640, side='longer', to_square=True, div=1,
              color=(114,114,114)):
    '''
    Resize and pad the input image to square.
    1. resize such that the longer side = tgt_size;
    2. pad to square.

    Args:
        img:        np.array, (H,W,3), uint8, 0-255
        tgt_size:   int, the width/height of the output square
        color:      (int,int,int)
    
    Returns:
        img, ratio, (top, left)
    '''
    assert isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.shape[2] == 3
    assert isinstance(tgt_size, int) and (tgt_size % div == 0)
    old_hw = img.shape[:2]  # current shape [height, width]

    # resize if needed
    _func = min if (side == 'shorter') else max
    ratio = tgt_size / _func(old_hw[0], old_hw[1]) # Scale ratio (new / old)
    if ratio != 1:
        new_h = round(old_hw[0] * ratio)
        new_w = round(old_hw[1] * ratio)
        img = cv2.resize(img, (new_w,new_h))
    assert _func(img.shape[:2]) == tgt_size

    # pad to square if needed
    if to_square:
        dh, dw = tgt_size - img.shape[0], tgt_size - img.shape[1]  # wh padding
    else:
        dh = round(div * np.ceil(img.shape[0] / div)) - img.shape[0]
        dw = round(div * np.ceil(img.shape[1] / div)) - img.shape[1]
    top, bottom = dh//2, dh - dh//2
    left, right = dw//2, dw - dw//2
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (top, left)
