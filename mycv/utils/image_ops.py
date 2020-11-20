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
    H_PAD = int(div * np.ceil(H_ORG / div))
    W_PAD = int(div * np.ceil(W_ORG / div))
    padded = np.zeros([H_PAD, W_PAD, ch], dtype=im.dtype)
    padded[:H_ORG, :W_ORG, :] = im
    return padded


def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114),
              square=False, scaleup=True) -> Tuple[np.ndarray, float, tuple]:
    '''
    Resize image to a 32-pixel-multiple rectangle
    https://github.com/ultralytics/yolov3/issues/232
    '''
    assert isinstance(img, np.ndarray) and img.dtype == np.uint8
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if not square:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)
