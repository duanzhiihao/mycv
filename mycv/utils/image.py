import numpy as np
import cv2
import torch


def is_image(im):
    flag = isinstance(im, np.ndarray) and im.dtype == np.uint8 \
           and im.ndim == 3 and im.shape[2] == 3
    return flag


def imread_tensor(img_path: str, div=16, color='RGB'):
    """ Read image and convert to tensor

    Args:
        img_path (str): image path
        div (int, optional): [description]. Defaults to 16.
        color (str, optional): RGB or BGR. Defaults to 'RGB'.
    """
    im = cv2.imread(img_path)
    assert im is not None, f'Failed loading image {img_path}'
    if color.upper() == 'RGB':
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        assert color.upper == 'BGR'
    im = pad_divisible(im, div=div)
    im = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0 # C,H,W
    # 0~1, float32, RGB, HWC
    im: torch.Tensor
    return im


def to_tensor(im: np.ndarray):
    ''' Convert uint8 [0,255] HxWx3 np.ndarray to float32 [0,1] 3xHxW torch.Tensor

    Args:
        im (np.ndarray): RGB, uint8, 0-255, [h,w,3]

    Returns:
        im (torch.Tensor): RGB, float32, 0-1, [3,h,w]
    '''
    assert is_image(im)
    assert im.shape[2] == 3 and im.dtype == np.uint8
    im = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
    im: torch.FloatTensor
    return im


def to_numpy_img(im: torch.Tensor, clamp=True):
    '''
    im: RGB, uin8, 0-255, [h,w,3]
    '''
    assert im.dtype == torch.float32
    im = im.cpu()
    if im.dim() == 4:
        assert im.shape[0] == 1
        im = im.squeeze(0)
    assert im.dim() == 3 and im.shape[0] == 3
    if clamp:
        im = torch.clamp(im, min=0, max=1)
    im = (im * 255).to(dtype=torch.uint8).permute(1, 2, 0).numpy()
    im: np.ndarray
    return im


def pad_divisible(im: np.ndarray, div: int):
    '''
    zero-pad the bottom and right border such that the image [h,w] are multiple of div
    '''
    h_old, w_old, ch = im.shape
    h_pad = round(div * np.ceil(h_old / div))
    w_pad = round(div * np.ceil(w_old / div))
    padded = np.zeros([h_pad, w_pad, ch], dtype=im.dtype)
    padded[:h_old, :w_old, :] = im
    return padded


def crop_divisible(im: np.ndarray, div: int):
    '''
    Crop the bottom and right border such that the image [h,w] are multiple of div
    '''
    assert len(im.shape) == 3 and isinstance(div, int)
    h_old, w_old, ch = im.shape
    h_crop = div * (h_old // div)
    w_crop = div * (w_old // div)
    cropped = im[:h_crop, :w_crop, :]
    return cropped


def scale(im: np.ndarray, size: int, side='shorter'):
    """ resize the image such that the shorter/longer side of the image = size

    Args:
        im (np.ndarray): image
        size (int): target size
    """
    assert is_image(im)
    old_hw = im.shape[:2]
    if side == 'longer':
        ratio = size / max(old_hw[0], old_hw[1]) # Scale ratio (new / old)
    elif side == 'shorter':
        ratio = size / min(old_hw[0], old_hw[1]) # Scale ratio (new / old)
    else:
        raise ValueError()
    if ratio != 1:
        new_h = round(old_hw[0] * ratio)
        new_w = round(old_hw[1] * ratio)
        im = cv2.resize(im, dsize=(new_w, new_h))
    return im


def center_crop(im: np.ndarray, crop_hw: tuple):
    """ center crop

    Args:
        im (np.ndarray): image
        crop_hw (tuple): target (height, width)
    """
    assert is_image(im)
    height, width = im.shape[:2]
    ch, cw = crop_hw
    if height < ch or width < cw:
        raise ValueError()
    y1 = (height - ch) // 2
    x1 = (width - cw) // 2
    im = im[y1:y1+ch, x1:x1+cw, :]
    return im


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
