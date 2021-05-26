import numpy as np
import cv2
from PIL import Image
import torch

_input_mean = torch.FloatTensor((0.485, 0.456, 0.406)).view(3, 1, 1)
_input_std  = torch.FloatTensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def is_image(im, cv2_ok=True, pil_ok=True):
    """ Check if the input is a valid image or not

    Args:
        im: image
        cv2_ok (bool, optional): check cv2. Defaults to True.
        pil_ok (bool, optional): check pil. Defaults to True.
    """
    assert cv2_ok or pil_ok
    if cv2_ok:
        flag_cv2 = isinstance(im, np.ndarray) and im.dtype == np.uint8 \
            and im.ndim == 3 and im.shape[2] == 3
    else:
        flag_cv2 = False
    if pil_ok:
        flag_pil = isinstance(im, Image.Image)
    else:
        flag_pil = False
    flag = flag_cv2 or flag_pil
    return flag


def save_tensor_images(images, save_path: str, is_normalized=False):
    assert isinstance(images, (list, torch.Tensor))
    imglist = []
    for img in images:
        assert img.dtype == torch.float and img.dim() == 3
        if is_normalized:
            img = (img * _input_std) + _input_mean
        im = torch.clamp(img.detach().cpu(), min=0, max=1)
        imglist.append(im)
    if len(imglist) == 1:
        im = imglist[0]
    elif len(imglist) == 2:
        im = torch.cat(imglist, dim=2)
    elif len(imglist) == 4:
        row1 = torch.cat(imglist[:2], dim=2)
        row2 = torch.cat(imglist[2:], dim=2)
        im = torch.cat([row1, row2], dim=1)
    else:
        raise NotImplementedError(f'Get {len(imglist)} images to save, not supported.')
    im = im.permute(1, 2, 0) * 255
    im = im.to(dtype=torch.uint8).numpy()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), im)


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


def pad_divisible(im: np.ndarray, div: int, mode='zero'):
    """ pad the image borders such that the [h,w] are multiples of div

    Args:
        im (np.ndarray): input image
        div (int): the output dimension will be multiples of div
        mode (str, optional): padding mode. Defaults to 'zero'.

    Returns:
        np.ndarray: padded image
    """
    assert is_image(im, cv2_ok=True, pil_ok=False)
    h_old, w_old, ch = im.shape
    if h_old % div == 0 and h_old % div == 0:
        return im
    h_tgt = round(div * np.ceil(h_old / div))
    w_tgt = round(div * np.ceil(w_old / div))
    if mode == 'zero':
        padded = np.zeros([h_tgt, w_tgt, ch], dtype=im.dtype)
        padded[:h_old, :w_old, :] = im
    elif mode == 'replicate':
        top, left = (h_tgt - h_old) // 2, (w_tgt - w_old) // 2
        padded = np.pad(im, [(top, h_tgt-h_old-top), (left, w_tgt-w_old-left), (0, 0)],
                        mode='edge')
    else:
        raise ValueError()
    return padded


def crop_divisible(im: np.ndarray, div: int):
    ''' center crop the image such that the [h,w] are multiples of div

    Args:
        im (np.ndarray): input image
        div (int): the output dimension will be multiples of div

    Returns:
        np.ndarray: padded image
    '''
    assert len(im.shape) == 3 and isinstance(div, int)
    h_old, w_old, _ = im.shape
    if h_old % div == 0 and w_old % div == 0:
        return im
    h_new = div * (h_old // div)
    w_new = div * (w_old // div)
    top = (h_old - h_new) // 2
    left = (w_old - w_new) // 2
    cropped = im[top:top+h_new, left:left+w_new, :]
    return cropped


def scale(im: np.ndarray, size: int, shorter=True):
    """ resize the image such that the shorter/longer side of the image = size

    Args:
        im (np.ndarray): image
        size (int): target size
    """
    assert is_image(im, cv2_ok=True, pil_ok=False)
    old_hw = im.shape[:2]
    if shorter:
        ratio = size / min(old_hw[0], old_hw[1]) # Scale ratio (new / old)
    else:
        ratio = size / max(old_hw[0], old_hw[1]) # Scale ratio (new / old)
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
    assert is_image(im, cv2_ok=True, pil_ok=False)
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


def psnr_dB(img1: np.ndarray, img2: np.ndarray):
    """ Compute the PSNR between two images

    Args:
        img1 (np.ndarray): image 1.
        img2 (np.ndarray): image 2.
    """
    assert is_image(img1, pil_ok=False) and is_image(img2, pil_ok=False)
    assert img1.shape == img2.shape

    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
    mse = np.mean(np.square(img1 - img2))
    if mse == 0:
        return 100
    return 10 * np.log10(255.0**2 / mse)


def msssim(img1: np.ndarray, img2: np.ndarray):
    """ Compute the MS-SSIM between two images

    Args:
        img1 (np.ndarray): image 1.
        img2 (np.ndarray): image 2.
    """
    raise NotImplementedError()
