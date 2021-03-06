import os
from numpy.lib.polynomial import polyval
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from mycv.datasets.constants import COCO_CATEGORY_LIST, IMAGENET_30_CATEGORY_LIST


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=15):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        raise NotImplementedError()
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)


def draw_xywha_(im, cx, cy, w, h, angle, color=(0,255,0), linewidth=5,
                linestyle='-'):
    '''
    Draw a single rotated bbox on an image in-place.

    Args:
        im: image numpy array, shape(h,w,3), preferably RGB
        cx, cy, w, h: center xy, width, and height of the bounding box
        angle: degrees that the bounding box rotated clockwisely
        color (optional): tuple in 0~255 range
        linewidth (optional): with of the lines
        linestyle (optional): '-' for solid lines, and ':' for dotted lines
    
    Returns:
        None
    '''
    assert isinstance(im, np.ndarray) and im.dtype == np.uint8 and im.shape[2] ==3
    c, s = np.cos(angle/180*np.pi), np.sin(angle/180*np.pi)
    R = np.asarray([[c, s], [-s, c]])
    pts = np.asarray([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    rot_pts = []
    for pt in pts:
        rot_pts.append(([cx, cy] + pt @ R).astype(int))
    if linestyle == '-':
        contours = np.array([rot_pts[0], rot_pts[1], rot_pts[2], rot_pts[3]])
        cv2.polylines(im, [contours], isClosed=True, color=color,
                    thickness=linewidth, lineType=cv2.LINE_4)
    elif linestyle == ':':
        contours = [rot_pts[0], rot_pts[1], rot_pts[2], rot_pts[3]]
        drawpoly(im, contours, color, thickness=linewidth)
    else:
        raise Exception('Unknown linestyle in function draw_xywha_()')


def draw_bboxes_(im, detections, print_dt=False, color=(255,0,0),
                 text_size=1, **kwargs):
    '''
    Args:
        im: image numpy array, shape(h,w,3), RGB
        detections: rows of [x,y,w,h,a,conf], angle in degree
    '''
    raise NotImplementedError()
    assert isinstance(im, np.ndarray) and im.dtype == np.uint8 and im.shape[2] ==3
    line_width = kwargs.get('line_width', im.shape[0] // 300)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = max(int(2*text_size), 1)
    for bb in detections:
        if len(bb) == 6:
            x,y,w,h,a,conf = bb
        elif len(bb) == 7:
            x,y,w,h,conf,cidx = bb
        else:
            x,y,w,h,a = bb[:5]
            conf = -1
        x1, y1 = x - w/2, y - h/2
        if print_dt:
            print(f'[{x} {y} {w} {h} {a}], confidence: {conf}')
        draw_xywha_(im, x, y, w, h, a, color=color, linewidth=line_width)
        if kwargs.get('show_conf', True):
            cv2.putText(im, f'{conf:.2f}', (int(x1),int(y1)), font, 1*text_size,
                        (255,255,255), font_bold, cv2.LINE_AA)
        if kwargs.get('show_angle', False):
            cv2.putText(im, f'{int(a)}', (x,y), font, 1*text_size,
                        (255,255,255), font_bold, cv2.LINE_AA)
    if kwargs.get('show_count', True):
        caption_w = int(im.shape[0] / 4.8)
        caption_h = im.shape[0] // 25
        start = (im.shape[1] - caption_w, im.shape[0] // 20)
        end = (im.shape[1], start[1] + caption_h)
        # cv2.rectangle(im, start, end, color=(0,0,0), thickness=-1)
        cv2.putText(im, f'Count: {len(detections)}',
                    (im.shape[1] - caption_w + im.shape[0]//100, end[1]-im.shape[1]//200),
                    font, 1.2*text_size,
                    (255,255,255), font_bold*2, cv2.LINE_AA)


def random_colors(num: int, order: str='RGB', dtype: str='uint8') -> np.ndarray:
    '''
    Generate random distinct colors

    Args:
        num: number of distinct colors
        order: 'RGB', 'BGR'
        dtype: 'uint8', 'float', 'float32'
    
    Return:
        colors: np.ndarray, shape[num, 3]
    '''
    assert isinstance(num, int) and num >= 1
    hues = np.linspace(0, 360, num+1, dtype=np.float32)
    np.random.shuffle(hues)
    hsvs = np.ones((1,num,3), dtype=np.float32)
    hsvs[0,:,0] = 2 if num==1 else hues[:-1]
    if order == 'RGB':
        colors = cv2.cvtColor(hsvs, cv2.COLOR_HSV2RGB)
    elif order == 'BGR':
        colors = cv2.cvtColor(hsvs, cv2.COLOR_HSV2BGR)
    if dtype == 'uint8':
        colors = (colors * 255).astype(np.uint8)
    else:
        assert dtype == 'float' or dtype == 'float32'
    colors: np.ndarray = colors.reshape(num,3)
    return colors


def tensor_to_npimg(tensor_img):
    tensor_img = tensor_img.squeeze()
    assert tensor_img.shape[0] == 3 and tensor_img.dim() == 3
    return tensor_img.permute(1,2,0).cpu().numpy()


def imshow_tensor(tensor_batch):
    batch = tensor_batch.clone().detach().cpu()
    if batch.dim() == 3:
        batch = batch.unsqueeze(0)
    for tensor_img in batch:
        np_img = tensor_to_npimg(tensor_img)
        plt.imshow(np_img)
    plt.show()


def colorize_semseg(gray: torch.LongTensor, palette='cityscapes'):
    """ Visualize sementic segmentation predictions

    Args:
        gray (torch.LongTensor): prediction int tensor
        palette (str, optional): dataset name or color tensor. Defaults to 'cityscapes'.

    Returns: a colored image
    """
    assert torch.is_tensor(gray) and gray.dim() == 2
    assert gray.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64)
    if palette == 'cityscapes':
        from mycv.datasets.cityscapes import TRAIN_COLORS
        colors = TRAIN_COLORS
    elif isinstance(palette, str):
        raise ValueError('Unsupported palette name. Please provide the color list instead.')
    else:
        colors = palette
    # assert gray.min() >= 0 and gray.max() <= len(colors)
    assert torch.is_tensor(colors) and colors.dim() == 2 and colors.shape[1] == 3

    painting = torch.zeros(gray.shape[0], gray.shape[1], 3, dtype=torch.uint8)
    fgmask = (gray >= 0) & (gray < len(colors))
    painting[fgmask] = colors[gray[fgmask].to(dtype=torch.int64)]
    # painting = colors[gray]
    return painting


def _entropy(layer):
    total = layer.numel()
    hist = torch.bincount(layer.view(-1))
    p = hist.float() / total
    H = p * torch.log2(p)
    H[torch.isnan(H)] = 0
    H = - H.sum()
    return H

def topk_entropy(x: torch.Tensor, k: int, save_dir=None):
    """ Get the k layers with the highest information entropy

    Args:
        x (torch.Tensor): feature map
        k (int): k
    """
    assert not x.requires_grad
    x = x.squeeze(0).cpu()
    assert x.dim() == 3
    x = x - x.min()
    x = x * 255 / x.max()
    x = x.to(dtype=torch.int64)
    entropys = []
    for i in range(x.shape[0]):
        H = _entropy(x[i])
        entropys.append(H)
    entropys = torch.Tensor(entropys)
    _, kidxs = torch.topk(entropys, k)
    xk = x[kidxs, :, :]
    # save visualization
    if save_dir is not None:
        assert os.path.isdir(save_dir)
        plt.close('all')
        for li, layer in enumerate(xk):
            layer = layer.to(dtype=torch.uint8).numpy()
            plt.imshow(layer, cmap='gray')
            plt.savefig(f'{save_dir}/{li}.png', bbox_inches='tight')
    return xk, kidxs


def zoom_in(im: np.ndarray, window_center_xy: tuple, window_wh, scale, loc='br'):
    """ zoom in a window (bounding box) of the image

    Args:
        im (np.ndarray): image
        window_center_xy (tuple): window center xy
        window_wh (tuple, int): window width, height
        scale (int): upsampling factor
        loc (str, optional): new window location. Defaults to 'br'.
    """
    assert isinstance(window_center_xy, tuple)
    im = np.ascontiguousarray(np.copy(im))

    # center h, center w
    cx, cy = window_center_xy
    # window h, window w
    if isinstance(window_wh, int):
       ww, wh = window_wh, window_wh
    else:
       ww, wh = window_wh
    # draw bounding box
    line_width = max(min(im.shape[:2]) // 300, 1)
    draw_xywha_(im, cx, cy, ww, wh, 0, linewidth=line_width)
    # extract window
    window = im[cy-wh//2:cy+wh-wh//2, cx-ww//2:cx+ww-ww//2, :]
    # target window h, w
    twh, tww = round(wh*scale), round(ww*scale)
    # resize window
    window = cv2.resize(window, (tww, twh), interpolation=cv2.INTER_NEAREST)
    # paste to the original image
    if loc == 'br': # bottom-right
        im[-twh:, -tww:, :] = window
    else:
        raise ValueError()

    return im
