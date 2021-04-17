import numpy as np
import torch
import torch.nn.functional as tnf


def backward_warp(x: torch.Tensor, flow: torch.Tensor):
    """ Warp an image/tensor (im2) back to im1, according to the optical flow
    https://github.com/NVlabs/PWC-Net/blob/master/Multi_Frame_Flow/models/PWCNet.py

    Args:
        x (torch.Tensor): [B, C, H, W] im2
        flow (torch.Tensor): [B, 2, H, W] flow
    """
    assert x.device == flow.device
    B, C, H, W = x.size()
    device = x.device

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    grid = grid.to(device=device)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = tnf.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).to(deivice=device)
    mask = tnf.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output, mask


def forward_warp_numpy(im1: np.ndarray, flow: np.ndarray):
    """ Forward warp image1 to image2 according to the optical flow
    """
    nH, nW = im1.shape[:2]
    assert flow.shape == (nH, nW, 2)

    # image1 coordinates
    x1, y1 = np.meshgrid(np.arange(nW), np.arange(nH))
    # image2 coordinates
    flow = flow.astype(np.int64)
    x2 = x1 + flow[:,:,0]
    y2 = y1 + flow[:,:,1]
    # select the new coordinates that are within the image
    valid = (x2 > 0) & (x2 < nW) & (y2 > 0) & (y2 < nH)
    x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
    # warp
    warped = np.zeros_like(im1)
    warped[y2, x2, ...] = im1[y1, x1, ...]
    if False:
        import matplotlib.pyplot as plt
        warped[:,:,1] = 255
        plt.figure(); plt.imshow(warped.astype(np.uint8)); plt.show()
    # simply 'interpolate' using the original image1
    interp_mask = np.ones((nH, nW), dtype=np.bool)
    interp_mask[y2, x2] = False
    warped[interp_mask, ...] = im1[interp_mask, ...]

    return warped, interp_mask
