from matplotlib.pyplot import isinteractive
import numpy as np
import torch
import torch.nn.functional as tnf

import mycv.utils.image as imgUtils


def psnr_dB(img1: np.ndarray, img2: np.ndarray):
    """ Calculate PSNR between two images in terms of dB

    Args:
        img1 (np.ndarray): image 1.
        img2 (np.ndarray): image 2
    """
    assert imgUtils.is_image(img1) and imgUtils.is_image(img2)
    assert img1.shape == img2.shape

    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
    mse = np.mean(np.square(img1 - img2))
    if mse == 0:
        return 100
    return 10 * np.log10(255.0**2 / mse)


def cal_bpp(prob: torch.Tensor, num_pixels: int):
    """ bitrate per pixel

    Args:
        prob (torch.Tensor): probabilities
        num_pixels (int): number of pixels
    """
    assert isinstance(prob, torch.Tensor), 'Invalid input type'
    bpp = torch.sum(torch.log(prob)) / (-np.log(2) * num_pixels)
    return bpp


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)]
    )
    return gauss/gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class MS_SSIM(torch.nn.Module):
    ''' Adapted from https://github.com/lizhengwei1992/MS_SSIM_pytorch
    '''
    def __init__(self, max_val=1.0, reduction='mean'):
        super(MS_SSIM, self).__init__()
        self.channel = 3
        self.max_val = max_val
        self.weight = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        assert reduction in {'mean'}, 'Invalid reduction'
        self.reduction = reduction

    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor):
        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11

        window = create_window(window_size, sigma, self.channel)
        window = window.to(device=img1.device)

        mu1 = tnf.conv2d(img1, window, padding=window_size //
                       2, groups=self.channel)
        mu2 = tnf.conv2d(img2, window, padding=window_size //
                       2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = tnf.conv2d(
            img1*img1, window, padding=window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = tnf.conv2d(
            img2*img2, window, padding=window_size//2, groups=self.channel) - mu2_sq
        sigma12 = tnf.conv2d(img1*img2, window, padding=window_size //
                           2, groups=self.channel) - mu1_mu2

        C1 = (0.01*self.max_val)**2
        C2 = (0.03*self.max_val)**2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
        mcs_map = V1 / V2
        if self.reduction == 'mean':
            ssim_map = ssim_map.mean()
            mcs_map = mcs_map.mean()
        return ssim_map, mcs_map

    def forward(self, img1, img2):
        assert img1.device == img2.device
        self.weight = self.weight.to(device=img1.device)
        levels = 5

        msssim = []
        mcs = []
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            msssim.append(ssim_map)
            mcs.append(mcs_map)
            filtered_im1 = tnf.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = tnf.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2
        msssim = torch.stack(msssim)
        mcs = torch.stack(mcs)
        value = torch.prod(mcs[0:levels-1] ** self.weight[0:levels-1]) \
                * (msssim[levels-1] ** self.weight[levels-1])
        value: torch.Tensor
        return value
