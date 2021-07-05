import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf
from torch.nn.parameter import Parameter


class SimpleGaussian(nn.Module):
    def __init__(self, channels=192):
        super().__init__()
        self._mean  = Parameter(torch.zeros(1, channels, 1, 1))
        self._scale = Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # scale = torch.exp(self._scale)
        scale = tnf.softplus(self._scale)
        p1 = gaussian_cdf(self._mean, scale, x+0.5)
        p0 = gaussian_cdf(self._mean, scale, x-0.5)
        p = p1 - p0

        _debug = p.detach()
        assert 0 <= _debug.min() <= _debug.max() <= 1
        return p


def gaussian_cdf(mean, scale, value):
    assert scale.min() > 0, f'{scale.min().item()}'
    p = 0.5 * (1 + torch.erf((value - mean) * scale.reciprocal() / math.sqrt(2)))
    _debug = p.detach()
    assert 0 <= _debug.min() <= _debug.max() <= 1
    return p
