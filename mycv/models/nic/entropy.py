import math
import numpy as np
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
        p = p.clamp_(1e-6)

        _debug = p.detach()
        assert 0 < _debug.min() <= _debug.max() <= 1
        return p

def gaussian_cdf(mean, scale, value):
    assert scale.min() > 0, f'{scale.min().item()}'
    p = 0.5 * (1 + torch.erf((value - mean) * scale.reciprocal() / math.sqrt(2)))
    _debug = p.detach()
    assert 0 <= _debug.min() <= _debug.max() <= 1
    return p



class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-6)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x<1e-6] = 0
        pass_through_if = np.logical_or(x.cpu().numpy() >= 1e-6,g.cpu().numpy()<0.0)
        t = torch.Tensor(pass_through_if+0.0).cuda()
        return grad1*t


class Entropy_bottleneck(nn.Module):
    def __init__(self, channel, init_scale=10, filters=(3,3,3), likelihood_bound=1e-6,
                 tail_mass=1e-9, optimize_integer_offset=True):
        super(Entropy_bottleneck, self).__init__()

        self.filters = tuple(int(t) for t in filters)
        self.init_scale = float(init_scale)
        self.likelihood_bound = float(likelihood_bound)
        self.tail_mass = float(tail_mass)

        self.optimize_integer_offset = bool(optimize_integer_offset)

        if not 0 < self.tail_mass < 1:
            raise ValueError(
                "`tail_mass` must be between 0 and 1")
        filters = (1,) + self.filters + (1,) # (1, 3, 3, 3, 1)
        scale = self.init_scale ** (1.0 / (len(self.filters) + 1))
        self._matrices = nn.ParameterList([])
        self._bias = nn.ParameterList([])
        self._factor = nn.ParameterList([])
        # print ('scale:',scale)
        for i in range(len(self.filters) + 1): # i = 0, 1, 2, 3
            matrix = Parameter(torch.FloatTensor(channel, filters[i + 1], filters[i]))
            # matrix: (nC,3,1), (nC,3,3), (nC,3,3), (nC,1,3)
            init = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            matrix.data.fill_(init)
            self._matrices.append(matrix)

            bias = Parameter(torch.FloatTensor(channel, filters[i + 1], 1))
            # bias: (nC,3,1), (nC,3,1), (nC,3,1), (nC,1,1)
            noise = np.random.uniform(-0.5, 0.5, bias.size())
            noise = torch.FloatTensor(noise)
            bias.data.copy_(noise)
            self._bias.append(bias)

            if i < len(self.filters):
                factor = Parameter(torch.FloatTensor(channel, filters[i + 1], 1))
                # factor: (nC,3,1), (nC,3,1), (nC,3,1), (nC,1,1)
                factor.data.fill_(0.0)
                self._factor.append(factor)

    def forward(self, x):
        # x: (nB, nC, nH, nW)
        x = x.permute(1,0,2,3).contiguous()
        # x: (nC, nB, nH, nW)
        shape = x.shape
        x = x.view(shape[0],1,-1)
        # x: (nC, 1, nB*nH*nW)
        lower = self._logits_cumulative(x - 0.5, stop_gradient=False)
        upper = self._logits_cumulative(x + 0.5, stop_gradient=False)

        sign = -torch.sign(torch.add(lower, upper))
        sign = sign.detach()
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

        if self.likelihood_bound > 0:
            likelihood = Low_bound.apply(likelihood)

        likelihood = likelihood.view(shape)
        likelihood = likelihood.permute(1, 0, 2, 3)
        return likelihood

    def _logits_cumulative(self, logits, stop_gradient):
        # logits: (nC, 1, nB*nH*nW)
        for i in range(len(self.filters) + 1): # i = 0, 1, 2, 3
            matrix = tnf.softplus(self._matrices[i])
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(matrix, logits) # (nC, 3, 1) matmul (nC, 1, nB*nH*nW)
            # logits: # (nC, 3, nB*nH*nW)

            bias = self._bias[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self._factor):
                factor = torch.tanh(self._factor[i])
                if stop_gradient:
                    factor = factor.detach()
                logits += factor * torch.tanh(logits)
        return logits
