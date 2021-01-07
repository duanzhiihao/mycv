import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


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

        return grad1 * t


class Distribution_for_entropy(nn.Module):
    def __init__(self):
        super(Distribution_for_entropy,self).__init__()

    def forward(self, x, p_dec):
        mean = p_dec[:,:192, :, :]
        scale= p_dec[:,192:, :, :]

        # to make the scale always positive
        scale[scale == 0] = 1e-9
        # scale1 = torch.clamp(scale1,min = 1e-6)
        m1 = torch.distributions.normal.Normal(mean,scale)

        lower = m1.cdf(x - 0.5)
        upper = m1.cdf(x + 0.5)

        #sign = -torch.sign(torch.add(lower, upper))
        #sign = sign.detach()
        #likelihood = torch.abs(f.sigmoid(sign * upper) - f.sigmoid(sign * lower))
        likelihood = torch.abs(upper - lower)

        likelihood = Low_bound.apply(likelihood)
        return likelihood


class Distribution_for_entropy2(nn.Module):
    def __init__(self):
        super(Distribution_for_entropy2,self).__init__()

    def forward(self, x, p_dec):
        mean = p_dec[:,0,:, :, :]
        scale= p_dec[:,1,:, :, :]

        # to make the scale always positive
        scale[scale == 0] = 1e-9
        #scale1 = torch.clamp(scale1,min = 1e-6)
        m1 = torch.distributions.normal.Normal(mean,scale)

        lower = m1.cdf(x - 0.5)
        upper = m1.cdf(x + 0.5)

        #sign = -torch.sign(torch.add(lower, upper))
        #sign = sign.detach()
        #likelihood = torch.abs(f.sigmoid(sign * upper) - f.sigmoid(sign * lower))
        likelihood = torch.abs(upper - lower)

        likelihood = Low_bound.apply(likelihood)
        return likelihood


class Laplace_for_entropy(nn.Module):
    def __init__(self):
        super(Laplace_for_entropy,self).__init__()

    def forward(self, x, p_dec):
        mean = p_dec[:, 0,:, :, :]
        scale= p_dec[:, 1,:, :, :]

        # to make the scale always positive
        scale[scale == 0] = 1e-9
        # scale1 = torch.clamp(scale1,min = 1e-6)
        m1 = torch.distributions.laplace.Laplace(mean,scale)

        lower = m1.cdf(x - 0.5)
        upper = m1.cdf(x + 0.5)
        likelihood = torch.abs(upper - lower)

        likelihood = Low_bound.apply(likelihood)
        return likelihood
