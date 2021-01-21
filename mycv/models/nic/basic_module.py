import torch
import torch.nn as nn
import torch.nn.functional as tnf
from torch.distributions.uniform import Uniform


def conv2d(in_channels: int, out_channels: int, kernel_size: int, stride=1,
           padding=0):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                     padding_mode='zeros')


class ResBlock(nn.Module):
    '''
    Basic residual block
    '''
    def __init__(self, inout, ks=3):
        super().__init__()
        pad = (ks - 1) // 2
        hidden = inout
        self.conv1 = conv2d(inout, hidden, ks, 1, padding=pad)
        self.conv2 = conv2d(hidden, inout, ks, 1, padding=pad)

    def forward(self, x):
        identity = x
        x = tnf.relu(self.conv1(x), inplace=True)
        x = self.conv2(x)
        out = x + identity
        # out = tnf.relu(out)
        return out


# here use embedded gaussian
class Non_local_Block(nn.Module):
    '''
    Non-local Attention Optimized Deep Image Compression:
    https://arxiv.org/abs/1904.09757
    '''
    def __init__(self,in_channel,out_channel):
        super(Non_local_Block,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.g = conv2d(self.in_channel,self.out_channel, 1, 1, 0)
        self.theta = conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.phi = conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.W = conv2d(self.out_channel, self.in_channel, 1, 1, 0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self,x):
        # x_size: (b c h w)

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size,self.out_channel,-1)
        g_x = g_x.permute(0,2,1)
        theta_x = self.theta(x).view(batch_size,self.out_channel,-1)
        theta_x = theta_x.permute(0,2,1)
        phi_x = self.phi(x).view(batch_size, self.out_channel, -1)

        f1 = torch.matmul(theta_x,phi_x)
        f_div_C = tnf.softmax(f1,dim=-1)
        y = torch.matmul(f_div_C,g_x)
        y = y.permute(0,2,1).contiguous()
        y = y.view(batch_size,self.out_channel,*x.size()[2:])
        W_y = self.W(y)
        z = W_y+x

        return z


class UniverseQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        b = torch.rand(1).item() * 2 - 1
        uniform_distribution = Uniform(
            -0.5*torch.ones(x.size())*(2**b), 0.5*torch.ones(x.size())*(2**b)
        ).sample().to(device=x.device)
        return torch.round(x+uniform_distribution)-uniform_distribution

    @staticmethod
    def backward(ctx, g):
        return g
