# This file contains modules common to various models
import math

import torch
import torch.nn as nn


def yolov5backbone(width_multiple=0.5, depth_multiple=0.33):
    '''
    https://github.com/ultralytics/yolov5
    '''
    scale = lambda x: max(round(x * depth_multiple), 1)
    ceil8 = lambda x: int(math.ceil(x/8) * 8)
    channels = [ceil8(ch*width_multiple) for ch in [64, 128, 256, 512, 1024]]

    backbone = nn.ModuleList()
    backbone.append(Focus(3, channels[0], k=3)) # 0
    # 2x
    backbone.append(Conv(channels[0], channels[1], k=3, s=2)) # 1
    # 4x
    backbone.append(BottleneckCSP(channels[1], channels[1], n=scale(3))) # 2
    backbone.append(Conv(channels[1], channels[2], k=3, s=2)) # 3
    # 8x
    backbone.append(BottleneckCSP(channels[2], channels[2], n=scale(9))) # 4
    backbone.append(Conv(channels[2], channels[3], k=3, s=2)) # 5
    # 16x
    backbone.append(BottleneckCSP(channels[3], channels[3], n=scale(9))) # 6
    backbone.append(Conv(channels[3], channels[4], k=3, s=2)) # 7
    # 32x
    backbone.append(SPP(channels[4], channels[4], k=[5,9,13])) # 8
    backbone.append(BottleneckCSP(channels[4], channels[4], n=scale(3), shortcut=False)) # 9
    # backbone.append(Conv(channels[4], channels[3], k=1, s=1)) # 10

    return backbone, channels


class CustomBackbone(nn.Module):
    '''
    https://github.com/ultralytics/yolov5
    '''
    def __init__(self, width_multiple=0.5, depth_multiple=0.33, in_ch=3):
        super().__init__()
        scale = lambda x: max(round(x * depth_multiple), 1)
        ceil8 = lambda x: int(math.ceil(x/8) * 8)
        channels = [ceil8(ch*width_multiple) for ch in [64, 128, 256, 512, 1024]]

        self.c1 = Conv(in_ch, channels[0], k=5, s=2)
        self.b1 = BottleneckCSP(channels[0], channels[0], n=1)
        # 2x
        self.c2 = Conv(channels[0], channels[1], k=3, s=2)
        self.b2 = BottleneckCSP(channels[1], channels[1], n=scale(3))
        # 4x
        self.c3 = Conv(channels[1], channels[2], k=3, s=2)
        self.b3 = BottleneckCSP(channels[2], channels[2], n=scale(9))
        # 8x
        self.c4 = Conv(channels[2], channels[3], k=3, s=2)
        self.b4 = BottleneckCSP(channels[3], channels[3], n=scale(9))
        # 16x
        self.c5 = Conv(channels[3], channels[4], k=3, s=2)
        self.b5 = BottleneckCSP(channels[4], channels[4], n=scale(3))
        # 32x
        self.channels = channels
    
    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.c3(x)
        p3 = self.b3(x)
        x = self.c4(p3)
        p4 = self.b4(x)
        x = self.c5(p4)
        p5 = self.b5(x)
        return [p3, p4, p5]


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    '''
    CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    '''
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat([y1, y2], dim=1))))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        x = torch.cat([
            x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]
        ], 1)
        return self.conv(x)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
