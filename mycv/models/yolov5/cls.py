import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf

from mycv.models.yolov5.common import Conv, Focus, SPP, BottleneckCSP


class YOLOv5Cls(nn.Module):
    setting = {
        's': (0.5,  0.33),
        'm': (0.75, 0.67),
        'l': (1.0,  1.0)
    }
    def __init__(self, model='s', num_class=1000):
        super().__init__()
        w, d = YOLOv5Cls.setting[model]
        self.backbone, channels = yolov5backbone(w, d)
        self.fc = nn.Linear(channels[-1], num_class)

        self.channels  = channels
        self.num_class = num_class

    def forward(self, x):
        nB, nC, nH, nW = x.shape
        assert nH % 32 == 0 and nW % 32 == 0
        # features = []
        for i, module in enumerate(self.backbone):
            x = module(x)
            # if i in {4, 6, 10}:
            #     features.append(x)
        assert x.shape == (nB, self.channels[-1], nH//32, nW//32)
        x = tnf.adaptive_avg_pool2d(x, 1)
        assert x.shape == (nB, self.channels[-1], 1, 1)
        x = torch.flatten(x, 1)
        assert x.shape == (nB, self.channels[-1])
        x = self.fc(x)
        assert x.shape == (nB, self.num_class)
        return x


class CSP(nn.Module):
    setting = {
        's': (0.5,  0.33),
        'm': (0.75, 0.67),
        'l': (1.0,  1.0)
    }
    def __init__(self, model='s', num_class=1000):
        super().__init__()
        w, d = CSP.setting[model]
        self.backbone = CustomBackbone(w, d, in_ch=3)
        channels = self.backbone.channels
        self.fc = nn.Linear(channels[-1], num_class)

        self.channels  = channels
        self.num_class = num_class

    def forward(self, x):
        nB, nC, nH, nW = x.shape
        assert nH % 32 == 0 and nW % 32 == 0
        _, _, x = self.backbone(x)
        assert x.shape == (nB, self.channels[-1], nH//32, nW//32)
        x = tnf.adaptive_avg_pool2d(x, 1)
        assert x.shape == (nB, self.channels[-1], 1, 1)
        x = torch.flatten(x, 1)
        assert x.shape == (nB, self.channels[-1])
        x = self.fc(x)
        assert x.shape == (nB, self.num_class)
        return x


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


if __name__ == "__main__":
    from mycv.utils.torch_utils import num_params
    model = YOLOv5Cls(model='s', num_class=1000)
    print(num_params(model.backbone))
    print(num_params(model))
