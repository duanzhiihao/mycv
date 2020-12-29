import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf

from mycv.models.yolov5.common import yolov5backbone


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
        w, d = YOLOv5Cls.setting[model]
        from mycv.models.yolov5.common import CustomBackbone
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


if __name__ == "__main__":
    from mycv.utils.torch_utils import num_params
    model = YOLOv5Cls(model='s', num_class=1000)
    print(num_params(model.backbone))
    print(num_params(model))
