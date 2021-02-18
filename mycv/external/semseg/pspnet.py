import torch
from torch import nn
import torch.nn.functional as F

from mycv.external.semseg._resnet import resnet50

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    """ PSPNet: https://arxiv.org/abs/1612.01105
    Copied from https://github.com/hszhao/semseg
    """
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, num_class=19):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert num_class > 1
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        if layers == 50:
            resnet = resnet50()
            from mycv.paths import MYCV_DIR
            wpath = MYCV_DIR / 'weights/semseg_resnet50_v2.pth'
            resnet.load_state_dict(torch.load(wpath))
        else:
            raise NotImplementedError()
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.conv2, resnet.bn2, resnet.relu,
            resnet.conv3, resnet.bn3, resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for k, m in self.layer3.named_modules():
            if 'conv2' in k:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in k:
                m.stride = (1, 1)
        for k, m in self.layer4.named_modules():
            if 'conv2' in k:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in k:
                m.stride = (1, 1)

        fea_dim = 2048
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
        fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.aux = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(256, num_class, kernel_size=1)
        )

    def forward(self, x, y=None):
        assert (x.shape[2]-1) % 8 == 0 and (x.shape[3]-1) % 8 == 0
        h, w = x.shape[2:4]

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        x = self.ppm(x)
        x = self.cls(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return torch.max(x.detach(), dim=1)[1], main_loss, aux_loss
        else:
            return x


if __name__ == '__main__':
    model = PSPNet()
    model = model.cuda()

    # x = torch.rand(4, 3, 713, 713).cuda()
    # p = model(x)

    from mycv.paths import MYCV_DIR
    from mycv.datasets.cityscapes import evaluate_semseg
    weights = torch.load(MYCV_DIR / 'weights/psp50_epoch_200.pt')
    model.load_state_dict(weights)
    model.eval()

    results = evaluate_semseg(model)
    print(results)
