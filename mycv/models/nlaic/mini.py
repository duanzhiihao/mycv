import torch
import torch.nn as nn

from mycv.models.nic.basic_module import conv2d, ResBlock, UniverseQuant
from mycv.models.nic.factorized_entropy_model import Entropy_bottleneck
from mycv.models.nic.context_model import P_Model, Complex_context


class MiniNIC(nn.Module):
    """
    10.4M parameters
    """
    def __init__(self, enable_bpp=False):
        super().__init__()
        self.encoder = miniEnc(hyper=enable_bpp) # 5.91M
        self.decoder = miniDec() # 2.7M

        self.enable_bpp = enable_bpp
        if not enable_bpp:
            return
        self.factorized_entropy_func = Entropy_bottleneck(96) # 4.1k
        self.hyper_dec = Hyper_Dec(96, 192, nums=[2,2,1,1]) # 2.1M
        self.p = P_Model(192, num=2) # 885k
        self.context = Complex_context(ch=192, nums=[3,2]) # 402k

    def forward(self, x):
        assert x.dim() == 4 and x.shape[1] == 3
        assert x.shape[2] % 16 == 0 and x.shape[3] % 16 == 0

        x, xp = self.encoder(x) # 3GB
        if self.training:
            xq = UniverseQuant.apply(x)
        else:
            xq = torch.round(x)

        output = self.decoder(xq) # 3GB
        if not self.enable_bpp:
            return output, None

        xq2, xp2 = self.factorized_entropy_func(xp)
        x3 = self.hyper_dec(xq2)
        hyper_dec = self.p(x3)
        xp1 = self.context(xq, hyper_dec) # 2GB
        return output, (xp1, xp2)

    def encode(self, x):
        """ Encode only
        """
        assert not self.training
        x, _ = self.encoder(x)
        return torch.round(x)

    def decode(self, x, ori_hw=None, return_type='tensor'):
        """ Reconstruct the original image from the feature
        """
        assert not self.training
        output = self.decoder(x)
        if ori_hw is not None:
            h, w = ori_hw
            output = output[:, :, 0:h, 0:w]
        output.clamp_(min=0, max=1)
        if return_type == 'tensor':
            return output

        assert return_type == 'array'
        if output.shape[0] == 1:
            img_rec = (output * 255).cpu().squeeze(0).permute(1,2,0)
            img_rec = img_rec.to(dtype=torch.uint8).numpy()
            return img_rec
        else:
            raise NotImplementedError()

    def forward_nic(self, imgs):
        assert not self.training
        rec, probs = self.forward(imgs)
        return rec, probs


class miniEnc(nn.Module):
    """ mini NLAIC encoder
    """
    def __init__(self, input_ch=3, channels=[32, 64, 128, 192], hyper=False):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.hyper = hyper

        self.trunk = nn.Sequential(
            conv2d(input_ch, c0, 5, 2, padding=2),
            *[ResBlock(c0) for _ in range(2)],
            # 2x
            conv2d(c0, c1, 5, 2, padding=2),
            *[ResBlock(c1) for _ in range(3)],
            # 4x
            conv2d(c1, c2, 5, 2, padding=2),
            *[ResBlock(c2) for _ in range(4)],
            # 8x
            conv2d(c2, c3, 5, 2, padding=2),
            *[ResBlock(c3) for _ in range(5)],
            # 16x
        )
        if not hyper:
            return
        # hyper
        self.trunk6 = nn.Sequential(
            *[ResBlock(c3) for _ in range(2)],
            conv2d(c3, c3, 5, 2, 2)
        )
        self.trunk7 = nn.Sequential(
            *[ResBlock(c3) for _ in range(2)],
            conv2d(c3, c3, 5, 2, 2)
        )
        self.trunk8 = nn.Sequential(
            *[ResBlock(c3) for _ in range(2)],
        )
        self.conv2 = conv2d(c3, 96, 3, 1, 1)

    def forward(self, x):
        x_comp = self.trunk(x)
        if not self.hyper:
            return x_comp, None
        x = self.trunk6(x_comp)
        x = self.trunk7(x)
        x = self.trunk8(x) + x
        x_prob = self.conv2(x)
        return x_comp, x_prob


class miniDec(nn.Module):
    """ mini NLAIC decoder
    """
    def __init__(self, channels=[192,160,128,96,64], input_features=3):
        super().__init__()
        c4, c3, c2, c1, c0 = channels
        # self.m1 = nn.Sequential(
        #     *[ResBlock(c4) for _ in range(3)]
        # )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, 5, 2, 2, 1),
            *[ResBlock(c3) for _ in range(3)]
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 5, 2, 2, 1),
            *[ResBlock(c2) for _ in range(3)]
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, 5, 2, 2, 1),
            *[ResBlock(c1) for _ in range(3)]
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(c1, c0, 3, 2, 1, 1),
            *[ResBlock(c0) for _ in range(3)],
        )
        self.cv1 = conv2d(c0, input_features, 5, 1, 2)

        self._feat_ch = c4

    def forward(self, x):
        assert x.dim() == 4 and x.shape[1] == self._feat_ch
        # x = self.m1(x)
        x = self.up1(x) # 0.09 0.07
        x = self.up2(x) # 0.39 0.34
        x = self.up3(x) # 0.96 0.86
        x = self.up4(x) # 2.12 1.91
        x = self.cv1(x) # 2.12 1.95
        return x


class Hyper_Dec(nn.Module):
    '''
    Hyper decoder
    '''
    def __init__(self, in_ch=128, out_ch=192, nums=[3,3,2,2]):
        super(Hyper_Dec, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,1,1)
        self.trunk1 = nn.Sequential(
            *[ResBlock(out_ch) for _ in range(nums[0])],
        )
        # self.mask1 = nn.Sequential(
        #     Non_local_Block(out_ch, out_ch // 2),
        #     *[ResBlock(out_ch) for _ in range(nums[1])],
        #     nn.Conv2d(out_ch, out_ch, 1, 1, 0)
        # )
        self.trunk2 = nn.Sequential(
            *[ResBlock(out_ch) for _ in range(nums[2])],
            nn.ConvTranspose2d(out_ch, out_ch, 5, 2, 2, 1)
        )
        self.trunk3 = nn.Sequential(
            *[ResBlock(out_ch) for _ in range(nums[3])],
            nn.ConvTranspose2d(out_ch, out_ch, 5, 2, 2, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        # x = self.trunk1(x) * torch.sigmoid(self.mask1(x)) + x
        x = self.trunk1(x)
        x = self.trunk2(x)
        x = self.trunk3(x)
        return x


if __name__ == "__main__":
    from mycv.paths import MYCV_DIR
    from mycv.utils.torch_utils import num_params
    model = MiniNIC(enable_bpp=False)
    model.load_state_dict(torch.load(MYCV_DIR / 'weights/miniMSE.pt')['model'])
    # checkpoint = torch.load('C:/Projects/yolov5/runs/nic/mini9_v1/weights/last.pt')
    model.eval()
    model = model.cuda()

    from mycv.datasets.imcoding import kodak_val
    results = kodak_val(model, input_norm=False)
    print(results)
