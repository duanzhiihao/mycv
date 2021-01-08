import torch
import torch.nn as nn

from mycv.models.nic.basic_module import ResBlock, UniverseQuant


class IMCoding(nn.Module):
    """
    docstring
    """
    def __init__(self, enable_bpp=False):
        super().__init__()
        self.encoder = miniEnc(hyper=enable_bpp)
        self.decoder = miniDec()

        self.enable_bpp = enable_bpp
        if not enable_bpp:
            return
        raise NotImplementedError()
        # self.factorized_entropy_func = Entropy_bottleneck(96) # 4.1k
        # self.hyper_dec = Hyper_Dec(96, 128, nums=[2,2,1,1]) # 2.7M
        # self.p = P_Model(128, num=2) # 885k
        # self.context = Complex_context(ch=128, nums=[3,2]) # 402k

    def forward(self, x):
        assert x.dim() == 4 and x.shape[1] == 3
        assert x.shape[2] % 16 == 0 and x.shape[3] % 16 == 0

        x, xp = self.encoder(x) # 3GB
        if self.training:
            # noise = torch.rand_like(x) - 0.5
            # xq = x + noise
            xq = UniverseQuant.apply(x)
        else:
            xq = torch.round(x)

        output = self.decoder(xq) # 3GB
        if not self.enable_bpp:
            return output, (None, None)

        raise NotImplementedError()
        xq2, xp2 = self.factorized_entropy_func(xp, self.training)
        x3 = self.hyper_dec(xq2)
        hyper_dec = self.p(x3)
        xp1 = self.context(xq, hyper_dec) # 2GB
        return output, (xp1, xp2)

    def encode(self, x):
        ''' Encode only
        '''
        assert not self.training
        x, _ = self.encoder(x)
        return torch.round(x)

    def decode(self, x, ori_hw=None, return_type='tensor'):
        ''' Reconstruct the original image from the feature
        '''
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


class miniEnc(nn.Module):
    ''' mini NLAIC encoder
    '''
    def __init__(self, input_ch=3, channels=[32, 64, 128], hyper=False):
        super().__init__()
        c0, c1, c2 = channels
        self.hyper = hyper

        self.trunk = nn.Sequential(
            nn.Conv2d(input_ch, c0, 5, 2, padding=2, padding_mode='reflect'),
            # 2x
            *[ResBlock(c0) for _ in range(2)],
            nn.Conv2d(c0, c1, 5, 2, padding=2, padding_mode='reflect'),
            # 4x
            *[ResBlock(c1) for _ in range(2)],
            nn.Conv2d(c1, c2, 5, 2, padding=2, padding_mode='reflect'),
            # 8x
            *[ResBlock(c2) for _ in range(3)],
            nn.Conv2d(c2, c2, 5, 2, padding=2, padding_mode='reflect'),
            # 16x
            *[ResBlock(c2) for _ in range(3)],
        )
        if not hyper:
            return
        raise NotImplementedError()
        # hyper
        self.trunk6 = nn.Sequential(
            *[ResBlock(c2) for _ in range(2)],
            nn.Conv2d(c2, c2, 5, 2, 2, padding_mode='reflect')
        )
        self.trunk7 = nn.Sequential(
            *[ResBlock(c2) for _ in range(3)],
            nn.Conv2d(c2, c2, 5, 2, 2, padding_mode='reflect')
        )
        self.trunk8 = nn.Sequential(
            *[ResBlock(c2) for _ in range(3)],
        )
        self.conv2 = nn.Conv2d(c2, 96, 3, 1, 1, padding_mode='reflect')

    def forward(self, x):
        x_comp = self.trunk(x)
        if not self.hyper:
            return x_comp, None
        x = self.trunk6(x_comp)
        x = self.trunk7(x)
        x = self.trunk8(x)
        x_prob = self.conv2(x)
        return x_comp, x_prob


class miniDec(nn.Module):
    ''' mini NLAIC decoder
    '''
    def __init__(self, channels=[128, 64, 32, 16], input_features=3):
        super().__init__()
        c3, c2, c1, c0 = channels
        self.m1 = nn.Sequential(
            *[ResBlock(c3) for _ in range(2)]
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 5, 2, 2, 1),
            *[ResBlock(c2) for _ in range(2)]
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, 5, 2, 2, 1),
            *[ResBlock(c1) for _ in range(2)]
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(c1, c1, 5, 2, 2, 1),
            *[ResBlock(c1) for _ in range(2)]
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(c1, c0, 3, 2, 1, 1),
            *[ResBlock(c0) for _ in range(1)],
        )
        self.cv1 = nn.Conv2d(c0, input_features, 5, 1, 2, padding_mode='reflect')

        self._feat_ch = c3

    def forward(self, x):
        assert x.dim() == 4 and x.shape[1] == self._feat_ch
        x = self.m1(x)
        x = self.up1(x) # 0.09 0.07
        x = self.up2(x) # 0.39 0.34
        x = self.up3(x) # 0.96 0.86
        x = self.up4(x) # 2.12 1.91
        x = self.cv1(x) # 2.12 1.95
        return x
