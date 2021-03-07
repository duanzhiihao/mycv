import torch
import torch.nn as nn

from mycv.models.nic.basic_module import conv2d, ResBlock, Non_local_Block, UniverseQuant
from mycv.models.nic.factorized_entropy_model import Entropy_bottleneck
from mycv.models.nic.context_model import P_Model, Weighted_Gaussian


class NLAIC(nn.Module):
    '''
    Non-local Attention Optimized Deep Image Compression:
    https://arxiv.org/abs/1904.09757

    N1=192, N2=128, M=192, M1=96: 48.5M parameters
    '''
    def __init__(self, enable_bpp=True):
        super().__init__()
        input_channels = 3
        N1, N2, M, M1 = 192, 128, 192, 96

        self.encoder = Enc(input_channels, N1, N2, M, M1,
                           enable_hyper=enable_bpp) # 22.5M
        self.decoder = Dec(input_channels, N1, M, M1) # 13.8M
        self.enable_bpp = enable_bpp
        if not enable_bpp:
            return

        self.factorized_entropy_func = Entropy_bottleneck(N2) # 5.5k
        self.hyper_dec = Hyper_Dec(N2, M) # 8.8M
        self.p = P_Model(M) # 2.6M
        self.context = Weighted_Gaussian(M) # 806k

    def forward(self, x):
        x, xhyp = self.encoder(x) # 3GB
        if self.training:
            xq = UniverseQuant.apply(x)
        else:
            xq = torch.round(x)
        rec = self.decoder(xq) # 3GB

        if self.enable_bpp:
            xq2, xp2 = self.factorized_entropy_func(xhyp)
            x3 = self.hyper_dec(xq2)
            hyper_dec = self.p(x3)
            xp3 = self.context(xq, hyper_dec) # 2GB
            return rec, (xp2, xp3)
        else:
            return rec, None

    def encode(self, x):
        '''
        Encode only
        '''
        assert not self.training
        x, _ = self.encoder(x)
        return torch.round(x)

    def decode(self, x, ori_hw=None):
        '''
        Reconstruct the original image from the feature
        '''
        assert not self.training
        x_rec = self.decoder(x)
        if ori_hw is not None:
            h, w = ori_hw
            x_rec = x_rec[:, :, 0:h, 0:w]
        x_rec.clamp_(min=0, max=1)

        if x_rec.shape[0] == 1:
            img_rec = (x_rec * 255).cpu().squeeze(0).permute(1,2,0)
            img_rec = img_rec.to(dtype=torch.uint8).numpy()
            return img_rec
        else:
            raise NotImplementedError()

    def forward_nic(self, imgs):
        assert not self.training
        rec, probs = self.forward(imgs)
        rec = rec.clamp_(min=0, max=1)
        return rec, probs


class Enc(nn.Module):
    def __init__(self, in_channels=3, N1=192, N2=128, M=192, M1=96,
                 enable_hyper=True):
        super().__init__()
        self.conv1 = conv2d(in_channels, M1, 5, 1, 2)
        self.trunk1 = nn.Sequential(
            *[ResBlock(M1) for _ in range(2)],
            conv2d(M1, 2*M1, 5, 2, 2)
        )
        self.trunk2 = nn.Sequential(
            *[ResBlock(2*M1) for _ in range(3)]
        )
        self.down1 = conv2d(2*M1, M, 5, 2, 2)
        self.trunk3 = nn.Sequential(
            *[ResBlock(M) for _ in range(3)],
            conv2d(M, M, 5, 2, 2)
        )
        self.trunk4 = nn.Sequential(
            *[ResBlock(M) for _ in range(3)],
            conv2d(M, M, 5, 2, 2)
        )
        self.trunk5 = nn.Sequential(
            *[ResBlock(M) for _ in range(3)],
        )
        self.mask2 = nn.Sequential(
            Non_local_Block(M, M // 2),
            *[ResBlock(M) for _ in range(3)],
            conv2d(M, M, 1, 1, 0)
        )
        self.enable_hyper = enable_hyper
        if not enable_hyper:
            return

        # hyper
        self.trunk6 = nn.Sequential(
            *[ResBlock(M) for _ in range(2)],
            conv2d(M, M, 5, 2, 2)
        )
        self.trunk7 = nn.Sequential(
            *[ResBlock(M) for _ in range(2)],
            conv2d(M, M, 5, 2, 2)
        )
        self.trunk8 = nn.Sequential(
            *[ResBlock(M) for _ in range(3)],
        )
        self.mask3 = nn.Sequential(
            Non_local_Block(M, M // 2),
            *[ResBlock(M) for _ in range(3)],
            conv2d(M, M, 1, 1, 0)
        )
        self.conv2 = conv2d(M, N2, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.trunk1(x)
        x = self.trunk2(x) + x
        x = self.down1(x)
        x = self.trunk3(x)
        x = self.trunk4(x)
        x_comp = self.trunk5(x) * torch.sigmoid(self.mask2(x)) + x
        if not self.enable_hyper:
            return x_comp, None
        # hyper
        x = self.trunk6(x_comp)
        x = self.trunk7(x)
        x = self.trunk8(x) * torch.sigmoid(self.mask3(x)) + x
        x_prob = self.conv2(x)
        return x_comp, x_prob


class Hyper_Dec(nn.Module):
    '''
    Hyper decoder
    '''
    def __init__(self, in_ch=128, out_ch=192, nums=[3,3,2,2]):
        super(Hyper_Dec, self).__init__()
        self.conv1 = conv2d(in_ch, out_ch, 3, 1, 1)
        self.trunk1 = nn.Sequential(
            *[ResBlock(out_ch) for _ in range(nums[0])],
        )
        self.mask1 = nn.Sequential(
            Non_local_Block(out_ch, out_ch // 2),
            *[ResBlock(out_ch) for _ in range(nums[1])],
            conv2d(out_ch, out_ch, 1, 1, 0)
        )
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
        x = self.trunk1(x) * torch.sigmoid(self.mask1(x)) + x
        x = self.trunk2(x)
        x = self.trunk3(x)
        return x


class Dec(nn.Module):
    def __init__(self, input_features=3, N1=192, M=192, M1=96):
        super(Dec,self).__init__()
        self.N1 = N1
        self.M = M
        self.M1 = M1
        self.input = input_features

        self.trunk1 = nn.Sequential(
            *[ResBlock(M) for _ in range(3)]
        )
        self.mask1 = nn.Sequential(
            Non_local_Block(self.M, self.M // 2),
            *[ResBlock(M) for _ in range(3)],
            conv2d(self.M, self.M, 1, 1, 0)
        )
        self.up1 = nn.ConvTranspose2d(M, M, 5, 2, 2, 1)
        self.trunk2 = nn.Sequential(
            *[ResBlock(M) for _ in range(3)],
            nn.ConvTranspose2d(M, M, 5, 2, 2, 1)
        )
        self.trunk3 = nn.Sequential(
            *[ResBlock(M) for _ in range(3)],
            nn.ConvTranspose2d(M, 2*M1, 5, 2, 2, 1)
        )
        self.trunk4 = nn.Sequential(
            *[ResBlock(2*M1) for _ in range(3)]
        )
        self.trunk5 = nn.Sequential(
            nn.ConvTranspose2d(2*M1, M1, 5, 2, 2, 1),
            *[ResBlock(M1) for _ in range(3)]
        )
        self.conv1 = conv2d(self.M1, self.input, 5, 1, 2)

    def forward(self, x):
        x = self.trunk1(x) * torch.sigmoid(self.mask1(x)) + x
        x = self.up1(x)
        x = self.trunk2(x)
        x = self.trunk3(x)
        x = self.trunk4(x) + x
        x = self.trunk5(x)
        x = self.conv1(x)
        return x


if __name__ == "__main__":
    from mycv.paths import MYCV_DIR
    from mycv.utils.torch_utils import load_partial, num_params
    model = NLAIC(enable_bpp=True)
    print(num_params(model))
    # load_partial(model, MYCV_DIR / 'weights/nlaic/msssim4.pkl')
    # load_partial(model.context, MYCV_DIR / 'weights/nlaic/msssim4p.pkl')
    # torch.save(model.state_dict(), MYCV_DIR / 'weights/nlaic/nlaic_ms4_2.pt')
    # exit()
    load_partial(model, MYCV_DIR / 'weights/nlaic/nlaic_ms4_2.pt')
    # load_partial(model, MYCV_DIR / 'weights/nlaic/nlaic_mse200_2.pt')
    model = model.cuda()
    model.eval()

    from mycv.datasets.imcoding import nic_evaluate
    # results = nic_evaluate(model, input_norm=False, dataset='kodak')
    results = nic_evaluate(model, input_norm=False, dataset='imagenet')
    print(results)
