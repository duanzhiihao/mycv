import torch
import torch.nn as nn

from compressai.models.priors import MeanScaleHyperprior
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.models.utils import conv, deconv


class MBT2018(nn.Module):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192, M=192, enable_bpp=False, **kwargs):
        super().__init__()

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.latent_channels = M
        self.enable_bpp = enable_bpp
        if not enable_bpp:
            return

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        # encoding
        y = self.g_a(x)

        # quantization
        if self.training:
            y_hat = self.gaussian_conditional.quantize(
                y, "noise" if self.training else "dequantize"
            )
        else:
            y_hat = torch.round(y)

        # decoding
        x_hat = self.g_s(y_hat)
        if not self.enable_bpp:
            return x_hat, None

        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        return x_hat, (y_likelihoods, z_likelihoods)

    def forward_nic(self, imgs):
        assert not self.training
        rec, probs = self.forward(imgs)
        rec = rec.clamp_(min=0, max=1)
        return rec, probs


if __name__ == '__main__':
    from mycv.paths import MYCV_DIR
    from mycv.utils.torch_utils import load_partial, num_params
    from compressai.zoo import mbt2018
    q = 5
    model = mbt2018(q, pretrained=True)
    torch.save(model.state_dict(), MYCV_DIR / f'weights/compressai/mbt2018-{q}.pt')

    # model = MBT2018(N=192, M=320, enable_bpp=True)
    # print(num_params(model))
    # load_partial(model, MYCV_DIR / 'weights/compressai/mbt2018-8.pt')
    # model = model.cuda()
    # model.eval()

    from mycv.datasets.imcoding import nic_evaluate
    results = nic_evaluate(model, input_norm=False, dataset='kodak')
    print(results)
