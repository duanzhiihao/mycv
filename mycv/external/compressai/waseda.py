# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as tnf

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)


class Cheng2020Anchor(nn.Module):
    """Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """
    def __init__(self, N=192, enable_bpp=False):
        super().__init__()
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )
        self.enable_bpp = enable_bpp
        self.latent_channels = N
        if not enable_bpp:
            return
        raise NotImplementedError()

    def forward(self, x):
        y = self.g_a(x)
        if self.training:
            raise NotImplementedError()
        else:
            y_hat = torch.round(y)
        x_hat = self.g_s(y_hat)

        if not self.enable_bpp:
            return x_hat, None
        raise NotImplementedError()
        # z = self.h_a(y)
        # z_hat, z_likelihoods = self.entropy_bottleneck(z)
        # params = self.h_s(z_hat)
        # ctx_params = self.context_prediction(y_hat)
        # gaussian_params = self.entropy_parameters(
        #     torch.cat((params, ctx_params), dim=1)
        # )
        # scales_hat, means_hat = gaussian_params.chunk(2, 1)
        # _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        # return x_hat, (y_likelihoods, z_likelihoods)


if __name__ == '__main__':
    from mycv.paths import MYCV_DIR
    from mycv.utils.torch_utils import load_partial

    # from compressai.zoo import cheng2020_anchor
    # model = cheng2020_anchor(4, metric='mse', pretrained=True)
    # torch.save(model.state_dict(), MYCV_DIR / f'weights/compressai/cheng2020-anchor-4.pt')

    model = Cheng2020Anchor(N=192, enable_bpp=False)
    weights_path = MYCV_DIR / f'weights/compressai/cheng2020-anchor-4.pt'
    load_partial(model, weights_path)
    model = model.cuda()
    model.eval()

    from mycv.datasets.imcoding import nic_evaluate
    results = nic_evaluate(model, input_norm=False, dataset='kodak')
    print(results)
