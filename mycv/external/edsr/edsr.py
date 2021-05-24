# Copied from https://github.com/sanghyun-son/EDSR-PyTorch

import torch
import torch.nn as nn

from mycv.external.edsr import common


url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}


class EDSR(nn.Module):
    """ Enhanced Deep Residual Networks for Single Image Super-Resolution
    https://github.com/sanghyun-son/EDSR-PyTorch

    Args:
        version (str): 'baseline' or 'paper'
        scale (int): 2, 3, or 4
        conv (optional): [description]. Defaults to common.default_conv.
    """
    def __init__(self, version, scale, conv=common.default_conv):
        super().__init__()
        if version == 'baseline':
            n_resblocks = 16
            n_feats = 64
            res_scale = 1
        elif version == 'paper':
            n_resblocks = 32
            n_feats = 256
            res_scale = 0.1
        else:
            raise ValueError('invalid version name')
        n_colors = 3
        kernel_size = 3
        rgb_range = 255
 
        act = nn.ReLU(inplace=True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def sr_numpy(self, lr):
        device = self.sub_mean.bias.device
        x = torch.from_numpy(lr).float().permute(2,0,1).unsqueeze(0)
        x = x.to(device=device)
        y = self.forward(x)
        hr = y.clamp_(min=0, max=255).round_().cpu().squeeze_(0).permute(1,2,0)
        hr = hr.to(dtype=torch.uint8).numpy()
        return hr

    # def load_state_dict(self, state_dict, strict=True):
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             if isinstance(param, nn.Parameter):
    #                 param = param.data
    #             try:
    #                 own_state[name].copy_(param)
    #             except Exception:
    #                 if name.find('tail') == -1:
    #                     raise RuntimeError('While copying the parameter named {}, '
    #                                        'whose dimensions in the model are {} and '
    #                                        'whose dimensions in the checkpoint are {}.'
    #                                        .format(name, own_state[name].size(), param.size()))
    #         elif strict:
    #             if name.find('tail') == -1:
    #                 raise KeyError('unexpected key "{}" in state_dict'
    #                                .format(name))


if __name__ == '__main__':
    from mycv.paths import MYCV_DIR
    scale = 2
    model = EDSR('baseline', scale=scale)

    wpath = MYCV_DIR / f'weights/edsr/edsr_baseline_x{scale}.pt'
    weights = torch.load(wpath)
    model.load_state_dict(weights)

    model = model.cuda()
    model.eval()

    model = model.cuda()
    from tqdm import tqdm
    for _ in tqdm(range(256)):
        lr = torch.rand(1, 3, 512, 512, device='cuda:0') * 255 - 128
        rec = model(lr)
        assert rec.shape == (1, 3, 1024, 1024)

    model = model.cpu()
    from thop import profile, clever_format
    from fvcore.nn import flop_count
    lr = torch.randn(1, 3, 512, 512)
    final_count, skipped_ops = flop_count(model, (lr, )) 
    print(final_count)
    macs, params = profile(model, inputs=(lr, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)

    exit()

    from mycv.datasets.superres import sr_evaluate
    results = sr_evaluate(model, dataset='set14', scale=scale)
    print(results)
