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
    def __init__(self, version, scale):
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
 
        # url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        # if url_name in url:
        #     self.url = url[url_name]
        # else:
        #     self.url = None

        act = nn.ReLU(inplace=True)
        conv=common.default_conv
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
        # register modules
        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)
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
        sr = y.clamp_(min=0, max=255).round_().cpu().squeeze_(0).permute(1,2,0)
        sr = sr.to(dtype=torch.uint8).numpy()
        return sr


if __name__ == '__main__':
    from mycv.paths import MYCV_DIR
    scale = 4
    model = EDSR('baseline', scale=scale)

    wpath = MYCV_DIR / f'weights/edsr/edsr_baseline_x{scale}.pt'
    weights = torch.load(wpath)
    model.load_state_dict(weights)

    model = model.cuda()
    model.eval()

    # study impulse response
    import matplotlib.pyplot as plt
    # lr = torch.zeros(1,3,5,5, device='cuda:0')
    # rec = model(lr)
    # model = model.cuda()
    for ch in range(64):
        latent = torch.zeros(1, 64, 16, 16, device='cuda:0')
        latent[0, ch, 7, 7] = 1000
        with torch.no_grad():
            rec1 = model.tail(latent)
            rec1 = model.add_mean(rec1)
        latent[0, ch, 7, 7] = -1000
        with torch.no_grad():
            rec2 = model.tail(latent)
            rec2 = model.add_mean(rec2)
        rec = torch.cat([rec1, rec2], dim=3)
        rec = rec.clamp_(min=0, max=255).round_().detach().cpu().squeeze_(0).permute(1,2,0)
        rec = rec.to(dtype=torch.uint8).numpy()
        plt.imshow(rec); plt.show()
    exit()

    # model = model.cuda()
    # from tqdm import tqdm
    # for _ in tqdm(range(256)):
    #     lr = torch.rand(1, 3, 512, 512, device='cuda:0') * 255 - 128
    #     rec = model(lr)
    #     assert rec.shape == (1, 3, 1024, 1024)

    # model = model.cpu()
    # from thop import profile, clever_format
    # from fvcore.nn import flop_count
    # lr = torch.randn(1, 3, 512, 512)
    # final_count, skipped_ops = flop_count(model, (lr, )) 
    # print(final_count)
    # macs, params = profile(model, inputs=(lr, ))
    # macs, params = clever_format([macs, params], "%.3f")
    # print(macs, params)
    # exit()

    from mycv.datasets.superres import sr_evaluate
    results = sr_evaluate(model, dataset='div2k_val', scale=scale)
    print(results)
