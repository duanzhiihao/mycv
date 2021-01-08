import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform

from models.imcoding.basic_module import ResBlock, Non_local_Block


class Enc(nn.Module):
    def __init__(self, in_channels=3, N1=192, N2=128, M=192, M1=96,
                 encode_only=False):
        super().__init__()
        self.encode_only = encode_only

        self.conv1 = nn.Conv2d(in_channels, M1, 5, 1, 2)
        self.trunk1 = nn.Sequential(
            *[ResBlock(M1, M1, 3, 1, 1) for _ in range(2)],
            nn.Conv2d(M1, 2*M1, 5, 2, 2)
        )
        self.trunk2 = nn.Sequential(
            *[ResBlock(2*M1, 2*M1, 3, 1, 1) for _ in range(3)]
        )
        self.down1 = nn.Conv2d(2*M1, M, 5, 2, 2)
        self.trunk3 = nn.Sequential(
            *[ResBlock(M, M, 3, 1, 1) for _ in range(3)],
            nn.Conv2d(M, M, 5, 2, 2)
        )
        self.trunk4 = nn.Sequential(
            *[ResBlock(M, M, 3, 1, 1) for _ in range(3)],
            nn.Conv2d(M, M, 5, 2, 2)
        )
        self.trunk5 = nn.Sequential(
            *[ResBlock(M, M, 3, 1, 1) for _ in range(3)],
        )
        self.mask2 = nn.Sequential(
            Non_local_Block(M, M // 2),
            *[ResBlock(M, M, 3, 1, 1) for _ in range(3)],
            nn.Conv2d(M, M, 1, 1, 0)
        )
        if encode_only:
            return
        # hyper
        self.trunk6 = nn.Sequential(
            *[ResBlock(M, M, 3, 1, 1) for _ in range(2)],
            nn.Conv2d(M, M, 5, 2, 2)
        )
        self.trunk7 = nn.Sequential(
            *[ResBlock(M, M, 3, 1, 1) for _ in range(2)],
            nn.Conv2d(M, M, 5, 2, 2)
        )
        self.trunk8 = nn.Sequential(
            *[ResBlock(M, M, 3, 1, 1) for _ in range(3)],
        )
        self.mask3 = nn.Sequential(
            Non_local_Block(M, M // 2),
            *[ResBlock(M, M, 3, 1, 1) for _ in range(3)],
            nn.Conv2d(M, M, 1, 1, 0)
        )
        self.conv2 = nn.Conv2d(M, N2, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.trunk1(x)
        x = self.trunk2(x)+x
        x = self.down1(x)
        x = self.trunk3(x)
        x = self.trunk4(x)
        x_comp = self.trunk5(x)*torch.sigmoid(self.mask2(x)) + x
        if self.encode_only:
            return x_comp, None
        # hyper
        x = self.trunk6(x_comp)
        x = self.trunk7(x)
        x = self.trunk8(x)*torch.sigmoid(self.mask3(x)) + x
        x_prob = self.conv2(x)
        return x_comp, x_prob


class Hyper_Dec(nn.Module):
    '''
    Hyper decoder
    '''
    def __init__(self, in_ch=128, out_ch=192, nums=[3,3,2,2]):
        super(Hyper_Dec, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,1,1)
        self.trunk1 = nn.Sequential(
            *[ResBlock(out_ch, out_ch, 3, 1, 1) for _ in range(nums[0])],
        )
        # self.mask1 = nn.Sequential(
        #     Non_local_Block(out_ch, out_ch // 2),
        #     *[ResBlock(out_ch, out_ch, 3, 1, 1) for _ in range(nums[1])],
        #     nn.Conv2d(out_ch, out_ch, 1, 1, 0)
        # )
        self.trunk2 = nn.Sequential(
            *[ResBlock(out_ch, out_ch, 3, 1, 1) for _ in range(nums[2])],
            nn.ConvTranspose2d(out_ch, out_ch, 5, 2, 2, 1)
        )
        self.trunk3 = nn.Sequential(
            *[ResBlock(out_ch, out_ch, 3, 1, 1) for _ in range(nums[3])],
            nn.ConvTranspose2d(out_ch, out_ch, 5, 2, 2, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        # x = self.trunk1(x) * torch.sigmoid(self.mask1(x)) + x
        x = self.trunk1(x)
        x = self.trunk2(x)
        x = self.trunk3(x)
        return x


class Dec(nn.Module):
    def __init__(self, input_features, N1, M, M1):
        super(Dec,self).__init__()
        self.N1 = N1
        self.M = M
        self.M1 = M1
        self.input = input_features

        self.trunk1 = nn.Sequential(
            *[ResBlock(M, M, 3, 1, 1) for _ in range(3)]
        )
        self.mask1 = nn.Sequential(
            Non_local_Block(self.M, self.M // 2),
            *[ResBlock(M, M, 3, 1, 1) for _ in range(3)],
            nn.Conv2d(self.M, self.M, 1, 1, 0)
        )
        self.up1 = nn.ConvTranspose2d(M, M, 5, 2, 2, 1)
        self.trunk2 = nn.Sequential(
            *[ResBlock(M, M, 3, 1, 1) for _ in range(3)],
            nn.ConvTranspose2d(M, M, 5, 2, 2, 1)
        )
        self.trunk3 = nn.Sequential(
            *[ResBlock(M, M, 3, 1, 1) for _ in range(3)],
            nn.ConvTranspose2d(M, 2*M1, 5, 2, 2, 1)
        )
        self.trunk4 = nn.Sequential(
            *[ResBlock(2*M1, 2*M1, 3, 1, 1) for _ in range(3)]
        )
        self.trunk5 = nn.Sequential(
            nn.ConvTranspose2d(2*M1, M1, 5, 2, 2, 1),
            *[ResBlock(M1, M1, 3, 1, 1) for _ in range(3)]
        )
        self.conv1 = nn.Conv2d(self.M1, self.input, 5, 1, 2)

    def forward(self, x):
        x = self.trunk1(x) * torch.sigmoid(self.mask1(x)) + x
        x = self.up1(x)
        x = self.trunk2(x)
        x = self.trunk3(x)
        x = self.trunk4(x)+x
        x = self.trunk5(x)
        x = self.conv1(x)
        return x


class UniverseQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        #b = np.random.uniform(-1,1)
        b = 0
        uniform_distribution = Uniform(-0.5*torch.ones(x.size())*(2**b),0.5*torch.ones(x.size())*(2**b)).sample().cuda()
        return torch.round(x+uniform_distribution)-uniform_distribution
    @staticmethod
    def backward(ctx, g):
        return g


if __name__ == "__main__":
    from tqdm import tqdm
    from models.detection import HalfBackbone
    model = Enc()
    # model = Dec()
    model.train()
    model.cuda()
    print(sum(torch.numel(p) for p in model.parameters()), 'parameters')
    
    # mem = torch.cuda.max_memory_allocated(0) / 1e9
    # print(f'{mem:.3g}G')
    # torch.cuda.reset_peak_memory_stats()

    # pbar = tqdm(range(128))
    # for _ in pbar:
    #     x = torch.rand(1,3,640,640).cuda()
    #     # x = torch.rand(1,128,40,40).cuda()

    #     # with torch.cuda.amp.autocast():
    #     x = model(x)
    #     torch.cuda.synchronize(0)
    #     mem = torch.cuda.memory_allocated(0) / 1e9
    #     pbar.set_description(f'{mem:.3g}G')

    # mem = torch.cuda.max_memory_allocated(0) / 1e9
    # print(f'{mem:.3g}G')
    # torch.cuda.reset_peak_memory_stats()
