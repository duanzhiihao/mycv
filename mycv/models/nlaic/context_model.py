import torch
import torch.nn as nn
import torch.nn.functional as f

from mycv.models.nlaic.basic_module import conv2d, ResBlock


class MaskConv3d(nn.Conv3d):
    def __init__(self, mask_type,in_ch, out_ch, kernel_size, stride, padding):
        super(MaskConv3d, self).__init__(in_ch, out_ch, kernel_size, stride, padding,bias=True)

        self.mask_type = mask_type
        ch_out, ch_in, k, k, k = self.weight.size()
        mask = torch.zeros(ch_out, ch_in, k, k, k)
        central_id = k*k*k//2+1
        current_id = 1
        if mask_type=='A':
            for i in range(k):
                for j in range(k):
                    for t in range(k):
                        if current_id < central_id:
                            mask[:, :, i, j, t] = 1
                        else:
                            mask[:, :, i, j, t] = 0
                        current_id = current_id + 1
        if mask_type == 'B':
            for i in range(k):
                for j in range(k):
                    for t in range(k):
                        if current_id <= central_id:
                            mask[:, :, i, j, t] = 1
                        else:
                            mask[:, :, i, j, t] = 0
                        current_id = current_id + 1
        self.register_buffer('mask', mask)
    def forward(self, x):

        self.weight.data *= self.mask
        return super(MaskConv3d,self).forward(x)


class Context(nn.Module):
    def __init__(self):
        raise DeprecationWarning()
        super(Context, self).__init__()
        self.conv1 = MaskConv3d('A', 1, 24, 5, 1, 2)

        self.conv2 = nn.Sequential(nn.Conv3d(25,48,1,1,0),nn.ReLU(),nn.Conv3d(48,96,1,1,0),nn.ReLU(),
                                   nn.Conv3d(96,2,1,1,0))
        self.conv3 = conv2d(384,192,3,1,1)

        self.gaussin_entropy_func = Distribution_for_entropy2()

    def forward(self, x,hyper):
        x = torch.unsqueeze(x, dim=1)
        hyper = torch.unsqueeze(self.conv3(hyper),dim=1)
        x1 = self.conv1(x)
        output = self.conv2(torch.cat((x1,hyper),dim=1))
        p3 = self.gaussin_entropy_func(torch.squeeze(x,dim=1),output)

        return p3, output


class Maskb_resblock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride,padding):
        # raise DeprecationWarning()
        super(Maskb_resblock,self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.conv1 = MaskConv3d('B', self.in_ch, self.out_ch, self.k, self.stride
                               , self.padding)
        self.conv2 = MaskConv3d('B', self.in_ch, self.out_ch, self.k, self.stride
                               , self.padding)

    def forward(self,x):
        x1 = self.conv2(f.relu(self.conv1(x)))
        out = x+x1
        return out

class Resblock_3D(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride,padding):
        # raise DeprecationWarning()
        super(Resblock_3D,self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.conv1 = nn.Conv3d(self.in_ch, self.out_ch, self.k, self.stride,
                               self.padding)
        self.conv2 = nn.Conv3d(self.in_ch, self.out_ch, self.k, self.stride,
                               self.padding)

    def forward(self,x):
        x1 = self.conv2(f.relu(self.conv1(x)))
        out = x+x1
        return out


class Complex_context(nn.Module):
    '''
    context
    '''
    def __init__(self, ch=192, nums=[4,3]):
        # raise DeprecationWarning()
        super(Complex_context, self).__init__()
        self.hyper_to_main = conv2d(ch*2, ch, 3, 1, 1)
        self.conv1 = MaskConv3d('A', 1, 24, 5, 1, 2)
        self.resblock1 = nn.Sequential(
            *[Maskb_resblock(24, 24, 3, 1, 1) for _ in range(nums[0])]
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(25, 48, 1, 1, 0),
            *[Resblock_3D(48, 48, 1, 1, 0) for _ in range(nums[1])],
            nn.Conv3d(48, 2, 1, 1, 0)
        )
        from mycv.models.nlaic.gaussian_entropy_model import Distribution_for_entropy2_old
        self.gaussin_entropy_func = Distribution_for_entropy2_old()

    def forward(self, x, hyper):
        x = torch.unsqueeze(x, dim=1)
        hyper = torch.unsqueeze(self.hyper_to_main(hyper),dim=1)
        x1 = self.conv1(x)
        x1 = self.resblock1(x1)
        output = self.fusion(torch.cat((x1,hyper),dim=1))
        p3 = self.gaussin_entropy_func(torch.squeeze(x,dim=1),output)
        return p3


class P_Model(nn.Module):
    def __init__(self, ch, num=3):
        super(P_Model,self).__init__()
        self.context_p = nn.Sequential(
            *[ResBlock(ch) for _ in range(num)],
            conv2d(ch, 2*ch, 3, 1, 1)
        )

    def forward(self,x):
        x = self.context_p(x)
        return x


class Weighted_Gaussian(nn.Module):
    def __init__(self, M):
        super(Weighted_Gaussian, self).__init__()
        self.conv1 = MaskConv3d('A', 1, 24, 11, 1, 5)
        self.conv2 = nn.Sequential(
            nn.Conv3d(25, 48, 1, 1, 0),
            nn.ReLU(),
            nn.Conv3d(48, 96, 1, 1, 0),
            nn.ReLU(),
            nn.Conv3d(96, 9, 1, 1, 0)
        )
        self.conv3 = nn.Conv2d(M*2, M, 3, 1, 1)
        from mycv.models.nlaic.gaussian_entropy_model import Distribution_for_entropy2
        self.gaussin_entropy_func = Distribution_for_entropy2()

    def forward(self, x, hyper):
        x = torch.unsqueeze(x, dim=1)
        hyper = torch.unsqueeze(self.conv3(hyper), dim=1)
        x1 = self.conv1(x)
        output = self.conv2(torch.cat((x1, hyper), dim=1))
        p3 = self.gaussin_entropy_func(torch.squeeze(x, dim=1), output)
        return p3 #, output
