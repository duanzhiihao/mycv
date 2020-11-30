from pathlib import Path
from copy import deepcopy
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)
    random.seed(random_seed)


def load_partial(model, weights, verbose=True):
    '''
    Load weights that have the same name
    '''
    if isinstance(weights, (str, Path)):
        print(f'Loading {type(model).__name__}() weights from {weights}...')
        external_state = torch.load(weights)
    else:
        external_state = weights
    if 'model' in external_state:
        external_state = external_state['model']

    self_state = model.state_dict()
    new_dic = dict()
    for k,v in external_state.items():
        if k in self_state and self_state[k].shape == v.shape:
            new_dic[k] = v
    model.load_state_dict(new_dic, strict=False)
    if verbose:
        print(f'{type(model).__name__}: {len(self_state)} layers,',
              f'saved: {len(external_state)} layers,',
              f'overlap & loaded: {len(new_dic)} layers')


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/

    # init
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def is_parallel(model):
    '''
    Check if the model is DP or DDP
    '''
    flag = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    return flag


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.final_decay = decay  # final decay
        # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
    
    def get_decay(self):
        decay = self.final_decay * (1 - np.exp(-self.updates / 2000))
        return decay

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.get_decay()
            # model state_dict
            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        raise NotImplementedError()
        # copy_attr(self.ema, model, include, exclude)
