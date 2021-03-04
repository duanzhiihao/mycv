from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
import math
import random
import numpy as np
import torch
import torch.nn as nn


def set_random_seeds(random_seed=1):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def num_params(model: nn.Module):
    """ Get the number of parameters of a model.

    Args:
        model (nn.Module): pytorch model
    """    
    num = sum([p.numel() for p in model.parameters()])
    return num


def summary_weights(state_dict: OrderedDict, save_path='model.txt'):
    if not isinstance(state_dict, OrderedDict):
        print('Warning: state_dict is not a OrderedDict. keys may not be ordered.')
    if Path(save_path).exists():
        print(f'Warning: overwriting {save_path}')
    with open(save_path, 'w') as f:
        for k, v in state_dict.items():
            line = f'{k:<48s}{v.shape}'
            print(line, file=f)


def load_partial(model, weights, verbose=True):
    ''' Load weights that have the same name
    
    Args:
        model (torch.nn.Module): model
        weights (str or dict): weights
        verbose (bool, optional): print if True. Defaults to True.
    '''
    if isinstance(weights, (str, Path)):
        if verbose:
            print(f'Loading {type(model).__name__}() weights from {weights}...')
        external_state = torch.load(weights)
    else:
        external_state = weights
    if 'model' in external_state:
        external_state = external_state['model']
    assert isinstance(external_state, (dict, OrderedDict))

    self_state = model.state_dict()
    new_dic = OrderedDict()
    for k,v in external_state.items():
        if k in self_state and self_state[k].shape == v.shape:
            new_dic[k] = v
        else:
            debug = 1
    model.load_state_dict(new_dic, strict=False)
    def _num(dic_):
        return sum([p.numel() for k,p in dic_.items()])
    if verbose:
        print(f'{type(model).__name__}: {len(self_state)} layers,',
              f'saved: {len(external_state)} layers,',
              f'overlap & loaded: {len(new_dic)} layers')


def load_partial_optimizer(optimizer, state, verbose=True):
    raise NotImplementedError()
    groups = self.param_groups
    saved_groups = state_dict['param_groups']
    # idmap = 
    debug = 1


def rename_weights(weights, old, new, verbose=True):
    """ replace old with new

    Args:
        weights (str or dict): weights
        verbose (bool, optional): print if True. Defaults to True.
    """
    if isinstance(weights, (str, Path)):
        if verbose:
            print(f'Loading weights from {weights}...')
        weights = torch.load(weights)
    if 'model' in weights:
        weights = weights['model']
    assert isinstance(weights, (dict, OrderedDict))

    count = 0
    new_dic = OrderedDict()
    for k,v in weights.items():
        if old in k:
            count += 1
        k = k.replace(old, new)
        new_dic[k] = v
    if verbose:
        print(f"Total {len(weights)}, renamed {count} '{old}' to '{new}'")
    return new_dic


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


def warmup_cosine(n, min_lrf, warmup_iter, total_iter):
    if n < warmup_iter:
        factor = n / warmup_iter
    else:
        _cur = n - warmup_iter + 1
        factor = min_lrf + 0.5 * (1 - min_lrf) * (1 + math.cos(_cur * math.pi / total_iter))
    return factor


def adjust_lr_threestep(optimizer, cur_epoch, base_lr, total_epoch):
    """ Sets the learning rate to the initial LR decayed by 10 every total/3 epochs

    Args:
        optimizer (torch.optim.Optimizer): optimizer
        cur_epoch (int): current epoch
        base_lr (float): base learning rate
        total_epoch (int): total epoch
    """
    assert total_epoch >= 3
    period = math.ceil(total_epoch / 3)
    lr = base_lr * (0.1 ** (cur_epoch // period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

    Args:
        decay: final decay when updates -> infinity
        updates: initial num. of updates
        warmup: num. of updates to reach 0.632 * decay
    """
    def __init__(self, model, decay=0.99, updates=0, warmup=2000):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.warmup = warmup
        self.final_decay = decay  # final decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def get_decay(self):
        # decay exponential ramp (to help early epochs)
        decay = self.final_decay * (1 - np.exp(-self.updates / self.warmup))
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
