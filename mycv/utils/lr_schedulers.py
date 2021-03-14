import math
import torch


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


def _adjust_lr_532(optimizer, cur_epoch, base_lr, total_epoch):
    assert total_epoch % 10 == 0
    if cur_epoch < round(total_epoch * 5/10):
        lrf = 1
    elif cur_epoch < round(total_epoch * 8/10):
        lrf = 0.1
    else:
        assert cur_epoch < total_epoch
        lrf = 0.01
    lr = base_lr * lrf
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
