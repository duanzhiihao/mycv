from mycv.utils.general import disable_multithreads
disable_multithreads()
import os
import time
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import defaultdict
import wandb
import cv2
import torch
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import LambdaLR

from mycv.datasets.cityscapes import Cityscapes, evaluate_semseg, TRAIN_COLORS
from mycv.utils.general import increment_dir
from mycv.utils.torch_utils import set_random_seeds, ModelEMA, warmup_cosine, is_parallel


def compute_metric(outputs: torch.Tensor, labels: torch.LongTensor, ignore_label=255):
    assert not outputs.requires_grad and outputs.device == labels.device
    assert outputs.dtype == labels.dtype == torch.int64
    assert outputs.dim() == 3 and outputs.shape == labels.shape

    valid = (labels != ignore_label)
    tpmask = (outputs[valid] == labels[valid])
    acc = tpmask.sum() / valid.sum()
    return acc.item()


def get_model(name, num_class):
    if name == 'psp50':
        from mycv.external.semseg import PSPNet
        model = PSPNet(num_class=num_class)
    else:
        raise ValueError()
    return model


def get_params(model, cfg):
    # different optimization setting for different layers
    names0 = ['layer0.', 'layer1.', 'layer2.', 'layer3.', 'layer4.']
    names1 = ['ppm.', 'cls.', 'aux.']
    pgb0, pgw0 = [], []
    pgb1, pgw1 = [], []
    for k, v in model.named_parameters():
        if any([msg in k for msg in names0]):
            if ('.bn' in k) or ('.bias' in k): # batchnorm or bias
                pgb0.append(v)
            else: # conv weights
                assert '.weight' in k
                pgw0.append(v)
        elif any([msg in k for msg in names1]):
            if ('.bn' in k) or ('.bias' in k): # batchnorm or bias
                pgb1.append(v)
            else: # conv weights
                assert '.weight' in k
                pgw1.append(v)
        else:
            print(k, v.shape)
            raise ValueError()

    parameters = [
        {'params': pgb0, 'lr': cfg.lr, 'weight_decay': 0.0},
        {'params': pgw0, 'lr': cfg.lr, 'weight_decay': cfg.weight_decay},
        {'params': pgb1, 'lr': cfg.lr*10, 'weight_decay': 0.0},
        {'params': pgw1, 'lr': cfg.lr*10, 'weight_decay': cfg.weight_decay}
    ]
    print('Parameter groups:', [len(pg['params']) for pg in parameters])
    return parameters


def train():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',    type=str,  default='cityscapes')
    parser.add_argument('--group',      type=str,  default='fine')
    parser.add_argument('--model',      type=str,  default='psp50')
    parser.add_argument('--resume',     type=str,  default='')
    parser.add_argument('--batch_size', type=int,  default=8)
    parser.add_argument('--amp',        type=bool, default=True)
    parser.add_argument('--ema',        type=bool, default=True)
    parser.add_argument('--epochs',     type=int,  default=120)
    parser.add_argument('--device',     type=int,  default=[0], nargs='+')
    parser.add_argument('--workers',    type=int,  default=2)
    parser.add_argument('--wbmode',     type=str,  default='disabled')
    cfg = parser.parse_args()
    # model
    cfg.img_size = 640
    cfg.input_norm = False
    cfg.aux_weight = 0.4
    # optimizer
    cfg.lr = 0.006
    cfg.momentum = 0.9
    cfg.weight_decay = 0.0001
    cfg.accum_batch_size = 16
    cfg.accum_num = max(1, round(cfg.accum_batch_size // cfg.batch_size))
    # lr scheduler
    cfg.lrf = 0.02 # min lr factor
    cfg.lr_warmup_epochs = 1
    # EMA
    cfg.ema_warmup_epochs = 16

    # check arguments
    metric: str = 'miou'
    epochs: int = cfg.epochs
    print(cfg, '\n')
    # fix random seeds for reproducibility
    set_random_seeds(1)
    torch.backends.cudnn.benchmark = True
    # device setting
    assert torch.cuda.is_available()
    for _id in cfg.device:
        print(f'Using device {_id}:', torch.cuda.get_device_properties(_id))
    device = torch.device(f'cuda:{cfg.device[0]}')
    bs_each = cfg.batch_size // len(cfg.device)
    print('Batch size on each single GPU =', bs_each, '\n')
    print(f'Gradient accmulation: {cfg.accum_num} backwards() -> one step()')
    print(f'Effective batch size: {cfg.accum_batch_size}')

    # Dataset
    print('Initializing Datasets and Dataloaders...')
    if cfg.group == 'fine':
        train_split = 'train_fine'
    else:
        raise ValueError()
    # training set
    trainset = Cityscapes(train_split, train_size=cfg.img_size, input_norm=cfg.input_norm)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size,
                    shuffle=True, num_workers=cfg.workers, pin_memory=True)
    # test set
    testloader = torch.utils.data.DataLoader(
        Cityscapes('val', input_norm=cfg.input_norm),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False
    )

    # Initialize model
    model = get_model(cfg.model, trainset.num_class)
    model = model.to(device)

    # optimizer
    parameters = get_params(model, cfg)
    optimizer = torch.optim.SGD(parameters, lr=cfg.lr, momentum=cfg.momentum)

    # AMP
    scaler = amp.GradScaler(enabled=cfg.amp)

    log_parent = Path(f'runs/{cfg.project}')
    results = defaultdict(float)
    if cfg.resume:
        # resume
        run_name = cfg.resume
        log_dir = log_parent / run_name
        assert log_dir.is_dir()
        checkpoint = torch.load(log_dir / 'last.pt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_fitness = checkpoint.get(metric, 0)
    else:
        # new experiment
        run_name = increment_dir(dir_root=log_parent, name=cfg.model)
        log_dir = log_parent / run_name # wandb logging dir
        os.makedirs(log_dir, exist_ok=False)
        print(str(model), file=open(log_dir / 'model.txt', 'w'))
        start_epoch = 0
        best_fitness = 0
    start_iter = start_epoch * len(trainloader)

    # initialize wandb
    wbrun = wandb.init(project=cfg.project, group=cfg.group, name=run_name, config=cfg,
                       dir='runs/', mode=cfg.wbmode)
    cfg = wbrun.config
    cfg.log_dir = log_dir
    cfg.wandb_id = wbrun.id
    with open(log_dir / 'wandb_id.txt', 'a') as f:
        print(wbrun.id, file=f)

    # lr scheduler
    _warmup = cfg.lr_warmup_epochs * len(trainloader)
    _total = epochs * len(trainloader)
    lr_func = lambda x: warmup_cosine(x, cfg.lrf, _warmup, _total)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_func, last_epoch=start_iter - 1)

    # Exponential moving average
    if cfg.ema:
        ema = ModelEMA(model, decay=0.9999)
        ema.updates = start_iter // cfg.accum_num  # set EMA updates
        ema.warmup = cfg.ema_warmup_epochs * len(trainloader) # set EMA warmup
    else:
        ema = None

    # DP mode
    if len(cfg.device) > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.device)

    # ======================== start training ========================
    pbar_title = ('%-10s' * 7) % (
        'Epoch', 'GPU_mem', 'lr', 'tr_loss', 'tr_acc', 'miou', 'acc',
    )
    niter = msg = None
    for epoch in range(start_epoch, epochs):
        model.train()

        epoch_loss, epoch_acc = 0.0, 0.0
        time.sleep(0.1)
        print('\n' + pbar_title) # title
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for bi, (imgs, labels) in pbar:
            niter = epoch * len(trainloader) + bi

            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            # nB, nC, nH, nW = imgs.shape

            # forward
            with amp.autocast(enabled=cfg.amp):
                outputs, main_loss, aux_loss = model(imgs, labels)
                loss = main_loss + cfg.aux_weight * aux_loss
                loss = loss / cfg.accum_num
                # loss is averaged within image, sumed over batch, and sumed over gpus
            # backward, update
            scaler.scale(loss).backward()

            if niter % cfg.accum_num == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if cfg.ema:
                    ema.update(model)
            # Scheduler
            scheduler.step()

            # logging
            cur_lr = optimizer.param_groups[0]['lr']
            acc = compute_metric(outputs.detach(), labels, trainset.ignore_label)
            epoch_loss = (epoch_loss*bi + loss.item()) / (bi+1)
            epoch_acc = (epoch_acc*bi + acc) / (bi+1)
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            msg = ('%-10s' * 2 + '%-10.4g' * 5) % (
                f'{epoch}/{epochs-1}', f'{mem:.3g}G', cur_lr, epoch_loss,
                100*epoch_acc, 100*results['miou'], 100*results['acc'],
            )
            pbar.set_description(msg)
            torch.cuda.reset_peak_memory_stats()
            # Weights & Biases logging
            if niter % 100 == 0:
                save_prediction(imgs[0], labels[0], outputs[0],
                                log_dir / 'prediction.png', cfg.input_norm)
                wbrun.log({
                    'general/lr': cur_lr,
                    'train/epoch_loss': epoch_loss,
                    'train/epoch_acc': epoch_acc,
                    'ema/n_updates': ema.updates if cfg.ema else 0,
                    'ema/decay': ema.get_decay() if cfg.ema else 0
                }, step=niter)
            # logging end
            # ----Mini batch end
        # ----Epoch end

        if epoch % 2 != 0:
            continue
        # Evaluation
        _log_dic = {'general/epoch': epoch}
        _eval_model = model.module if is_parallel(model) else model
        results = evaluate_semseg(_eval_model, testloader=testloader)
        _log_dic.update({'metric/plain_val_'+k: v for k,v in results.items()})

        if cfg.ema:
            results = evaluate_semseg(ema.ema, testloader=testloader)
            _log_dic.update({f'metric/ema_val_'+k: v for k,v in results.items()})
            # select best result among all emas
            _save_model = ema.ema
        else:
            _save_model = model

        _cur_fitness = results[metric]
        # wandb log
        wbrun.log(_log_dic, step=niter)
        # Write evaluation results
        res = msg + '||' + '%10.4g' * 1 % (_cur_fitness)
        with open(log_dir / 'results.txt', 'a') as f:
            f.write(res + '\n')
        # save last checkpoint
        checkpoint = {
            'model'     : _save_model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scaler'    : scaler.state_dict(),
            'epoch'     : epoch,
            metric      : _cur_fitness,
        }
        torch.save(checkpoint, log_dir / 'last.pt')
        # save best checkpoint
        if _cur_fitness > best_fitness:
            best_fitness = _cur_fitness
            torch.save(checkpoint, log_dir / 'best.pt')
        del checkpoint
        # ----Epoch end
    # ----Training end


def save_prediction(img: torch.Tensor, label: torch.Tensor, output: torch.Tensor,
                    save_path: str, is_normalized=False):
    assert img.dtype == torch.float and img.dim() == 3
    nB, nH, nW = img.shape
    assert label.shape == output.shape == (nH, nW)
    assert label.dtype == output.dtype == torch.int64

    img, label, output = img.cpu(), label.cpu(), output.detach().cpu()
    if is_normalized:
        img = (img * Cityscapes.input_std) + Cityscapes.input_mean

    # input image
    im = torch.clamp(img, min=0, max=1) * 255
    im = im.to(dtype=torch.uint8).permute(1, 2, 0)
    # pred-gt difference
    igmask = (label == Cityscapes.ignore_label)
    tpmask = (output == label)
    diff = torch.ones(3, nH, nW, dtype=torch.uint8) + 50
    diff[1, tpmask] = 255
    diff[0, ~tpmask] = 255
    diff[:, igmask] = 0
    diff = diff.permute(1, 2, 0)
    # ground truth
    label[igmask] = 0
    gt = TRAIN_COLORS[label]
    gt[igmask, :] = 0
    # prediction
    pred = TRAIN_COLORS[output]

    row1 = torch.cat([im, gt], dim=1)
    row2 = torch.cat([diff, pred], dim=1)
    im = torch.cat([row1, row2], dim=0)

    im = cv2.cvtColor(im.numpy(), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), im)


if __name__ == '__main__':
    print()
    train()

    # from mycv.models.cls.resnet import resnet50
    # model = resnet50(num_classes=1000)
    # weights = torch.load('weights/resnet50-19c8e357.pth')
    # model.load_state_dict(weights)
    # model = model.cuda()
    # model.eval()
    # results = imagenet_val(model, img_size=224, batch_size=64, workers=4)
    # print(results['top1'])
