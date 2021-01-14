from mycv.utils.general import disable_multithreads
disable_multithreads()
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import LambdaLR
import wandb

from mycv.utils.general import increment_dir
from mycv.utils.torch_utils import set_random_seeds, ModelEMA, warmup_cosine, is_parallel
from mycv.datasets.imagenet import ImageNetCls, imagenet_val


def cal_acc(p: torch.Tensor, labels: torch.LongTensor):
    assert not p.requires_grad and p.device == labels.device
    assert p.dim() == 2 and p.shape[0] == labels.shape[0]
    _, p_cls = torch.max(p, dim=1)
    tp = (p_cls == labels)
    acc = tp.sum() / len(tp)
    return acc


def train():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',    type=str,  default='imagenet')
    parser.add_argument('--group',      type=str,  default='mini200')
    parser.add_argument('--model',      type=str,  default='res50')
    parser.add_argument('--resume',     type=str,  default='')
    parser.add_argument('--batch_size', type=int,  default=128)
    parser.add_argument('--amp',        type=bool, default=True)
    parser.add_argument('--ema',        type=bool, default=True)
    parser.add_argument('--epochs',     type=int,  default=100)
    parser.add_argument('--device',     type=int,  default=[0], nargs='+')
    parser.add_argument('--workers',    type=int,  default=4)
    # parser.add_argument('--dryrun',   type=bool, default=True)
    parser.add_argument('--dryrun',     action='store_true')
    cfg = parser.parse_args()
    # model
    cfg.img_size = 224
    cfg.input_norm = False
    # optimizer
    cfg.lr = 0.01
    cfg.momentum = 0.9
    cfg.weight_decay = 0.0001
    cfg.nesterov = False
    # lr scheduler
    cfg.lrf = 0.2 # min lr factor
    cfg.lr_warmup_epochs = 1
    # EMA
    cfg.ema_warmup_epochs = 4

    # check arguments
    metric: str = 'top1_real'
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

    # Dataset
    print('Initializing Datasets and Dataloaders...')
    if cfg.group == 'default':
        train_split = 'train'
        val_split = 'val'
        cfg.num_class = 1000
    elif cfg.group == 'mini200':
        train_split = 'train200_600'
        val_split = 'val200_600'
        cfg.num_class = 200
    else:
        raise ValueError()
    # training set
    trainset = ImageNetCls(train_split, img_size=cfg.img_size, input_norm=cfg.input_norm)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size,
                    shuffle=True, num_workers=cfg.workers, pin_memory=True)
    # test set
    testloader = torch.utils.data.DataLoader(
        ImageNetCls(split=val_split, img_size=cfg.img_size, input_norm=cfg.input_norm),
        batch_size=bs_each//2, shuffle=False, num_workers=cfg.workers//2,
        pin_memory=True, drop_last=False
    )

    # Initialize model
    model = get_model(cfg.model, cfg.num_class)
    model = model.to(device)

    # loss function
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

    # different optimization setting for different layers
    pgb, pgw = [], []
    for k, v in model.named_parameters():
        if ('.bn' in k) or ('.bias' in k): # batchnorm or bias
            pgb.append(v)
        else: # conv weights
            assert '.weight' in k
            pgw.append(v)
    parameters = [
        {'params': pgb, 'lr': cfg.lr, 'weight_decay': 0.0},
        {'params': pgw, 'lr': cfg.lr, 'weight_decay': cfg.weight_decay}
    ]
    print('Parameter groups:', [len(pg['params']) for pg in parameters])
    del pgb, pgw

    # optimizer
    optimizer = torch.optim.SGD(parameters, lr=cfg.lr,
                                momentum=cfg.momentum, nesterov=cfg.nesterov)
    # optimizer = torch.optim.Adam(parameters, lr=cfg.lr)

    # AMP
    scaler = amp.GradScaler(enabled=cfg.amp)

    log_parent = Path(f'runs/{cfg.project}')
    wb_id = None
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
        wb_id = open(log_dir / 'wandb_id.txt', 'r').read()
    else:
        # new experiment
        run_name = increment_dir(dir_root=log_parent, name=cfg.model)
        log_dir = log_parent / run_name # wandb logging dir
        os.makedirs(log_dir, exist_ok=False)
        print(str(model), file=open(log_dir / 'model.txt', 'w'))
        start_epoch = 0
        best_fitness = 0

    # initialize wandb
    if cfg.dryrun:
        os.environ["WANDB_MODE"] = "dryrun"
    wbrun = wandb.init(project=cfg.project, group=cfg.group, name=run_name,
                        config=cfg, dir='runs/', resume='allow', id=wb_id)
    cfg = wbrun.config
    cfg.log_dir = log_dir
    cfg.wandb_id = wbrun.id
    if not (log_dir / 'wandb_id.txt').exists():
        with open(log_dir / 'wandb_id.txt', 'w') as f:
            f.write(wbrun.id)

    # lr scheduler
    _warmup = cfg.lr_warmup_epochs * len(trainloader)
    _total = epochs * len(trainloader)
    lr_func = lambda x: warmup_cosine(x, cfg.lrf, _warmup, _total)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_func, last_epoch=start_epoch - 1)

    # Exponential moving average
    if cfg.ema:
        ema = ModelEMA(model, decay=0.9999)
        ema.updates = start_epoch * len(trainloader)  # set EMA updates
        ema.warmup = cfg.ema_warmup_epochs * len(trainloader) # set EMA warmup
    else:
        ema = None

    # DP mode
    if len(cfg.device) > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.device)

    # ======================== start training ========================
    pbar_title = ('%-10s' * 7) % (
        'Epoch', 'GPU_mem', 'lr', 'tr_loss', 'tr_acc', 'top1_real', 'top1_old'
    )
    niter = s = None
    for epoch in range(start_epoch, epochs):
        model.train()

        train_loss, train_acc = 0.0, 0.0
        print('\n' + pbar_title) # title
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for i, (imgs, labels) in pbar:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            nB, nC, nH, nW = imgs.shape

            # forward
            with amp.autocast(enabled=cfg.amp):
                p = model(imgs)
                loss = loss_func(p, labels) * nB
                # loss is averaged within image, sumed over batch, and sumed over gpus
            # backward, update
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if cfg.ema:
                ema.update(model)
            # Scheduler
            scheduler.step()

            # logging
            niter = epoch * len(trainloader) + i
            cur_lr = optimizer.param_groups[0]['lr']
            loss = loss.detach().cpu().item()
            acc = cal_acc(p.detach(), labels)
            train_loss = (train_loss*i + loss/nB) / (i+1)
            train_acc = (train_acc*i + acc) / (i+1)
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            s = ('%-10s' * 2 + '%-10.4g' * 5) % (
                f'{epoch}/{epochs-1}', f'{mem:.3g}G', cur_lr, train_loss,
                100*train_acc, 100*results['top1_real'], 100*results['top1_old']
            )
            pbar.set_description(s)
            torch.cuda.reset_peak_memory_stats()
            # Weights & Biases logging
            if niter % 100 == 0:
                wbrun.log({
                    'general/lr': cur_lr,
                    'loss/train_loss': train_loss,
                    'metric/train_acc': train_acc,
                    'ema/n_updates': ema.updates if cfg.ema else 0,
                    'ema0/decay': ema.get_decay() if cfg.ema else 0
                }, step=niter)
            # logging end
            # ----Mini batch end
        # ----Epoch end

        # Evaluation
        _log_dic = {'general/epoch': epoch}
        _eval_model = model.module if is_parallel(model) else model
        results = imagenet_val(_eval_model, split=val_split, testloader=testloader)
        _log_dic.update({'metric/plain_val_'+k: v for k,v in results.items()})

        if cfg.ema:
            results = imagenet_val(ema.ema, split=val_split, testloader=testloader)
            _log_dic.update({f'metric/ema_val_'+k: v for k,v in results.items()})
            # select best result among all emas
            _save_model = ema.ema
        else:
            _save_model = model

        _cur_fitness = results[metric]
        # wandb log
        wbrun.log(_log_dic, step=niter)
        # Write evaluation results
        res = s + '||' + '%10.4g' * 1 % (_cur_fitness)
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


def get_model(name, num_class):
    if name == 'res50':
        from mycv.models.cls.resnet import resnet50
        model = resnet50(num_classes=num_class)
    elif name == 'res101':
        from mycv.models.cls.resnet import resnet101
        model = resnet101(num_classes=num_class)
    elif name.startswith('yolov5'):
        from mycv.models.yolov5.cls import YOLOv5Cls
        assert name[-1] in ['s', 'm', 'l']
        model = YOLOv5Cls(model=name[-1], num_class=num_class)
    elif name.startswith('csp'):
        from mycv.models.yolov5.cls import CSP
        assert name[-1] in ['s', 'm', 'l']
        model = CSP(model=name[-1], num_class=num_class)
    else:
        raise ValueError()
    return model


if __name__ == '__main__':
    train()

    # from mycv.models.cls.resnet import resnet50
    # model = resnet50(num_classes=1000)
    # weights = torch.load('weights/resnet50-19c8e357.pth')
    # model.load_state_dict(weights)
    # model = model.cuda()
    # model.eval()
    # results = imagenet_val(model, img_size=224, batch_size=64, workers=4)
    # print(results['top1'])
