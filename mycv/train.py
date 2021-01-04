import os
from pathlib import Path
import argparse
from tqdm import tqdm
import math
import random
import torch
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from mycv.utils.general import increment_dir
from mycv.utils.torch_utils import set_random_seeds, ModelEMA
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
    parser.add_argument('--model',      type=str,  default='csp_s')
    parser.add_argument('--resume',     type=str,  default='')
    parser.add_argument('--batch_size', type=int,  default=128)
    parser.add_argument('--amp',        type=bool, default=True)
    parser.add_argument('--ema',        type=bool, default=True)
    parser.add_argument('--optimizer',  type=str,  default='SGD', choices=['Adam', 'SGD'])
    parser.add_argument('--epochs',     type=int,  default=100)
    parser.add_argument('--metric',     type=str,  default='top1', choices=['top1'])
    parser.add_argument('--device',     type=int,  default=0)
    parser.add_argument('--workers',    type=int,  default=4)
    parser.add_argument('--local_rank', type=int,  default=-1, help='DDP arg, do not modify')
    # parser.add_argument('--dryrun',   type=bool, default=True)
    parser.add_argument('--dryrun',     action='store_true')
    cfg = parser.parse_args()
    # model
    cfg.img_size = 224
    cfg.input_norm = False
    cfg.sync_bn = False
    # optimizer
    cfg.lr = 0.01
    cfg.momentum = 0.9
    cfg.weight_decay = 0.0001
    cfg.nesterov = True
    # lr scheduler
    cfg.lrf = 0.2 # min lr factor
    cfg.lr_warmup_epochs = 1
    # EMA
    # cfg.ema_decay = 0.999
    cfg.ema_warmup_epochs = 4
    # Main process
    IS_MAIN = (cfg.local_rank in [-1, 0])

    # check arguments
    metric:     str = cfg.metric.lower()
    epochs:     int = cfg.epochs
    local_rank: int = cfg.local_rank
    world_size: int = int(os.environ.get('WORLD_SIZE', 1))
    assert local_rank == int(os.environ.get('RANK', -1)), 'Only support single node'
    assert cfg.batch_size % world_size == 0, 'batch_size must be multiple of device count'
    batch_size: int = cfg.batch_size // world_size
    if IS_MAIN:
        print(cfg, '\n')
        print('Batch size on each single GPU =', batch_size, '\n')
    # fix random seeds for reproducibility
    set_random_seeds(1)
    torch.backends.cudnn.benchmark = True
    # device setting
    assert torch.cuda.is_available()
    if local_rank == -1: # Single GPU
        device = torch.device(f'cuda:{cfg.device}')
    else: # DDP mode
        assert torch.cuda.device_count() > local_rank and torch.distributed.is_available()
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://', world_size=world_size, rank=local_rank
        )
    print(f'Local rank: {local_rank}, using device {device}:', 'device property:',
          torch.cuda.get_device_properties(device))

    # Dataset
    if IS_MAIN:
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
    sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=world_size, rank=local_rank, shuffle=True
    ) if local_rank != -1 else None
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.workers, pin_memory=True
    )
    # test set
    testloader = torch.utils.data.DataLoader(
        ImageNetCls(split=val_split, img_size=cfg.img_size, input_norm=cfg.input_norm),
        batch_size=batch_size, shuffle=False, num_workers=cfg.workers//2,
        pin_memory=True, drop_last=False
    )

    # Initialize model
    if cfg.model == 'res50':
        from mycv.models.cls.resnet import resnet50
        model = resnet50(num_classes=cfg.num_class)
    elif cfg.model == 'res101':
        from mycv.models.cls.resnet import resnet101
        model = resnet101(num_classes=cfg.num_class)
    elif cfg.model.startswith('yolov5'):
        from mycv.models.yolov5.cls import YOLOv5Cls
        assert cfg.model[-1] in ['s', 'm', 'l']
        model = YOLOv5Cls(model=cfg.model[-1], num_class=cfg.num_class)
    elif cfg.model.startswith('csp'):
        from mycv.models.yolov5.cls import CSP
        assert cfg.model[-1] in ['s', 'm', 'l']
        model = CSP(model=cfg.model[-1], num_class=cfg.num_class)
    else:
        raise NotImplementedError()
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
    if IS_MAIN:
        print('Parameter groups:', [len(pg['params']) for pg in parameters])
    del pgb, pgw

    # optimizer
    if cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=cfg.lr)
    elif cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=cfg.lr)
    else:
        raise ValueError()
    # AMP
    scaler = amp.GradScaler(enabled=cfg.amp)

    log_parent = Path(f'runs/{cfg.project}')
    wb_id = None
    cur_fitness = best_fitness = 0
    results = {metric: 0}
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
        if IS_MAIN:
            cur_fitness = best_fitness = checkpoint.get(metric, 0)
            wb_id = open(log_dir / 'wandb_id.txt', 'r').read()
    else:
        # new experiment
        run_name = increment_dir(dir_root=log_parent, name=cfg.model)
        log_dir = log_parent / run_name # wandb logging dir
        if IS_MAIN:
            os.makedirs(log_dir, exist_ok=False)
            print(str(model), file=open(log_dir / 'model.txt', 'w'))
        start_epoch = 0

    # initialize wandb
    if cfg.dryrun:
        os.environ["WANDB_MODE"] = "dryrun"
    if IS_MAIN:
        wbrun = wandb.init(project=cfg.project, group=cfg.group, name=run_name,
                           config=cfg, dir='runs/', resume='allow', id=wb_id)
        cfg = wbrun.config
        cfg.log_dir = log_dir
        cfg.wandb_id = wbrun.id
        if not (log_dir / 'wandb_id.txt').exists():
            with open(log_dir / 'wandb_id.txt', 'w') as f:
                f.write(wbrun.id)
    else:
        wbrun = None

    # lr scheduler
    def warmup_cosine(x):
        warmup_iter = cfg.lr_warmup_epochs * len(trainloader)
        if x < warmup_iter:
            factor = x / warmup_iter
        else:
            _cur = x - warmup_iter + 1
            _total = epochs * len(trainloader)
            factor = cfg.lrf + 0.5 * (1 - cfg.lrf) * (1 + math.cos(_cur * math.pi / _total))
        return factor
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine, last_epoch=start_epoch - 1)

    # SyncBatchNorm
    if local_rank != -1 and cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Exponential moving average
    if IS_MAIN and cfg.ema:
        emas = [
            ModelEMA(model, decay=0.99),
            ModelEMA(model, decay=0.999),
            ModelEMA(model, decay=0.9999)
        ]
    else:
        emas = None

    # DDP mode
    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if emas:
        for ema in emas:
            ema.updates = start_epoch * len(trainloader)  # set EMA updates
            ema.warmup = cfg.ema_warmup_epochs * len(trainloader) # 4 epochs

    # ======================== start training ========================
    niter = s = None
    for epoch in range(start_epoch, epochs):
        model.train()
        if local_rank != -1:
            trainloader.sampler.set_epoch(epoch)
        optimizer.zero_grad()

        pbar = enumerate(trainloader)
        train_loss, train_acc = 0.0, 0.0
        if IS_MAIN:
            pbar_title = ('%-10s' * 6) % (
                'Epoch', 'GPU_mem', 'lr', 'tr_loss', 'tr_acc', metric
            )
            print('\n' + pbar_title) # title
            pbar = tqdm(pbar, total=len(trainloader))
        for i, (imgs, labels) in pbar:
            # debugging
            # if True:
            #     import matplotlib.pyplot as plt
            #     from mycv.datasets.food101 import CLASS_NAMES
            #     for im, lbl in zip(imgs, labels):
            #         im = im * trainset._input_std + trainset._input_mean
            #         im = im.permute(1,2,0).numpy()
            #         print(CLASS_NAMES[lbl])
            #         plt.imshow(im); plt.show()
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            # forward
            with amp.autocast(enabled=cfg.amp):
                p = model(imgs)
                loss = loss_func(p, labels) * imgs.shape[0]
                if local_rank != -1:
                    loss = loss * world_size
                # loss is averaged within image, sumed over batch, and sumed over gpus
            # backward, update
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if emas:
                for ema in emas:
                    ema.update(model)
            # Scheduler
            scheduler.step()

            # logging
            if IS_MAIN:
                niter = epoch * len(trainloader) + i
                cur_lr = optimizer.param_groups[0]['lr']
                loss = loss.detach().cpu().item()
                acc = cal_acc(p.detach(), labels)
                train_loss = (train_loss*i + loss) / (i+1)
                train_acc  = (train_acc*i + acc) / (i+1)
                mem = torch.cuda.max_memory_allocated(device) / 1e9
                s = ('%-10s' * 2 + '%-10.4g' * 4) % (
                    f'{epoch}/{epochs-1}', f'{mem:.3g}G',
                    cur_lr, train_loss, 100*train_acc, 100*cur_fitness
                )
                pbar.set_description(s)
                torch.cuda.reset_peak_memory_stats()
                # Weights & Biases logging
                if niter % 100 == 0:
                    wbrun.log({
                        'general/lr': cur_lr,
                        'metric/train_loss': train_loss,
                        'metric/train_acc': train_acc,
                        'ema/n_updates': emas[0].updates if emas is not None else 0,
                        'ema0/decay': emas[0].get_decay() if emas is not None else 0,
                        'ema1/decay': emas[1].get_decay() if emas is not None else 0,
                        'ema2/decay': emas[2].get_decay() if emas is not None else 0,
                    }, step=niter)
                # logging end
            # ----Mini batch end
        # ----Epoch end
        # If DDP mode, synchronize model parameters on all gpus
        if local_rank != -1:
            model._sync_params_and_buffers(authoritative_rank=0)

        # Evaluation
        if IS_MAIN:
            # results is like {'top1': xxx, 'top5': xxx}
            _log_dic = {'general/epoch': epoch}
            results = imagenet_val(model, split=val_split, testloader=testloader)
            _log_dic.update({'metric/plain_val_'+k: v for k,v in results.items()})

            res_emas = torch.zeros(len(emas))
            if emas is not None:
                for ei, ema in enumerate(emas):
                    results = imagenet_val(ema.ema, split=val_split, testloader=testloader)
                    _log_dic.update({f'metric/ema{ei}_val_'+k: v for k,v in results.items()})
                    res_emas[ei] = results[metric]
                # select best result among all emas
                _idx = torch.argmax(res_emas)
                cur_fitness = res_emas[_idx]
                _save_model = emas[_idx].ema
                best_decay  = emas[_idx].final_decay
            else:
                cur_fitness = results[metric]
                _save_model = model
                best_decay  = 0
            # wandb log
            wbrun.log(_log_dic, step=niter)
            # Write evaluation results
            res = s + '||' + '%10.4g' * 1 % (results[metric])
            with open(log_dir / 'results.txt', 'a') as f:
                f.write(res + '\n')
            # save last checkpoint
            checkpoint = {
                'model'     : _save_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler'    : scaler.state_dict(),
                'epoch'     : epoch,
                metric      : cur_fitness,
                'best_decay': best_decay
            }
            torch.save(checkpoint, log_dir / 'last.pt')
            # save best checkpoint
            if cur_fitness > best_fitness:
                best_fitness = cur_fitness
                torch.save(checkpoint, log_dir / 'best.pt')
            del checkpoint
        # ----Epoch end
    # ----Training end


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
