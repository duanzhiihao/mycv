from mycv.utils.general import disable_multithreads
disable_multithreads()
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import math
import cv2
import torch
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from mycv.utils.general import increment_dir
from mycv.utils.torch_utils import set_random_seeds, ModelEMA, is_parallel
from mycv.utils.coding import MS_SSIM, cal_bpp
from mycv.datasets.loadimgs import LoadImages, kodak_val


def train():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',    type=str,  default='imcoding')
    parser.add_argument('--group',      type=str,  default='default')
    parser.add_argument('--datasets',   type=str,  default=['imagenet200'], nargs='+')
    parser.add_argument('--model',      type=str,  default='mini')
    parser.add_argument('--loss',       type=str,  default='mse', choices=['mse','msssim'])
    parser.add_argument('--lmbda',      type=float,default=32)
    parser.add_argument('--resume',     type=str,  default='')
    parser.add_argument('--batch_size', type=int,  default=64)
    parser.add_argument('--amp',        type=bool, default=False)
    parser.add_argument('--ema',        type=bool, default=True)
    parser.add_argument('--optimizer',  type=str,  default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--epochs',     type=int,  default=80)
    parser.add_argument('--device',     type=int,  default=0)
    parser.add_argument('--workers',    type=int,  default=4)
    parser.add_argument('--local_rank', type=int,  default=-1, help='DDP arg, do not modify')
    parser.add_argument('--wbmode',     type=str,  default='online')
    cfg = parser.parse_args()
    # model
    cfg.img_size = 192
    cfg.input_norm = False
    cfg.sync_bn = False
    # optimizer
    cfg.lr = 1e-5
    cfg.momentum = 0.9
    cfg.weight_decay = 0.0
    cfg.nesterov = True
    # lr scheduler
    cfg.lrf = 0.2 # min lr factor
    cfg.lr_warmup_epochs = 0
    # EMA
    cfg.ema_warmup_epochs = 2
    # Main process
    IS_MAIN = (cfg.local_rank in [-1, 0])

    # check arguments
    _loss2metric = {'mse':'psnr', 'msssim':'msssim'}
    metric:     str = _loss2metric[cfg.loss]
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
    # training set
    trainset = LoadImages(datasets=cfg.datasets, img_size=cfg.img_size,
                          input_norm=cfg.input_norm, verbose=IS_MAIN)
    sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=world_size, rank=local_rank, shuffle=True
    ) if local_rank != -1 else None
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.workers, pin_memory=True
    )

    # Initialize model
    if cfg.model == 'mini':
        from mycv.models.nic.mini import MiniNIC
        model = MiniNIC(enable_bpp=True)
    elif cfg.model == 'nlaic':
        from mycv.models.nic.nlaic import NLAIC
        model = NLAIC(enable_bpp=True)
    else:
        raise NotImplementedError()
    model = model.to(device)
    # loss function
    if cfg.loss == 'mse':
        loss_func = torch.nn.MSELoss(reduction='mean')
    elif cfg.loss == 'msssim':
        raise NotImplementedError()
        loss_func = MS_SSIM(max_val=1.0, reduction='mean')
    else:
        raise ValueError()

    # different optimization setting for different layers
    pgb, pgw = [], []
    for k, v in model.named_parameters():
        if ('.bn' in k) or ('.bias' in k): # batchnorm or bias
            pgb.append(v)
        else: # conv weights
            # assert k.endswith('.weight')
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
        cur_fitness = best_fitness = checkpoint.get(metric, 0)
        cur_bpp = checkpoint.get('bpp', 0)
        if IS_MAIN:
            wb_id = open(log_dir / 'wandb_id.txt', 'r').read()
    else:
        # new experiment
        run_name = increment_dir(dir_root=log_parent, name=cfg.model)
        log_dir = log_parent / run_name # wandb logging dir
        if IS_MAIN:
            os.makedirs(log_dir, exist_ok=False)
            print(str(model), file=open(log_dir / 'model.txt', 'w'))
        start_epoch = 0
        cur_fitness = best_fitness = 0
        cur_bpp = 0

    # initialize wandb
    if IS_MAIN:
        wbrun = wandb.init(project=cfg.project, group=cfg.group, name=run_name, config=cfg,
                           dir='runs/', resume='allow', id=wb_id, mode=cfg.wbmode)
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
        ema = ModelEMA(model, decay=0.9999)
        ema.updates = start_epoch * len(trainloader)  # set EMA updates
        ema.warmup = cfg.ema_warmup_epochs * len(trainloader) # 4 epochs
    else:
        ema = None

    # DDP mode
    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ======================== start training ========================
    niter = s = None
    for epoch in range(start_epoch, epochs):
        model.train()
        if local_rank != -1:
            trainloader.sampler.set_epoch(epoch)
        optimizer.zero_grad()

        pbar = enumerate(trainloader)
        train_loss, train_rec, train_bpp = 0.0, 0.0, 0.0
        if IS_MAIN:
            pbar_title = ('%-10s' * 8) % (
                'Epoch', 'GPU_mem', 'lr',
                f'tr_{cfg.loss}', 'tr_bpp', 'loss', metric, 'bpp'
            )
            print('\n' + pbar_title) # title
            pbar = tqdm(pbar, total=len(trainloader))
        for i, imgs in pbar:
            niter = epoch * len(trainloader) + i
            imgs = imgs.to(device=device)
            nB, nC, nH, nW = imgs.shape
            # forward
            with amp.autocast(enabled=cfg.amp):
                rec, probs = model(imgs)
                l_rec = loss_func(rec, imgs) * nB
                if probs is not None:
                    p1, p2 = probs
                    l_bpp = cal_bpp(p1, nH*nW) + cal_bpp(p2, nH*nW)
                else:
                    l_bpp = torch.zeros(1).to(device=device)
                loss = cfg.lmbda * l_rec + 0.01 * l_bpp
                if local_rank != -1:
                    loss = loss * world_size
                # loss is averaged within image, sumed over batch, and sumed over gpus
            # backward, update
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)
            # Scheduler
            scheduler.step()

            # save output
            if IS_MAIN and niter % 100 == 0:
                save_output(imgs, rec, log_dir)
            # logging
            if IS_MAIN:
                cur_lr = optimizer.param_groups[0]['lr']
                loss = loss.detach().cpu().item()
                train_rec  = (train_rec*i + l_rec/nB) / (i+1)
                train_bpp  = (train_bpp*i + l_bpp/nB) / (i+1)
                train_loss = (train_loss*i + loss/nB) / (i+1)
                mem = torch.cuda.max_memory_allocated(device) / 1e9
                s = ('%-10s' * 2 + '%-10.4g' * 6) % (
                    f'{epoch}/{epochs-1}', f'{mem:.3g}G', cur_lr,
                    train_rec, train_bpp, train_loss, cur_fitness, cur_bpp
                )
                pbar.set_description(s)
                torch.cuda.reset_peak_memory_stats()
                # Weights & Biases logging
                if niter % 100 == 0:
                    wbrun.log({
                        'general/lr': cur_lr,
                        'metric/train_rec': train_rec,
                        'metric/train_bpp': train_bpp,
                        'metric/train_loss': train_loss,
                        'ema/n_updates': ema.updates if ema is not None else 0,
                        'ema2/decay': ema.get_decay() if ema is not None else 0,
                    }, step=niter)
                # logging end
            # Evaluation
            if IS_MAIN and niter % 200 == 0:
                _log_dic = {'general/epoch': epoch}
                _val_model = model.module if is_parallel(model) else model
                results = kodak_val(_val_model, input_norm=cfg.input_norm, verbose=False)
                _log_dic.update({'metric/plain_val_'+k: v for k,v in results.items()})
                cur_fitness, cur_bpp = results[metric], results['bpp']
                _save_model = model
                if ema is not None:
                    results = kodak_val(ema.ema, input_norm=cfg.input_norm, verbose=False)
                    _log_dic.update({f'metric/ema2_val_'+k: v for k,v in results.items()})
                    # select best result among all emas
                    ema_fitness = results[metric]
                    if ema_fitness > cur_fitness:
                        cur_fitness, cur_bpp = ema_fitness, results['bpp']
                        _save_model = ema.ema
                # wandb log
                wbrun.log(_log_dic, step=niter)
                # Write evaluation results
                res = s + '||' + '%10.4g' * 2 % (results['psnr'], results['msssim'])
                with open(log_dir / 'results.txt', 'a') as f:
                    f.write(res + '\n')
                # save last checkpoint
                checkpoint = {
                    'model'     : _save_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler'    : scaler.state_dict(),
                    'epoch'     : epoch,
                    metric      : cur_fitness,
                }
                torch.save(checkpoint, log_dir / 'last.pt')
                # save best checkpoint
                if cur_fitness > best_fitness:
                    best_fitness = cur_fitness
                    torch.save(checkpoint, log_dir / 'best.pt')
                del checkpoint
            model.train()
            # ----Mini batch end
        # ----Epoch end
        # If DDP mode, synchronize model parameters on all gpus
        if local_rank != -1:
            model._sync_params_and_buffers(authoritative_rank=0)

        # ----Epoch end
    # ----Training end


def save_output(input_: torch.Tensor, output: torch.Tensor, log_dir: Path):
    imt, imp = input_[0].cpu(), output[0].detach().cpu().float()
    imp.clamp_(min=0, max=1)
    assert imt.shape == imp.shape and imt.dim() == 3
    im = torch.cat([imt, imp], dim=2)
    im = im.permute(1,2,0) * 255
    im = im.to(dtype=torch.uint8).numpy()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(log_dir / 'out.png'), im)


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
