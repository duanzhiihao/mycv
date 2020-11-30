import os
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import random
import torch
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from mycv.utils.timer import now
from mycv.utils.torch_utils import load_partial, set_random_seeds
# from mycv.datasets.imagenet import ImageNetCls, imagenet_val
from mycv.datasets.food101 import Food101, food101_val


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
    parser.add_argument('--model',      type=str,  default='res101')
    parser.add_argument('--resume',     type=str,  default='')
    parser.add_argument('--batch_size', type=int,  default=64)
    parser.add_argument('--amp',        type=bool, default=True)
    parser.add_argument('--ema',        type=bool, default=False)
    parser.add_argument('--optimizer',  type=str,  default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--epochs',     type=int,  default=100)
    parser.add_argument('--metric',     type=str,  default='top1', choices=['top1'])
    parser.add_argument('--log_root',   type=str,  default='runs/food101')
    parser.add_argument('--device',     type=int,  default=0)
    parser.add_argument('--workers',    type=int,  default=8)
    parser.add_argument('--local_rank', type=int,  default=-1, help='DDP arg, do not modify')
    cfg = parser.parse_args()
    cfg.lr = 0.0002
    cfg.momentum = 0.937
    cfg.nesterov = True
    cfg.img_size = 320
    IS_MAIN = (cfg.local_rank in [-1, 0])
    # initialize wandb
    if IS_MAIN:
        wbrun = wandb.init(project='food101', group=cfg.model, name=now(),
                           config=cfg)
        cfg = wbrun.config

    # check arguments
    metric:     str = cfg.metric.lower()
    epochs:     int = cfg.epochs
    local_rank: int = cfg.local_rank
    world_size: int = int(os.environ.get('WORLD_SIZE', 1))
    assert local_rank == int(os.environ.get('RANK', -1)), 'Only support single node'
    assert cfg.batch_size % world_size == 0, 'batch_size must be multiple of device count'
    batch_size: int = cfg.batch_size // world_size
    if IS_MAIN:
        print(cfg)
        print('Batch size on each single GPU =', batch_size)
    # fix random seeds for reproducibility
    set_random_seeds(1)
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

    # Initialize model
    if cfg.model == 'res50':
        from mycv.models.cls.resnet import resnet50
        model = resnet50(num_classes=101)
    elif cfg.model == 'res101':
        from mycv.models.cls.resnet import resnet101
        model = resnet101(num_classes=101)
        # load_partial(model, 'weights/resnet101-5d3b4d8f.pth', verbose=IS_MAIN)
    model = model.to(device)
    # loss function
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

    # optimizer
    if cfg.optimizer == 'SGD':
        raise NotImplementedError()
        # optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr'])
    elif cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = amp.GradScaler(enabled=cfg.amp)

    if cfg.resume:
        raise NotImplementedError()
        # resume
        log_dir = Path(cfg.log_root) / cfg.resume
        assert os.path.isdir(log_dir)
        checkpoint = torch.load(log_dir / 'weights/last.pt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_fitness = checkpoint.get(metric, 0)
    else:
        # new experiment
        if IS_MAIN:
            log_dir = Path(wbrun.dir) # wandb logging dir
            assert log_dir.exists()
            print(str(model), file=open(log_dir / 'model.txt', 'w'))
        start_epoch = 0
        best_fitness = 0

    # Exponential moving average
    if cfg.ema:
        raise NotImplementedError()
        # ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Dataset
    if IS_MAIN:
        print('Initializing Datasets and Dataloaders...')
    # training set
    # trainset = ImageNetCls(split='train', img_size=hyp['img_size'], augment=True)
    trainset = Food101(split='train', img_size=cfg.img_size, augment=True)
    sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=world_size, rank=local_rank, shuffle=True
    ) if local_rank != -1 else None
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.workers, pin_memory=True
    )

    # ======================== start training ========================
    for epoch in range(epochs):
        model.train()
        if local_rank != -1:
            trainloader.sampler.set_epoch(epoch)
        optimizer.zero_grad()

        pbar = enumerate(trainloader)
        if IS_MAIN:
            train_loss, train_acc = 0.0, 0.0
            cur_lr = optimizer.param_groups[0]['lr']
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
            #         im = im * trainset._input_std + trainset._input_std
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

            # logging
            if IS_MAIN:
                niter = epoch * len(trainloader) + i
                loss = loss.detach().cpu().item()
                acc = cal_acc(p.detach(), labels)
                train_loss = (train_loss*i + loss) / (i+1)
                train_acc  = (train_acc*i + acc) / (i+1)
                mem = torch.cuda.max_memory_allocated() / 1e9
                s = ('%-10s' * 2 + '%-10.4g' * 4) % (
                    f'{epoch}/{epochs}', f'{mem:.3g}G',
                    cur_lr, train_loss, 100*train_acc, 100*results[metric]
                )
                pbar.set_description(s)
                torch.cuda.reset_peak_memory_stats()
                # Tensorboard
                if niter % 200 == 0:
                    wbrun.log({
                        'metric/train_loss': train_loss,
                        'metric/train_acc':  train_acc
                    }, step=niter)
                    # model.eval()
                    # results = food101_val(model, img_size=hyp['img_size'],
                    #             batch_size=4*batch_size, workers=cfg.workers)
                    # val_acc = results['top1']
                    # tb_writer.add_scalar('metric/val_acc', val_acc,  global_step=niter)
                    # model.train()
                # logging end
            # ----Mini batch end
        # ----Epoch end
        # If DDP mode, synchronize model parameters on all gpus
        if local_rank != -1:
            model._sync_params_and_buffers(authoritative_rank=0)

        # Evaluation
        if IS_MAIN:
            model.eval()
            results = food101_val(model, img_size=cfg.img_size,
                        batch_size=4*batch_size, workers=cfg.workers)
            # results is like {'top1': xxx, 'top5': xxx}
            wbrun.log(
                {'metric/val_'+k: v for k,v in results.items()}, step=niter
            )
            # Write evaluation results
            res = s + '||' + '%10.4g' * 1 % (results[metric])
            with open(log_dir / 'results.txt', 'a') as f:
                f.write(res + '\n')
            # save last checkpoint
            checkpoint = {
                'model'    : model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler'   : scaler.state_dict(),
                'epoch'    : epoch,
                metric     : results[metric]
            }
            torch.save(checkpoint, log_dir / 'last.pt')
            # save best checkpoint
            if results[metric] > best_fitness:
                best_fitness = results[metric]
                torch.save(checkpoint, log_dir / 'best.pt')
            del checkpoint
        # ----Epoch end
    # ----Training end


if __name__ == '__main__':
    train()

    # from mycv.models.cls.resnet import resnet50
    # model = resnet50(num_classes=101)
    # weights = torch.load('runs/food101/res50_2/weights/last.pt')
    # model.load_state_dict(weights['model'])
    # model = model.cuda()
    # model.eval()
    # results = food101_val(model, img_size=256, batch_size=4, workers=0)
    # print(results['top1'])
