import os
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import random
import torch
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torch.utils.tensorboard import SummaryWriter

from mycv.utils.torch_utils import load_partial, set_random_seeds
from mycv.utils.general import increment_dir
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
    args = parser.parse_args()
    hyp = {
        'lr': 0.0002,
        'momentum': 0.937, # SGD
        'nesterov': True, # SGD
        'img_size': 320
    }
    # arguments
    metric:     str = args.metric.lower()
    epochs:     int = args.epochs
    local_rank: int = args.local_rank
    world_size: int = int(os.environ.get('WORLD_SIZE', 1))
    assert local_rank == int(os.environ.get('RANK', -1)), 'Currently only support single node'
    assert args.batch_size % world_size == 0, 'batch_size must be multiple of CUDA device count'
    batch_size: int = args.batch_size// world_size
    if local_rank in [-1, 0]:
        print(args)
        print(hyp)
    # fix random seeds for reproducibility
    set_random_seeds(1)
    # device setting
    assert torch.cuda.is_available()
    if local_rank == -1: # Single GPU
        device = torch.device(f'cuda:{args.device}')
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
    if args.model == 'res50':
        from mycv.models.cls.resnet import resnet50
        model = resnet50(num_classes=101)
    elif args.model == 'res101':
        from mycv.models.cls.resnet import resnet101
        model = resnet101(num_classes=101)
        # load_partial(model, 'weights/resnet101-5d3b4d8f.pth')
    model = model.to(device)
    # loss function
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

    # optimizer
    if args.optimizer == 'SGD':
        raise NotImplementedError()
        # optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr'])
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=hyp['lr'])
    scaler = amp.GradScaler(enabled=args.amp)

    if args.resume:
        # resume
        log_dir = Path(args.log_root) / args.resume
        assert os.path.isdir(log_dir)
        checkpoint = torch.load(log_dir / 'weights/last.pt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_fitness = checkpoint.get(metric, 0)
        if local_rank in [-1, 0]:
            yaml.dump(hyp, open(log_dir / f'hyp{start_epoch}.yaml', 'w'), sort_keys=False)
            yaml.dump(vars(args), open(log_dir/f'args{start_epoch}.yaml','w'), sort_keys=False)
    else:
        # new experiment
        log_dir = increment_dir(dir_root=args.log_root, name=args.model)
        assert not os.path.exists(log_dir)
        # make dir and save configs
        os.makedirs(log_dir / 'weights')
        if local_rank in [-1, 0]:
            yaml.dump(hyp, open(log_dir / 'hyp.yaml', 'w'), sort_keys=False)
            yaml.dump(vars(args), open(log_dir / 'args.yaml', 'w'), sort_keys=False)
        print(str(model), file=open(log_dir / 'model.txt', 'w'))
        start_epoch = 0
        best_fitness = 0

    # Exponential moving average
    if args.ema:
        raise NotImplementedError()
        # ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Dataset
    if local_rank in [-1, 0]:
        print('Initializing Datasets and Dataloaders...')
    # training set
    # trainset = ImageNetCls(split='train', img_size=hyp['img_size'], augment=True)
    trainset = Food101(split='train', img_size=hyp['img_size'], augment=True)
    sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=world_size, rank=local_rank, shuffle=True
    ) if local_rank != -1 else None
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.workers, pin_memory=True
    )

    # Tensorboard
    if local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir)
        print(f'Start Tensorboard with "tensorboard --logdir {log_dir}",',
            'view at http://localhost:6006/')

    # ======================== start training ========================
    val_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = 0.0, 0.0
        cur_lr = optimizer.param_groups[0]['lr']
        model.train()

        if local_rank != -1:
            trainloader.sampler.set_epoch(epoch)
        pbar = enumerate(trainloader)
        if local_rank in [-1, 0]:
            pbar_title = ('%-10s' * 6) % (
                'Epoch', 'GPU_mem', 'lr', 'tr_loss', 'tr_acc', 'val_acc'
            )
            print('\n' + pbar_title) # title
            pbar = tqdm(pbar, total=len(trainloader))
        optimizer.zero_grad()
        for i, (imgs, labels) in pbar:
            niter = epoch * len(trainloader) + i
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
            with amp.autocast(enabled=args.amp):
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
            if local_rank in [-1, 0]:
                loss = loss.detach().cpu().item()
                acc = cal_acc(p.detach(), labels)
                train_loss = (train_loss*i + loss) / (i+1)
                train_acc  = (train_acc*i + acc) / (i+1)
                mem = torch.cuda.max_memory_allocated() / 1e9
                s = ('%-10s' * 2 + '%-10.4g' * 4) % (
                    f'{epoch}/{epochs}', f'{mem:.3g}G',
                    cur_lr, train_loss, 100*train_acc, 100*val_acc
                )
                pbar.set_description(s)
                torch.cuda.reset_peak_memory_stats()
                # Tensorboard
                if niter % 200 == 0:
                    tb_writer.add_scalar('metric/train_loss', train_loss, global_step=niter)
                    tb_writer.add_scalar('metric/train_acc',  train_acc,  global_step=niter)
                    # model.eval()
                    # results = food101_val(model, img_size=hyp['img_size'],
                    #             batch_size=4*batch_size, workers=args.workers)
                    # val_acc = results['top1']
                    # tb_writer.add_scalar('metric/val_acc', val_acc,  global_step=niter)
                    # model.train()
                # logging end
            # ----Mini batch end
        # ----Epoch end

        # Evaluation
        if local_rank in [-1, 0]:
            model.eval()
            results = food101_val(model, img_size=hyp['img_size'],
                        batch_size=4*batch_size, workers=args.workers)
            val_acc = results['top1']
            tb_writer.add_scalar('metric/val_acc', val_acc,  global_step=niter)
            # Write evaluation results
            res = s + '||' + '%10.4g' * 1 % (results['top1'])
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
            torch.save(checkpoint, log_dir / 'weights/last.pt')
            # save best checkpoint
            if results[metric] > best_fitness:
                best_fitness = results[metric]
                torch.save(checkpoint, log_dir / 'weights/best.pt')
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
