import os
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.cuda.amp as amp
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter

from mycv.utils.torch_utils import load_partial
from mycv.utils.general import increment_dir


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
    parser.add_argument('--model',      type=str,  default='res50')
    parser.add_argument('--resume',     type=str,  default='')
    parser.add_argument('--batch_size', type=int,  default=64)
    parser.add_argument('--amp',        type=bool, default=True)
    parser.add_argument('--ema',        type=bool, default=False)
    parser.add_argument('--optimizer',  type=str,  default='Adam')
    parser.add_argument('--epochs',     type=int,  default=4)
    parser.add_argument('--metric',     type=str,  default='top1')
    parser.add_argument('--log_root',   type=str,  default='runs/ilsvrc')
    parser.add_argument('--device',     nargs='+', default=[0])
    parser.add_argument('--workers',    type=int,  default=8)
    args = parser.parse_args()
    hyp = {
        'lr': 0.0001,
        'momentum': 0.937, # SGD
        'nesterov': True, # SGD
        'img_size': 256
    }
    print(args)
    print(hyp)
    # fix random seeds for reproducibility
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    # device setting
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    print('Using device', torch.cuda.get_device_properties(device))
    torch.backends.cudnn.benchmark = True
    # argument settings
    batch_size: int = args.batch_size
    metric:     str = args.metric.lower()
    epochs:     int = args.epochs

    # Dataset
    print('Initializing Datasets and Dataloaders...')
    from mycv.datasets.imagenet import ImageNetCls
    trainset = ImageNetCls(split='train', img_size=hyp['img_size'], augment=True)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=False
    )

    # Initialize model
    from mycv.models.cls.resnet import resnet50
    model = resnet50(num_classes=1000)
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
    else:
        # new experiment
        log_dir = increment_dir(dir_root=args.log_root, name=args.model)
        assert not os.path.exists(log_dir)
        # make dir and save configs
        os.makedirs(log_dir / 'weights')
        yaml.dump(hyp, open(log_dir / 'hyp.yaml', 'w'), sort_keys=False)
        yaml.dump(vars(args), open(log_dir / 'args.yaml', 'w'), sort_keys=False)
        print(str(model), file=open(log_dir / 'model.txt', 'w'))
        start_epoch = 0
        best_fitness = 0

    # Tensorboard
    tb_writer = SummaryWriter(log_dir)

    # ======================== start training ========================
    for epoch in range(epochs):
        train_loss, train_acc = 0.0, 0.0
        val_acc = 0
        cur_lr = optimizer.param_groups[0]['lr']
        model.train()

        pbar_title = ('%-10s' * 6) % (
            'Epoch', 'GPU_mem', 'lr', 'tr_loss', 'tr_acc', 'val_acc'
        )
        print('\n' + pbar_title) # title
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for i, (imgs, labels) in pbar:
            niter = epoch * len(trainloader) + i
            # debugging
            # if True:
            #     import matplotlib.pyplot as plt
            #     im = imgs[0].permute(1,2,0).numpy()
            #     plt.imshow(im); plt.show()
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            # forward
            with amp.autocast(enabled=args.amp):
                p = model(imgs)
                loss = loss_func(p, labels)
                loss = loss * batch_size
            # backward, update
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # logging
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
                model.eval()
                results = test_ucf101_1stframe(
                    model, split='test', fold=args.fold, img_hw=hyp['img_hw']
                )
                val_acc = results['top1']
                tb_writer.add_scalar('metric/val_acc', val_acc,  global_step=niter)
                model.train()
            # ----Mini batch end
        # ----Epoch end

        # Evaluation
        model.eval()
        results = test_ucf101_1stframe(
            model, split='test', fold=args.fold, img_hw=hyp['img_hw']
        )
        tb_writer.add_scalar('metric/val_acc', val_acc,  global_step=niter)
        # Write evaluation results
        res = s + '||' + '%10.4g' * 1 % (results['top1'])
        with open(log_dir / 'results', 'a') as f:
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
            torch.save(checkpoint, log_dir / 'weights/last.pt')
        del checkpoint
        # ----Epoch end
    # ----Training end


if __name__ == '__main__':
    train()
