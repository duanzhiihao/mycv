from mycv.utils.general import disable_multithreads
disable_multithreads()
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import defaultdict
import cv2
import torch
import torch.nn.functional as tnf
from torch.optim.lr_scheduler import LambdaLR
import wandb

from mycv.utils.general import increment_dir
from mycv.utils.torch_utils import set_random_seeds
from mycv.datasets.superres import SRDataset, sr_evaluate


def _psnr(sr, hr):
    with torch.no_grad():
        mse = torch.mean(torch.square_(sr - hr))
    if mse == 0:
        psnr = 100
    else:
        psnr = 10 * torch.log10(255.0**2 / mse).item()
    return psnr


def train():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',    type=str,  default='sr')
    parser.add_argument('--group',      type=str,  default='default')
    parser.add_argument('--dataset',    type=str,  default='div2k_train')
    parser.add_argument('--scale',      type=int,  default=2)
    parser.add_argument('--model',      type=str,  default='edsr_base')
    parser.add_argument('--batch_size', type=int,  default=32)
    parser.add_argument('--epochs',     type=int,  default=100)
    parser.add_argument('--device',     type=int,  default=[0], nargs='+')
    parser.add_argument('--workers',    type=int,  default=4)
    parser.add_argument('--wbmode',     type=str,  default='disabled')
    cfg = parser.parse_args()
    # model
    cfg.lr_size = 192
    cfg.input_norm = False
    # optimizer
    cfg.lr = 1e-4

    # check arguments
    epochs: int = cfg.epochs
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
    trainset = SRDataset(dataset=cfg.dataset, scale=cfg.scale, lr_size=cfg.lr_size,
                         one_mean_std=(False,False,False))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, 
                    shuffle=True, num_workers=cfg.workers, pin_memory=True)

    # Initialize model
    if cfg.model == 'edsr_base':
        from mycv.external.edsr.edsr import EDSR
        model = EDSR('baseline', scale=2)
    model = model.to(device)

    # set device
    if len(cfg.device) > 1:
        print(f'Using DataParallel on {len(cfg.device)} devices...')
        model = torch.nn.DataParallel(model, device_ids=cfg.device)

    # loss function
    # loss = torch.nn.MSELoss(reduction='mean')
    # loss_func = torch.nn.L1Loss(reduction='mean')
    # msssim_func = MS_SSIM(max_val=1.0, reduction='mean')

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    log_parent = Path(f'runs/{cfg.project}')
    # new experiment
    _name = f'{cfg.model}_{cfg.dataset}_s{cfg.scale}'
    run_name = increment_dir(dir_root=log_parent, name=_name)
    log_dir = log_parent / run_name # wandb logging dir
    os.makedirs(log_dir, exist_ok=False)
    print(str(model), file=open(log_dir / 'model.txt', 'w'))

    # initialize wandb
    wbrun = wandb.init(project=cfg.project, group=cfg.group, name=run_name, config=cfg,
                       dir='runs/', resume='allow', id=None, mode=cfg.wbmode)
    cfg = wbrun.config
    cfg.log_dir = log_dir
    cfg.wandb_id = wbrun.id
    if not (log_dir / 'wandb_id.txt').exists():
        with open(log_dir / 'wandb_id.txt', 'w') as f:
            f.write(wbrun.id)

    # lr scheduler

    # ======================== start training ========================
    start_epoch, best_fitness = 0, 0.0
    results = defaultdict(float)
    niter = msg = None
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()

        epoch_loss, epoch_psnr = 0.0, 0.0
        pbar_title = ('%-10s' * 6) % (
            'Epoch', 'GPU_mem', 'lr',
            'loss', 'tr_psnr', 'psnr'
        )
        print('\n' + pbar_title) # title
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for bi, (lowr, highr) in pbar:
            niter = epoch * len(trainloader) + bi

            # adjust learning rate
            if (niter + 1) % 4000 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 2

            # data to device
            lowr = lowr.to(device=device)
            highr = highr.to(device=device)
            nB = lowr.shape[0]
            # forward
            superr = model(lowr)
            # loss
            loss = tnf.l1_loss(superr, highr, reduction='mean')
            # loss = loss_func(rec, imgs)
            # backward, update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # logging
            cur_lr = optimizer.param_groups[0]['lr']
            trps = _psnr(superr, highr)
            epoch_loss = (epoch_loss*bi + loss.item()) / (bi+1)
            epoch_psnr = (epoch_psnr*bi + trps) / (bi+1)
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            msg = ('%-10s' * 2 + '%-10.4g' * 4) % (
                f'{epoch}/{epochs-1}', f'{mem:.3g}G', cur_lr,
                epoch_loss, epoch_psnr, results['psnr']
            )
            pbar.set_description(msg)
            torch.cuda.reset_peak_memory_stats()

            # Weights & Biases logging
            if bi % 100 == 0:
                wbrun.log({
                    'general/lr': cur_lr,
                    'train/epoch_loss': epoch_loss,
                    'train/epoch_psnr': epoch_psnr,
                }, step=niter)

            # save output
            if bi % 100 == 0:
                save_output(lowr[0], superr[0], highr[0], log_dir / 'out0.png')
                save_output(lowr[1], superr[1], highr[1], log_dir / 'out1.png')

        # Evaluation
        # if bi % 200 == 0:
        if True:
            # message
            _msg = msg + ' Evaluating...'
            pbar.set_description(_msg)
            # evaluate
            _log_dic = {'general/epoch': epoch}
            results = sr_evaluate(model, 'div2k_val', cfg.scale, verbose=True)
            _log_dic.update({'metric/val_'+k: v for k,v in results.items()})
            # wandb log
            wbrun.log(_log_dic, step=niter)
            # Write evaluation results
            _msg = msg + '||' + '%10.4g'*1 % (results['psnr'])
            with open(log_dir / 'results.txt', 'a') as f:
                f.write(_msg + '\n')
            # save last checkpoint
            checkpoint = {
                'model'     : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler'    : None,
                'epoch'     : epoch,
                'psnr'      : results['psnr'],
                # 'msssim'    : results['msssim'],
            }
            torch.save(checkpoint, log_dir / 'last.pt')
            # save best checkpoint
            if results['psnr'] > best_fitness:
                best_fitness = results['psnr']
                torch.save(checkpoint, log_dir / 'best.pt')
            del checkpoint
        model.train()
            # ----Mini batch end
        # ----Epoch end
    # ----Training end


def get_optimizer(model, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    return optimizer


def save_output(lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor, save_path):
    lr = lr.detach().cpu().clamp_(0,255).round_().to(dtype=torch.uint8)
    sr = sr.detach().cpu().clamp_(0,255).round_().to(dtype=torch.uint8)
    hr = hr.detach().cpu().clamp_(0,255).round_().to(dtype=torch.uint8)
    lrh, hrh = lr.shape[1], hr.shape[1]
    assert hrh % lrh == 0
    scale = hrh // lrh
    lr = tnf.interpolate(lr.unsqueeze(0), scale_factor=(scale,scale), mode='nearest')
    lr = lr.squeeze(0)
    im = torch.cat([lr, sr, hr], dim=2)
    im = im.permute(1,2,0).numpy()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), im)


if __name__ == '__main__':
    print()
    train()
