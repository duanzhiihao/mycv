from mycv.utils.general import disable_multithreads
disable_multithreads()
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import defaultdict
import cv2
import torch
from torch.optim.lr_scheduler import LambdaLR
import wandb

from mycv.utils.general import increment_dir
from mycv.utils.torch_utils import set_random_seeds, warmup_cosine
from mycv.utils.coding import MS_SSIM, cal_bpp
from mycv.datasets.loadimgs import LoadImages, kodak_val


def train():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',    type=str,  default='imcoding')
    parser.add_argument('--group',      type=str,  default='default')
    parser.add_argument('--datasets',   type=str,  default=['COCO','CLIC400'], nargs='+')
    parser.add_argument('--model',      type=str,  default='mini')
    parser.add_argument('--loss',       type=str,  default='mse', choices=['mse','msssim'])
    parser.add_argument('--lmbda',      type=float,default=32)
    parser.add_argument('--batch_size', type=int,  default=16)
    parser.add_argument('--epochs',     type=int,  default=80)
    parser.add_argument('--device',     type=int,  default=[0], nargs='+')
    parser.add_argument('--workers',    type=int,  default=2)
    parser.add_argument('--wbmode',     type=str,  default='disabled')
    cfg = parser.parse_args()
    # model
    cfg.img_size = 256
    cfg.input_norm = False
    # optimizer
    cfg.lr = 5e-5
    # lr scheduler
    cfg.lrf = 0.2 # min lr factor
    cfg.lr_warmup_epochs = 0

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
    trainset = LoadImages(datasets=cfg.datasets, img_size=cfg.img_size,
                          input_norm=cfg.input_norm)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, 
                    shuffle=True, num_workers=cfg.workers, pin_memory=True)

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

    # set device
    if len(cfg.device) > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.device)

    # loss function
    mse_func = torch.nn.MSELoss(reduction='mean')
    msssim_func = MS_SSIM(max_val=1.0, reduction='mean')

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    log_parent = Path(f'runs/{cfg.project}')
    wb_id = None
    results = defaultdict(float)
    # new experiment
    run_name = increment_dir(dir_root=log_parent, name=cfg.model)
    log_dir = log_parent / run_name # wandb logging dir
    os.makedirs(log_dir, exist_ok=False)
    print(str(model), file=open(log_dir / 'model.txt', 'w'))
    start_epoch, best_fitness = 0, 0.0

    # initialize wandb
    wbrun = wandb.init(project=cfg.project, group=cfg.group, name=run_name, config=cfg,
                       dir='runs/', resume='allow', id=wb_id, mode=cfg.wbmode)
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

    # ======================== start training ========================
    niter = s = None
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()

        epoch_rec, epoch_bpp, epoch_loss = [0.0] * 3
        pbar_title = ('%-10s' * 9) % (
            'Epoch', 'GPU_mem', 'lr',
            'l_rec', 'l_bpp', 'loss', 'psnr', 'msssim', 'bpp'
        )
        print('\n' + pbar_title) # title
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for bi, imgs in pbar:
            niter = epoch * len(trainloader) + bi
            imgs = imgs.to(device=device)
            nB, nC, nH, nW = imgs.shape
            # forward
            rec, probs = model(imgs)
            if cfg.loss == 'mse':
                l_rec = mse_func(rec, imgs)
            else:
                raise NotImplementedError()
            if probs is not None:
                p1, p2 = probs
                l_bpp = cal_bpp(p1, nB*nH*nW) + cal_bpp(p2, nB*nH*nW)
            else:
                l_bpp = torch.zeros(1, device=device)
            loss = cfg.lmbda * l_rec + 0.01*l_bpp
            # backward, update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Scheduler
            scheduler.step()

            # logging
            cur_lr = optimizer.param_groups[0]['lr']
            epoch_rec  = (epoch_rec*bi + l_rec.item()) / (bi+1)
            epoch_bpp  = (epoch_bpp*bi + l_bpp.item()) / (bi+1)
            epoch_loss = (epoch_loss*bi + loss.item()) / (bi+1)
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            s = ('%-10s' * 2 + '%-10.4g' * 7) % (
                f'{epoch}/{epochs-1}', f'{mem:.3g}G', cur_lr,
                epoch_rec, epoch_bpp, epoch_loss,
                results['psnr'], results['msssim'], results['bpp']
            )
            pbar.set_description(s)
            torch.cuda.reset_peak_memory_stats()

            # Weights & Biases logging
            if bi % 100 == 0:
                wbrun.log({
                    'general/lr': cur_lr,
                    'train/epoch_rec': epoch_rec,
                    'train/epoch_bpp': epoch_bpp,
                    'train/epoch_loss': epoch_loss,
                }, step=niter)

            # save output
            if bi % 100 == 0:
                save_output(imgs[0], rec[0], log_dir / 'out0.png')
                save_output(imgs[1], rec[1], log_dir / 'out1.png')

            # Evaluation
            if bi % 200 == 0:
                _log_dic = {'general/epoch': epoch}
                results = kodak_val(model, input_norm=cfg.input_norm, verbose=False)
                _log_dic.update({'metric/plain_val_'+k: v for k,v in results.items()})
                # wandb log
                wbrun.log(_log_dic, step=niter)
                # Write evaluation results
                _res = s + '||' + '%10.4g'*3 % (results['psnr'],results['msssim'],results['bpp'])
                with open(log_dir / 'results.txt', 'a') as f:
                    f.write(_res + '\n')
                # save last checkpoint
                checkpoint = {
                    'model'     : model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler'    : None,
                    'epoch'     : epoch,
                    'psnr'      : results['psnr'],
                    'msssim'    : results['msssim'],
                    'bpp'       : results['bpp'],
                }
                torch.save(checkpoint, log_dir / 'last.pt')
                # save best checkpoint
                if results['msssim'] > best_fitness:
                    best_fitness = results['msssim']
                    torch.save(checkpoint, log_dir / 'best.pt')
                del checkpoint
            model.train()
            # ----Mini batch end
        # ----Epoch end
    # ----Training end


def save_output(input_: torch.Tensor, output: torch.Tensor, save_path):
    imt, imp = input_.cpu(), output.detach().cpu().float()
    imp.clamp_(min=0, max=1)
    assert imt.shape == imp.shape and imt.dim() == 3
    im = torch.cat([imt, imp], dim=2)
    im = im.permute(1,2,0) * 255
    im = im.to(dtype=torch.uint8).numpy()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), im)


if __name__ == '__main__':
    print()
    train()
