import argparse
import os
pjoin = os.path.join
import yaml
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
import torchvision as tv

from mycv.utils.general import increment_dir

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str,  default='')
    parser.add_argument('--train_dir',  type=str,  default='path')
    parser.add_argument('--test_dir',   type=str,  default='path')
    parser.add_argument('--resume',     type=str,  default='')
    parser.add_argument('--img_size',   type=int,  default=256)
    parser.add_argument('--batch_size', type=int,  default=64)
    parser.add_argument('--amp',        type=bool, default=False)
    parser.add_argument('--optimizer',  type=str,  default='Adam')
    parser.add_argument('--epochs',     type=int,  default=16)
    parser.add_argument('--workers',    type=int,  default=8)
    parser.add_argument('--metric',     type=str,  default='msssim')
    parser.add_argument('--log_root',   type=str,  default='runs/nic/')
    args = parser.parse_args()
    print(args)

    np.random.seed(1); torch.manual_seed(1)
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    metric: str = args.metric.lower()
    epochs: int = args.epochs

    hyp = {
        'lr': 1e-4,
        'momentum': 0.937, # SGD
        'nesterov': True, # SGD
        'msssim_lambda': 0.64
    }

    # model
    if args.name == 'nlaic':
        from models.imcoding.nlaic import NLAIC
        model = NLAIC(test_only=False)
    elif args.name == 'mini':
        from models.imcoding.mininlaic import MiniNLAIC
        model = MiniNLAIC(encode_only=args.enc_only)
    model = model.cuda()

    # Dataset
    dataset = tv.datasets.ImageFolder(
        root=args.train_dir,
        transform=tv.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=False
    )

    # optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=hyp['lr'])
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr'],
                                    momentum=hyp['momentum'], nesterov=hyp['nesterov'])
    else:
        raise NotImplementedError()
    optimizer.zero_grad()
    # mixed precision training
    scaler = amp.GradScaler(enabled=args.amp)

    if args.resume:
        # resume
        log_dir = pjoin(args.log_root, args.resume)
        assert os.path.isdir(log_dir)
        checkpoint = torch.load(pjoin(log_dir, 'weights/last.pt'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_fitness = checkpoint.get(metric, 0)
    else:
        # new experiment
        log_dir = increment_dir(name=args.name, log_root=args.log_root)
        assert not os.path.exists(log_dir)
        # make dir and save configs
        os.makedirs(pjoin(log_dir,'weights'))
        yaml.dump(hyp, open(pjoin(log_dir,'hyp.yaml'), 'w'), sort_keys=False)
        yaml.dump(vars(args), open(pjoin(log_dir, 'args.yaml'), 'w'), sort_keys=False)
        print(str(model), file=open(pjoin(log_dir, 'model.txt'), 'w'))
        start_epoch = 0
        best_fitness = 0
    print(f'Logging to {log_dir}')
    last_path = pjoin(log_dir, 'weights/last.pt')
    best_path = pjoin(log_dir, 'weights/best.pt')
    txt_path  = pjoin(log_dir, 'results.txt')

    # initialize tensorboard
    print(f'Start Tensorboard with "tensorboard --logdir {args.log_root}"',
          'view at http://localhost:6006/')
    tb_writer = SummaryWriter(log_dir=log_dir)

    # loss functions
    msssim_func = torch_msssim.MS_SSIM(max_val=1.).cuda()
    mse_func = torch.nn.MSELoss(reduction='mean')

    for epoch in range(start_epoch, epochs):
        avg_loss = torch.zeros(4, dtype=torch.float32)
        cur_lr = optimizer.param_groups[0]['lr']

        pbar_title = ('%-10s' * 4) % ('Epoch', 'GPU_mem', 'lr', 'Loss')
        print('\n' + pbar_title) # title
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (imgs, _) in pbar:
            niter = epoch * len(train_loader) + i
            assert imgs.shape[2:4] == (args.img_size, args.img_size)
            # debugging
            # if False:
            #     import matplotlib.pyplot as plt
            #     im = imgs[0].permute(1,2,0).numpy()
            #     plt.imshow(im); plt.show()
            num_pixels = imgs.shape[0] * imgs.shape[2] * imgs.shape[3]
            imgs = imgs.cuda()

            with amp.autocast(enabled=args.amp):
                rec_imgs, (xp1, xp2) = model(imgs, if_training=1)
                # reconstruction loss
                if args.metric == 'mse':
                    l_rec = mse_func(rec_imgs, imgs)
                    lmb = hyp['msssim_lambda'] * 50
                elif args.metric == 'msssim':
                    if epoch < 1:
                        l_rec = mse_func(rec_imgs, imgs)
                        lmb = hyp['msssim_lambda'] * 50
                    else:
                        l_rec = 1.0 - msssim_func(rec_imgs, imgs)
                        lmb = hyp['msssim_lambda']
                else:
                    raise NotImplementedError()
                # bit rate loss
                if not args.enc_only:
                    bpp1 = torch.sum(torch.log(xp1)) / (-np.log(2) * num_pixels)
                    bpp2 = torch.sum(torch.log(xp2)) / (-np.log(2) * num_pixels)
                else:
                    bpp1, bpp2 = torch.zeros(2).cuda()
                # total loss
                Loss = lmb * l_rec + 0.01 * (bpp1 + bpp2)

            scaler.scale(Loss).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # gradient clip
            scaler.step(optimizer)
            scaler.update()
            # Loss.backward()
            # optimizer.step()
            optimizer.zero_grad()

            # logging
            mem = torch.cuda.max_memory_allocated() / 1e9
            s = ('%-10s' * 2 + '%-10.4g' * 4) % (
                 f'{epoch}/{epochs}', f'{mem:.3g}G',
                 cur_lr, l_rec, (bpp1+bpp2), Loss
            )
            pbar.set_description(s)
            torch.cuda.reset_peak_memory_stats()

            # total training loss of the whole epoch
            cur_loss = torch.stack([l_rec, bpp1, bpp2, Loss]).detach().cpu()
            avg_loss = (avg_loss*i + cur_loss) / (i+1)

            # save model weights
            if niter % 1000 == 0 and niter > 0:
                # validation
                results = test(model, test_dir=args.test_dir, bpp_metric=(not args.enc_only))
                # logging to tensorboard
                for s in ['msssim', 'psnr', 'bpp']:
                    tb_writer.add_scalar(f'compression/{s}', results[s], global_step=niter)
                for i, s in enumerate(['rec', 'bpp1', 'bpp2', 'total']):
                    tb_writer.add_scalar(f'loss/{s}', avg_loss[i], global_step=niter)
                # logging to file
                s = ('%-10s' * 2 + '%-10.4g' * 4) % (f'{epoch}/{epochs-1}', mem, *avg_loss)
                s += '||'
                s += ('%-10.4g' * 3) % (results['msssim'], results['psnr'], results['bpp'])
                print(s, file=open(txt_path, 'a'))

                # save last checkpoint
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch,
                    metric: results[metric]
                }
                torch.save(checkpoint, last_path)
                # save best checkpoint
                if results[metric] > best_fitness:
                    torch.save(checkpoint, best_path)
                    best_fitness = results[metric]
                del checkpoint

                # demo
                im = cv2.imread('images/zidane.jpg')
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = torch.from_numpy(im).float() / 255
                im = im.permute(2, 0, 1).unsqueeze(0).cuda()
                model.eval()
                with torch.no_grad():
                    x = model.encode(im)
                    im = model.decode(x, return_type='array')
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                cv2.imwrite(pjoin(log_dir, f'zidane_rec{epoch}.jpg'), im)

                print(pbar_title) # title
            model.train()
        # test


if __name__ == '__main__':
    train()
