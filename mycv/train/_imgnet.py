from mycv.utils.general import disable_multithreads
disable_multithreads()
import os
from pathlib import Path
import argparse
import time
from tqdm import tqdm
from collections import defaultdict
import math
import torch
import torch.cuda.amp as amp

from mycv.utils.general import increment_dir
from mycv.utils.torch_utils import set_random_seeds, ModelEMA, is_parallel
from mycv.utils.lr_schedulers import adjust_lr_threestep
from mycv.utils.image import save_tensor_images
from mycv.datasets.imagenet import ImageNetCls, imagenet_val


def main():
    print()
    trainw = TrainWrapper()
    trainw.train()


def get_config():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    # wandb setting
    parser.add_argument('--project',    type=str,  default='imagenet')
    parser.add_argument('--group',      type=str,  default='default')
    parser.add_argument('--wbmode',     type=str,  default='disabled')
    # model setting
    parser.add_argument('--model',      type=str,  default='res50')
    # resume setting
    parser.add_argument('--resume',     type=str,  default='')
    parser.add_argument('--initial',    type=str,  default='')
    # optimization setting
    parser.add_argument('--batch_size', type=int,  default=256)
    parser.add_argument('--accum_bs',   type=int,  default=None)
    parser.add_argument('--lr',         type=float,default=0.1)
    parser.add_argument('--amp',        type=bool, default=True)
    parser.add_argument('--ema',        type=bool, default=True)
    parser.add_argument('--epochs',     type=int,  default=100)
    # device setting
    parser.add_argument('--fixseed',    action='store_true')
    parser.add_argument('--device',     type=int,  default=[0], nargs='+')
    parser.add_argument('--workers',    type=int,  default=8)
    cfg = parser.parse_args()

    # model
    cfg.img_size = 224
    cfg.input_norm = False
    # optimizer
    cfg.momentum = 0.9
    cfg.weight_decay = 0.0001
    cfg.nesterov = False
    cfg.accum_batch_size = cfg.accum_bs or cfg.batch_size
    cfg.accum_num = max(1, round(cfg.accum_batch_size // cfg.batch_size))
    # EMA
    cfg.ema_warmup_epochs = round(cfg.epochs / 20)
    # logging
    cfg.metric = 'top1' # metric to save best model

    if cfg.fixseed: # fix random seeds for reproducibility
        set_random_seeds(1)
    torch.backends.cudnn.benchmark = True
    return cfg


class TrainWrapper():
    def __init__(self) -> None:
        # config
        self.cfg = get_config()

        # core
        self.set_device_()
        self.set_dataset_()
        self.set_model_()
        self.set_loss_()
        self.set_optimizer_()
        self.set_ema_()

        # logging
        self.prepare_training_()
        self.set_wandb_()

    def set_device_(self):
        cfg = self.cfg

        # device setting
        assert torch.cuda.is_available()
        for _id in cfg.device:
            print(f'Using device {_id}:', torch.cuda.get_device_properties(_id))
        device = torch.device(f'cuda:{cfg.device[0]}')
        bs_each = cfg.batch_size // len(cfg.device)
        print('Batch size on each single GPU =', bs_each)
        print(f'Gradient accmulation: {cfg.accum_num} backwards() -> one step()')
        print(f'Effective batch size: {cfg.accum_batch_size}', '\n')

        self.device = device
        self.cfg.bs_each = bs_each

    def set_dataset_(self):
        cfg = self.cfg

        print('Initializing Datasets and Dataloaders...')
        if cfg.group == 'default':
            train_split = 'train'
            val_split = 'val'
        elif cfg.group == 'mini200':
            train_split = 'train200_600'
            val_split = 'val200_600'
        else:
            train_split = f'train_{cfg.group}'
            val_split = f'val_{cfg.group}'
        # training set
        trainset = ImageNetCls(train_split, img_size=cfg.img_size, input_norm=cfg.input_norm)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size,
                        shuffle=True, num_workers=cfg.workers, pin_memory=True)
        cfg.num_class = trainset.num_class
        # test set
        testloader = torch.utils.data.DataLoader(
            ImageNetCls(split=val_split, img_size=cfg.img_size, input_norm=cfg.input_norm),
            batch_size=cfg.bs_each//2, shuffle=False, num_workers=cfg.workers//2,
            pin_memory=True, drop_last=False
        )
        print(f'Training on {train_split}, val on {val_split}, {cfg.num_class} classes', '\n')
        print(f'First training image path: {trainset._img_paths[0]}')
        print(f'First val image path: {testloader.dataset._img_paths[0]}', '\n')

        self.trainset    = trainset
        self.trainloader = trainloader
        self.testloader  = testloader

    def set_model_(self, cfg):
        name, num_class = cfg.name, cfg.nu

        if name == 'res50':
            from mycv.models.cls.resnet import resnet50
            model = resnet50(num_classes=num_class)
        elif name == 'res101':
            from mycv.models.cls.resnet import resnet101
            model = resnet101(num_classes=num_class)
        elif name == 'res152':
            from mycv.models.cls.resnet import resnet152
            model = resnet152(num_classes=num_class)
        elif name.startswith('yolov5'):
            from mycv.models.yolov5.cls import YOLOv5Cls
            assert name[-1] in ['s', 'm', 'l']
            model = YOLOv5Cls(model=name[-1], num_class=num_class)
        elif name.startswith('csp'):
            from mycv.models.yolov5.cls import CSP
            assert name[-1] in ['s', 'm', 'l']
            model = CSP(model=name[-1], num_class=num_class)
        elif name == 'efb0':
            from mycv.external.efficientnet.model import EfficientNet
            model = EfficientNet.from_name('efficientnet-b0', num_class=num_class)
        else:
            raise ValueError()
        print(f'Using model {type(model)}, {num_class} classes', '\n')

        self.model = model.to(self.device)
        if len(cfg.device) > 1: # DP mode
            model = torch.nn.DataParallel(model, device_ids=cfg.device)

    def set_loss_(self):
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

    def set_optimizer_(self):
        cfg, model = self.cfg, self.model

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
        # optimizer
        optimizer = torch.optim.SGD(parameters, lr=cfg.lr, momentum=cfg.momentum,
                                    nesterov=cfg.nesterov)

        self.optimizer = optimizer
        self.scaler = amp.GradScaler(enabled=cfg.amp) # Automatic mixed precision

    def set_ema_(self):
        cfg = self.cfg

        # Exponential moving average
        if cfg.ema:
            warmup = cfg.ema_warmup_epochs * len(self.trainloader)
            ema = ModelEMA(self.model, decay=0.9999, warmup=warmup)
        else:
            ema = None

        self.ema = ema

    def prepare_training_(self):
        cfg = self.cfg

        log_parent = Path(f'runs/{cfg.project}')
        if cfg.resume: # resume
            run_name = cfg.resume
            log_dir = log_parent / run_name
            assert log_dir.is_dir()
            checkpoint = torch.load(log_dir / 'last.pt')
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['epoch']
            results = checkpoint['results']
            if self.ema is not None:
                start_iter = start_epoch * len(self.trainloader)
                self.ema.updates = start_iter // cfg.accum_num # set EMA update number
        else: # new experiment
            _base = f'{cfg.model}'
            run_name = increment_dir(dir_root=log_parent, name=_base)
            log_dir = log_parent / run_name # wandb logging dir
            os.makedirs(log_dir, exist_ok=False)
            print(str(self.model), file=open(log_dir / 'model.txt', 'w'))
            start_epoch = 0
            results = defaultdict(float)
            if cfg.initial: # initialize from weights
                checkpoint = torch.load(log_parent / cfg.initial / 'best.pt')
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scaler.load_state_dict(checkpoint['scaler'])
        
        self._run_name = run_name
        self._log_dir = log_dir
        self._start_epoch = start_epoch
        self._results = results
        self._best_fitness = results[cfg.metric]

    def set_wandb_(self):
        cfg = self.cfg

        # initialize wandb
        import wandb
        wbrun = wandb.init(project=cfg.project, group=cfg.group, name=self._run_name,
                           config=cfg, dir='runs/', mode=cfg.wbmode)
        cfg = wbrun.config
        cfg.log_dir = str(self._log_dir)
        cfg.wandb_id = wbrun.id
        with open(self._log_dir / 'wandb_id.txt', 'a') as f:
            print(wbrun.id, file=f)

        self.wbrun = wbrun
        self.cfg = cfg

    def train(self):
        cfg = self.cfg
        model = self.model

        # evaluate before training
        self.evaluate()
        # ======================== start training ========================
        for epoch in range(self._start_epoch, cfg.epochs):
            time.sleep(0.1)
            model.train()
            self.adjust_lr_(epoch)

            self.init_logging_()
            print('\n' + self._pbar_title)
            pbar = tqdm(enumerate(self.trainloader), total=len(self.trainloader))
            for bi, (imgs, labels) in pbar:
                niter = epoch * len(self.trainloader) + bi

                imgs = imgs.to(device=self.device)
                labels = labels.to(device=self.device)

                # forward
                with amp.autocast(enabled=cfg.amp):
                    p = model(imgs)
                    assert p.shape == (imgs.shape[0], cfg.num_class)
                    loss = self.loss_func(p, labels)
                    loss = loss / cfg.accum_num / len(cfg.device)
                # loss is averaged over batch and gpus
                self.scaler.scale(loss).backward() # backward, update
                if niter % cfg.accum_num == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if cfg.ema:
                        self.ema.update(model)

                self.logging(pbar, epoch, bi, niter, imgs, labels, p, loss)

            self.evaluate()
        print('Training finished. results:', self._results)

    def adjust_lr_(self, epoch):
        cfg = self.cfg
        assert cfg.epochs >= 3
        period = math.ceil(cfg.epochs / 3)
        lr = cfg.lr * (0.1 ** (epoch // period))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def init_logging_(self):
        self._epoch_stats = [
            ('tr_loss', 0),
            ('tr_acc', 0)
        ]
        keys = [k for k,v in self._epoch_stats]
        sn = 5 + len(keys)
        self._pbar_title = ('%-10s' * sn) % (
            'Epoch', 'GPU_mem', 'lr', *keys, 'top1', 'top1_real',
        )

    def logging(self, pbar, epoch, bi, niter, imgs, labels, p, loss):
        cfg = self.cfg
        cur_lr = self.optimizer.param_groups[0]['lr']
        acc = compute_acc(p.detach(), labels)
        stats = torch.Tensor([loss.item(), acc])
        self._epoch_stats.mul_(bi).add_(stats).div_(bi+1)
        mem = torch.cuda.max_memory_allocated(self.device) / 1e9
        torch.cuda.reset_peak_memory_stats()
        sn = 1 + len(self._epoch_stats) + 2
        msg = ('%-10s' * 2 + '%-10.4g' * sn) % (
            f'{epoch}/{cfg.epochs-1}', f'{mem:.3g}G', cur_lr,
            *self._epoch_stats,
            100*self._results['top1'], 100*self._results['top1_real'],
        )
        pbar.set_description(msg)
        self._msg = msg

        # Weights & Biases logging
        if niter % 100 == 0:
            save_tensor_images(imgs[:4], save_path=self._log_dir / 'imgs.png')
            _log_dic = {
                'general/lr': cur_lr,
                'ema/n_updates': self.ema.updates if cfg.ema else 0,
                'ema0/decay': self.ema.get_decay() if cfg.ema else 0
            }
            _log_dic.update({'train/'+k: v for k,v in self._epoch_stats})
            self.wbrun.log(_log_dic, step=niter)

    def evaluate(self, epoch, niter):
        # Evaluation
        _log_dic = {'general/epoch': epoch}
        _eval_model = self.model
        results = imagenet_val(_eval_model, testloader=self.testloader)
        _log_dic.update({'metric/plain_val_'+k: v for k,v in results.items()})

        if self.cfg.ema:
            _eval_model = self.ema.ema
            results = imagenet_val(_eval_model, testloader=self.testloader)
            _log_dic.update({f'metric/ema_val_'+k: v for k,v in results.items()})

        _cur_fitness = results[self.cfg.metric]
        # wandb log
        self.wbrun.log(_log_dic, step=niter)
        # Write evaluation results
        msg = self._msg + '||' + '%10.4g' * 1 % (_cur_fitness)
        with open(self._log_dir / 'results.txt', 'a') as f:
            f.write(msg + '\n')
        # save last checkpoint
        checkpoint = {
            'model'     : _eval_model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'scaler'    : self.scaler.state_dict(),
            'epoch'     : epoch,
            'results'   : results,
        }
        torch.save(checkpoint, self._log_dir / 'last.pt')
        # save best checkpoint
        if _cur_fitness > self._best_fitness:
            self._best_fitness = _cur_fitness
            torch.save(checkpoint, self._log_dir / 'best.pt')
        return results


def compute_acc(p: torch.Tensor, labels: torch.LongTensor):
    assert not p.requires_grad and p.device == labels.device
    assert p.dim() == 2 and p.shape[0] == labels.shape[0]
    _, p_cls = torch.max(p, dim=1)
    tp = (p_cls == labels)
    acc = float(tp.sum()) / len(tp)
    assert 0 <= acc <= 1
    return acc * 100.0


if __name__ == '__main__':
    main()

    # from mycv.models.cls.resnet import resnet50
    # model = resnet50(num_classes=1000)
    # weights = torch.load('weights/resnet50-19c8e357.pth')
    # model.load_state_dict(weights)
    # model = model.cuda()
    # model.eval()
    # results = imagenet_val(model, img_size=224, batch_size=64, workers=4)
    # print(results['top1'])
