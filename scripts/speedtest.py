import argparse
from tqdm import tqdm
import torch
import torch.cuda.amp as amp
from torch.nn.functional import mse_loss
import torchvision


def test1(model, shape, device, enable_amp=False):
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scaler = amp.GradScaler(enabled=enable_amp)

    x = torch.randn(shape, device=device)
    y = model(x)
    y.sum().backward()

    pbar = tqdm(range(128))
    for _ in pbar:
        x = torch.randn(shape, device=device)
        with amp.autocast(enabled=enable_amp):
            y = model(x)
            label = torch.randn_like(y)
            loss = mse_loss(y, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        mem = torch.cuda.max_memory_allocated(device) / 1e9
        s = '%-10s' % (f'CUDA mem: {mem:.3g}G')
        pbar.set_description(s)
        torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=[0], nargs='+')
    args = parser.parse_args()

    assert torch.cuda.is_available()

    for id_ in args.device:
        _dvc = torch.device(f'cuda:{id_}')
        print(f'Using device {_dvc}:', 'device property:',
            torch.cuda.get_device_properties(_dvc))

    device = torch.device(f'cuda:{args.device[0]}')
    model = torchvision.models.resnet50()
    model = model.to(device=device)

    if len(args.device) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device)

    shape = (256, 3, 224, 224)
    print(f'shape: {shape}')

    print('Training speed, no AMP:')
    test1(model, shape, device, enable_amp=False)

    print('Training speed, AMP:')
    test1(model, shape, device, enable_amp=True)

    print('no_grad speed:')
    pbar = tqdm(range(128))
    with torch.no_grad():
        for _ in pbar:
            x = torch.randn(shape, device=device)
            y = model(x)

            mem = torch.cuda.max_memory_allocated(device) / 1e9
            s = '%-10s' % (f'{mem:.3g}G')
            pbar.set_description(s)
            torch.cuda.reset_peak_memory_stats()
