import argparse
from tqdm import tqdm
import torch
from torch.nn.functional import mse_loss
import torchvision


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dp', action='store_true')
    args = parser.parse_args()

    assert torch.cuda.is_available()
    # device = torch.device('cuda:0')

    if args.dp:
        device_ids = [0, 1]
    else:
        device_ids = [0]
    for id_ in device_ids:
        _dvc = torch.device(f'cuda:{id_}')
        print(f'Using device {_dvc}:', 'device property:',
            torch.cuda.get_device_properties(_dvc))

    model = torchvision.models.resnet50()
    model = model.to(device=device_ids[0])
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()

    shape = (256, 3, 224, 224)
    print(f'shape: {shape}')

    print('Training speed:')
    model.train()
    pbar = tqdm(range(128))
    for _ in pbar:
        x = torch.randn(shape, device=device_ids[0])
        y = model(x)
        label = torch.randn_like(y)
        loss = mse_loss(y, label)
        loss.backward()
        model.zero_grad()

        mem = torch.cuda.max_memory_allocated(device_ids[0]) / 1e9
        s = '%-10s' % (f'{mem:.3g}G')
        pbar.set_description(s)
        torch.cuda.reset_peak_memory_stats()

    print('no_grad speed:')
    pbar = tqdm(range(128))
    with torch.no_grad():
        for _ in pbar:
            x = torch.randn(shape, device=device_ids[0])
            y = model(x)

            mem = torch.cuda.max_memory_allocated(device_ids[0]) / 1e9
            s = '%-10s' % (f'{mem:.3g}G')
            pbar.set_description(s)
            torch.cuda.reset_peak_memory_stats()
