import os
from collections import OrderedDict
import torch

from mycv.paths import MYCV_DIR

def main():
    # wname = 'resnet50-19c8e357.pth'
    wname = 'res50_imgnet200.pt'
    wpath = MYCV_DIR / 'weights' / wname

    weights = torch.load(wpath)
    if 'model' in weights:
        weights = weights['model']
    assert isinstance(weights, OrderedDict)

    svname = wname.split('.')[0] + '.txt'
    svpath = MYCV_DIR / 'weights' / svname
    if svpath.is_file():
        print(f'Warning: {svpath} already exists. Removing it...')
        os.remove(svpath)

    for k, v in weights.items():
        vs = str(v.shape)
        msg = f'{k:<48}{vs:<20}'
        # print(msg)
        with open(svpath, 'a') as f:
            print(msg, file=f)


if __name__ == '__main__':
    main()
