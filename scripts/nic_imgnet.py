import os
from tqdm import tqdm
import cv2
import torch

from mycv.models.nic.nlaic import NLAIC
from mycv.datasets.imagenet import ImageNetCls
from mycv.utils.torch_utils import load_partial
from mycv.utils.image import pad_divisible
from mycv.paths import MYCV_DIR, IMAGENET_DIR


def get_img_paths(split):
    list_path = IMAGENET_DIR / f'annotations/{split}.txt'
    with open(list_path, 'r') as f:
        lines = f.read().strip().split('\n')
    lines = [s.split()[0] for s in lines]
    return lines


def main():
    print()
    weights_path = MYCV_DIR / 'weights/nlaic_msssim64.pt'
    save_dir = 'train_ms64'

    model = NLAIC(enable_bpp=False)
    load_partial(model, weights_path)
    model = model.cuda()
    model.eval()

    # from mycv.datasets.imcoding import nic_evaluate
    # results = nic_evaluate(model, input_norm=False, bar=True)
    # print(results)

    print('Loading image paths...')
    img_names = get_img_paths('train')

    for imname in tqdm(img_names):
        # print(impath)
        impath = IMAGENET_DIR / 'train' / imname

        im = cv2.imread(str(impath))
        # cv2.imwrite('ori.png', im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imh, imw = im.shape[:2]
        im = pad_divisible(im, div=16)
        im = torch.from_numpy(im).permute(2,0,1).float() / 255.0
        im = im.unsqueeze(0)

        input = im.cuda()
        try:
            with torch.no_grad():
                rec, _ = model.forward_nic(input)
        except Exception as e:
            print(e)
            print(impath, imh, imw, im.shape)
            exit()

        rec = rec.cpu().squeeze(0) * 255
        rec = rec.to(dtype=torch.uint8).permute(1,2,0).numpy()
        rec = rec[:imh, :imw, :]

        rec = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('rec.png', rec)

        savepath = IMAGENET_DIR / save_dir / imname
        if not savepath.parent.is_dir():
            savepath.parent.mkdir(parents=True)

        cv2.imwrite(str(savepath), rec)
        # break

if __name__ == '__main__':
    main()
