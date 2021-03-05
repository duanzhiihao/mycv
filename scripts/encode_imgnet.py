import os
from tqdm import tqdm
import cv2
import torch

from mycv.models.nic.nlaic import NLAIC
from mycv.utils.torch_utils import load_partial
from mycv.utils.image import pad_divisible, crop_divisible
from mycv.paths import MYCV_DIR, IMAGENET_DIR


def get_img_paths(split):
    list_path = IMAGENET_DIR / f'annotations/{split}.txt'
    with open(list_path, 'r') as f:
        lines = f.read().strip().split('\n')
    lines = [s.split()[0] for s in lines]
    return lines


def main():
    split = 'val'
    weights_path = MYCV_DIR / 'weights/nlaic/nlaic_ms64.pt'
    save_dir = f'{split}_ms64'
    print(f'Saving to {save_dir}')

    model = NLAIC(enable_bpp=False)
    load_partial(model, weights_path)
    model = model.cuda()
    model.eval()

    # from mycv.datasets.imcoding import nic_evaluate
    # results = nic_evaluate(model, input_norm=False, bar=True)
    # print(results)

    print('Loading image paths...')
    img_names = get_img_paths(split)

    for imname in tqdm(img_names):
        # print(impath)
        impath = IMAGENET_DIR / split / imname

        im = cv2.imread(str(impath))
        # cv2.imwrite('ori.png', im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imh, imw = im.shape[:2]
        im = crop_divisible(im, div=16)
        im = torch.from_numpy(im).permute(2,0,1).float() / 255.0
        im = im.unsqueeze(0)

        input = im.cuda()
        try:
            with torch.no_grad():
                comp, xp = model.encoder(input)
                assert xp is None
        except Exception as e:
            print(e)
            print(impath, imh, imw, im.shape)
            exit()

        svname = imname.replace('.JPEG', '.pt')
        savepath = IMAGENET_DIR / save_dir / svname
        # if not savepath.parent.is_dir():
        #     savepath.parent.mkdir(parents=True)
        torch.save(comp, savepath)
        # break


if __name__ == '__main__':
    print()
    main()
