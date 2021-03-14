import argparse
import os
from tqdm import tqdm
from PIL import Image
import cv2

from mycv.paths import IMAGENET_DIR


def get_imgpaths(dataset, split):
    if dataset == 'imagenet':
        raise NotImplementedError()
    elif dataset == 'cityscapes':
        from mycv.paths import CITYSCAPES_DIR
        ann_path = CITYSCAPES_DIR / f'annotations/{split}.txt'
        data_list = open(ann_path, 'r').read().strip().split('\n')
        img_paths = [CITYSCAPES_DIR / s.split()[0] for s in data_list]
    else:
        raise ValueError()
    return img_paths


def get_bpp(dataset='cityscapes', split='val'):
    img_paths = get_imgpaths(dataset, split)

    total_bits = 0
    total_pixels = 0
    pbar = tqdm(img_paths)
    for impath in pbar:
        bytes = os.path.getsize(impath)

        im = Image.open(impath)
        pixels = im.height * im.width
        im.close()

        bits = bytes * 8
        total_bits += bits
        total_pixels += pixels
        bpp = total_bits / total_pixels
        pbar.set_description(f'bpp: {bpp:.4g}')

    print(total_bits / total_pixels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cityscapes')
    parser.add_argument('--split', type=str, default='val')
    args = parser.parse_args()

    get_bpp(dataset=args.dataset, split=args.split)
