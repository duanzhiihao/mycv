import argparse
import os
from tqdm import tqdm
from PIL import Image
import cv2

from mycv.paths import IMAGENET_DIR
from mycv.datasets.imagenet import ImageNetCls


def get_bpp(split='val_jpeg50'):
    img_dir = IMAGENET_DIR / split

    img_names = os.listdir(img_dir)
    img_names.sort()

    total_bits = 0
    total_pixels = 0
    pbar = tqdm(img_names)
    for imname in pbar:
        impath = img_dir / imname
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


def jpeg(split='val', quality=50):
    img_dir = IMAGENET_DIR / split
    save_dir = IMAGENET_DIR / f'{split}_jpeg{quality}'
    if not save_dir.is_dir():
        print(f'{save_dir} does not exist. Creating it...')
        save_dir.mkdir(parents=False, exist_ok=False)

    imlabels = ImageNetCls.get_image_label_pairs(split)
    pbar = tqdm(imlabels)
    for imname, _ in pbar:
        impath = img_dir / imname
        im = cv2.imread(str(impath))
        assert im is not None
        svpath = save_dir / imname
        if not svpath.parent.is_dir():
            svpath.parent.mkdir(parents=False, exist_ok=False)
        cv2.imwrite(str(svpath), im, [cv2.IMWRITE_JPEG_QUALITY, quality])
        wnid = svpath.parent.stem
        pbar.set_description(f'Processing: {wnid}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--quality', type=int, default=23)
    args = parser.parse_args()

    jpeg(split=args.split, quality=args.quality)
    # get_bpp(split=f'{args.split}_jpeg{args.quality}')
