import os
from tqdm import tqdm
from PIL import Image

from mycv.paths import IMAGENET_DIR

if __name__ == '__main__':
    img_dir = IMAGENET_DIR / 'val'

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
    debug = 1
