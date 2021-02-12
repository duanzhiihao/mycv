import os
from tqdm import tqdm
import numpy as np
import piexif
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from mycv.paths import IMAGENET_DIR


def fix_jpeg():
    """ fix broken JPEG file
    https://discuss.pytorch.org/t/corrupt-exif-data-messages-when-training-imagenet/17313/5
    """
    jpgpath = str(IMAGENET_DIR / 'train/n02105855/n02105855_2933.JPEG')
    pngpath = jpgpath.replace('.JPEG', '.PNG')
    os.rename(jpgpath, pngpath)
    im = cv2.imread(pngpath)
    assert im is not None
    cv2.imwrite(jpgpath, im)
    os.remove(pngpath)


def remove_exif(split: str='train'):
    """ remove EXIF data
    """
    annpath = IMAGENET_DIR / f'annotations/{split}.txt'
    with open(annpath, 'r') as f:
        file_list = f.read().strip().split('\n')
    img_names = [line.split()[0] for line in file_list]
    img_dir = IMAGENET_DIR / split
    for imname in tqdm(img_names):
        impath = img_dir / imname
        piexif.remove(str(impath))


if __name__ == '__main__':
    # fix_jpeg()
    remove_exif('train')
    remove_exif('val')
