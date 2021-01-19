import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt
import PIL.Image
import torchvision.transforms.functional as tvf

from mycv.paths import CLIC_DIR


def check(img_dir: Path, file_num):
    if not img_dir.exists():
        print(f'{img_dir} does not exist.')
        return
    print(f'Checking {img_dir}...')
    img_names = os.listdir(img_dir)
    img_names.sort()
    if len(img_names) != file_num:
        print(f'Warning: number of files in {img_dir} is {len(img_names)}, should be {file_num}.')

    for imname in tqdm(img_names):
        impath = img_dir / imname
        im_pil = PIL.Image.open(impath)
        im_pil.verify()
        # _ = cv2.imread(str(impath))


if __name__ == "__main__":
    assert CLIC_DIR.is_dir(), f'CLIC_DIR = {CLIC_DIR} does not exist.'
    check(CLIC_DIR / 'train', 1633)
    check(CLIC_DIR / 'train400', 34136)
