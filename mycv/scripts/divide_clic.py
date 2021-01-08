import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

from mycv.paths import CLIC_DIR


def main():
    divide(400)


def divide(tgt_size=256):
    assert CLIC_DIR.is_dir()
    img_dir = CLIC_DIR / 'train'
    img_names = os.listdir(img_dir)

    new_dir = CLIC_DIR / f'train{tgt_size}'
    assert not new_dir.exists(), 'target dir exists. Please doublecheck.'
    os.mkdir(new_dir)

    for imname in tqdm(img_names):
        assert imname.endswith('.png')
        impath = str(img_dir / imname)
        im = cv2.imread(impath)
        assert im is not None, impath

        imh, imw = im.shape[:2]
        # save the whole image if it's too small
        if min(imh, imw) < tgt_size:
            save_path = new_dir / imname
            cv2.imwrite(str(save_path), im)
            continue

        h_num = imh // tgt_size
        h_starts = np.linspace(0, imh - tgt_size, num=h_num+1)
        w_num = imw // tgt_size
        w_starts = np.linspace(0, imw - tgt_size, num=w_num+1)
        for i in h_starts:
            i = int(i)
            for j in w_starts:
                j = int(j)
                window = im[i:i+tgt_size, j:j+tgt_size, :]
                # plt.imshow(window); plt.show()
                assert window.shape == (tgt_size, tgt_size, 3)
                save_path = new_dir / (f'{imname[:-4]}_{i}_{j}.png')
                cv2.imwrite(str(save_path), window)
                debug = 1


if __name__ == "__main__":
    main()
