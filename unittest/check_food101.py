import os
from pathlib import Path
from tqdm import tqdm

from mycv.paths import FOOD101_DIR


if __name__ == "__main__":
    # check images
    root = Path(FOOD101_DIR)
    # check training set
    lines = open(root / 'ImageSets/CLS-LOC/train_cls.txt', 'r').read().strip().split('\n')
    img_paths = [root / f'Data/CLS-LOC/train/{s.split()[0]}' for s in lines]
    print(len(img_paths))
    for impath in tqdm(img_paths):
        impath = str(impath) + '.JPEG'
        assert os.path.exists(impath)
    # check val set
    lines = open(root / 'ImageSets/CLS-LOC/val.txt', 'r').read().strip().split('\n')
    img_paths = [root / f'Data/CLS-LOC/val/{s.split()[0]}' for s in lines]
    print(len(img_paths))
    for impath in tqdm(img_paths):
        impath = str(impath) + '.JPEG'
        assert os.path.exists(impath)
    assert os.path.exists(root / 'Annotations' / 'cls_val_real.json')
