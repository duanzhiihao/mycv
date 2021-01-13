from pathlib import Path
from tqdm import tqdm
from PIL import Image

from mycv.paths import ILSVRC_DIR


if __name__ == "__main__":
    # check images
    root = Path(ILSVRC_DIR)

    # check training set
    lines = open(root / 'ImageSets/CLS-LOC/train_cls.txt', 'r').read().strip().split('\n')
    img_paths = [root / f'Data/CLS-LOC/train/{s.split()[0]}' for s in lines]
    print(f'Checking imagenet training set... total {len(img_paths)} images')
    for impath in tqdm(img_paths):
        impath = str(impath) + '.JPEG'
        img = Image.open(impath)
        img.verify()

    # check val set
    lines = open(root / 'ImageSets/CLS-LOC/val.txt', 'r').read().strip().split('\n')
    img_paths = [root / f'Data/CLS-LOC/val/{s.split()[0]}' for s in lines]
    print(len(img_paths))
    for impath in tqdm(img_paths):
        impath = str(impath) + '.JPEG'
        img = Image.open(impath)
        img.verify()

    # check labels
    assert (root / 'Annotations' / 'cls_val_real.json').exists()
    assert (root / 'Annotations' / 'val.txt').exists()
