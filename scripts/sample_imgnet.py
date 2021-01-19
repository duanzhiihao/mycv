import os
from tqdm import tqdm
from pathlib import Path
import random

from mycv.paths import IMAGENET_DIR
from mycv.datasets.imagenet import WNIDS, WNID_TO_IDX


def main():
    sample(200, 600, 50)


def sample(num_cls=200, num_train=600, num_val=50):
    assert IMAGENET_DIR.is_dir()

    train_root = IMAGENET_DIR / 'train'
    # check if imageset file already exist
    trainlabel_path = IMAGENET_DIR / f'annotations/train{num_cls}_{num_train}.txt'
    vallabel_path = IMAGENET_DIR / f'annotations/val{num_cls}_{num_train}.txt'
    if trainlabel_path.exists():
        print(f'Warning: {trainlabel_path} already exist. Removing it...')
        os.remove(trainlabel_path)
    if vallabel_path.exists():
        print(f'Warning: {vallabel_path} already exist. Removing it...')
        os.remove(vallabel_path)

    wnid_subset = random.sample(WNIDS, k=num_cls)
    for cls_idx, wnid in tqdm(enumerate(wnid_subset)):
        img_dir = train_root / wnid
        assert img_dir.is_dir()
        img_names = os.listdir(img_dir)
        # selelct the num_train and num_val images
        assert len(img_names) > num_train + num_val
        imname_subset = random.sample(img_names, num_train + num_val)
        train_names = imname_subset[:num_train]
        val_names = imname_subset[num_train:num_train+num_val]

        # write names to an annotation file
        with open(trainlabel_path, 'a', newline='\n') as f:
            for imname in train_names:
                assert imname.endswith('.JPEG')
                f.write(f'{wnid}/{imname} {cls_idx}\n')
        with open(vallabel_path, 'a', newline='\n') as f:
            for imname in val_names:
                assert imname.endswith('.JPEG')
                f.write(f'{wnid}/{imname} {cls_idx}\n')


if __name__ == "__main__":
    main()
