import os
from tqdm import tqdm
from pathlib import Path

from mycv.paths import ILSVRC_DIR
from mycv.datasets.imagenet import WNIDS


def main():
    sample(200, 600, 100)


def sample(num_cls=200, num_train=600, num_val=100):
    assert ILSVRC_DIR.is_dir()

    train_root = ILSVRC_DIR / 'Data/CLS-LOC/train'
    # check if imageset file already exist
    trainset_path = ILSVRC_DIR / f'ImageSets/CLS-LOC/train_{num_train}_{num_val}.txt'
    valset_path = ILSVRC_DIR / f'ImageSets/CLS-LOC/val_{num_train}_{num_val}.txt'
    if trainset_path.exists():
        print(f'Warning: {trainset_path} already exist. Removing it...')
        os.remove(trainset_path)
    if valset_path.exists():
        print(f'Warning: {valset_path} already exist. Removing it...')
        os.remove(valset_path)

    train_count = 0
    val_count = 0
    for wnid in tqdm(WNIDS[0:num_cls]):
        img_dir = train_root / wnid
        assert img_dir.is_dir()
        img_names = os.listdir(img_dir)
        # selelct the num_train and num_val images
        assert len(img_names) > num_train + num_val
        train_names = img_names[:num_train]
        val_names = img_names[num_train:num_train+num_val]
        
        # write names to imageset file
        with open(trainset_path, 'a') as f:
            for imname in train_names:
                assert imname.endswith('.JPEG')
                s = imname.replace('.JPEG', '')
                f.write(f'{wnid}/{s} {train_count}\n')
                train_count += 1
        with open(valset_path, 'a') as f:
            for imname in val_names:
                assert imname.endswith('.JPEG')
                s = imname.replace('.JPEG', '')
                f.write(f'{wnid}/{s} {val_count}\n')
                val_count += 1


if __name__ == "__main__":
    main()
