import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms.functional as tvf

from mycv.paths import ILSVRC_DIR

class Check(Dataset):
    def __init__(self, img_paths) -> None:
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        impath = self.img_paths[index]
        img = Image.open(impath)
        im = tvf.to_tensor(img)
        assert im.dim() == 3 and im.shape[0] in {1, 3}
        return 0

def check_images(img_paths, workers=4):
    dataset = Check(img_paths)
    dataloader = DataLoader(dataset, batch_size=workers*8, num_workers=workers)
    for imgs in tqdm(dataloader):
        do_nothing = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()

    # imagenet root
    root = Path(ILSVRC_DIR)

    # check training set
    lines = open(root / 'ImageSets/CLS-LOC/train_cls.txt', 'r').read().strip().split('\n')
    img_paths = [root / f'Data/CLS-LOC/train/{s.split()[0]}' for s in lines]
    img_paths = [str(s)+'.JPEG' for s in img_paths]
    print(f'Checking imagenet training set... total {len(img_paths)} images')
    check_images(img_paths, workers=args.workers)

    # check val set
    lines = open(root / 'ImageSets/CLS-LOC/val.txt', 'r').read().strip().split('\n')
    img_paths = [root / f'Data/CLS-LOC/val/{s.split()[0]}' for s in lines]
    img_paths = [str(s)+'.JPEG' for s in img_paths]
    print(f'Checking imagenet validation set... total {len(img_paths)} images')
    check_images(img_paths, workers=args.workers)

    # check labels
    assert (root / 'Annotations' / 'cls_val_real.json').exists()
    assert (root / 'Annotations' / 'cls_val.txt').exists()
