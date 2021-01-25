import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms.functional as tvf

from mycv.paths import IMAGENET_DIR

class Check(Dataset):
    def __init__(self, img_paths) -> None:
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        impath = self.img_paths[index]
        img = Image.open(impath).convert('RGB')
        h, w = img.height, img.width
        im = tvf.to_tensor(img)
        assert im.shape == (3, h, w), f'{im.shape} {impath}'
        return 0

def check_images(img_paths, workers=4):
    dataset = Check(img_paths)
    dataloader = DataLoader(dataset, batch_size=workers*8, num_workers=workers)
    for imgs in tqdm(dataloader):
        do_nothing = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()

    # imagenet root
    root = Path(IMAGENET_DIR)

    # check training set
    lines = open(root / 'annotations/train.txt', 'r').read().strip().split('\n')
    img_paths = [root / f'train/{s.split()[0]}' for s in lines]
    print(f'Checking imagenet training set... total {len(img_paths)} images')
    check_images(img_paths, workers=args.workers)

    # check val set
    lines = open(root / 'annotations/val.txt', 'r').read().strip().split('\n')
    img_paths = [root / f'val/{s.split()[0]}' for s in lines]
    print(f'Checking imagenet validation set... total {len(img_paths)} images')
    check_images(img_paths, workers=args.workers)

    # check labels
    assert (root / 'annotations' / 'val_real.json').exists()
