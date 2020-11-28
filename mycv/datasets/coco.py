from pathlib import Path
import json
import cv2
import torch

from mycv.paths import COCO_DIR
from mycv.utils.image_ops import letterbox


COCO2017_CLS_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]

class COCOEvalDataset(torch.utils.data.Dataset):
    def __init__(self, split='val', img_size=640, to_square=False):
        if split == 'val':
            ann_path = COCO_DIR / 'annotations/instances_val2017.json'
            img_dir = COCO_DIR / 'val2017'
        else:
            raise NotImplementedError()
        json_data = json.load(open(ann_path, 'r'))

        self.json_data = json_data
        self.img_dir   = img_dir
        self.img_infos = json_data['images']
        self.img_size  = img_size
        self.to_square = to_square

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, index):
        imgInfo = self.img_infos[index]
        imname = imgInfo['file_name']
        impath = self.img_dir / imname
        # read image
        assert impath.exists()
        im = cv2.imread(str(impath))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # resize image
        im, ratio, (top,left) = letterbox(im, tgt_size=self.img_size,
                                          to_square=self.to_square, div=32)
        # to tensor
        im = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        # information for evaluation
        imgid = imgInfo['id']
        # resize and padding information
        pad_info = torch.Tensor([ratio, top, left])
        return im, imgid, pad_info


if __name__ == "__main__":
    dataset = COCOEvalDataset()
