from pathlib import Path
import json
import cv2
import torch

from mycv.paths import COCO_DIR
from mycv.utils.image_ops import letterbox


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
        if self.to_square:
            im, ratio, (top,left) = letterbox(im, tgt_size=self.img_size,
                                              side='longer', to_square=True)
        else:
            im, ratio, (top,left) = letterbox(im, tgt_size=self.img_size,
                                              side='shorter', to_square=False, div=32)
        # to tensor
        im = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        # information for evaluation
        imgid = imgInfo['id']
        # resize and padding information
        pad_info = torch.Tensor([ratio, top, left])
        return im, imgid, pad_info
