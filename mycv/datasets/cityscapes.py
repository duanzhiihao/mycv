import collections
import cv2
import torch

from mycv.datasets.imagenet import RGB_MEAN, RGB_STD
from mycv.paths import CITYSCAPES_DIR
from mycv.external.semseg import transform

# Copied from torchvision and https://github.com/mcordts/cityscapesScripts
_CityClass = collections.namedtuple(
    '_CityClass',
    ['name', 'id', 'train_id', 'category', 'has_instances', 'ignore_in_eval', 'color']
)
CLASS_INFO = [
    _CityClass('unlabeled',     0,  255, 'void',         False, True,  (0, 0, 0)),
    _CityClass('ego vehicle',   1,  255, 'void',         False, True,  (0, 0, 0)),
    _CityClass('rectification border', 2, 255, 'void',   False, True,  (0, 0, 0)),
    _CityClass('out of roi',    3,  255, 'void',         False, True,  (0, 0, 0)),
    _CityClass('static',        4,  255, 'void',         False, True,  (0, 0, 0)),
    _CityClass('dynamic',       5,  255, 'void',         False, True,  (111, 74, 0)),
    _CityClass('ground',        6,  255, 'void',         False, True,  (81, 0, 81)),
    _CityClass('road',          7,  0,   'flat',         False, False, (128, 64, 128)),
    _CityClass('sidewalk',      8,  1,   'flat',         False, False, (244, 35, 232)),
    _CityClass('parking',       9,  255, 'flat',         False, True,  (250, 170, 160)),
    _CityClass('rail track',    10, 255, 'flat',         False, True,  (230, 150, 140)),
    _CityClass('building',      11, 2,   'construction', False, False, (70, 70, 70)),
    _CityClass('wall',          12, 3,   'construction', False, False, (102, 102, 156)),
    _CityClass('fence',         13, 4,   'construction', False, False, (190, 153, 153)),
    _CityClass('guard rail',    14, 255, 'construction', False, True,  (180, 165, 180)),
    _CityClass('bridge',        15, 255, 'construction', False, True,  (150, 100, 100)),
    _CityClass('tunnel',        16, 255, 'construction', False, True,  (150, 120, 90)),
    _CityClass('pole',          17, 5,   'object',       False, False, (153, 153, 153)),
    _CityClass('polegroup',     18, 255, 'object',       False, True,  (153, 153, 153)),
    _CityClass('traffic light', 19, 6,   'object',       False, False, (250, 170, 30)),
    _CityClass('traffic sign',  20, 7,   'object',       False, False, (220, 220, 0)),
    _CityClass('vegetation',    21, 8,   'nature',       False, False, (107, 142, 35)),
    _CityClass('terrain',       22, 9,   'nature',       False, False, (152, 251, 152)),
    _CityClass('sky',           23, 10,  'sky',          False, False, (70, 130, 180)),
    _CityClass('person',        24, 11,  'human',        True,  False, (220, 20, 60)),
    _CityClass('rider',         25, 12,  'human',        True,  False, (255, 0, 0)),
    _CityClass('car',           26, 13,  'vehicle',      True,  False, (0, 0, 142)),
    _CityClass('truck',         27, 14,  'vehicle',      True,  False, (0, 0, 70)),
    _CityClass('bus',           28, 15,  'vehicle',      True,  False, (0, 60, 100)),
    _CityClass('caravan',       29, 255, 'vehicle',      True,  True,  (0, 0, 90)),
    _CityClass('trailer',       30, 255, 'vehicle',      True,  True,  (0, 0, 110)),
    _CityClass('train',         31, 16,  'vehicle',      True,  False, (0, 80, 100)),
    _CityClass('motorcycle',    32, 17,  'vehicle',      True,  False, (0, 0, 230)),
    _CityClass('bicycle',       33, 18,  'vehicle',      True,  False, (119, 11, 32)),
    # _CityClass('license plate', -1, -1,  'vehicle',      False, True,  (0, 0, 142)),
]
COLORS = torch.Tensor(
    [c.color for c in CLASS_INFO if (not c.ignore_in_eval)]
).to(dtype=torch.uint8)


class Cityscapes(torch.utils.data.Dataset):
    input_mean = torch.FloatTensor(RGB_MEAN).view(3, 1, 1)
    input_std  = torch.FloatTensor(RGB_STD).view(3, 1, 1)
    def __init__(self, split='val', img_hw=(713,713)):
        assert split in {'train_fine', 'val', 'test'}
        # raed datalist
        list_path = CITYSCAPES_DIR / f'annotations/{split}.txt'
        data_list = open(list_path, 'r').read().strip().split('\n')
        self.img_gt_paths = [tuple(pair.split()) for pair in data_list]

        # data augmentation setting
        self.img_hw = img_hw
        if 'train' in split:
            _mean = [v*255 for v in RGB_MEAN]
            self.transform = transform.Compose([
                transform.RandScale([0.5, 2.0]),
                transform.RandRotate([-10, 10], padding=_mean, ignore_label=255),
                transform.RandomGaussianBlur(),
                transform.RandomHorizontalFlip(),
                transform.Crop(img_hw, crop_type='rand', padding=_mean, ignore_label=255),
                # ToTensor(),
                # Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
            ])
        else:
            self.transform = None

        # build mapping from cityscapes id to training id
        mapping = torch.zeros(len(CLASS_INFO), dtype=torch.uint8)
        for cinfo in CLASS_INFO:
            mapping[cinfo.id] = cinfo.train_id
        self.mapping = mapping

    def __len__(self):
        return len(self.img_gt_paths)

    def __getitem__(self, index):
        pair = self.img_gt_paths[index]
        assert len(pair) in [1,2]
        impath = str(CITYSCAPES_DIR / pair[0])
        gtpath = str(CITYSCAPES_DIR / pair[1]) if len(pair) == 2 else None

        # load the image and label
        im = cv2.cvtColor(cv2.imread(impath), cv2.COLOR_BGR2RGB)
        label = cv2.imread(gtpath, cv2.IMREAD_GRAYSCALE)
        assert im.shape[:2] == label.shape, f'image {im.shape}, label {label.shape}'
        # data augmentation during training
        if self.transform is not None:
            im, label = self.transform(im, label)
        # numpy to tensor
        im = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).to(dtype=torch.int64)
        # map the label id to training id
        label = self.mapping[label]

        return im, label


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mycv.utils.visualization import colorize_semseg
    dataset = Cityscapes()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0)

    for imgs, labels in dataloader:
        for im_, seg_ in zip(imgs, labels):
            seg_ = colorize_semseg(seg_)
            # fgmask = (seg_ <= 18)
            # segcolor = torch.zeros()
            plt.figure(); plt.imshow(im_.permute(1,2,0).numpy())
            plt.figure(); plt.imshow(seg_.numpy()); plt.show()
            debug = 1
    debug = 1


# CLASS_NAMES = [
#     'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
#     'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
#     'truck', 'bus', 'train', 'motorcycle', 'bicycle'
# ]
# COLORS = torch.Tensor([
#     (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
#     (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
#     (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
#     (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
# ]).to(dtype=torch.uint8)
