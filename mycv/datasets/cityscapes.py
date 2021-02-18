from tqdm import tqdm
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
TRAIN_COLORS = torch.Tensor(
    [c.color for c in CLASS_INFO if (not c.ignore_in_eval)]
).to(dtype=torch.uint8)

def label_to_train_id_mapping():
    # build mapping from cityscapes id to training id
    mapping = torch.zeros(len(CLASS_INFO), dtype=torch.int64)
    for cinfo in CLASS_INFO:
        mapping[cinfo.id] = cinfo.train_id
    return mapping

class Cityscapes(torch.utils.data.Dataset):
    input_mean = torch.FloatTensor(RGB_MEAN).view(3, 1, 1)
    input_std  = torch.FloatTensor(RGB_STD).view(3, 1, 1)
    num_class = 19
    mapping = label_to_train_id_mapping()
    ignore_label = 255

    def __init__(self, split='train_fine', train_size=713, input_norm=True):
        assert split in {'train_fine', 'val', 'test'}
        # raed datalist
        list_path = CITYSCAPES_DIR / f'annotations/{split}.txt'
        data_list = open(list_path, 'r').read().strip().split('\n')
        self.img_gt_paths = [tuple(pair.split()) for pair in data_list]

        # data augmentation setting
        if 'train' in split:
            _mean = [v*255 for v in RGB_MEAN]
            self.transform = transform.Compose([
                transform.RandScale([0.5, 2.0]),
                transform.RandRotate([-10, 10], padding=_mean, ignore_label=0),
                transform.RandomGaussianBlur(),
                transform.RandomHorizontalFlip(),
                transform.Crop(train_size, crop_type='rand', padding=_mean, ignore_label=0),
            ])
        else:
            self.transform = None
        self.input_norm = input_norm

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
        # image: np to tensor, normalization
        im = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        if self.input_norm:
            im = (im - self.input_mean) / self.input_std
        # label: np to tensor, map the label id to training id
        # label[label == 255] = 0
        label = torch.from_numpy(label).to(dtype=torch.int64)
        label = self.mapping[label]

        return im, label


def evaluate_semseg(model, testloader=None, input_norm=None):
    model.eval()
    device = next(model.parameters()).device
    forward_ = getattr(model, 'forward_cls', model.forward)

    if testloader is None:
        assert input_norm is not None
        testloader = torch.utils.data.DataLoader(
            Cityscapes(split='val', input_norm=input_norm),
            batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False
        )
    num_cls = testloader.dataset.num_class
    ignore_label = testloader.dataset.ignore_label

    sum_inter, sum_union, sum_tgt = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(testloader):
            # _debug(labels[0])
            assert imgs.shape[2:] == labels.shape[1:] == (1024, 2048)
            imgs, labels = imgs.to(device=device), labels.to(device=device)
            new = torch.zeros(1, 3, 1025, 2049, device=device)
            new[:, :, :1024, :2048] = imgs
            output = forward_(new)
            assert output.dim() == 4

            output, label = output.squeeze(0), labels.squeeze(0)
            output = output[:, :1024, :2048]
            output = torch.argmax(output, dim=0)
            output[label == ignore_label] = ignore_label
            tpmask = (output == label) # true positive (ie. intersection) for all classes
            intersection = output[tpmask]
            inter = torch.histc(intersection, bins=num_cls, min=0, max=num_cls-1)
            area_p = torch.histc(output, bins=num_cls, min=0, max=num_cls-1)
            area_t = torch.histc(label,  bins=num_cls, min=0, max=num_cls-1)
            union = area_p + area_t - inter
            # ious = area_inter / (area_output + area_target - area_inter)
            sum_inter += inter
            sum_union += union
            sum_tgt += area_t
    ious = sum_inter / sum_union
    miou = ious.mean()
    acc = (sum_inter / sum_tgt).mean()

    results = {'miou': miou.item(), 'acc': acc.item()}
    return results


def _debug(label):
    import matplotlib.pyplot as plt
    from mycv.utils.visualization import colorize_semseg
    # label = label.to(dtype=torch.int64)
    # painting = TRAIN_COLORS[label].numpy()
    painting = colorize_semseg(label)
    plt.imshow(painting); plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mycv.utils.visualization import colorize_semseg
    dataset = Cityscapes()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=4)
    for imgs, labels in tqdm(dataloader, total=len(dataloader)):
        for im_, seg_ in zip(imgs, labels):
            # im_ = im_ * dataset.input_std + dataset.input_mean
            # seg_ = colorize_semseg(seg_)
            # plt.figure(); plt.imshow(im_.permute(1,2,0).numpy())
            # plt.figure(); plt.imshow(seg_.numpy()); plt.show()
            # overlay = 0.7 * im_.permute(1,2,0).numpy() + 0.3 * seg_.float().numpy() / 255.0
            # plt.imshow(overlay); plt.show()
            debug = 1
    debug = 1

    # from mycv.paths import MYCV_DIR
    # from mycv.external.semseg import PSPNet
    # model = PSPNet()
    # model = model.cuda()
    # model.eval()
    # weights = torch.load(MYCV_DIR / 'weights/psp50_epoch_200.pt')
    # model.load_state_dict(weights)

    # results = evaluate_semseg(model)
    # print(results)


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
