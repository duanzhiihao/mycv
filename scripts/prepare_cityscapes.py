import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from mycv.paths import CITYSCAPES_DIR


if __name__ == '__main__':
    img_root = 'leftImg8bit/val'
    gt_root = 'gtFine/val'
    list_path = CITYSCAPES_DIR / 'annotations/val.txt'

    if list_path.exists():
        print(f'Warning: {list_path} exists. Removing the old file...')
    file = open(list_path, 'w')

    city_names = os.listdir(CITYSCAPES_DIR / img_root)
    city_names.sort()
    for cname in tqdm(city_names):
        cname: str
        img_dir = CITYSCAPES_DIR / img_root / cname
        gt_dir = CITYSCAPES_DIR / gt_root / cname
        assert gt_dir.is_dir()

        img_names = os.listdir(img_dir)
        for imname in img_names:
            # 'aachen_000000_000019_leftImg8bit.png'
            # 'aachen_000000_000019_gtFine_labelIds.png'

            impath = img_dir / imname
            # im = cv2.imread(str(impath))
            # plt.figure(); plt.imshow(im)

            gtname = imname.replace('leftImg8bit', 'gtFine_labelIds')
            gtpath = gt_dir / gtname
            assert gtpath.is_file()
            # gt = cv2.imread(str(gtpath), cv2.IMREAD_GRAYSCALE)
            # plt.figure(); plt.imshow(gt)
            # plt.show()

            line = f'{img_root}/{cname}/{imname} {gt_root}/{cname}/{gtname}'
            print(line, file=file)
    file.close()
