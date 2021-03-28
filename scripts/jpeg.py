import argparse
import os
from tqdm import tqdm
from PIL import Image
import cv2
import torch
import torchvision.transforms.functional as tvf

from mycv.paths import IMAGENET_DIR
from mycv.datasets.imagenet import ImageNetCls
from mycv.utils.coding import psnr_dB, MS_SSIM


def get_bpp(split='val_jpeg50'):
    img_dir = IMAGENET_DIR / split

    img_names = os.listdir(img_dir)
    img_names.sort()

    total_bits = 0
    total_pixels = 0
    pbar = tqdm(img_names)
    for imname in pbar:
        impath = img_dir / imname
        bytes = os.path.getsize(impath)

        im = Image.open(impath)
        pixels = im.height * im.width
        im.close()

        bits = bytes * 8
        total_bits += bits
        total_pixels += pixels
        bpp = total_bits / total_pixels
        pbar.set_description(f'bpp: {bpp:.4g}')

    print(total_bits / total_pixels)


def jpeg(split='val', quality=50):
    img_dir = IMAGENET_DIR / split
    # save_dir = IMAGENET_DIR / f'{split}_jpeg{quality}'
    # if not save_dir.is_dir():
    #     print(f'{save_dir} does not exist. Creating it...')
    #     save_dir.mkdir(parents=False, exist_ok=False)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    msssim_func = MS_SSIM(max_val=1.0)
    imlabels = ImageNetCls.get_image_label_pairs(split)
    pbar = tqdm(enumerate(imlabels), total=len(imlabels))
    psnr_sum, msssim_sum, bpp_sum = 0, 0, 0
    for bi, (imname, _) in pbar:
        impath = img_dir / imname
        im = cv2.imread(str(impath))
        assert im is not None
        nH, nW, _ = im.shape

        # encode and decode
        flag, bits = cv2.imencode('.jpg', im, params=encode_param)
        assert flag
        im_dec = cv2.imdecode(bits, 1)
        # compute metrics
        ps = psnr_dB(im, im_dec)
        with torch.no_grad():
            ms = msssim_func(
                img1=tvf.to_tensor(im_dec).unsqueeze(0).cuda(),
                img2=tvf.to_tensor(im).unsqueeze(0).cuda()
            )
        bpp = len(bits)*8 / (nH*nW)

        # save image to a new folder
        # svpath = save_dir / imname
        # if not svpath.parent.is_dir():
        #     svpath.parent.mkdir(parents=False, exist_ok=False)
        # cv2.imwrite(str(svpath), im, [cv2.IMWRITE_JPEG_QUALITY, quality])

        # loggging
        wnid = impath.parent.stem
        _num = bi + 1
        msssim_sum += ms
        psnr_sum += ps
        bpp_sum += bpp
        msssim = msssim_sum / _num
        psnr   = psnr_sum   / _num
        bpp    = bpp_sum    / _num
        msg = f'{wnid}... MS: {msssim:.4g}, PSNR: {psnr:.4g}, bpp: {bpp:.4g}'
        pbar.set_description(msg)
    print(bpp, psnr, msssim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--quality', type=int, default=2)
    args = parser.parse_args()

    jpeg(split=args.split, quality=args.quality)
    # get_bpp(split=f'{args.split}_jpeg{args.quality}')
