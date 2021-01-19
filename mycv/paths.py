'''
This is the global settings of dataset paths.
'''
import os
from pathlib import Path


# Project root dir
MYCV_DIR = Path(os.path.dirname(__file__))

# Network weights dir
WEIGHTS_DIR = Path(os.path.dirname(__file__)) / 'weights'

# CLIC dataset: http://www.compression.cc/
CLIC_DIR = Path('D:/Datasets/CLIC')

# Kodak dataset: http://www.cs.albany.edu/~xypan/research/snr/Kodak.html
KODAK_DIR = Path('D:/Datasets/Kodak')

# ImageNet dataset
IMAGENET_DIR = Path('D:/Datasets/imagenet')

# Food-101 dataset
FOOD101_DIR = Path('D:/Datasets/food-101')

# COCO dataset: http://cocodataset.org/#home
COCO_DIR = Path('D:/Datasets/COCO')

# MW-R, HABBOF, and CEPDOF dataset: http://vip.bu.edu/projects/vsns/cossy/datasets/
COSSY_DIR = Path('/home/duanzh/Projects/Datasets/COSSY')

# Global Wheat Head Detection dataset
GWHD_DIR = Path('/home/duanzh/Projects/Datasets/GWHD')

# UCF-101 dataset
UCF101_DIR = Path('D:/Datasets/UCF-101')
