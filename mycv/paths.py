'''
This is the global settings of dataset paths.
'''
import os
from pathlib import Path


# Project root dir
MYCV_DIR = os.path.dirname(__file__)

# Network weights dir
WEIGHTS_DIR = Path(os.path.dirname(__file__)) / 'weights'

# COCO dataset: http://cocodataset.org/#home
COCO_DIR = Path('C:/Projects/Datasets/COCO')

# MW-R, HABBOF, and CEPDOF dataset: http://vip.bu.edu/projects/vsns/cossy/datasets/
COSSY_DIR = Path('/home/duanzh/Projects/Datasets/COSSY')

# ImageNet dataset
ILSVRC_DIR = Path('D:/Datasets/ILSVRC')

# Global Wheat Head Detection dataset
GWHD_DIR = Path('/home/duanzh/Projects/Datasets/GWHD')

# UCF-101 dataset
UCF101_DIR = Path('D:/Datasets/UCF-101')
