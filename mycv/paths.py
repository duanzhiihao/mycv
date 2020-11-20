'''
This is the global settings of dataset paths.
'''
import os
from pathlib import Path


# Project root dir
MYCV_DIR = os.path.dirname(__file__)

# COCO dataset: http://cocodataset.org/#home
COCO_DIR = Path('C:/Projects/Datasets/COCO')

# MW-R, HABBOF, and CEPDOF dataset: http://vip.bu.edu/projects/vsns/cossy/datasets/
COSSY_DIR = Path('/home/duanzh/Projects/Datasets/COSSY')

# ImageNet dataaset
ILSVRC_DIR = Path('/home/duanzh/Projects/Datasets/ILSVRC')

# Global Wheat Head Detection dataset
GWHD_DIR = Path('/home/duanzh/Projects/Datasets/GWHD')
