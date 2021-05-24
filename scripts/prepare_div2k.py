import os
from pathlib import Path
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from mycv.paths import DIV2K_DIR


if __name__ == '__main__':
    for fname in [
        'DIV2K_train_LR_bicubic/X2','DIV2K_train_LR_bicubic/X3','DIV2K_train_LR_bicubic/X4',
        'DIV2K_train_LR_unknown/X2','DIV2K_train_LR_unknown/X3','DIV2K_train_LR_unknown/X4',
        'DIV2K_train_HR'
    ]:
        old_dir = DIV2K_DIR / fname
        new_dir = DIV2K_DIR / fname.lower().replace('div2k_', '')

        if old_dir.is_dir():
            assert not new_dir.is_dir()
            if not new_dir.parent.is_dir():
                new_dir.parent.mkdir()
            os.rename(old_dir, new_dir)
        else:
            assert new_dir.is_dir()

        img_names = os.listdir(new_dir)
        img_names.sort()

        assert len(img_names) == 900
        val_names = img_names[800:]

        for vname in val_names:
            old_path = str(new_dir / vname)
            new_path = old_path.replace('train', 'val')
            new_path = Path(new_path)
            if not new_path.parent.is_dir():
                new_path.parent.mkdir(parents=True)
            os.rename(old_path, new_path)
            debug = 1
