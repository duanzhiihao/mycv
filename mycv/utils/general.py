import os
from pathlib import Path
import re


def increment_dir(dir_root='runs/', name='exp'):
    '''
    increament directory name
    '''
    assert isinstance(dir_root, (str, Path))
    dir_root = Path(dir_root)
    if not os.path.exists(dir_root):
        print(f'Warning: {dir_root} does not exist.')
    dnames = [s for s in os.listdir(dir_root) if s.startswith(name)]
    if len(dnames) > 0:
        dnames = [s[len(name):] for s in dnames]
        ids = [int(re.search(r'\d+', s).group()) for s in dnames]
        n = max(ids) + 1
    else:
        n = 0
    return dir_root / f'{name}_{n}'
