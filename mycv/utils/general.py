import os
from pathlib import Path
import re


def increment_dir(dir_root='runs/', name='exp'):
    """ Increament directory name. E.g., exp1, exp2, exp3, ...

    Args:
        dir_root (str, optional): root directory. Defaults to 'runs/'.
        name (str, optional): dir prefix. Defaults to 'exp'.
    """
    assert isinstance(dir_root, (str, Path))
    dir_root = Path(dir_root)
    if not dir_root.exists():
        print(f'Warning: {dir_root} does not exist. Creating it...')
        os.makedirs(dir_root)
    assert dir_root.is_dir()
    dnames = [s for s in os.listdir(dir_root) if s.startswith(name)]
    if len(dnames) > 0:
        dnames = [s[len(name):] for s in dnames]
        ids = [int(re.search(r'\d+', s).group()) for s in dnames]
        n = max(ids) + 1
    else:
        n = 0
    name = f'{name}_{n}'
    return name


# def wb_increment_id(dir_root='wandb/', name='exp'):
#     """ Increament wandb run name. E.g., xxx-exp1, yyy-exp2, zzz-exp3, ...
#     """
#     assert isinstance(dir_root, (str, Path))
#     dir_root = Path(dir_root)
#     if not dir_root.exists():
#         print(f'Warning: {dir_root} does not exist. Creating it...')
#         os.makedirs(dir_root)
#     assert dir_root.is_dir()

#     wandb_ids = []
#     for fname in os.listdir(dir_root):
#         assert fname.startswith('run-') or fname.startswith('offline-run-')
#         fname = remove_prefix(fname, 'run-')
#         fname = remove_prefix(fname, 'offline-run-')
#         assert fname[8] == '_' and fname[15] == '-'
#         fname = fname[16:]
#         assert len(fname) > 0
#         wandb_ids.append(fname)
#     dnames = [s for s in wandb_ids if s.startswith(name)]
#     if len(dnames) > 0:
#         dnames = [s[len(name):] for s in dnames]
#         ids = [int(re.search(r'\d+', s).group()) for s in dnames]
#         n = max(ids) + 1
#     else:
#         n = 0
#     name = f'{name}_{n}' if name[-1].isdigit() else f'{name}{n}'
#     return name


# def remove_prefix(s: str, prefix: str):
#     assert isinstance(s, str) and isinstance(prefix, str)
#     if s.startswith(prefix):
#         s = s[len(prefix):]
#     return s
