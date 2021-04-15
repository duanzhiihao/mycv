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


def disable_multithreads():
    """ Disable multi-processing in numpy and cv2
    """    
    os.environ["OMP_NUM_THREADS"]      = "1" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"]      = "1" # export MKL_NUM_THREADS=6
    os.environ["NUMEXPR_NUM_THREADS"]  = "1" # export NUMEXPR_NUM_THREADS=6
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)


def warning(msg: str):
    print('=======================================================================')
    print('Warning:', msg)
    print('=======================================================================')


class SimpleConfig():
    """ A simple config class
    """
    def __init__(self) -> None:
        pass

    def __setattr__(self, name: str, value) -> None:
        self.__dict__[name] = value


# def remove_prefix(s: str, prefix: str):
#     assert isinstance(s, str) and isinstance(prefix, str)
#     if s.startswith(prefix):
#         s = s[len(prefix):]
#     return s


if __name__ == '__main__':
    cfg = SimpleConfig()
    cfg.abc = '123'
    print(cfg.abc)
