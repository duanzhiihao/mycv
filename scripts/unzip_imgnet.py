import os
from tqdm import tqdm
import argparse
import tarfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str)
    parser.add_argument('--tgt', type=str)
    args = parser.parse_args()

    src_dir = args.src
    tgt_dir = args.tgt

    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    assert os.path.isdir(tgt_dir)

    # tar files
    tar_names = os.listdir(src_dir)
    assert len(tar_names) == 1000, len(tar_names)
    # wordnet ids
    wdids = [s[:-4] for s in tar_names]
    # check files are either tar or folder
    print('checking files and folders...')
    existing = os.listdir(tgt_dir)

    # wordnet id to img count
    count_file = os.path.join('annotations/train_counts.txt')
    assert os.path.exists(count_file)
    lines = open(count_file, 'r').read().strip().split('\n')
    wnid2count = [line.split() for line in lines]
    wnid2count = {k:int(v) for k,v in wnid2count}

    for fname in tqdm(tar_names):
        # skip if it is already unziped
        wnid = fname.replace('.tar', '')
        img_dir = os.path.join(tgt_dir, wnid)
        if os.path.isdir(img_dir) and len(os.listdir(img_dir)) == wnid2count[wnid]:
            continue
        assert fname.endswith(".tar")
        fpath = os.path.join(src_dir, fname)
        tar = tarfile.open(fpath, "r:")
        tar.extractall(img_dir)
        tar.close()
