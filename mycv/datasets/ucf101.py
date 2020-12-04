import os
import json
from pathlib import Path
from tqdm import tqdm
import cv2
import torch

VIDEO_ROOT = 'D:/Datasets/UCF-101/videos/'
FRAME_ROOT = 'D:/Datasets/UCF-101/frames/'
ANN_ROOT   = 'D:/Datasets/UCF-101/annotations/'
CLASS_PATH = 'D:/Datasets/UCF-101/annotations/classInd.txt'

def get_class_info():
    '''
    Return:
        clsNames: list
        clsName2idx: dict
    '''
    lines = open(CLASS_PATH, 'r').read().strip().split('\n')
    clsNames = [s.split()[1] for s in lines]
    clsName2idx = {s:i for i,s in enumerate(clsNames)}
    return clsNames, clsName2idx

CLASS_NAMES, CLS_NAME_TO_IDX = get_class_info()


class UCF101Dataset(torch.utils.data.Dataset):
    '''
    UCF-101 dataset
    '''
    def __init__(self, split='train', fold=None, img_hw=(240, 320), augment=True, sample_step=1):
        super().__init__()
        assert fold is not None
        self.root = Path(VIDEO_ROOT)
        # image-label pairs
        ann_path = Path(ANN_ROOT) / f'{split}list0{fold}.txt'
        self.get_all_images(ann_path, sample_step)
        self.cls_num = len(CLASS_NAMES)

        self.augment = augment
        assert img_hw == (240, 320)
        self.img_hw = (240, 320)

    def get_all_images(self, ann_path: str, sample_step: int):
        img_info_path = str(ann_path).replace('.txt', f'_{sample_step}.json')
        if os.path.exists(img_info_path):
            json_data = json.load(open(img_info_path, 'r'))
            self.imgs_info = json_data['imgs_info']
            self.labels    = json_data['labels']
        else:
            lines = open(ann_path, 'r').read().strip().split('\n')
            img_label_pairs = [s.split() for s in lines]
            self.imgs_info = []
            self.labels    = []
            for vname, label in tqdm(img_label_pairs):
                vpath = self.root / vname
                assert os.path.exists(vpath)
                vcap = cv2.VideoCapture(str(vpath))
                fnum = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
                vcap.release()
                for i in range(0, fnum-25, sample_step):
                    self.imgs_info.append((vname, i))
                    self.labels.append(int(label) - 1)
            json_data = {
                'imgs_info': self.imgs_info,
                'labels': self.labels
            }
            json.dump(json_data, open(img_info_path, 'w'), indent=1)

    def __len__(self):
        assert len(self.imgs_info) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, index):
        vname, fi = self.imgs_info[index]
        label = self.labels[index]
        assert CLS_NAME_TO_IDX[vname.split('/')[0]] == label
        assert isinstance(label, int) and 0 <= label < self.cls_num
        # read video and frame
        vpath = str(self.root / vname)
        assert os.path.exists(vpath)
        vcap = cv2.VideoCapture(vpath)
        if vcap.isOpened() == False:
            raise Exception(f'Failed to open video {vpath}')
        vcap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        flag, im = vcap.read()
        if not flag:
            raise Exception(f'Failed to read frame {fi} from video {vpath}')
        vcap.release()
        # resize image
        th, tw = self.img_hw
        if im.shape[1:3] != (th, tw):
            im = cv2.resize(im, (tw,th))
        # random aug
        if self.augment and torch.rand(1) > 0.5:
            im = cv2.flip(im, 1)
        # to tensor
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = torch.from_numpy(im).permute(2,0,1).float() / 255.0
        return im, label


class UCF101FirstFrames(torch.utils.data.Dataset):
    '''
    UCF-101 first frames
    '''
    def __init__(self, split='test', fold=None, img_hw=(240, 320)):
        super().__init__()
        if split != 'test':
            raise NotImplementedError()
        assert fold is not None
        self.root = Path(FRAME_ROOT)
        ann_path = Path(ANN_ROOT) / f'{split}list0{fold}.txt'
        self.lines = open(ann_path, 'r').read().strip().split('\n')
        assert img_hw == (240, 320)
        self.img_hw = (240, 320)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        vname = self.lines[index]
        assert len(vname.split()) == 1
        label = CLS_NAME_TO_IDX[vname.split('/')[0]]
        # read image
        assert vname.endswith('.avi')
        impath = self.root / vname.replace('.avi', '.png')
        assert os.path.exists(impath)
        im = cv2.imread(str(impath))
        # resize image
        th, tw = self.img_hw
        if im.shape[1:3] != (th, tw):
            im = cv2.resize(im, (tw,th))
        # to tensor
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = torch.from_numpy(im).permute(2,0,1).float() / 255.0
        return im, label


def test_ucf101_1stframe(model, split, fold, img_hw=(240,320)):
    '''
    first frames
    '''
    dataset = UCF101FirstFrames(split=split, fold=fold, img_hw=img_hw)
    testloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, drop_last=False,
        pin_memory=True, num_workers=0
    )
    model.eval()
    device = next(model.parameters()).device
    tp, total = 0, 0
    for imgs, labels in tqdm(testloader, total=len(testloader)):
        imgs = imgs.to(device=device)
        # forward pass
        with torch.no_grad():
            logits = model(imgs)
        assert logits.dim() == 2
        # find class index
        _, p_idx = torch.max(logits, dim=1)
        p_idx = p_idx.cpu()
        # compare
        tp_mask = (p_idx == labels)
        tp += tp_mask.sum()
        total += tp_mask.shape[0]
    assert total == len(dataset)
    acc_top1 = tp / total
    results = {
        'top1': acc_top1
    }
    return results


def test_ucf101(model, split='test', fold=1):
    '''
    evaluation
    '''
    assert split == 'test'
    ann_path = Path(ANN_ROOT) / f'{split}list0{fold}.txt'
    lines = open(ann_path, 'r').read().strip().split('\n')
    tp = 0
    for vname in tqdm(lines):
        label = CLS_NAME_TO_IDX[vname.split('/')[0]]
        # run on video
        vpath = Path(VIDEO_ROOT) / vname
        logits = model.run_video(vpath)
        assert logits.dim() == 1
        p = torch.argmax(logits)
        assert 0 <= p <= 100 and 0 <= label <= 100
        if p == label:
            tp += 1
    acc = tp / len(lines)
    results = {
        'top1': acc
    }
    return results


if __name__ == "__main__":
    import torchvision as tv
    # from models import SingleFrameTest
    model = tv.models.resnet152(num_classes=101)
    model.eval()
    model = model.cuda()
    results = test_ucf101_1stframe(model, split='test', fold=1)
    print(results['top1'])
    # model = SingleFrameTest(model)
    # results = test_ucf101(
    #     model, data_root=Path('D:/Datasets/UCF-101/videos'),
    #     ann_path='D:/Datasets/UCF-101/annotations/testlist01.txt',
    #     clsName2idx=clsName2idx,
    #     frame_step=10
    # )
    # print(results['top1'])
