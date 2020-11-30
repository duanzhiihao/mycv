import numpy as np
import cv2

import mycv.utils.image as imgUtils


def load_frames(video_path, max_length=100):
    '''
    Load RGB frames using cv2
    '''
    vcap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(max_length):
        flag, im = vcap.read()
        if not flag:
            break
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        frames.append(im)
    # read frames end
    info = {
        'fps': vcap.get(cv2.CAP_PROP_FPS)
    }
    vcap.release()
    return frames, info


def play_frames(frames: list):
    '''
    Args:
        frames: list of RGB frames
    '''
    for i, im in enumerate(frames):
        assert imgUtils.is_image(im), f"The {i}'th frame is not a valid image"
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imshow('Press q to exit', im)
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break
