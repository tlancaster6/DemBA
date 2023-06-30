from math import radians, degrees
import numpy as np
import cv2

bodypart2abbreviation_mapping = {'nose': 'no',
                                 'leftEye': 'le',
                                 'rightEye': 're',
                                 'stripe1': 'sa',
                                 'stripe2': 'sb',
                                 'stripe3': 'sc',
                                 'stripe4': 'sd',
                                 'tailBase': 'tb',
                                 'tailTip': 'tt'
                                 }

abbreviation2bodypart_mapping = {v: k for k, v in bodypart2abbreviation_mapping.items()}


fullid2shortid_mapping = {'individual1': '1',
                          'individual2': '2',
                          'individual3': '3'}

shortid2fullid_mapping = {v: k for k, v in fullid2shortid_mapping.items()}


def avg_heading(headings):
    headings = [radians(h) for h in headings]
    sin_sum = np.nansum([np.sin(h) for h in headings])
    cos_sum = np.nansum([np.cos(h) for h in headings])
    return degrees(np.atan2(sin_sum, cos_sum))


def get_frame_size(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    h, w = frame.shape[:-1]
    return h, w
