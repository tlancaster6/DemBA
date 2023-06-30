from pathlib import Path
import pandas as pd
import cv2
import os
from demba.utils.roi_utils import estimate_roi
from itertools import permutations
import numpy as np
import ruptures as rpt
idx = pd.IndexSlice


def threshed_pelt_event_detection(data: pd.Series, upper_thresh: float, min_size: int, penalty=3):
    binary_data = (data < upper_thresh).astype(int).values
    bkpts = rpt.Pelt(model='l2', min_size=min_size, jump=1).fit_predict(binary_data, penalty)
    if bkpts[0] != 0:
        bkpts = [0] + bkpts
    if bkpts[-1] != len(data):
        bkpts = bkpts + [len(data)]
    event_ids = pd.Series(-1, index=data.index)
    current_event = 0
    for start, stop in list(zip(bkpts[:-1], bkpts[1:])):
        if np.mean(binary_data[start:stop]) > 0.5:
            event_ids.iloc[start:stop] = current_event
            current_event += 1
    return event_ids


class BasicAnalyzer:

    def __init__(self, video_path):
        self.video_path = video_path
        self.h5_path = str(next(Path(self.video_path).parent.glob(Path(self.video_path).stem + '*_filtered.h5')))
        self.output_dir = str(Path(self.video_path).parent / 'output')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self._load_pose_data()
        self._estimate_roi()

    def _load_pose_data(self):
        self.pose_df = pd.read_hdf(self.h5_path)
        scorer = self.pose_df.columns.get_level_values(0)[0]
        self.pose_df = self.pose_df.loc[:, scorer]
        self.pose_df = self.pose_df.loc[:, idx[:, :, ('x', 'y')]]
        self.pose_df.columns = self.pose_df.columns.remove_unused_levels()
        self.pose_df = self.pose_df.sort_index(axis=1)
        self.individuals, self.bodyparts = [list(self.pose_df.columns.levels[i]) for i in [0, 1]]

    def _estimate_roi(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if not ret:
            print(f'could not extract a reference frame from {Path(self.video_path).name}')
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi_vis_path = str(Path(self.video_path).parent / (Path(self.video_path).stem + '_roi.png'))
        self.roi_x, self.roi_y, self.roi_r = estimate_roi(frame, output_path=roi_vis_path)
        self.frame_height, self.frame_width = frame.shape[:-1]
        cap.release()

    def calc_nfish_frame(self):
        nfish_frame = self.pose_df.groupby('individuals', axis=1).any().sum(axis=1)
        nfish_frame.name = 'nfish_frame'
        return nfish_frame

    def calc_nfish_pipe(self):
        tmp_df = self.pose_df.loc[:, idx[:, 'stripe1', :]]
        tmp_df.loc[:, idx[:, :, 'x']] -= self.roi_x
        tmp_df.loc[:, idx[:, :, 'y']] -= self.roi_y
        tmp_df = (tmp_df ** 2).groupby('individuals', axis=1).sum(min_count=2) ** 0.5
        nfish_pipe = (tmp_df <= self.roi_r).sum(axis=1)
        nfish_pipe.name = 'nfish_pipe'
        return nfish_pipe

    def calc_min_dist_nose_to_stripe4(self):
        candidate_dists = []
        for id1, id2 in list(permutations(self.individuals, 2)):
            dists = self.pose_df.loc[:, idx[id1, 'nose', :]].values - self.pose_df.loc[:, idx[id2, 'stripe4', :]].values
            candidate_dists.append(np.hypot(dists[:, 0], dists[:, 1]))
        return pd.Series(np.nanmin(np.vstack(candidate_dists), axis=0), name='min_dist_nose_to_stripe4')

    def detect_mouthing_events(self, dist_thresh=25, min_frames=30):
        event_ids = threshed_pelt_event_detection(self.calc_min_dist_nose_to_stripe4(), dist_thresh, min_frames)
        return event_ids


vid = r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\testclip\testclip.mp4"
ba = BasicAnalyzer(vid)
