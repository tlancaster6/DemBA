from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from demba.utils.roi_utils import estimate_roi
from itertools import combinations
idx = pd.IndexSlice

class FeatureExtractionUnit:

    def __init__(self, data, roi_x, roi_y, roi_r):
        self.data = data
        self.roi_x, self.roi_y, self.roi_r = roi_x, roi_y, roi_r
        self.features = {'start': data.index[0], 'stop': data.index[-1]}

    def calc_features(self):
        self.features.update({'med_nfish': self.calc_med_nfish()})
        self.features.update({'med_nfish_pipe': self.calc_med_nfish_pipe()})
        self.features.update({'min_separation': self.calc_min_separation()})

    def calc_med_nfish(self):
        return np.nanmedian(self.data.groupby('individuals', axis=1).any().sum(axis=1))

    def calc_med_nfish_pipe(self):
        tmp_df = self.data.loc[:, idx[:, 'stripe1', :]].copy()
        tmp_df.loc[:, idx[:, :, 'x']] -= self.roi_x
        tmp_df.loc[:, idx[:, :, 'y']] -= self.roi_y
        tmp_df = (tmp_df ** 2).groupby('individuals', axis=1).sum(min_count=2) ** 0.5
        return np.nanmedian((tmp_df <= self.roi_r).sum(axis=1))

    def calc_min_separation(self):
        candidate_dists = []
        for id1, id2 in combinations(list(self.data.columns.levels[0]), 2):
            dists = self.data.loc[:, idx[id1, 'stripe1', :]].values - self.data.loc[:, idx[id2, 'stripe1', :]].values
            candidate_dists.append(np.nanmin(np.hypot(dists[:, 0], dists[:, 1])))
        retval = np.nanmin(candidate_dists)
        return 1000 if np.isnan(retval) else retval

    def summarize_centroid_kinetics(self):
        velocities = self.data.loc[:, idx[:, 'stripe1', :]].diff()
        accelerations = velocities.diff()
        v_magnitudes = (velocities ** 2).groupby('individuals', axis=1).sum(min_count=2) ** 0.5
        v_mean = np.nanmean(v_magnitudes)





class Featurizer:

    def __init__(self, video_path):
        self.video_path = video_path
        self.h5_path = str(next(Path(self.video_path).parent.glob(Path(self.video_path).stem + '*_filtered.h5')))
        self.feature_matrix = None

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
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        if not ret:
            print(f'could not extract a reference frame from {Path(self.video_path).name}')
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi_vis_path = str(Path(self.video_path).parent / (Path(self.video_path).stem + '_roi.png'))
        self.roi_x, self.roi_y, self.roi_r = estimate_roi(frame, output_path=roi_vis_path)
        self.frame_height, self.frame_width = frame.shape[:-1]
        cap.release()

    def generate_feature_matrix(self, window_length=30):
        bkpts = np.arange(0, len(self.pose_df) + 1, window_length)
        rows = []
        for start, stop in list(zip(bkpts[:-1], bkpts[1:])):
            feu = FeatureExtractionUnit(self.pose_df.loc[start: stop-1], self.roi_x, self.roi_y, self.roi_r)
