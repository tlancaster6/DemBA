from deeplabcut.utils.auxiliaryfunctions import read_config
from pathlib import Path
import pandas as pd
import cv2
from demba.utils.roi_utils import estimate_roi
import numpy as np
import cmath
idx = pd.IndexSlice
import math


def xy_to_theta(xy):
    return np.arctan2(xy[1], xy[0])


def rotate_around_origin(df_x, df_y, df_theta):
    cos_theta, sin_theta = np.cos(df_theta).values, np.sin(df_theta).values
    return pd.concat([(df_x * cos_theta) + (df_y * sin_theta), (df_y * cos_theta) - (df_x * sin_theta)],
                     keys=['x', 'y'], names=['coords'])


class FeatureExtractor:

    def __init__(self, config_path, video_path):
        self.config_path, self.video_path = config_path, video_path
        self.tail_points = ['stripe2', 'stripe3', 'stripe4', 'tailBase', 'tailTip']
        self.origin_ref, self.heading_ref = 'stripe1', 'nose'
        self.config_dict = read_config(config_path)
        self.h5_path = str(next(Path(self.video_path).parent.glob(Path(self.video_path).stem + '*_filtered.h5')))
        self._load_pose_data()
        self._estimate_roi()
        self._shift_reference_frames()
        # self._extract_all_features()

    def _extract_all_features(self):
        feature_df = []
        feature_df.extend(self._calc_nfish())
        self.feature_df = pd.DataFrame(feature_df).T

    def _load_pose_data(self):
        self.pose_df = pd.read_hdf(self.h5_path).T
        scorer = self.pose_df.index.get_level_values(0)[0]
        self.pose_df = self.pose_df.loc[scorer]
        self.pose_df = self.pose_df.loc[idx[:, :, ('x', 'y')], ]
        self.pose_df.index = self.pose_df.index.remove_unused_levels()
        self.pose_df.dropna(axis=1, how='all', inplace=True)
        self.pose_df.index.set_names('feature', level='bodyparts', inplace=True)
        self.pose_df = self.pose_df.sort_index()
        self.individuals, self.bodyparts = [list(self.pose_df.index.levels[i]) for i in [0, 1]]

    def _estimate_roi(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if not ret:
            print(f'could not extract a reference frame from {Path(self.video_path).name}')
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi_vis_path = str(Path(self.video_path).parent / (Path(self.video_path).stem + '_roi.png'))
        self.roi_x, self.roi_y, self.roi_r = estimate_roi(frame, output_path=roi_vis_path)
        cap.release()

    def _shift_reference_frames(self):
        origin_ref, heading_ref = self.origin_ref, self.heading_ref
        # define fish origins relative to roi center
        self.origin_df = self.pose_df.loc[idx[:, origin_ref, :]].copy()
        self.origin_df.loc[idx[:, 'x'], ] = (self.origin_df.loc[idx[:, 'x'], ] - self.roi_x).values
        self.origin_df.loc[idx[:, 'y'], ] = (self.origin_df.loc[idx[:, 'y'], ] - self.roi_y).values
        # define heading vectors relative to fish origins
        heading_vectors = self.pose_df.loc[idx[:, heading_ref, :]] - self.pose_df.loc[idx[:, origin_ref, :]]
        # define remaining bodypart positions in a coordinate system originating at the fish origin with the
        # x-axis parallel to the heading vector
        self.pose_df = self.pose_df.loc[idx[:, self.tail_points, :]] - self.pose_df.loc[idx[:, origin_ref, :]]
        rotation_angles = heading_vectors.groupby('individuals').agg(xy_to_theta)
        rotation_results = []
        for fish_id in self.individuals:
            df_x = self.pose_df.loc[idx[fish_id, 'x', :]]
            df_y = self.pose_df.loc[idx[fish_id, 'y', :]]
            df_theta = rotation_angles.loc[[fish_id, ]]
            rotation_results.append(rotate_around_origin(df_x, df_y, df_theta))
        self.pose_df = pd.concat(rotation_results, keys=self.individuals, names=['individuals'])
        self.pose_df = self.pose_df.reorder_levels(['individuals', 'feature', 'coords']).sort_index()

        # new_index = self.origin_df.index.set_levels(['rho', 'theta'], level='coords')
        # columns = self.origin_df.columns
        # self.origin_df = self.origin_df.groupby('individuals').agg(lambda x: cmath.polar(complex(*x)))
        # self.origin_df = self.origin_df.explode(column=list(columns)).values
        # self.origin_df = pd.DataFrame(self.origin_df, index=new_index, columns=columns)
        # self.origin_df = pd.concat([self.origin_df], keys=['origin'], names=['feature'])
        # self.origin_df = self.origin_df.reorder_levels(['individuals', 'feature', 'coords']).sort_index()

    def _calc_nfish(self):
        sub_df = self.pose_df.loc[idx[:, 'origin', 'rho']]
        nfishframe = sub_df.notna().astype(int).sum(axis=0)
        nfishframe.name = 'nfishframe'
        nfishroi = (sub_df < self.roi_r).astype(int).sum(axis=0)
        nfishroi.name = 'nfishroi'
        return [nfishframe, nfishroi]















