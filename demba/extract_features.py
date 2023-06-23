from demba.utils.dlc_utils import h5_to_df
from demba.utils.roi_utils import estimate_roi
from demba.utils import gen_utils
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from itertools import combinations
import warnings
import deeplabcut as dlc
from math import atan2, degrees
from scipy.stats import circmean, circstd

framerate = 30  #fps

class FeatureExtractor:

    def __init__(self, config_path, video_path):
        self.config_path, self.video_path = config_path, video_path
        self.config_dict = dlc.utils.auxiliaryfunctions.read_config(config_path)
        self.h5_path = str(next(Path(self.video_path).parent.glob(Path(self.video_path).stem + '*_filtered.h5')))
        self._load_pose_estimation()
        self._estimate_roi()
        self.feature_df = None

    def extract_all_features(self):
        component_dfs = []
        component_dfs.append(self._calc_all_distances())
        component_dfs.append(self._calc_nfish_in_frame())
        component_dfs.append(self._calc_skeleton_angles())
        bp_velocities_df = self._estimate_bp_velocities()
        centroids_df = self._estimate_centroids()
        component_dfs.extend([centroids_df, bp_velocities_df])
        component_dfs.append(self._calc_roi_relationships(centroids_df))
        component_dfs.append(self._calc_intercentroid_angles(centroids_df))
        component_dfs.append(self._calc_rolling_window_features(bp_velocities_df=bp_velocities_df))
        self.feature_df = pd.concat(component_dfs, axis=1)

    def save_feature_matrix(self):
        self.feature_df.to_csv(self.video_path.replace('.mp4', '_features.csv'))

    def _load_pose_estimation(self):
        bp_map = gen_utils.bodypart2abbreviation_mapping
        id_map = gen_utils.fullid2shortid_mapping
        val_map = {'x': 'x', 'y': 'y', 'likelihood': 'c'}
        self.raw_pose_df = h5_to_df(self.h5_path)
        flat_ax = self.raw_pose_df.columns.map(lambda x: bp_map[x[1]] + id_map[x[0]] + val_map[x[2]])
        self.pose_df = self.raw_pose_df.set_axis(flat_ax, axis=1)
        self.pose_df_cols = list(self.pose_df.columns)
        self.fish_ids = list(gen_utils.fullid2shortid_mapping.values())

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

    def _calc_all_distances(self):
        unique_points = list(pd.unique([x[:-1] for x in self.pose_df.columns]))
        unique_combos = combinations(unique_points, 2)
        dists_df = []
        pose_df = self.pose_df
        for p1, p2 in unique_combos:
            dists = pose_df.apply(lambda row: np.linalg.norm([row[p1+'x'] - row[p2+'x'], row[p1+'y'] - row[p2+'y']]), axis=1)
            dists.name = f'dist_{p1}{p2}'
            dists_df.append(dists)
        return pd.DataFrame(dists_df).T

    # def _calc_all_angles(self):
    #     unique_points = list(pd.unique([x[:-1] for x in self.pose_df.columns]))
    #     unique_combos = combinations(unique_points, 2)
    #     angles_df = []
    #     pose_df = self.pose_df
    #     for p1, p2 in unique_combos:
    #         angles = pose_df.apply(lambda row: degrees(atan2(row[p1+'y'] - row[p2+'y'], row[p1+'x'] - row[p2+'x'])), axis=1)
    #         angles.name = f'angle_{p1}{p2}'
    #         angles_df.append(angles)
    #     return pd.DataFrame(angles_df).T

    def _calc_skeleton_angles(self):
        angles_df = []
        bp_map = gen_utils.bodypart2abbreviation_mapping
        skeleton_pairs = [[bp_map[x[0]], bp_map[x[1]]] for x in self.config_dict['skeleton']]
        for fid in self.fish_ids:
            for p in skeleton_pairs:
                p1, p2 = f'{p[0]}{fid}', f'{p[1]}{fid}'
                angles = self.pose_df.apply(lambda row: degrees(atan2(row[p1+'y']-row[p2+'y'], row[p1+'x']-row[p2+'x'])), axis=1)
                angles[angles < 0] += 360
                angles.name = f'angle_{p1}{p2}'
                angles_df.append(angles)
        return pd.DataFrame(angles_df).T

    def _estimate_centroids(self):
        centroids_df = []
        for fid in self.fish_ids:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                x_centroids = self.pose_df[[c for c in self.pose_df_cols if c.endswith(fid+'x')]].apply(np.nanmean, axis=1)
                y_centroids = self.pose_df[[c for c in self.pose_df_cols if c.endswith(fid+'y')]].apply(np.nanmean, axis=1)
            x_centroids.name = f'centx_{fid}'
            y_centroids.name = f'centy_{fid}'
            centroids_df.extend([x_centroids, y_centroids])
        return pd.DataFrame(centroids_df).T

    def _calc_intercentroid_angles(self, centroids_df=None):
        if centroids_df is None:
            centroids_df = self._estimate_centroids()
        fish_id_combos = list(combinations(self.fish_ids, 2))
        angles_df = []
        for fid1, fid2 in fish_id_combos:
            angles = centroids_df.apply(
                lambda row: degrees(atan2(row[f'centy_{fid1}'] - row[f'centy_{fid2}'],
                                          row[f'centx_{fid1}'] - row[f'centx_{fid2}'])), axis=1)
            angles[angles < 0] += 360
            angles_df.append(angles)
        return pd.DataFrame(angles_df).T

    def _calc_nfish_in_frame(self):
        nfish_df = pd.DataFrame(0, index=self.pose_df.index, columns=['nfishframe'])
        for fid in self.fish_ids:
            sub_df = self.pose_df[[c for c in self.pose_df_cols if c.endswith(fid+'x')]]
            nfish_df['nfishframe'] += sub_df.apply(lambda row: int(row.notnull().any()), axis=1)
        return nfish_df

    def _calc_roi_relationships(self, centroids_df=None):
        if centroids_df is None:
            centroids_df = self._estimate_centroids()
        roi_df = []
        nfish_roi = pd.Series(0, index=self.pose_df.index, name='nfishroi')
        roi_x, roi_y, roi_r = self.roi_x, self.roi_y, self.roi_r
        for fid in self.fish_ids:
            sub_df = centroids_df[[f'centx_{fid}', f'centy_{fid}']]
            disttoroicenter = sub_df.apply(lambda row: np.linalg.norm([row[0]-roi_x, row[1]-roi_y]), axis=1)
            disttoroicenter.name = f'disttoroicenter_{fid}'
            angletoroicenter = sub_df.apply(lambda row: degrees(atan2(row[1]-roi_y, row[0]-roi_x)), axis=1)
            angletoroicenter[angletoroicenter < 0] += 360
            angletoroicenter.name = f'angletoroicenter_{fid}'
            roi_df.extend([disttoroicenter, angletoroicenter])
            nfish_roi += (disttoroicenter <= roi_r).astype(int)
        roi_df.append(nfish_roi)
        return pd.DataFrame(roi_df).T

    def _estimate_bp_velocities(self):
        sub_df = self.pose_df[[c for c in self.pose_df_cols if not c.endswith('c')]]
        movement_df = []
        unique_points = list(pd.unique([x[:-1] for x in self.pose_df.columns]))
        component_velocities = sub_df.rolling(window=2).apply(np.diff)
        for p in unique_points:
            speeds = component_velocities.apply(lambda row: np.linalg.norm([row[p+'x'], row[p+'y']]), axis=1)
            speeds.name = f'speed_{p}'
            headings = component_velocities.apply(lambda row: degrees(atan2(row[p+'x'], row[p+'y'])), axis=1)
            headings.name = f'heading_{p}'
            headings[headings < 0] += 360
            movement_df.extend([speeds, headings])
        return pd.DataFrame(movement_df).T

    def _calc_rolling_window_features(self, bp_velocities_df=None, window=31):
        if bp_velocities_df is None:
            bp_velocities_df = self._estimate_bp_velocities()

        sub_df = bp_velocities_df[[c for c in bp_velocities_df.columns if c.startswith('speed')]]
        roll = sub_df.rolling(window=window, center=True)
        rolling_avg_speeds = roll.apply(np.nanmean)
        rolling_avg_speeds.columns = ['rollavg' + c for c in rolling_avg_speeds.columns]
        rolling_std_speeds = roll.apply(np.nanstd)
        rolling_std_speeds.columns = ['rollstd' + c for c in rolling_std_speeds.columns]

        sub_df = bp_velocities_df[[c for c in bp_velocities_df.columns if c.startswith('heading')]]
        roll = sub_df.rolling(window=window, center=True)
        rolling_avg_headings = roll.apply(lambda row: circmean(row, high=360, nan_policy='omit'))
        rolling_avg_headings.columns = ['rollavg' + c for c in rolling_avg_headings.columns]
        rolling_std_headings = roll.apply(lambda row: circstd(row, high=360, nan_policy='omit'))
        rolling_std_headings.columns = ['rollstd' + c for c in rolling_std_headings.columns]

        return pd.concat([rolling_avg_speeds, rolling_std_speeds, rolling_avg_headings, rolling_std_headings], axis=1)

















