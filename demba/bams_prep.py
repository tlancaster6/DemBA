from pathlib import Path
import pandas as pd
import cv2
from demba.utils.roi_utils import estimate_roi
import re
from itertools import permutations, combinations
import numpy as np
import matplotlib as mpl
from dbscan1d.core import DBSCAN1D
idx = pd.IndexSlice



class FeatureExtractor:

    def __init__(self, video_path, quivering_annotation_path):
        self.video_path = str(video_path)
        self.quivering_annotation_path = str(quivering_annotation_path)
        self.file_stem = Path(video_path).stem
        self.trial_id = '_'.join(self.file_stem.split('_')[:2])
        self.start_frame, self.end_frame = [int(x) for x in self.file_stem.split('_')[-1].split('-')]
        self.framefeatures_path = str(video_path).replace('.mp4', '_framefeatures.csv')
        self.clipfeatures_path = str(video_path).replace('.mp4', '_clipfeatures.csv')
        self.h5_path = str(next(Path(self.video_path).parent.glob(Path(self.video_path).stem + '*_filtered.h5')))
        self.pose_df = self._load_pose_data()
        self.individuals, self.bodyparts = [list(self.pose_df.columns.levels[i]) for i in [0, 1]]
        self.roi_x, self.roi_y, self.roi_r, self.frame_height, self.frame_width = self._estimate_roi()
        self.framefeatures_df, self.clipfeatures_df = None, None

    def _load_pose_data(self):
        pose_df = pd.read_hdf(self.h5_path)
        scorer = pose_df.columns.get_level_values(0)[0]
        pose_df = pose_df.loc[:, scorer]
        pose_df = pose_df.loc[:, idx[:, :, ('x', 'y')]]
        pose_df.columns = pose_df.columns.remove_unused_levels()
        pose_df = pose_df.sort_index(axis=1)
        return pose_df

    def _estimate_roi(self):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
        ret, frame = cap.read()
        if not ret:
            print(f'could not extract a reference frame from {Path(self.video_path).name}')
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi_vis_path = str(self.video_path).replace('.mp4', '_roi.png')
        roi_x, roi_y, roi_r = estimate_roi(frame, output_path=roi_vis_path)
        frame_height, frame_width = frame.shape[:-1]
        cap.release()
        return roi_x, roi_y, roi_r, frame_height, frame_width

    def extract_all_features(self):
        # extract frame-level features
        framefeatures_df = []
        framefeatures_df.append(self._calc_nfish_frame())
        framefeatures_df.append(self._calc_nfish_pipe())
        framefeatures_df.append(self._detect_mouthing_events())
        framefeatures_df = pd.concat(framefeatures_df, axis=1)
        self.framefeatures_df = framefeatures_df
        self.framefeatures_df.to_csv(self.framefeatures_path)

        # extract clip-level features
        clipfeatures_series = pd.Series(dtype=float)
        clipfeatures_series['n_mouthing_events'] = self._calc_n_mouthing_events()
        clipfeatures_series['nfish_frame_max'] = framefeatures_df.nfish_frame.max()
        clipfeatures_series['nfish_pipe_max'] = framefeatures_df.nfish_pipe.max()
        clipfeatures_series['roi_x'], clipfeatures_series['roi_y'], clipfeatures_series['roi_r'] = self.roi_x, self.roi_y, self.roi_r
        clipfeatures_series = pd.concat([clipfeatures_series, self._calc_roi_occupancy_fractions()])
        self.clipfeatures_df = pd.DataFrame(clipfeatures_series, columns=[self.file_stem]).T
        self.clipfeatures_df.to_csv(self.clipfeatures_path)

    def _detect_mouthing_events(self, dist_thresh=25, eps=15, min_samples=15):
        candidate_dists = []
        for id1, id2 in list(permutations(self.individuals, 2)):
            dists = self.pose_df.loc[:, idx[id1, 'nose', :]].values - self.pose_df.loc[:, idx[id2, 'stripe4', :]].values
            candidate_dists.append(np.hypot(dists[:, 0], dists[:, 1]))
        if not candidate_dists:
            return pd.Series(data=-1, index=self.pose_df.index, name='mouthing_event_id')
        dists = pd.Series(np.nanmin(np.vstack(candidate_dists), axis=0), name='min_dist_nose_to_stripe4')
        subthresh_frames = dists.loc[dists < dist_thresh].index.values
        labels = DBSCAN1D(eps, min_samples).fit_predict(subthresh_frames)
        event_ids = pd.Series(data=labels, index=subthresh_frames).reindex(dists.index, fill_value=-1)
        event_ids.name = 'mouthing_event_id'
        return event_ids

    def _calc_nfish_frame(self):
        nfish_frame = self.pose_df.groupby('individuals', axis=1).any().sum(axis=1)
        nfish_frame.name = 'nfish_frame'
        return nfish_frame

    def _calc_nfish_pipe(self):
        tmp_df = self.pose_df.loc[:, idx[:, 'stripe1', :]].copy()
        tmp_df.loc[:, idx[:, :, 'x']] -= self.roi_x
        tmp_df.loc[:, idx[:, :, 'y']] -= self.roi_y
        tmp_df = (tmp_df ** 2).groupby('individuals', axis=1).sum(min_count=2) ** 0.5
        nfish_pipe = (tmp_df <= self.roi_r).sum(axis=1)
        nfish_pipe.name = 'nfish_pipe'
        return nfish_pipe

    def _calc_n_mouthing_events(self):
        event_ids = self.framefeatures_df[self.framefeatures_df.mouthing_event_id >=0].mouthing_event_id
        n_events = len(event_ids.unique())
        return n_events

    def _calc_roi_occupancy_fractions(self):
        value_counts = self.framefeatures_df.nfish_pipe.value_counts(normalize=True)
        value_counts = value_counts.reindex([0, 1, 2, 3], fill_value=0.0)
        value_counts = value_counts.rename(index={0: 'zero_occupancy_fraction',
                                                  1: 'single_occupancy_fraction',
                                                  2: 'double_occupancy_fraction',
                                                  3: 'triple_occupancy_fraction'})
        return value_counts

    def _map_quivering_annotations(self):
        ref_df = pd.read_excel(self.quivering_annotation_path, sheet_name=self.trial_id, skiprows=1)
        ref_df = ref_df[['temporal_segment_start', 'temporal_segment_end']]
        ref_df = (ref_df * 30).apply(np.round)





parent_dir = Path('/home/tlancaster/DLC/demasoni_singlenuc/BAMS_set1')
quivering_annotation_path = '/home/tlancaster/DLC/demasoni_singlenuc/quivering_annotations/Mbuna_behavior_annotations.xlsx'
vid_paths = list(parent_dir.glob('**/*.mp4'))
pattern = '((CTRL)|(BHVE))_group\d_\d{6}-\d{6}.mp4'
vid_paths = [p for p in vid_paths if re.fullmatch(pattern, p.name)]
for vp in vid_paths:
    feature_extractor = FeatureExtractor(vp, quivering_annotation_path)
    feature_extractor.extract_all_features()
    break
