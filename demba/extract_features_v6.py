import os.path
from pathlib import Path
import pandas as pd
import cv2
from demba.utils.roi_utils import estimate_roi
import re
from itertools import permutations, combinations
import numpy as np
import matplotlib.pyplot as plt
from dbscan1d.core import DBSCAN1D
idx = pd.IndexSlice
from matplotlib.animation import FuncAnimation


class FeatureExtractor:

    def __init__(self, video_path, quivering_annotation_path, overwrite=False):
        self.video_path = str(video_path)
        self.quivering_annotation_path = str(quivering_annotation_path)
        self.file_stem = Path(video_path).stem
        self.framefeatures_path = str(video_path).replace('.mp4', '_framefeatures.csv')
        self.clipfeatures_path = str(video_path).replace('.mp4', '_clipfeatures.csv')
        try:
            self.h5_path = str(next(Path(self.video_path).parent.glob(Path(self.video_path).stem + '*_filtered.h5')))
        except StopIteration:
            self.h5_path = None
        self.pose_df, self.individuals, self.bodyparts = self._load_pose_data()
        self.roi_x, self.roi_y, self.roi_r, self.frame_height, self.frame_width = self._estimate_roi()
        if overwrite or not (os.path.exists(self.framefeatures_path) and os.path.exists(self.clipfeatures_path)):
            self.extract_all_features()
        self._load_feature_csvs()

    def _load_pose_data(self):
        if self.h5_path is not None:
            pose_df = pd.read_hdf(self.h5_path)
            scorer = pose_df.columns.get_level_values(0)[0]
            pose_df = pose_df.loc[:, scorer]
            pose_df = pose_df.loc[:, idx[:, :, ('x', 'y')]]
            pose_df.columns = pose_df.columns.remove_unused_levels()
            pose_df = pose_df.sort_index(axis=1)
            individuals, bodyparts = [list(pose_df.columns.levels[i]) for i in [0, 1]]
            return pose_df, individuals, bodyparts
        return None, None, None

    def _load_feature_csvs(self):
        self.framefeatures_df = pd.read_csv(self.framefeatures_path, index_col=0)
        self.clipfeatures_df = pd.read_csv(self.clipfeatures_path, index_col=0)

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
        if self.pose_df is not None:
            framefeatures_df.append(self._calc_nfish_frame())
            framefeatures_df.append(self._calc_nfish_pipe())
            framefeatures_df.append(self._detect_mouthing_events())
            framefeatures_df.append(self._detect_double_occupancy_events())
            framefeatures_df.append(self._detect_spawning_events())
        framefeatures_df.append(self._map_quivering_annotations())
        framefeatures_df = pd.concat(framefeatures_df, axis=1)
        self.framefeatures_df = framefeatures_df
        self.framefeatures_df.to_csv(self.framefeatures_path)

        # extract clip-level features
        clipfeatures_series = pd.Series(dtype=float)
        clipfeatures_series['quivering_fraction'] = self._calc_quivering_fraction()
        if self.pose_df is not None:
            clipfeatures_series['n_mouthing_events'] = self._calc_n_mouthing_events()
            clipfeatures_series['n_double_occupancy_events'] = self._calc_n_double_occupancy_events()
            clipfeatures_series['n_spawning_events'] = self._calc_n_spawning_events()
            clipfeatures_series['nfish_frame_max'] = framefeatures_df.nfish_frame.max()
            clipfeatures_series['nfish_pipe_max'] = framefeatures_df.nfish_pipe.max()
            clipfeatures_series['roi_x'], clipfeatures_series['roi_y'], clipfeatures_series['roi_r'] = self.roi_x, self.roi_y, self.roi_r
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
        for eid in event_ids.unique():
            if eid != -1:
                start_idx = event_ids[event_ids == eid].index.min()
                end_idx = event_ids[event_ids == eid].index.max()
                event_ids.loc[start_idx:end_idx] = eid
        event_ids.name = 'mouthing_event_id'
        return event_ids

    def _detect_double_occupancy_events(self, eps=15, min_samples=15):
        nfish_pipe = self._calc_nfish_pipe()
        double_occupancy_frames = nfish_pipe[nfish_pipe == 2].index.values
        labels = DBSCAN1D(eps, min_samples).fit_predict(double_occupancy_frames)
        event_ids = pd.Series(data=labels, index=double_occupancy_frames).reindex(nfish_pipe.index, fill_value=-1)
        for eid in event_ids.unique():
            if eid != -1:
                start_idx = event_ids[event_ids == eid].index.min()
                end_idx = event_ids[event_ids == eid].index.max()
                event_ids.loc[start_idx:end_idx] = eid
        event_ids.name = 'double_occupancy_event_id'
        return event_ids

    def _detect_spawning_events(self, dist_thresh=25, eps=150, min_samples=150):
        candidate_dists = []
        for id1, id2 in list(permutations(self.individuals, 2)):
            dists = self.pose_df.loc[:, idx[id1, 'nose', :]].values - self.pose_df.loc[:, idx[id2, 'stripe4', :]].values
            candidate_dists.append(np.hypot(dists[:, 0], dists[:, 1]))
        if not candidate_dists:
            return pd.Series(data=-1, index=self.pose_df.index, name='spawning_event_id')
        dists = pd.Series(np.nanmin(np.vstack(candidate_dists), axis=0), name='min_dist_nose_to_stripe4')
        subthresh_frames = dists.loc[dists < dist_thresh].index.values
        labels = DBSCAN1D(eps, min_samples).fit_predict(subthresh_frames)
        event_ids = pd.Series(data=labels, index=subthresh_frames).reindex(dists.index, fill_value=-1)
        for eid in event_ids.unique():
            if eid != -1:
                start_idx = event_ids[event_ids == eid].index.min()
                end_idx = event_ids[event_ids == eid].index.max()
                event_ids.loc[start_idx:end_idx] = eid
        event_ids.name = 'spawning_event_id'
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
        event_ids = self.framefeatures_df[self.framefeatures_df.mouthing_event_id >= 0].mouthing_event_id
        n_events = len(event_ids.unique())
        return n_events

    def _calc_n_double_occupancy_events(self):
        event_ids = self.framefeatures_df[self.framefeatures_df.double_occupancy_event_id >=0 ].double_occupancy_event_id
        n_events = len(event_ids.unique())
        return n_events

    def _calc_n_spawning_events(self):
        event_ids = self.framefeatures_df[self.framefeatures_df.spawning_event_id >= 0].spawning_event_id
        n_events = len(event_ids.unique())
        return n_events

    def _calc_spawning_fraction(self):
        n_spawning_frames = len(self.framefeatures_df[self.framefeatures_df.spawning_event_id >= 0])
        spawning_fraction = n_spawning_frames / len(self.framefeatures_df)
        return spawning_fraction

    def _calc_roi_occupancy_fractions(self):
        value_counts = self.framefeatures_df.nfish_pipe.value_counts(normalize=True)
        value_counts = value_counts.reindex([0, 1, 2, 3], fill_value=0.0)
        value_counts = value_counts.rename(index={0: 'zero_occupancy_fraction',
                                                  1: 'single_occupancy_fraction',
                                                  2: 'double_occupancy_fraction',
                                                  3: 'triple_occupancy_fraction'})
        return value_counts

    def _calc_quivering_fraction(self):
        quivering_fraction = self.framefeatures_df.quivering.sum() / len(self.framefeatures_df)
        return quivering_fraction

    def _map_quivering_annotations(self):
        ref_df = pd.read_excel(self.quivering_annotation_path, sheet_name=self.file_stem, skiprows=1)
        ref_df = ref_df[['temporal_segment_start', 'temporal_segment_end']]
        ref_df = (ref_df * 30).apply(np.round)
        quivering = pd.Series(False, index=pd.RangeIndex(0, 378000), name='quivering')
        for event in ref_df.iterrows():
            event = event[1]
            start_in_range = 0 < event.temporal_segment_start < quivering.index.max()
            end_in_range = 0 < event.temporal_segment_end < quivering.index.max()
            if start_in_range and end_in_range:
                quivering.iloc[int(event.temporal_segment_start): int(event.temporal_segment_end)] = True
            elif start_in_range:
                quivering.iloc[int(event.temporal_segment_start): quivering.index.max()] = True
            elif end_in_range:
                quivering.iloc[0: int(event.temporal_segment_end)] = True
        return quivering

    def visualize_features(self, overwrite=True):
        def grab_frame(vid_cap, frame_number):
            vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = vid_cap.read()
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        def grab_framefeatures_as_strings(frame):
            feats = [f'frame = {frame}']
            feats.extend([f'{feat} = {val}' for feat, val in list(self.framefeatures_df.loc[frame].items())])
            return feats

        out_path = str(self.video_path).replace('.mp4', '_featurevis.mp4')
        if not overwrite and os.path.exists(out_path):
            return
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[1].axis('off')
        cap = cv2.VideoCapture(self.video_path)
        im = axes[0].imshow(grab_frame(cap, 0))
        feat_strings = grab_framefeatures_as_strings(0)
        y_positions = np.linspace(0.1, 0.9, len(feat_strings))
        txt_array = [axes[1].text(0.1, y_positions[i], feat_strings[i]) for i in range(len(feat_strings))]

        def animate(i):
            im.set_data(grab_frame(cap, i))
            feat_strings = grab_framefeatures_as_strings(i)
            [txt.set_text(feat) for txt, feat in list(zip(txt_array, feat_strings))]
            return *txt_array, im

        n_frames = len(self.framefeatures_df)
        anim = FuncAnimation(
            fig,
            animate,
            frames=n_frames,
            interval=1000 / 30,
            )

        anim.save(out_path, writer='ffmpeg', fps=30)
        cap.release()
        plt.close('all')


def concat_clipfeature_csvs(parent_dir):
    parent_dir = Path(parent_dir)
    clipfeature_csv_paths = list(parent_dir.glob('**/*_clipfeatures.csv'))
    rows = []
    for csv_path in clipfeature_csv_paths:
        rows.append(pd.read_csv(str(csv_path), index_col=0))
    df = pd.concat(rows, axis=0)
    df.to_csv(str(parent_dir / 'collated_clipfeatures.csv'))
    pd.concat(rows, axis=0)


def process_all(parent_dir, quivering_annotation_path, overwrite=False, visualize=False):
    parent_dir = Path(parent_dir)
    vid_paths = list(parent_dir.glob('**/*.mp4'))
    pattern = '((CTRL)|(BHVE))_group\d.mp4'
    vid_paths = [p for p in vid_paths if re.fullmatch(pattern, p.name)]
    for vp in vid_paths:
        print(f'processing {vp.stem}')
        fe = FeatureExtractor(vp, quivering_annotation_path, overwrite=overwrite)
        if visualize:
            print(f'generating visualization for {vp.stem}')
            fe.visualize_features()


def delete_outputs(parent_dir, keep_pose_data=True):
    parent_dir = Path(parent_dir)
    vid_paths = list(parent_dir.glob('**/*.mp4'))
    pattern = r'((CTRL)|(BHVE))_group\d.mp4'
    vid_paths = [p for p in vid_paths if re.fullmatch(pattern, p.name)]
    targets = ['*_assemblies.pickle', '*_el.h5', '*_el.pickle', '*_filtered.csv', '*_filtered.h5', '*_labeled.mp4',
               '*_framefeatures.csv', '*_clipfeatures.csv', '*_featurevis.mp4', '*_roi.png']
    if not keep_pose_data:
        targets.extend(['*_full.pickle', '*_meta.pickle', '*_full.mp4'])
    for vp in vid_paths:
        vid_parent = vp.parent
        for target in targets:
            if list(vid_parent.glob(target)):
                list(vid_parent.glob(target))[0].unlink()


# scratch code

# parent_dir = Path('/home/tlancaster/DLC/demasoni_singlenuc/Analysis/Videos')
# quivering_annotation_path = '/home/tlancaster/DLC/demasoni_singlenuc/quivering_annotations/Mbuna_behavior_annotations.xlsx'
# process_all(parent_dir, quivering_annotation_path, overwrite=True, visualize=False)
# concat_clipfeature_csvs(parent_dir)

#process_all(parent_dir, quivering_annotation_path, overwrite=False, visualize=True)
# vid_path = r'C:\Users\tucke\DLC_Projects\demasoni_singlenuc\BAMS_set1\test\1_fish\BHVE_group1_345600-347399.mp4'
# fe = FeatureExtractor(vid_path, quivering_annotation_path, overwrite=True)
# fe.visualize_features()


vid_path = '/home/tlancaster/DLC/demasoni_singlenuc/Analysis/Videos/CTRL_group1/CTRL_group1.mp4'
quivering_annotation_path = '/home/tlancaster/DLC/demasoni_singlenuc/quivering_annotations/Mbuna_behavior_annotations.xlsx'
fe = FeatureExtractor(vid_path, quivering_annotation_path, overwrite=True)