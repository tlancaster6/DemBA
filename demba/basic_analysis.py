import time
from pathlib import Path
import pandas as pd
import cv2
import os
from demba.utils.roi_utils import estimate_roi
from itertools import permutations, combinations
import numpy as np
import matplotlib as mpl
from dbscan1d.core import DBSCAN1D

mpl.use('TkAgg')
idx = pd.IndexSlice


class Track:

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.start, self.stop = self.data.index.min(), self.data.index.max()
        self.individual = data.columns.get_level_values(0)[0]
        self.data = self.data.droplevel('individuals', axis=1)


class BasicAnalyzer:

    def __init__(self, video_path):
        self.video_path = str(video_path)
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
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
        ret, frame = cap.read()
        if not ret:
            print(f'could not extract a reference frame from {Path(self.video_path).name}')
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi_vis_path = str(Path(self.output_dir) / (Path(self.video_path).stem + '_roi.png'))
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

    def detect_mouthing_events(self, dist_thresh=25, eps=15, min_samples=15):
        dists = self.calc_min_dist_nose_to_stripe4()
        subthresh_frames = dists.loc[dists < dist_thresh].index.values
        labels = DBSCAN1D(eps, min_samples).fit_predict(subthresh_frames)
        event_ids = pd.Series(data=labels, index=subthresh_frames).reindex(dists.index, fill_value=-1)
        return event_ids

    def summarize_mouthing_events(self, event_ids=None):
        output_path = str(Path(self.output_dir) / 'mouthing_events.csv')
        if not event_ids:
            event_ids = self.detect_mouthing_events()
        event_ids = pd.DataFrame({'frame': event_ids.index, 'event_id': event_ids.values})
        actual_events = event_ids[event_ids.event_id >= 0]
        starts = actual_events.groupby(actual_events.event_id).frame.min() / 30
        stops = actual_events.groupby(actual_events.event_id).frame.max() / 30
        event_lens = (stops - starts).values
        n_mouthing_events = len(actual_events.event_id.unique())
        if n_mouthing_events:
            summary_dict = {
                'n_mouthing_events': n_mouthing_events,
                'mean_event_len_secs': np.mean(event_lens),
                'max_event_len_secs': np.max(event_lens),
                'min_event_len_secs': np.min(event_lens),
                'med_event_len_secs': np.median(event_lens),
                'std_event_len_secs': np.std(event_lens)
            }
        else:
            summary_dict = {'n_mouthing_events': 0}
        pd.Series(summary_dict).to_csv(output_path, header=False)

    def parse_tracks(self):
        tracks = []
        for individual in self.individuals:
            sub_df = self.pose_df.loc[:, idx[individual, :, :]]
            deltas = sub_df.notnull().any(axis=1).astype(int).diff()
            starts = list(deltas[deltas == 1].index)
            stops = list(deltas[deltas == -1].index)
            if sub_df.iloc[0].notnull().any():
                starts = [sub_df.iloc[0].name] + starts
            if sub_df.iloc[-1].notnull().any():
                stops = stops + [sub_df.iloc[-1].name + 1]
            for start, stop in list(zip(starts, stops)):
                tracks.append(Track(sub_df.iloc[start:stop]))
        tracks = sorted(tracks, key=lambda x: x.start)
        return tracks

    def summarize_tracks(self):
        tracks = self.parse_tracks()
        track_lengths = [len(t.data) / 30 for t in tracks]
        summary_dict = {
            'n_tracks': len(tracks),
            'mean_len_secs': np.mean(track_lengths),
            'max_len_secs': np.max(track_lengths),
            'min_len_secs': np.min(track_lengths),
            'med_len_secs': np.median(track_lengths),
            'std_len_secs': np.std(track_lengths)
        }
        output_path = str(Path(self.output_dir) / 'track_summaries.csv')
        pd.Series(summary_dict).to_csv(output_path, header=False)
        return summary_dict

    def calc_keypoint_pair_dists(self, multiindex=True):
        sub_df = self.pose_df.loc[self.calc_nfish_frame() >= 2, :]
        if len(sub_df) == 0:
            print('no frames found with 2 or more fish. skipping keypoint pair distance calculations')
            return
        new_df_column_index = []
        for i1, i2 in list(combinations(self.individuals, 2)):
            for bp in self.bodyparts:
                new_df_column_index.append(tuple((i1, i2, bp)))
        new_df_data = []
        for i1, i2, bp in new_df_column_index:
            dists = sub_df.loc[:, idx[i1, bp, :]].values - sub_df.loc[:, idx[i2, bp, :]].values
            new_df_data.append(np.hypot(dists[:, 0], dists[:, 1]))
        if multiindex:
            new_df_column_index = pd.MultiIndex.from_tuples(new_df_column_index)
        else:
            new_df_column_index = ['_'.join([i1, i2, bp]) for i1, i2, bp in new_df_column_index]
        new_df_data = np.vstack(new_df_data).T
        new_df = pd.DataFrame(data=new_df_data, columns=new_df_column_index, index=sub_df.index)
        return new_df










analysis_dir = Path(r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\analysis")
for video_dir in analysis_dir.glob('*'):
    video_file = video_dir / f'{video_dir.name}.mp4'
    if video_file.exists() and 'BHVE_group8' not in video_file.name:
        print(f'analyzing {video_file.name}')
        ba = BasicAnalyzer(video_file)
        ba.summarize_tracks()
        ba.summarize_mouthing_events()

#vid = r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\testclip\testclip.mp4"
# vid = r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\Analysis\Videos\BHVE_group3\BHVE_group3.mp4"
# vid = r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\analysis\CTRL_group1\CTRL_group1.mp4"
# ba = BasicAnalyzer(vid)
# ba.summarize_tracks()
# ba.summarize_mouthing_events()

