import os.path
from pathlib import Path
import pandas as pd
import cv2
from demba.roi_utils import estimate_roi
from demba.dlc_utils import load_poses
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
from dbscan1d.core import DBSCAN1D
from datetime import timedelta

idx = pd.IndexSlice
from matplotlib.animation import FuncAnimation

ROI_RADIUS_MM = 76
VIDEO_FPS = 30

class FeatureExtractor:

    def __init__(self, video_path, quivering_annotation_path=None, pose_h5_path=None, mouthing_dist_mm=10, min_likelihood=0.1, n_minutes=None):
        self.video_path = str(video_path)
        self.n_minutes = n_minutes # if not None, only analyze the last n_minutes of the video
        self.quivering_annotation_path = None if quivering_annotation_path is None else str(quivering_annotation_path)
        self.file_stem = Path(video_path).stem
        self.framefeatures_path = str(video_path).replace('.mp4', '_framefeatures.csv')
        self.clipfeatures_path = str(video_path).replace('.mp4', '_clipfeatures.csv')
        self.h5_path = str(pose_h5_path)
        self.pose_df, self.individuals, self.bodyparts = load_poses(self.h5_path, min_likelihood=min_likelihood)
        self.roi_x, self.roi_y, self.roi_r, self.frame_height, self.frame_width = self._estimate_roi()
        self.mouthing_dist_mm = mouthing_dist_mm
        self.mouthing_dist_pixels = self._calc_mouthing_dist_pixels()

    def load_feature_csvs(self):
        self.framefeatures_df = pd.read_csv(self.framefeatures_path, index_col=0, low_memory=False)
        self.clipfeatures_df = pd.read_csv(self.clipfeatures_path, index_col=0, low_memory=False)

    def _estimate_roi(self):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        roi_vis_path = str(self.video_path).replace('.mp4', '_roi.png')
        roi_x, roi_y, roi_r, frame_height, frame_width = estimate_roi(frame, output_path=roi_vis_path)
        return roi_x, roi_y, roi_r, frame_height, frame_width

    def _calc_mouthing_dist_pixels(self):
        conversion_factor = (self.roi_r / ROI_RADIUS_MM)
        return self.mouthing_dist_mm * conversion_factor

    def extract_all_features(self):
        # extract frame-level features
        framefeatures_df = []
        if self.pose_df is not None:
            framefeatures_df.append(self._calc_nfish_frame())
            framefeatures_df.append(self._calc_nfish_pipe())
            framefeatures_df.append(self._detect_mouthing_events())
            framefeatures_df.append(self._detect_spawning_events(mouthing_event_ids=framefeatures_df[-1]))
            framefeatures_df.append(self._detect_double_occupancy_events())
        if self.quivering_annotation_path is not None:
            male_lead_quiver, male_circle_quiver, female_circle_quiver = self._map_quivering_annotations()
            framefeatures_df.append(male_lead_quiver)
            framefeatures_df.append(male_circle_quiver)
            framefeatures_df.append(female_circle_quiver)
        framefeatures_df = pd.concat(framefeatures_df, axis=1)
        if self.n_minutes is not None:
            n_frames = self.n_minutes * 60 * VIDEO_FPS
            framefeatures_df = framefeatures_df.tail(n_frames)
        self.framefeatures_df = framefeatures_df
        self.framefeatures_df.to_csv(self.framefeatures_path)

        # extract clip-level features
        clipfeatures_series = pd.Series(dtype=float)
        if self.quivering_annotation_path is not None:
            clipfeatures_series['male_lead_quivering_fraction'] = self._calc_quivering_fraction("male_lead_quiver")
            clipfeatures_series['male_circle_quivering_fraction'] = self._calc_quivering_fraction("male_circle_quiver")
            clipfeatures_series['female_circle_quivering_fraction'] = self._calc_quivering_fraction("female_circle_quiver")
            clipfeatures_series['n_male_lead_quivers'] = self._calc_n_quivers("male_lead_quiver")
            clipfeatures_series['n_male_circle_quivers'] = self._calc_n_quivers("male_circle_quiver")
            clipfeatures_series['n_female_circle_quivers'] = self._calc_n_quivers("female_circle_quiver")
        if self.pose_df is not None:
            clipfeatures_series['n_mouthing_events'] = self._calc_n_mouthing_events()
            clipfeatures_series['n_double_occupancy_events'] = self._calc_n_double_occupancy_events()
            clipfeatures_series['n_spawning_events'] = self._calc_n_spawning_events()
            clipfeatures_series['mouthing_event_fraction'] = self._calc_mouthing_event_fraction()
            clipfeatures_series['double_occupancy_event_fraction'] = self._calc_double_occupancy_event_fraction()
            clipfeatures_series['spawning_event_fraction'] = self._calc_spawning_event_fraction()
            clipfeatures_series['nfish_frame_max'] = framefeatures_df.nfish_frame.max()
            clipfeatures_series['nfish_pipe_max'] = framefeatures_df.nfish_pipe.max()
            clipfeatures_series['roi_x'], clipfeatures_series['roi_y'], clipfeatures_series['roi_r'] = self.roi_x, self.roi_y, self.roi_r
            clipfeatures_series = pd.concat([clipfeatures_series, self._calc_roi_occupancy_fractions()])
        self.clipfeatures_df = pd.DataFrame(clipfeatures_series, columns=[self.file_stem]).T
        self.clipfeatures_df.to_csv(self.clipfeatures_path)

    def _detect_mouthing_events(self, eps=15, min_samples=15):
        candidate_dists = []
        for id1, id2 in list(permutations(self.individuals, 2)):
            dists = self.pose_df.loc[:, idx[id1, 'nose', :]].values - self.pose_df.loc[:, idx[id2, 'stripe4', :]].values
            candidate_dists.append(np.hypot(dists[:, 0], dists[:, 1]))
        if not candidate_dists:
            return pd.Series(data=-1, index=self.pose_df.index, name='mouthing_event_id')
        dists = pd.Series(np.nanmin(np.vstack(candidate_dists), axis=0), name='min_dist_nose_to_stripe4')
        subthresh_frames = dists.loc[dists < self.mouthing_dist_pixels].index.values
        labels = DBSCAN1D(eps, min_samples).fit_predict(subthresh_frames)
        event_ids = pd.Series(data=labels, index=subthresh_frames).reindex(dists.index, fill_value=-1)
        for eid in event_ids.unique():
            if eid >= 0:
                start_idx = event_ids[event_ids == eid].index.min()
                end_idx = event_ids[event_ids == eid].index.max()
                event_ids.loc[start_idx:end_idx] = eid
        event_ids.name = 'mouthing_event_id'
        return event_ids

    def _detect_double_occupancy_events(self, eps=30, min_samples=30):
        nfish_pipe = self._calc_nfish_pipe()
        double_occupancy_frames = nfish_pipe[nfish_pipe == 2].index.values
        labels = DBSCAN1D(eps, min_samples).fit_predict(double_occupancy_frames)
        event_ids = pd.Series(data=labels, index=double_occupancy_frames).reindex(nfish_pipe.index, fill_value=-1)
        for eid in event_ids.unique():
            if eid >= 0:
                start_idx = event_ids[event_ids == eid].index.min()
                end_idx = event_ids[event_ids == eid].index.max()
                event_ids.loc[start_idx:end_idx] = eid
        event_ids.name = 'double_occupancy_event_id'
        return event_ids

    def _detect_spawning_events(self, mouthing_event_ids=None, eps=150, min_samples=6):
        if mouthing_event_ids is None:
            mouthing_event_ids = self._detect_mouthing_events()
        mouthing_event_start_frames = mouthing_event_ids.reset_index().groupby('mouthing_event_id').first().loc[0:]['index'].values
        mouthing_event_end_frames = mouthing_event_ids.reset_index().groupby('mouthing_event_id').last().loc[0:]['index'].values
        combined_mouthing_event_frames = np.concatenate((mouthing_event_start_frames, mouthing_event_end_frames))
        labels = DBSCAN1D(eps, min_samples).fit_predict(combined_mouthing_event_frames)
        event_ids = pd.Series(data=labels, index=combined_mouthing_event_frames).reindex(mouthing_event_ids.index, fill_value=-1)
        for eid in event_ids.unique():
            if eid >= 0:
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
        event_ids = self.framefeatures_df[self.framefeatures_df.double_occupancy_event_id >= 0].double_occupancy_event_id
        n_events = len(event_ids.unique())
        return n_events

    def _calc_n_spawning_events(self):
        event_ids = self.framefeatures_df[self.framefeatures_df.spawning_event_id >= 0].spawning_event_id
        n_events = len(event_ids.unique())
        return n_events

    def _calc_spawning_event_fraction(self):
        n_spawning_frames = len(self.framefeatures_df[self.framefeatures_df.spawning_event_id >= 0])
        spawning_fraction = n_spawning_frames / len(self.framefeatures_df)
        return spawning_fraction

    def _calc_double_occupancy_event_fraction(self):
        n_double_occupancy_frames = len(self.framefeatures_df[self.framefeatures_df.double_occupancy_event_id >= 0])
        double_occupancy_fraction = n_double_occupancy_frames / len(self.framefeatures_df)
        return double_occupancy_fraction

    def _calc_mouthing_event_fraction(self):
        n_mouthing_frames = len(self.framefeatures_df[self.framefeatures_df.mouthing_event_id >= 0])
        mouthing_fraction = n_mouthing_frames / len(self.framefeatures_df)
        return mouthing_fraction

    def _calc_roi_occupancy_fractions(self):
        value_counts = self.framefeatures_df.nfish_pipe.value_counts(normalize=True)
        value_counts = value_counts.reindex([0, 1, 2, 3], fill_value=0.0)
        value_counts = value_counts.rename(index={0: 'raw_zero_occupancy_fraction',
                                                  1: 'raw_single_occupancy_fraction',
                                                  2: 'raw_double_occupancy_fraction',
                                                  3: 'raw_triple_occupancy_fraction'})
        return value_counts

    def _calc_quivering_fraction(self, quivering):
        n_quivering_frames = len(self.framefeatures_df[self.framefeatures_df[quivering] >= 0])
        quivering_fraction = n_quivering_frames / len(self.framefeatures_df)
        return quivering_fraction

    def _calc_n_quivers(self, quivering):
        event_ids = self.framefeatures_df[self.framefeatures_df[quivering] >= 0][quivering]
        n_events = len(event_ids.unique())
        return n_events

    def _clean_metadata_string(self, meta_str):
        meta_dict = eval(meta_str)
        return meta_dict["TEMPORAL-SEGMENTS"]
        
    def _map_quivering_annotations(self):
        try:
            ref_df = pd.read_excel(self.quivering_annotation_path, sheet_name=self.file_stem, skiprows=1)
        except ValueError:
            ref_df = pd.read_excel(self.quivering_annotation_path, sheet_name=self.file_stem.split('cropped')[0], skiprows=1)
        ref_df = ref_df[['temporal_segment_start', 'temporal_segment_end', 'metadata']]
        ref_df["temporal_segment_start"] = (ref_df["temporal_segment_start"]  * 30).apply(np.round)
        ref_df["temporal_segment_end"] = (ref_df["temporal_segment_end"]  * 30).apply(np.round)
        ref_df["metadata"] = (ref_df["metadata"]).apply(self._clean_metadata_string)
        male_lead_quiver = pd.Series(False, index=pd.RangeIndex(0, 378000), name='male_lead_quiver')
        male_circle_quiver = pd.Series(False, index=pd.RangeIndex(0, 378000), name='male_circle_quiver')
        female_circle_quiver = pd.Series(False, index=pd.RangeIndex(0, 378000), name='female_circle_quiver')
        MIN_FRAME = 0
        MAX_FRAME = 378000 - 1 # 3 hr 30 min video at 30 FPS is 378000 Frames in total and minus to maintain similarity to index.max()
        for event in ref_df.iterrows(): # for every event of quivering how should we update the series 
            event = event[1]
            metadata = event.metadata
            start_in_range = 0 < event.temporal_segment_start < MAX_FRAME
            end_in_range = 0 < event.temporal_segment_end < MAX_FRAME
            # NOTE ILOC has been deprecated but still works. If it does break in the future move to arr[start:end] 
            if start_in_range and end_in_range:
                if metadata == "male-lead-quiver":
                    male_lead_quiver.iloc[int(event.temporal_segment_start): int(event.temporal_segment_end)] = True
                elif metadata == "male-circle-quiver":
                    male_circle_quiver.iloc[int(event.temporal_segment_start): int(event.temporal_segment_end)] = True
                elif metadata == "female-circle-quiver":
                    female_circle_quiver.iloc[int(event.temporal_segment_start): int(event.temporal_segment_end)] = True
            elif start_in_range:
                if metadata == "male-lead-quiver":
                    male_lead_quiver.iloc[int(event.temporal_segment_start): MAX_FRAME] = True
                elif metadata == "male-circle-quiver":
                    male_circle_quiver.iloc[int(event.temporal_segment_start): MAX_FRAME] = True
                elif metadata == "female-circle-quiver":
                    female_circle_quiver.iloc[int(event.temporal_segment_start): MAX_FRAME] = True
            elif end_in_range:
                if metadata == "male-lead-quiver":
                    male_lead_quiver.iloc[0: int(event.temporal_segment_end)] = True
                elif metadata == "male-circle-quiver":
                    male_circle_quiver.iloc[0: int(event.temporal_segment_end)] = True
                elif metadata == "female-circle-quiver":
                    female_circle_quiver.iloc[0: int(event.temporal_segment_end)] = True
        transformed_series = []
        for data_series in male_lead_quiver, male_circle_quiver, female_circle_quiver:
            event_ids = (data_series != data_series.shift()).cumsum() * data_series - 1
            event_ids = event_ids.where(data_series, -1)
            transformed_series.append(event_ids)
        male_lead_quiver, male_circle_quiver, female_circle_quiver = transformed_series
        return male_lead_quiver, male_circle_quiver, female_circle_quiver

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

    def generate_predicted_spawning_summary(self):
        if self.pose_df is None:
            return
        outfile_path = str(self.video_path).replace('.mp4', '_predicted_spawning_summary.csv')
        event_ids = self.framefeatures_df.spawning_event_id
        df = []
        for eid in event_ids.unique():
            if eid >= 0:
                start_time = str(timedelta(seconds=event_ids[event_ids == eid].index.min()/30))[:7]
                stop_time = str(timedelta(seconds=event_ids[event_ids == eid].index.max()/30))[:7]
                df.append({'event_id': eid, 'start': start_time, 'stop': stop_time})
        df = pd.DataFrame.from_records(df)
        df.to_csv(outfile_path, index='event_id')

    def generate_predicted_double_occupancy_summary(self):
        if self.pose_df is None:
            return
        outfile_path = str(self.video_path).replace('.mp4', '_predicted_double_occupancy_summary.csv')
        event_ids = self.framefeatures_df.double_occupancy_event_id
        df = []
        for eid in event_ids.unique():
            if eid >= 0:
                start_time = str(timedelta(seconds=event_ids[event_ids == eid].index.min()/30))[:7]
                stop_time = str(timedelta(seconds=event_ids[event_ids == eid].index.max()/30))[:7]
                df.append({'event_id': eid, 'start': start_time, 'stop': stop_time})
        df = pd.DataFrame.from_records(df)
        df.to_csv(outfile_path, index='event_id')


def concat_clipfeature_csvs(parent_dir):
    parent_dir = Path(parent_dir)
    clipfeature_csv_paths = list(parent_dir.glob('**/*_clipfeatures.csv'))
    rows = []
    for csv_path in clipfeature_csv_paths:
        rows.append(pd.read_csv(str(csv_path), index_col=0))
    df = pd.concat(rows, axis=0)
    df.to_csv(str(parent_dir / 'collated_clipfeatures.csv'))
    pd.concat(rows, axis=0)

def process_video(video_path, quivering_annotation_path=None, pose_h5_path=None, visualize=False):
    video_path = Path(video_path)
    print(f'processing {video_path.stem}')
    fe = FeatureExtractor(video_path, quivering_annotation_path, pose_h5_path)
    fe.extract_all_features()
    fe.generate_predicted_double_occupancy_summary()
    fe.generate_predicted_spawning_summary()
    if visualize:
        print(f'generating visualization for {video_path.stem}')
        fe.visualize_features()

def process_all(parent_dir, quivering_annotation_path, visualize=False):
    parent_dir = Path(parent_dir)
    vid_paths = list(parent_dir.glob('**/*cropped.mp4'))
    for vp in vid_paths:
        try:
            pose_h5_path = list(vp.parent.glob(f'{vp.stem}*[0-9].h5'))[0]
        except IndexError:
            pose_h5_path = None
        process_video(vp, quivering_annotation_path, pose_h5_path, visualize=visualize)
    print('all videos processed')

def delete_outputs(parent_dir, keep_pose_data=True):
    parent_dir = Path(parent_dir)
    vid_paths = list(parent_dir.glob('**/*cropped.mp4'))
    targets = ['*_assemblies.pickle', '*_el.h5', '*_el.pickle', '*_filtered.csv', '*_filtered.h5', '*_labeled.mp4',
               '*_framefeatures.csv', '*_clipfeatures.csv', '*_featurevis.mp4', '*_roi.png']
    if not keep_pose_data:
        targets.extend(['*_full.pickle', '*_meta.pickle', '*_full.mp4'])
    for vp in vid_paths:
        vid_parent = vp.parent
        for target in targets:
            if list(vid_parent.glob(target)):
                list(vid_parent.glob(target))[0].unlink()


