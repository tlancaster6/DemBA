from pathlib import Path
import pandas as pd
from demba.utils.gen_utils import get_frame_size
from demba.utils.roi_utils import estimate_roi
import numpy as np
import cv2
import re
idx = pd.IndexSlice


class BamsPrepper:

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.frame_h, self.frame_w = get_frame_size(video_path)
        self.h5_path = str(next(self.video_path.parent.glob(self.video_path.stem + '*_filtered.h5')))
        self.pose_df = self._load_pose_data()
        self.refined_pose_df = self.pose_df.copy()
        self.individuals, self.bodyparts = [list(self.pose_df.columns.levels[i]) for i in [0, 1]]

    def refine_pose_data(self, min_track_len=10, min_keypoints=4, border_crop_width=10):
        """
        pseudocode
        -----------
        for each individual:
            1) find frames where they have less than min_keypoints in their skeleton and set all keypoints for that
               that individual in that frame to null
            2) find frames with one-or-more keypoints located within border_crop_width of the edge of the frame, and
               set all keypoints for that individual in that frame to null
            3) remove any remaining tracklets that span less than min_track_len consecutive frames
        """
        for ind in self.individuals:
            # refine by min_keypoints
            n_kp = self.refined_pose_df.xs((ind, 'x'), level=('individuals', 'coords'), axis=1).count(axis=1)
            self.refined_pose_df.loc[n_kp < min_keypoints, ind] = np.nan
            # refine by border_crop_width
            x_df = self.refined_pose_df.xs((ind, 'x'), level=('individuals', 'coords'), axis=1)
            y_df = self.refined_pose_df.xs((ind, 'y'), level=('individuals', 'coords'), axis=1)
            xmin, ymin = border_crop_width, border_crop_width
            xmax, ymax = (self.frame_w - border_crop_width), (self.frame_h - border_crop_width)
            self.refined_pose_df.loc[(x_df < xmin).any(axis=1), ind] = np.nan
            self.refined_pose_df.loc[(y_df < ymin).any(axis=1), ind] = np.nan
            self.refined_pose_df.loc[(x_df > xmax).any(axis=1), ind] = np.nan
            self.refined_pose_df.loc[(y_df > ymax).any(axis=1), ind] = np.nan
            # refine by min_track_len
            is_in_frame = self.refined_pose_df.xs((ind, 'x'), level=('individuals', 'coords'), axis=1).any(axis=1)
            deltas = is_in_frame.astype(int).diff()
            starts = list(deltas[deltas == 1].index)
            stops = list(deltas[deltas == -1].index)
            if is_in_frame.values[0]:
                starts = [0] + starts
            if is_in_frame.values[-1]:
                stops = stops + is_in_frame.index[-1]
            for start, stop in list(zip(starts, stops)):
                if (stop - start) < min_track_len:
                    self.refined_pose_df.loc[start:stop, ind] = np.nan

    def segment_data(self, min_segment_len=30):
        """
        psuedocode
        ----------
        1) begin by splitting the data into contiguous segments -- the longest possible segments uninterruped by
           all-null rows (i.e., rows with no detected animals)
        2) filter out any segments that are already shorter than min_segment length
        3) for each segment:
            a) split the segment whenever a new fish appears (i.e., the number of fish in frame increases) unless
               the number increases from 2 to 3 (since only 3 fish are in the tank, the id of the out-of-frame
               fish was theoretically preserved)

        """
        valid_segments = []
        occupied_frames = self.refined_pose_df[self.refined_pose_df.any(axis=1)].index.values
        occupied_segments = np.split(occupied_frames, np.where(np.diff(occupied_frames) != 1)[0]+1)
        occupied_segments = [seg for seg in occupied_segments if (len(seg) >= min_segment_len)]
        for seg in occupied_segments:
            sub_df = self.refined_pose_df.loc[seg]
            indices = sub_df.index.values
            nfish_frame = sub_df.groupby('individuals', axis=1).any().sum(axis=1).values
            subsegments = np.split(indices, np.where((np.diff(nfish_frame) > 0) & (nfish_frame[1:] != 3))[0] + 1)
            subsegments = [seg for seg in subsegments if (len(seg) >= min_segment_len)]
            valid_segments.extend(subsegments)
        segment_ids = pd.Series(-1, index=self.refined_pose_df.index)
        curr_id = 0
        for seg in valid_segments:
            segment_ids.loc[seg] = curr_id
            curr_id += 1
        self.refined_pose_df['segment_id'] = segment_ids

    def shift_to_roi_reference_frame(self):
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi_x, roi_y, _ = estimate_roi(frame)
        cap.release()
        self.refined_pose_df.loc[:, idx[:, :, 'x']] = self.refined_pose_df.loc[:, idx[:, :, 'x']] - roi_x
        self.refined_pose_df.loc[:, idx[:, :, 'y']] = self.refined_pose_df.loc[:, idx[:, :, 'y']] - roi_y

    def format_for_saving(self, only_valid_segments=True):
        reformatted_pose_df = self.refined_pose_df.copy()
        if only_valid_segments:
            reformatted_pose_df = reformatted_pose_df[reformatted_pose_df.segment_id >= 0]
        reformatted_pose_df.loc[:, 'video_name'] = self.video_path.name
        reformatted_pose_df.loc[:, 'frame'] = reformatted_pose_df.index
        reformatted_pose_df.set_index(['video_name', 'segment_id', 'frame'], inplace=True)
        return reformatted_pose_df

    def _load_pose_data(self):
        pose_df = pd.read_hdf(self.h5_path)
        scorer = pose_df.columns.get_level_values(0)[0]
        pose_df = pose_df.loc[:, scorer]
        pose_df = pose_df.loc[:, idx[:, :, ('x', 'y')]]
        pose_df.columns = pose_df.columns.remove_unused_levels()
        pose_df = pose_df.sort_index(axis=1)
        return pose_df


def process_all(video_paths, collated_data_file, roi_ref_frame=False):
    collated_df = []
    for vp in video_paths:
        print(f'processing {vp.name}')
        bp = BamsPrepper(vp)
        bp.refine_pose_data()
        bp.segment_data()
        if roi_ref_frame:
            bp.shift_to_roi_reference_frame()
        collated_df.append(bp.format_for_saving())
    print('collating and saving data')
    collated_df = pd.concat(collated_df)
    collated_df.to_excel(str(collated_data_file))


parent_dir = Path('/home/tlancaster/DLC/demasoni_singlenuc/BAMS_set1/')
vid_paths = list(parent_dir.glob('**/*.mp4'))
pattern = r'((CTRL)|(BHVE))_group\d_\d{6}-\d{6}.mp4'
video_paths = [p for p in vid_paths if re.fullmatch(pattern, p.name)]

collated_data_file = Path('/home/tlancaster/DLC/demasoni_singlenuc/BAMS_set1/collated_pose_data.xlsx')
process_all(video_paths, collated_data_file)
collated_data_file = Path('/home/tlancaster/DLC/demasoni_singlenuc/BAMS_set1/collated_pose_data_shifted.xlsx')
process_all(video_paths, collated_data_file, roi_ref_frame=True)
