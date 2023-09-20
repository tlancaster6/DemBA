from pathlib import Path
import pandas as pd
from demba.utils.roi_utils import estimate_roi
import numpy as np
import cv2
import warnings
from tables import NaturalNameWarning
from itertools import permutations
from dbscan1d.core import DBSCAN1D
from tqdm import tqdm
import re
idx = pd.IndexSlice
warnings.filterwarnings('ignore', category=NaturalNameWarning)


class DataManager:

    def __init__(self,
                 collated_data_file,
                 segmentation_file=None,
                 feature_file=None,
                 quivering_annotation_file=None,
                 roi_location_file=None):
        """
        simplifies reading and cross-referencing the pose, segmentation, and feature data files
        :param collated_data_file: path to pose hdf file produced collate_data function
        :param segmentation_file: path to segmentation hdf file produced by Segmenter.segment_all_videos method
        :param feature_file: path to feature file produced by Featurizer.featurize_all_videos method
        :param roi_location_file: path to roi location file produced by locate_rois function
        """
        self.collated_data_file = collated_data_file
        self.segmentation_file = segmentation_file
        self.feature_file = feature_file
        self.quivering_annotation_file = quivering_annotation_file
        self.roi_location_file = roi_location_file

    def get_keys(self):
        """
        get the hdf store keys for self.collated_data_file
        :return: a list of keys
        """
        with pd.HDFStore(str(self.collated_data_file), 'r') as store:
            keys = store.keys()
        return keys

    def get_pose_data(self, key):
        """
        retrieve the pose data for the video associated with the given key
        :param key: hdf key for the data being accessed
        :return: dataframe of pose data
        """
        with pd.HDFStore(str(self.collated_data_file), 'r') as store:
            data = store.get(key)
        return data

    def get_segmentation_data(self, key):
        """
        retrieve the segmentation data for the video associated with the given key
        :param key: hdf key for the data being accessed
        :return: dataframe of segmentation data
        """
        with pd.HDFStore(str(self.segmentation_file), 'r') as store:
            data = store.get(key)
        return data

    def get_segmented_pose_data(self, key):
        """
        for the video associated with the given key, use the segmentation data to slice the pose data
        :param key: hdf key for the data being accessed
        :return: a list of dataframes, with each dataframe containing the pose data for a single individual during a
            single segment
        """
        pose_data = self.get_pose_data(key)
        segmentation_data = self.get_segmentation_data(key)
        segmented_pose_data = []
        for ind, start, stop in segmentation_data.itertuples(index=False):
            segmented_pose_data.append(pose_data.loc[start:stop, [ind]])
        return segmented_pose_data

    def get_feature_data(self, key):
        """
        retrieve the feature data for the video associated with the given key
        :param key: hdf key for the data being accessed
        :return: dataframe of feature data
        """
        with pd.HDFStore(str(self.feature_file), 'r') as store:
            data = store.get(key)
        return data

    def get_segmented_feature_data(self, key):
        """
        for the video associated with the given key, use the segmentation data to slice the pose data
        :param key: hdf key for the data being accessed
        :return: a list of dataframes, with each dataframe containing the feature data for a single individual over
            a single segment as well as the values of the global features during that segment
        """
        feature_data = self.get_feature_data(key)
        segmentation_data = self.get_segmentation_data(key)
        segmented_feature_data = []
        for ind, start, stop in segmentation_data.itertuples(index=False):
            segmented_feature_data.append(feature_data.loc[start:stop, [ind, 'global']])
        return segmented_feature_data

    def get_roi_location(self, key):
        """
        retrieve the roi location associated with the given key
        :param key: the key you would normally use to access one of the hdf stores (even though this method reads from
            a csv)
        :return: a pandas series containing the x and y coordinates (roi_x, roi_y) and radius (roi_r) of the roi, as
            well as the overall frame dimensions (frame_w and frame_h)
        """
        return pd.read_csv(str(self.roi_location_file), index_col='video_key').loc[key.strip('/')]

    def get_quivering_annotations(self, key):
        """
        retrieve manual quivering annotations for mapping onto the video associated with the given key

        the returned dataframe will be in frame units (rather than seconds). If the frame range can be inferred from the
        key, it will be used to align and clip the returned dataframe to match. Otherwise, the full unfishifted
        dataframe will be returned

        :param key: hdf key for the data being accessed
        :return: dataframe of quivering annotations for alignment with other features
        """
        trial_id = '_'.join(key.strip('/').split('_')[:2])
        df = pd.read_excel(self.quivering_annotation_file, sheet_name=trial_id, skiprows=1)
        df = df[['temporal_segment_start', 'temporal_segment_end']]
        df = (df * 30).apply(np.round)
        if re.fullmatch(r'((CTRL)|(BHVE))_group\d_\d{6}-\d{6}.mp4', key):
            start_frame, end_frame = [int(x) for x in key.strip('/').split('_')[-1].split('-')]
            df = df - start_frame
            df = df[df.temporal_segment_end >= 0]
            df = df[df.temporal_segment_start <= end_frame]
            df[df.temporal_segment_start < 0] = 0
            df[df.temporal_segment_end > end_frame] = end_frame
        return df

    def summarize_segment_lengths(self):
        segment_lengths = []
        for key in self.get_keys():
            seg_data = self.get_segmentation_data(key)
            lengths = list(seg_data.stop - seg_data.start + 1)
            segment_lengths.extend(lengths)
        print(f'total number of segments: {len(segment_lengths)}')
        print(f'mean segment length: {np.mean(segment_lengths)}')
        print(f'median segment length: {np.median(segment_lengths)}')
        print(f'max segment length: {np.max(segment_lengths)}')
        print(f'min segment length: {np.min(segment_lengths)}')


class Segmenter:

    def __init__(self, data_manager: DataManager):
        self.dm = data_manager

    def segment_one_video(self, key, min_segment_len=60):
        data = self.dm.get_pose_data(key)
        individuals = list(data.columns.levels[0])
        df = []
        for ind in individuals:
            view = data[ind].dropna(how='all')
            segments = np.split(view.index, np.where(np.diff(view.index) != 1)[0]+1)
            segments = [seg for seg in segments if (len(seg) >= min_segment_len)]
            starts = [seg.min() for seg in segments]
            stops = [seg.max() for seg in segments]
            ind_list = [ind] * len(starts)
            df.extend(list(zip(ind_list, starts, stops)))
        df = pd.DataFrame(df, columns=['individuals', 'start', 'stop'])
        return df

    def segment_all_videos(self):
        for key in self.dm.get_keys():
            df = self.segment_one_video(key)
            df.to_hdf(str(self.dm.segmentation_file), key, mode='a')


class Featurizer:

    def __init__(self, data_manager: DataManager):
        self.dm = data_manager

    def featurize_one_video(self, key):
        data = self.dm.get_pose_data(key)
        roi = self.dm.get_roi_location(key)
        quivering_ref = dm.get_quivering_annotations(key)
        feats = {}
        feats.update(self.calc_roi_occupancy_features(data, roi))
        feats.update(self.calc_mouthing_features(data))
        feats.update(self.map_quivering_events(data, quivering_ref))
        feats = pd.concat(list(feats.values()), axis=1, keys=list(feats.keys()), names=['feature', 'individual'])
        feats = feats.swaplevel(axis=1).sort_index(axis=1)
        feats.index.name = 'frame'
        return feats

    def featurize_all_videos(self):
        print('featurizing all videos')
        for key in tqdm(self.dm.get_keys()):
            df = self.featurize_one_video(key)
            df.to_hdf(str(self.dm.feature_file), key, mode='a')

    def calc_roi_occupancy_features(self, data, roi):
        occ_df = data.loc[:, idx[:, 'stripe1', :]].copy()
        occ_df.loc[:, idx[:, :, 'x']] -= roi.roi_x
        occ_df.loc[:, idx[:, :, 'y']] -= roi.roi_y
        occ_df = (occ_df ** 2).groupby('individuals', axis=1).sum(min_count=2) ** 0.5
        occ_df = occ_df < roi.roi_r
        return {'in_roi': occ_df, 'roi_occupancy': pd.DataFrame(occ_df.sum(axis=1), columns=['global'])}

    def calc_mouthing_features(self, data, dist_thresh=30, eps=15, min_samples=15):

        individuals = list(data.columns.levels[0])
        if len(individuals) == 1:
            mouther_df = pd.DataFrame(False, columns=[individuals[0]], index=data.index)
            mouthee_df = pd.DataFrame(False, columns=[individuals[0]], index=data.index)
            mouthing_df = pd.DataFrame(False, columns=['global'], index=data.index)
            return {'mouther': mouther_df, 'mouthee': mouthee_df, 'mouthing': mouthing_df}

        mouther_df = []
        mouthee_df = []
        for mouther, mouthee in list(permutations(individuals, 2)):
            dists = data.loc[:, idx[mouther, 'nose', :]].values - data.loc[:, idx[mouthee, 'stripe4', :]].values
            dists = pd.Series(np.hypot(dists[:, 0], dists[:, 1]), index=data.index)
            subthresh_frames = dists[dists < dist_thresh].index.values
            if len(subthresh_frames) == 0:
                mouther_df.append(pd.Series(False, name=mouther, index=data.index))
                mouthee_df.append(pd.Series(False, name=mouthee, index=data.index))
                continue
            labels = DBSCAN1D(eps, min_samples).fit_predict(subthresh_frames)
            labels = pd.Series(data=labels, index=subthresh_frames).reindex(data.index, fill_value=-1)
            for lid in labels.unique():
                if lid != -1:
                    start_idx = labels[labels == lid].index.min()
                    end_idx = labels[labels == lid].index.max()
                    labels.loc[start_idx:end_idx] = lid
            binary_labels = (labels != -1)
            mouther_df.append(pd.Series(binary_labels, name=mouther))
            mouthee_df.append(pd.Series(binary_labels, name=mouthee))
        mouther_df = pd.concat(mouther_df, axis=1).groupby(level=0, axis=1).any()
        mouthee_df = pd.concat(mouthee_df, axis=1).groupby(level=0, axis=1).any()
        mouthing_df = pd.DataFrame(mouther_df.any(axis=1), columns=['global'])
        return {'mouther': mouther_df, 'mouthee': mouthee_df, 'mouthing': mouthing_df}

    def map_quivering_events(self, data, quivering_reference):
        quivering = pd.DataFrame(False, index=data.index, columns=['global'])
        for event in quivering_reference.iterrows():
            start, end = int(event[1].temporal_segment_start), int(event[1].temporal_segment_end)
            quivering.loc[start:end] = True
        return {'quivering': quivering}


def collate_data(h5_paths, collated_data_file):
    for h5p in h5_paths:
        pose_df = pd.read_hdf(h5p)
        scorer = pose_df.columns.get_level_values(0)[0]
        pose_df = pose_df.loc[:, scorer]
        pose_df = pose_df.loc[:, idx[:, :, ('x', 'y')]]
        pose_df.columns = pose_df.columns.remove_unused_levels()
        pose_df = pose_df.sort_index(axis=1)
        pose_df.dropna(how='all', inplace=True)
        key = Path(str(h5p).split('DLC_dlcrnet')[0]).name
        pose_df.to_hdf(str(collated_data_file), key=key, mode='a')


def locate_rois(video_paths, roi_location_file):
    roi_locations = []
    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi_x, roi_y, roi_r = estimate_roi(frame)
        frame_h, frame_w = frame.shape[:-1]
        cap.release()
        roi_locations.append([vp.stem, roi_x, roi_y, roi_r, frame_w, frame_h])
    df = pd.DataFrame(roi_locations, columns=['video_key', 'roi_x', 'roi_y', 'roi_r', 'frame_w', 'frame_h'])
    df.to_csv(str(roi_location_file), index=False)


parent_dir = Path(r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\BAMS_set1")
h5_paths = list(parent_dir.glob('**/*filtered.h5'))
video_paths = [Path(str(h5p).replace('DLC_dlcrnetms5_demasoni_singlenucMay23shuffle1_50000_el_filtered.h5', '.mp4')) for h5p in h5_paths]
collated_data_file = Path(r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\BAMS_set1\collated_pose_data.h5")
segmentation_file = Path(r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\BAMS_set1\segmentation_data.h5")
feature_file = Path(r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\BAMS_set1\features.h5")
roi_location_file = Path(r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\BAMS_set1\roi_locations.csv")
quivering_annotation_file = r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\quivering_annotations\Mbuna_behavior_annotations.xlsx"

# collate_data(h5_paths, collated_data_file)
# locate_rois(video_paths, roi_location_file)
dm = DataManager(collated_data_file, segmentation_file, feature_file, quivering_annotation_file, roi_location_file)
# segmenter = Segmenter(dm)
# segmenter.segment_all_videos()

# featurizer = Featurizer(dm)
# featurizer.featurize_all_videos()

# key = 'BHVE_group3_142200-143999'
# df = featurizer.featurize_one_video(key)

