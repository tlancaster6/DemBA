from pathlib import Path
import pandas as pd
from demba.utils.roi_utils import estimate_roi
import numpy as np
import cv2
import re
import warnings
from tables import NaturalNameWarning

idx = pd.IndexSlice
warnings.filterwarnings('ignore', category=NaturalNameWarning)


class DataManager:

    def __init__(self, collated_data_file,
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
        :return: a list of dataframes, with each dataframe containing the pose data for a single individual over
            a single segment
        """
        pose_data = self.get_pose_data(key)
        segmentation_data = self.get_segmentation_data(key)
        segmented_pose_data = []
        for ind, start, stop in segmentation_data.itertuples(index=False):
            segmented_pose_data.append(pose_data.loc[start:stop, ind])
        return segmented_pose_data

    def get_roi_location(self, key):
        return pd.read_csv(str(self.roi_location_file))

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
        individuals = list(data.columns.levels[0])
        feat_columns = ['frame', 'individual', 'feat_name', 'value']
        feats = []
        for ind in individuals:


    def featurize_all_videos(self):
        for key in self.dm.get_keys():
            df = self.featurize_one_video(key)
            df.to_hdf(str(self.dm.feature_file), key, mode='a')



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
        roi_locations.append([vp.name, roi_x, roi_y, roi_r, frame_w, frame_h])
    df = pd.DataFrame(roi_locations, columns=['video_name', 'roi_x', 'roi_y', 'roi_r', 'frame_w', 'frame_h'])
    df.to_csv(str(roi_location_file))


# parent_dir = Path(r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\BAMS_set1")
# h5_paths = list(parent_dir.glob('**/*filtered.h5'))
# collated_data_file = Path(r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\BAMS_set1\collated_pose_data.h5")
# segmentation_file = Path(r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\BAMS_set1\segmentation_data.h5")
# collate_data(h5_paths, collated_data_file)
# segmenter = Segmenter(collated_data_file, segmentation_file)
# segmenter.segment_all_videos()
# quivering_annotation_path = r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\quivering_annotations\Mbuna_behavior_annotations.xlsx"