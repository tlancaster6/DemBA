import pandas as pd
import numpy as np
import warnings
from tables import NaturalNameWarning
import re
from pathlib import Path
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
        simplifies reading and cross-referencing the pose, segmentation, and feature data files. Only the
        collated_data_file is strictly required, but most features require that one-or-more of the remaining arguments
        are supplied.

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



### sample usage ###
# define paths
parent_dir = Path('~/Data')
collated_data_file = parent_dir / 'collated_pose_data.h5'
segmentation_file = parent_dir / 'segmentation_data.h5'
feature_file = parent_dir / 'features.h5'
quivering_annotation_file = parent_dir / 'quivering_annotations.xlsx'
roi_location_file = parent_dir / 'roi_locations.csv'
# initiate data manager
dm = DataManager(collated_data_file, segmentation_file, feature_file, quivering_annotation_file, roi_location_file)
# use the data manager -- for example, to retrieve data for a particular video clip
key = dm.get_keys()[4]
print(f'collecting data for {key}')
roi = dm.get_roi_location(key)
print(f'roi centered at ({roi.roi_x}, {roi.roi_y}) with radius={roi.roi_r}')
print(f'summary of segment lengths in {key}:')
dm.summarize_segment_lengths()
segmented_features = dm.get_segmented_feature_data(key)
print(f'segmented feature data for the {len(segmented_features)} segments in {key}')
for i, seg in enumerate(segmented_features):
    print(f'\n\nfeatures for segment {i}:')
    print(seg)



