from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from demba.utils.roi_utils import estimate_roi
from itertools import combinations
idx = pd.IndexSlice


class Featurizer:

    def __init__(self, video_path):
        self.video_path = video_path
        self.h5_path = str(next(Path(self.video_path).parent.glob(Path(self.video_path).stem + '*_filtered.h5')))
        self.feature_matrix = None

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
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        if not ret:
            print(f'could not extract a reference frame from {Path(self.video_path).name}')
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi_vis_path = str(Path(self.video_path).parent / (Path(self.video_path).stem + '_roi.png'))
        self.roi_x, self.roi_y, self.roi_r = estimate_roi(frame, output_path=roi_vis_path)
        self.frame_height, self.frame_width = frame.shape[:-1]
        cap.release()

    def generate_feature_matrix(self):
        """generate the full feature matrix. each feature matrix corresponds to a single individual in a single frame,
        but includes information about that individuals spatiotemporal relationship to other animals and itself

        -normalize features to theoretical max? observed max?
        """
        feature_matrix = []
        index = []
        for frame_number in range(len(self.pose_df)):
            feature_matrix_segment = []
            for individual in self.individuals:
                if self.pose_df.loc[frame_number, individual].notnull().any():
                    individual_features = self.extract_individual_features(frame_number, individual)
                    social_features = self.extract_individualized_social_features(frame_number, individual)
                    feature_matrix_segment.append(individual_features + social_features)
                    index.append(frame_number)
                bulk_social_features = self.extract_bulk_social_features(feature_matrix_segment)
                [row.extend(bulk_social_features) for row in feature_matrix_segment]
            feature_matrix.extend(feature_matrix_segment)
        columns = []
        df = pd.DataFrame(data=feature_matrix, index=index, columns=columns)
        self.extract_temporal_features(df)
        return df

    def extract_individual_features(self, frame_numer, individual):
        """extract features that describe the location and pose of the focal individual in a given frame.

        -body curvature?
        -heading?
        -velocity components perpendicular and parallel to heading?
        -rolling windows?
        """
        centroid_x, centroid_y = self.estimate_centroid(frame_numer, individual)
        dist_from_roi_center = self.calc_dist([centroid_x, centroid_y], [self.roi_x, self.roi_y])
        is_inside_roi = 1 if dist_from_roi_center < self.roi_r else 0
        return [centroid_x, centroid_y, dist_from_roi_center, is_inside_roi]
    def extract_individualized_social_features(self, frame_number, individual):
        """extract features that describe the spatial relationship of the focal individual to other individuals in the
        frame, and which can vary between individuals in the frame
        -interaction strength as inverse square of distance?
        -other features scaled by interaction strength?
        -cache calculations that repeat for each individual in a frame? Or calculate social features after individual
         calculations complete?
        """
        return []

    def extract_bulk_social_features(self, feature_matrix_segment):
        """extract features that describe the overall social environment for a given frame -- i.e., features that are
        the same regardless of which individual is focal.

        -n_fish_frame
        -n_fish_pipe
        """
        return []

    def extract_temporal_features(self, feature_matrix):
        """extract features that estimate the change in other features over time
        -individual velocity
        -individual acceleration
        @return:
        """

    def estimate_centroid(self, frame_number, individual):
        """

        @param frame_number:
        @param individual:
        @return:
        """
        return self.pose_df.loc[frame_number, individual].groupby('coords').mean().tolist()

    def calc_dist(self, xy1, xy2):
        """
        Euclidean distance between two points. xy1 and xy2 can be any type that support indexing
        @param xy1: first point
        @param xy2: second point
        @return: distance
        """
        return np.hypot(xy1[0] - xy2[0], xy1[1] - xy2[1])







