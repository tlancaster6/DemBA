from pathlib import Path
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from demba.utils.roi_utils import estimate_roi
from shutil import rmtree
idx = pd.IndexSlice



class Featurizer:

    def __init__(self, video_path):
        self.video_path = video_path
        self.h5_path = str(next(Path(self.video_path).parent.glob(Path(self.video_path).stem + '*_filtered.h5')))
        self.output_dir = str(Path(self.video_path).parent / 'output')
        self._load_pose_data()
        self._estimate_roi()
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

    def generate_feature_matrix(self):
        """generate the full feature matrix. each feature matrix corresponds to a single individual in a single frame,
        but includes information about that individuals spatiotemporal relationship to other animals and itself

        -normalize features to theoretical max? observed max?
        """
        individual_feature_matrices = []
        for individual in self.individuals:
            individual_feature_matrices.append(self.extract_individual_features(individual))
        # add additional feature calculation functions here
        self.feature_matrix = pd.concat(individual_feature_matrices, axis=0)


    def generate_feature_matrix_old(self):
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

    def extract_individual_features(self, individual):
        """extract features that describe the location and pose of the focal individual in each frame, and the dynamics
        of these values across frames.

        -body curvature?
        -heading?
        -velocity components perpendicular and parallel to heading?
        -rolling windows?
        """

        centroid_x = self.calc_centroid(individual, 'x')
        centroid_y = self.calc_centroid(individual, 'y')
        dist_from_roi_center = self.calc_dist_from_roi_center(centroid_x, centroid_y)
        is_inside_roi = self.calc_is_inside_roi(dist_from_roi_center)
        speed, tangential_acceleration, centripetal_acceleration = self.calc_instantaneous_centroid_kinetics(centroid_x, centroid_y)
        id_number = pd.Series(int(individual[-1]), index=centroid_x.index)
        id_number.name = 'id_number'
        df = pd.concat([id_number, centroid_x, centroid_y, dist_from_roi_center, is_inside_roi, speed,
                        tangential_acceleration, centripetal_acceleration], axis=1)
        return df

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

    def calc_centroid(self, individual, coord):
        """
        estimate one coordinate of the centroid (i.e., either centroid_x or centroid_y) for a given individual based
        on the average position of all visible body-parts for that individual in each frame. Frames with no available
        data will not appear in the returned Series object.
        @param individual: Must match a key from the "individuals" level of self.pose_df
        @param coord: either 'x' or 'y'
        @return: a series containing an estimate for the desired centroid coordinate (x or y) in each frame
        """
        centroid_component = self.pose_df.loc[:, idx[individual, :, coord]].mean(axis=1).dropna()
        centroid_component.name = f'centroid_{coord}'
        return centroid_component

    def calc_dist_from_roi_center(self, centroid_x, centroid_y):
        dist_from_roi_center = np.hypot(centroid_x-self.roi_x, centroid_y-self.roi_y)
        dist_from_roi_center.name = 'dist_from_roi_center'
        return dist_from_roi_center

    def calc_is_inside_roi(self, dist_from_roi_center):
        is_inside_roi = (dist_from_roi_center < self.roi_r).astype(int)
        is_inside_roi.name = 'is_inside_roi'
        return is_inside_roi

    def calc_
    def calc_instantaneous_centroid_kinetics(self, centroid_x, centroid_y):
        # should probably improve this module with a kalman filter or similar
        dz = (centroid_x + 1j * centroid_y).diff()
        ddz = dz.diff()

        speed = np.abs(dz)
        speed.name = 'centroid_speed'

        # should the order of dz and ddz be flipped here?
        theta = np.angle(ddz) - np.angle(dz)
        acc = np.abs(ddz)
        tangential_acceleration = acc * np.cos(theta)
        tangential_acceleration.name = 'centroid_tangential_acceleration'
        centripetal_acceleration = acc * np.sin(theta)
        centripetal_acceleration.name = 'centroid_centripetal_acceleration'

        return [speed, tangential_acceleration, centripetal_acceleration]

    def visualize_features(self, n_frames=900):
        cap = cv2.VideoCapture(self.video_path)
        tmp_dir = Path(self.output_dir) / 'tmp'
        tmp_dir.mkdir(exist_ok=True)
        count = 0
        fig, ax = plt.subplots()

        while count <= n_frames:
            ax.clear()
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                break
            count += 1
            ax.imshow(frame)
            features = self.feature_matrix.loc[count]
            for row in features.iterrows():
                centroid = [row[1].centroid_x, row[1].centroid_y]
                ax.plot(*centroid, 'ro')
                vector_origins = [[centroid[0]]*3, [centroid[1]]*3]

                ax.quiver(vector_origins, )
            fig.tight_layout()
            fig.savefig(str(tmp_dir / f'{count:05}.png'))

        plt.close(fig)
        cap.release()
        out_path = str(Path(self.output_dir) / 'feature_visualization.mp4')
        ImageSequenceClip(str(tmp_dir), 30).write_videofile(out_path, fps=30, audio=False)
        rmtree(tmp_dir)




vid = r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\testclip\testclip.mp4"
feat = Featurizer(vid)
feat.generate_feature_matrix()
feat.visualize_features(n_frames=30)

