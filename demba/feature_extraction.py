import os.path
from pathlib import Path
import pandas as pd
import cv2
from demba.utils.roi import estimate_roi
from demba.utils.dlc import load_poses
from demba.utils.metrics import phi_coefficient, jaccard_index
from demba.config import ROI_RADIUS_MM, VIDEO_FPS
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
from dbscan1d.core import DBSCAN1D
from datetime import timedelta

idx = pd.IndexSlice
from matplotlib.animation import FuncAnimation

class FeatureExtractor:
    """
    Extracts behavioral features from zebrafish courtship videos with DeepLabCut pose estimation.

    Analyzes videos to detect and quantify spawning-related behaviors including mouthing events
    (nose-to-genital proximity), double occupancy of breeding pipe, and spawning bouts. Integrates
    optional manual quivering annotations with automated pose-based detections.

    Frame-level features track instantaneous behaviors across all frames. Clip-level features
    aggregate statistics over the entire video. Supports temporal windowing to analyze specific
    portions (e.g., last N minutes).

    Attributes:
        video_path (str): Path to input video file
        pose_df (pd.DataFrame): Multi-index DataFrame with shape (n_frames, n_individuals*n_bodyparts*3)
                                where columns are (individual, bodypart, coord) with coord in {x, y, likelihood}
        individuals (list): Names of tracked individuals (e.g., ['individual1', 'individual2'])
        bodyparts (list): Names of tracked body parts (e.g., ['nose', 'stripe1', ..., 'stripe4'])
        roi_x, roi_y, roi_r (float): Breeding pipe ROI center coordinates and radius in pixels
        mouthing_dist_pixels (float): Distance threshold for mouthing detection in pixels
        framefeatures_df (pd.DataFrame): Frame-by-frame behavioral features
        clipfeatures_df (pd.DataFrame): Aggregated clip-level statistics
    """

    def __init__(self, video_path, quivering_annotation_path=None, pose_h5_path=None, mouthing_dist_mm=10, min_likelihood=0.5, n_minutes=None):
        """
        Initialize FeatureExtractor with video, pose data, and processing parameters.

        Args:
            video_path (str/Path): Path to .mp4 video file
            quivering_annotation_path (str/Path, optional): Path to Excel file with manual quivering annotations
            pose_h5_path (str/Path, optional): Path to DeepLabCut pose .h5 file
            mouthing_dist_mm (float): Maximum nose-to-genital distance in mm to count as mouthing event
            min_likelihood (float): Minimum keypoint confidence threshold (0-1) for pose filtering
            n_minutes (int, optional): If provided, only analyze the last n_minutes of video
        """
        self.video_path = str(video_path)
        self.n_minutes = n_minutes # if not None, only analyze the last n_minutes of the video
        self.quivering_annotation_path = None if quivering_annotation_path is None else str(quivering_annotation_path)
        self.file_stem = Path(video_path).stem
        self.mouthing_dist_mm = mouthing_dist_mm

        # Create parameter suffix for unique file naming
        self.param_suffix = f"_mdist{mouthing_dist_mm}mm_likelihood{min_likelihood}"
        if n_minutes is not None:
            self.param_suffix += f"_last{n_minutes}min"

        self.framefeatures_path = str(video_path).replace('.mp4', f'{self.param_suffix}_framefeatures.csv')
        self.clipfeatures_path = str(video_path).replace('.mp4', f'{self.param_suffix}_clipfeatures.csv')
        self.h5_path = str(pose_h5_path)
        self.pose_df, self.individuals, self.bodyparts = load_poses(self.h5_path, min_likelihood=min_likelihood)
        self.quivering_annotation_df = self._load_quivering_annotations()
        self.roi_x, self.roi_y, self.roi_r, self.frame_height, self.frame_width = self._estimate_roi()
        self.mouthing_dist_pixels = self._calc_mouthing_dist_pixels()

    def load_feature_csvs(self):
        """
        Load previously computed frame-level and clip-level feature CSVs from disk.

        Populates self.framefeatures_df and self.clipfeatures_df attributes. Use this to reload
        features without re-computing, enabling downstream analysis without full re-extraction.
        """
        self.framefeatures_df = pd.read_csv(self.framefeatures_path, index_col=0, low_memory=False)
        self.clipfeatures_df = pd.read_csv(self.clipfeatures_path, index_col=0, low_memory=False)

    def extract_all_features(self):
        """
        Extract all frame-level and clip-level behavioral features from video.

        Frame-level features (computed per frame):
            - nfish_frame: Number of individuals with any detected keypoints
            - nfish_pipe: Number of individuals with stripe1 inside breeding pipe ROI
            - min_dist_nose_to_stripe4: Minimum distance across all nose-stripe4 pairs (pixels)
            - mouthing_event_id: Event ID for mouthing bouts (-1 if not mouthing)
            - spawning_event_id: Event ID for spawning bouts (-1 if not spawning)
            - double_occupancy_event_id: Event ID for double occupancy bouts (-1 if single/zero)
            - male_lead_quiver, male_circle_quiver, female_circle_quiver: Event IDs from annotations

        Clip-level features (aggregated over video):
            - n_mouthing_events, n_spawning_events, n_double_occupancy_events: Event counts
            - mouthing_event_fraction, spawning_event_fraction, double_occupancy_event_fraction: Time fractions
            - raw_*_occupancy_fraction: Fraction of frames with 0/1/2/3 fish in pipe
            - roi_x, roi_y, roi_r: Pipe location and size
            - n_*_quivers: Count of quivering events by type
            - *_quivering_fraction: Fraction of time spent quivering
            - mouthing_quivering_phi, mouthing_quivering_jaccard: Co-occurrence metrics

        Saves framefeatures_df to *_framefeatures.csv and clipfeatures_df to *_clipfeatures.csv.
        """
        # extract frame-level features
        framefeatures_df = []
        if self.pose_df is not None:
            framefeatures_df.append(self._calc_nfish_frame())
            framefeatures_df.append(self._calc_nfish_pipe())
            framefeatures_df.append(self._calc_interaction_distances())
            framefeatures_df.append(self._detect_mouthing_events(dists=framefeatures_df[-1]))
            framefeatures_df.append(self._detect_spawning_events(mouthing_event_ids=framefeatures_df[-1]))
            framefeatures_df.append(self._detect_double_occupancy_events())
        if self.quivering_annotation_df is not None:
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
        if self.quivering_annotation_df is not None:
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
        if (self.quivering_annotation_df is not None) and (self.pose_df is not None):
            clipfeatures_series['mouthing_quivering_phi'] = self._calc_mouthing_quivering_phi()
            clipfeatures_series['mouthing_quivering_jaccard'] = self._calc_mouthing_quivering_jaccard()
        self.clipfeatures_df = pd.DataFrame(clipfeatures_series, columns=[self.file_stem]).T
        self.clipfeatures_df.to_csv(self.clipfeatures_path)

    def _load_quivering_annotations(self):
        """
        Load manual quivering annotations from Excel file.

        Searches for sheet matching video stem (with or without 'cropped' suffix). Returns DataFrame
        with columns: temporal_segment_start, temporal_segment_end, metadata (quivering type).

        Returns:
            pd.DataFrame or None: Annotation data if found, None otherwise
        """
        if self.quivering_annotation_path is None:
            return None
        sheet_names = pd.ExcelFile(self.quivering_annotation_path).sheet_names
        if self.file_stem in sheet_names:
            return pd.read_excel(self.quivering_annotation_path, sheet_name=self.file_stem, skiprows=1)
        elif self.file_stem.rstrip('cropped') in sheet_names:
            return pd.read_excel(self.quivering_annotation_path, sheet_name=self.file_stem.rstrip('cropped'), skiprows=1)
        else:
            return None

    def _estimate_roi(self):
        """
        Automatically detect breeding pipe ROI from middle frame of video.

        Uses Hough circle detection to find circular pipe structure. Saves visualization
        of detected ROI to *_roi.png file.

        Returns:
            tuple: (roi_x, roi_y, roi_r, frame_height, frame_width) in pixels
        """
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        roi_vis_path = str(self.video_path).replace('.mp4', '_roi.png')
        roi_x, roi_y, roi_r, frame_height, frame_width = estimate_roi(frame, output_path=roi_vis_path)
        return roi_x, roi_y, roi_r, frame_height, frame_width

    def _calc_mouthing_dist_pixels(self):
        """
        Convert mouthing distance threshold from mm to pixels using ROI as reference.

        Uses breeding pipe radius as known reference (79.375 mm) to calibrate pixel-to-mm conversion.

        Returns:
            float: Mouthing distance threshold in pixels
        """
        conversion_factor = (self.roi_r / ROI_RADIUS_MM)
        return self.mouthing_dist_mm * conversion_factor

    def _calc_interaction_distances(self):
        """
        Calculate minimum nose-to-genital (stripe4) distance across all individual pairs.

        Mouthing behavior involves male approaching female genital region. Computes Euclidean
        distance between each individual's nose and every other individual's stripe4 (genital marker).
        Takes minimum across all directional pairs.

        Returns:
            pd.Series: Per-frame minimum distance in pixels, name='min_dist_nose_to_stripe4'
        """
        candidate_dists = []
        for id1, id2 in list(permutations(self.individuals, 2)):
            # Use explicit x,y column selection to avoid column ordering issues
            nose_coords = self.pose_df.loc[:, (id1, 'nose', ['x', 'y'])].values
            stripe4_coords = self.pose_df.loc[:, (id2, 'stripe4', ['x', 'y'])].values
            dists = nose_coords - stripe4_coords
            candidate_dists.append(np.hypot(dists[:, 0], dists[:, 1]))
        if not candidate_dists:
            return pd.Series(data=-1, index=self.pose_df.index, name='min_dist_nose_to_stripe4')
        dists = pd.Series(np.nanmin(np.vstack(candidate_dists), axis=0), name='min_dist_nose_to_stripe4')
        return dists

    def _detect_mouthing_events(self, dists=None, eps=5, min_samples=10):
        """
        Detect mouthing events using temporal clustering of sub-threshold nose-genital distances.

        Uses DBSCAN1D to group nearby frames where distance < mouthing_dist_pixels. Fills gaps
        within events to handle brief tracking failures. Events must span at least min_samples frames.

        Args:
            dists (pd.Series, optional): Pre-computed interaction distances. If None, computes them.
            eps (int): Maximum gap in frames to consider same event (DBSCAN epsilon)
            min_samples (int): Minimum frames required to qualify as event

        Returns:
            pd.Series: Per-frame event ID (>=0 during events, -1 otherwise), name='mouthing_event_id'
        """
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
        """
        Detect sustained double occupancy of breeding pipe using temporal clustering.

        Both fish entering pipe together is prerequisite for spawning. Uses DBSCAN1D to identify
        continuous bouts where exactly 2 fish are inside pipe ROI. Fills gaps and requires minimum duration.

        Args:
            eps (int): Maximum gap in frames to consider same event (default 30 = 1 second at 30fps)
            min_samples (int): Minimum frames required to qualify as event (default 30 = 1 second)

        Returns:
            pd.Series: Per-frame event ID (>=0 during events, -1 otherwise), name='double_occupancy_event_id'
        """
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

    def _detect_spawning_events(self, mouthing_event_ids=None, eps=300, min_samples=6):
        """
        Detect spawning events as clusters of mouthing events in temporal proximity.

        Spawning consists of multiple mouthing bouts in quick succession. Uses DBSCAN1D on start/end
        frames of mouthing events to identify temporal clusters indicating spawning bouts.

        Args:
            mouthing_event_ids (pd.Series, optional): Pre-computed mouthing events. If None, computes them.
            eps (int): Maximum gap in frames between mouthing events in same spawning bout (default 300 = 10s)
            min_samples (int): Minimum mouthing event endpoints required to qualify as spawning

        Returns:
            pd.Series: Per-frame event ID (>=0 during events, -1 otherwise), name='spawning_event_id'
        """
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
        """
        Count number of individuals detected in frame based on any visible keypoint.

        Individual is considered present if ANY of their bodyparts has valid (non-NaN) pose estimate
        after likelihood filtering. More permissive than nfish_pipe which requires specific location.

        Returns:
            pd.Series: Per-frame count of detected individuals, name='nfish_frame'
        """
        nfish_frame = self.pose_df.groupby('individuals', axis=1).any().sum(axis=1)
        nfish_frame.name = 'nfish_frame'
        return nfish_frame

    def _calc_nfish_pipe(self):
        """
        Count number of individuals inside breeding pipe ROI based on stripe1 position.

        Uses stripe1 (mid-body keypoint) to determine if fish is inside circular pipe ROI.
        Computes Euclidean distance from stripe1 to ROI center and checks if <= roi_r.
        More conservative than nfish_frame, requires fish to be in specific breeding location.

        Returns:
            pd.Series: Per-frame count of individuals inside pipe, name='nfish_pipe'
        """
        # Get only x,y coordinates for stripe1 to avoid including likelihood in distance calculation
        tmp_df = self.pose_df.loc[:, idx[:, 'stripe1', ['x', 'y']]].copy()
        tmp_df.loc[:, idx[:, :, 'x']] -= self.roi_x
        tmp_df.loc[:, idx[:, :, 'y']] -= self.roi_y
        tmp_df = (tmp_df ** 2).groupby('individuals', axis=1).sum(min_count=2) ** 0.5
        nfish_pipe = (tmp_df <= self.roi_r).sum(axis=1)
        nfish_pipe.name = 'nfish_pipe'
        return nfish_pipe

    def _calc_n_mouthing_events(self):
        """
        Count total number of distinct mouthing events in video.

        Returns:
            int: Number of unique mouthing event IDs
        """
        event_ids = self.framefeatures_df[self.framefeatures_df.mouthing_event_id >= 0].mouthing_event_id
        n_events = len(event_ids.unique())
        return n_events

    def _calc_n_double_occupancy_events(self):
        """
        Count total number of distinct double occupancy events in video.

        Returns:
            int: Number of unique double occupancy event IDs
        """
        event_ids = self.framefeatures_df[self.framefeatures_df.double_occupancy_event_id >= 0].double_occupancy_event_id
        n_events = len(event_ids.unique())
        return n_events

    def _calc_n_spawning_events(self):
        """
        Count total number of distinct spawning events in video.

        Returns:
            int: Number of unique spawning event IDs
        """
        event_ids = self.framefeatures_df[self.framefeatures_df.spawning_event_id >= 0].spawning_event_id
        n_events = len(event_ids.unique())
        return n_events

    def _calc_spawning_event_fraction(self):
        """
        Calculate fraction of video time spent in spawning events.

        Returns:
            float: Proportion of frames with spawning_event_id >= 0 (range 0-1)
        """
        n_spawning_frames = len(self.framefeatures_df[self.framefeatures_df.spawning_event_id >= 0])
        spawning_fraction = n_spawning_frames / len(self.framefeatures_df)
        return spawning_fraction

    def _calc_double_occupancy_event_fraction(self):
        """
        Calculate fraction of video time spent in double occupancy events.

        Returns:
            float: Proportion of frames with double_occupancy_event_id >= 0 (range 0-1)
        """
        n_double_occupancy_frames = len(self.framefeatures_df[self.framefeatures_df.double_occupancy_event_id >= 0])
        double_occupancy_fraction = n_double_occupancy_frames / len(self.framefeatures_df)
        return double_occupancy_fraction

    def _calc_mouthing_event_fraction(self):
        """
        Calculate fraction of video time spent in mouthing events.

        Returns:
            float: Proportion of frames with mouthing_event_id >= 0 (range 0-1)
        """
        n_mouthing_frames = len(self.framefeatures_df[self.framefeatures_df.mouthing_event_id >= 0])
        mouthing_fraction = n_mouthing_frames / len(self.framefeatures_df)
        return mouthing_fraction

    def _calc_roi_occupancy_fractions(self):
        """
        Calculate fraction of time with 0, 1, 2, or 3 fish inside breeding pipe.

        Returns:
            pd.Series: Fractions for each occupancy level with keys:
                'raw_zero_occupancy_fraction', 'raw_single_occupancy_fraction',
                'raw_double_occupancy_fraction', 'raw_triple_occupancy_fraction'
        """
        value_counts = self.framefeatures_df.nfish_pipe.value_counts(normalize=True)
        value_counts = value_counts.reindex([0, 1, 2, 3], fill_value=0.0)
        value_counts = value_counts.rename(index={0: 'raw_zero_occupancy_fraction',
                                                  1: 'raw_single_occupancy_fraction',
                                                  2: 'raw_double_occupancy_fraction',
                                                  3: 'raw_triple_occupancy_fraction'})
        return value_counts

    def _calc_quivering_fraction(self, quivering):
        """
        Calculate fraction of video time spent in specified quivering behavior.

        Args:
            quivering (str): Column name ('male_lead_quiver', 'male_circle_quiver', or 'female_circle_quiver')

        Returns:
            float: Proportion of frames with quivering event ID >= 0
        """
        n_quivering_frames = len(self.framefeatures_df[self.framefeatures_df[quivering] >= 0])
        quivering_fraction = n_quivering_frames / len(self.framefeatures_df)
        return quivering_fraction

    def _calc_n_quivers(self, quivering):
        """
        Count number of distinct quivering events of specified type.

        Args:
            quivering (str): Column name ('male_lead_quiver', 'male_circle_quiver', or 'female_circle_quiver')

        Returns:
            int: Number of unique quivering event IDs
        """
        event_ids = self.framefeatures_df[self.framefeatures_df[quivering] >= 0][quivering]
        n_events = len(event_ids.unique())
        return n_events

    def _calc_mouthing_quivering_phi(self):
        """
        Calculate Phi coefficient (correlation) between mouthing and quivering behaviors.

        Measures association between automated mouthing detection and manual quivering annotations.
        Phi coefficient ranges from -1 (perfect negative correlation) to +1 (perfect positive correlation).

        Returns:
            float: Phi coefficient between mouthing and any quivering (male or female circle)
        """
        mouthing_events = self.framefeatures_df.mouthing_event_id >= 0
        male_circle_quivers = self.framefeatures_df.male_circle_quiver >= 0
        female_circle_quivers = self.framefeatures_df.female_circle_quiver >= 0
        all_quivers = male_circle_quivers | female_circle_quivers
        phi = phi_coefficient(mouthing_events, all_quivers)
        return phi

    def _calc_mouthing_quivering_jaccard(self):
        """
        Calculate Jaccard index (overlap) between mouthing and quivering behaviors.

        Measures temporal overlap as: (intersection) / (union). Jaccard index ranges from
        0 (no overlap) to 1 (perfect overlap) between mouthing and quivering frames.

        Returns:
            float: Jaccard index between mouthing and any quivering (male or female circle)
        """
        mouthing_events = self.framefeatures_df.mouthing_event_id >= 0
        male_circle_quivers = self.framefeatures_df.male_circle_quiver >= 0
        female_circle_quivers = self.framefeatures_df.female_circle_quiver >= 0
        all_quivers = male_circle_quivers | female_circle_quivers
        jaccard = jaccard_index(mouthing_events, all_quivers)
        return jaccard

    def _clean_metadata_string(self, meta_str):
        """
        Extract quivering type from annotation metadata string.

        Args:
            meta_str (str): Raw metadata string containing TEMPORAL-SEGMENTS field

        Returns:
            str: Quivering type ('male-lead-quiver', 'male-circle-quiver', or 'female-circle-quiver')
        """
        meta_dict = eval(meta_str)
        return meta_dict["TEMPORAL-SEGMENTS"]
        
    def _map_quivering_annotations(self):
        """
        Convert manual quivering annotations from temporal segments to frame-level event IDs.

        Reads Excel annotations with start/end times (in seconds) and metadata labels, converts to
        frame indices (30 fps), and creates binary event series. Consecutive frames of same behavior
        get unique sequential event IDs. Handles annotations extending beyond video bounds.

        Returns:
            tuple: (male_lead_quiver, male_circle_quiver, female_circle_quiver) as pd.Series with
                   event IDs (>=0 during events, -1 otherwise)
        """
        ref_df = self.quivering_annotation_df.copy()
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

    def visualize_features(self, overwrite=True, batch_size=300, full_vis=False):
        """
        Generate annotated video visualization of spawning events with pose overlays.

        Creates video showing spawning event frames with pose keypoints (nose and stripe4),
        interaction lines color-coded by distance (green=mouthing, red=normal), and text overlay
        of frame features. By default shows only spawning events; use full_vis=True for entire video.
        Black separator frames inserted between distinct spawning events.

        Args:
            overwrite (bool): If False, skip if output video already exists
            batch_size (int): Number of frames to read at once for efficiency (default 300)
            full_vis (bool): If True, visualize entire video; if False, only spawning event frames

        Saves:
            *_featurevis.mp4: Annotated video with 2-panel layout (video + text features)
        """
        def stream_frames_batch(cap, start_frame, batch_size):
            """Read batch_size frames sequentially starting from start_frame"""
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Seek once per batch
            frames = {}
            for i in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames[start_frame + i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frames

        def grab_framefeatures_as_strings(frame_idx, is_separator=False):
            if is_separator:
                return [f'--- EVENT SEPARATOR ---', f'Next spawning event starting soon...']
            elif frame_idx in self.framefeatures_df.index:
                feats = [f'frame = {frame_idx}']
                feats.extend([f'{feat} = {val}' for feat, val in list(self.framefeatures_df.loc[frame_idx].items())])
                return feats
            else:
                return [f'frame = {frame_idx} (no features)']

        def create_black_frame(shape=(480, 640, 3)):
            """Create a black frame for separators"""
            return np.zeros(shape, dtype=np.uint8)

        def overlay_poses_on_frame(frame, frame_idx, min_likelihood=0.1):
            """Overlay nose and stripe4 points, with mouthing interaction lines"""
            if frame_idx not in self.pose_df.index:
                return frame

            # Make a copy to avoid modifying original
            frame_with_poses = frame.copy()

            # Colors for keypoints (BGR format for cv2)
            nose_color = (0, 0, 255)        # Red
            stripe4_color = (0, 128, 255)   # Orange
            normal_line_color = (255, 0, 0)     # Red
            mouthing_line_color = (0, 255, 0)       # Green

            # Store coordinates for drawing lines
            coordinates = {}

            # Draw nose and stripe4 points only
            for individual in self.individuals:
                for bodypart in ['nose', 'stripe4']:
                    try:
                        likelihood = self.pose_df.loc[frame_idx, (individual, bodypart, 'likelihood')]

                        if likelihood >= min_likelihood:
                            x = self.pose_df.loc[frame_idx, (individual, bodypart, 'x')]
                            y = self.pose_df.loc[frame_idx, (individual, bodypart, 'y')]

                            # Store coordinates for line drawing (keep full precision for distance calculation)
                            coordinates[(individual, bodypart)] = (x, y)

                            # Choose color based on bodypart
                            color = nose_color if bodypart == 'nose' else stripe4_color

                            # Draw point (radius 6 for both nose and stripe4) - convert to int for drawing
                            cv2.circle(frame_with_poses, (int(x), int(y)), 6, color, -1)

                            # Draw individual label for nose only
                            if bodypart == 'nose':
                                cv2.putText(frame_with_poses, individual[-1], (int(x)+8, int(y)-8),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    except (KeyError, ValueError):
                        # Handle missing data gracefully
                        continue

            # Draw lines between nose of one individual and stripe4 of another
            if len(self.individuals) >= 2:
                for ind1 in self.individuals:
                    for ind2 in self.individuals:
                        if ind1 != ind2:
                            nose_key = (ind1, 'nose')
                            stripe4_key = (ind2, 'stripe4')

                            if nose_key in coordinates and stripe4_key in coordinates:
                                nose_pos = coordinates[nose_key]
                                stripe4_pos = coordinates[stripe4_key]

                                # Calculate distance
                                dx = nose_pos[0] - stripe4_pos[0]
                                dy = nose_pos[1] - stripe4_pos[1]
                                distance = np.sqrt(dx**2 + dy**2)

                                # Choose line color based on distance
                                line_color = mouthing_line_color if distance < self.mouthing_dist_pixels else normal_line_color
                                line_thickness = 3 if distance < self.mouthing_dist_pixels else 1

                                # Draw line (convert to int for drawing)
                                cv2.line(frame_with_poses, (int(nose_pos[0]), int(nose_pos[1])),
                                        (int(stripe4_pos[0]), int(stripe4_pos[1])), line_color, line_thickness)

            return frame_with_poses

        out_path = str(self.video_path).replace('.mp4', f'{self.param_suffix}_featurevis.mp4')
        if not overwrite and os.path.exists(out_path):
            return

        # Get spawning event frames only
        spawning_frames = self.framefeatures_df[self.framefeatures_df.spawning_event_id >= 0]

        if len(spawning_frames) == 0:
            print("No spawning events found - skipping visualization")
            return
        elif full_vis:
            print('creating full visualization. Not recommended for long videos.')
            visualization_sequence = list(self.framefeatures_df.index)
        else:
            print(f"Found {len(spawning_frames)} spawning event frames")

            # Build visualization sequence with separators between events
            visualization_sequence = []
            current_event_id = None
            separator_frames = 30  # 1 second at 30fps

            for frame_idx in spawning_frames.index:
                event_id = spawning_frames.loc[frame_idx, 'spawning_event_id']

                # Add separator when switching to new event
                if current_event_id is not None and event_id != current_event_id:
                    # Add separator frames
                    for _ in range(separator_frames):
                        visualization_sequence.append('separator')

                visualization_sequence.append(frame_idx)
                current_event_id = event_id

        print(f"Visualization sequence length: {len(visualization_sequence)} frames")

        cap = cv2.VideoCapture(self.video_path)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize frame cache
        frame_cache = {}
        current_batch_start = -1

        def get_frame(frame_idx):
            """Get frame from cache, loading batch if necessary, with pose overlay"""
            nonlocal frame_cache, current_batch_start

            if frame_idx == 'separator':
                return create_black_frame()

            # Check if we need to load a new batch
            batch_start = (frame_idx // batch_size) * batch_size
            if batch_start != current_batch_start or frame_idx not in frame_cache:
                # print(f"Loading batch starting at frame {batch_start}")
                # Clear old cache to free memory
                frame_cache.clear()
                # Load new batch
                frame_cache = stream_frames_batch(cap, batch_start, batch_size)
                current_batch_start = batch_start

            # Get base frame
            if frame_idx in frame_cache:
                base_frame = frame_cache[frame_idx]
            elif frame_idx < total_video_frames:
                # Frame exists but wasn't read - create black frame
                base_frame = create_black_frame()
            else:
                # Frame beyond video length - create black frame
                base_frame = create_black_frame()

            # Add pose overlay
            frame_with_poses = overlay_poses_on_frame(base_frame, frame_idx)
            return frame_with_poses

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[1].axis('off')

        # Initialize with first frame
        first_item = visualization_sequence[0]
        im = axes[0].imshow(get_frame(first_item))
        feat_strings = grab_framefeatures_as_strings(first_item, is_separator=(first_item == 'separator'))
        y_positions = np.linspace(0.1, 0.9, len(feat_strings))
        txt_array = [axes[1].text(0.1, y_positions[i], feat_strings[i] if i < len(feat_strings) else '')
                     for i in range(max(len(feat_strings), 6))]  # Ensure minimum text objects

        def animate(i):
            if i >= len(visualization_sequence):
                return *txt_array, im

            frame_item = visualization_sequence[i]
            is_separator = (frame_item == 'separator')

            # Update image
            im.set_data(get_frame(frame_item))

            # Update feature text
            feat_strings = grab_framefeatures_as_strings(frame_item, is_separator=is_separator)
            for j, txt in enumerate(txt_array):
                if j < len(feat_strings):
                    txt.set_text(feat_strings[j])
                else:
                    txt.set_text('')

            return *txt_array, im

        print(f"Creating visualization for {len(visualization_sequence)} frames using batch size {batch_size}")
        anim = FuncAnimation(
            fig,
            animate,
            frames=len(visualization_sequence),
            interval=1000 / 30,
            )

        anim.save(out_path, writer='ffmpeg', fps=30)
        cap.release()
        plt.close('all')
        print(f"Visualization saved to {out_path}")

        # Print summary
        spawning_frame_count = sum(1 for item in visualization_sequence if item != 'separator')
        separator_count = sum(1 for item in visualization_sequence if item == 'separator')
        print(f"Summary: {spawning_frame_count} spawning event frames + {separator_count} separator frames")

    def generate_predicted_spawning_summary(self):
        """
        Generate CSV summary of detected spawning events with timestamps.

        Creates human-readable table with event_id, start time, and stop time (HH:MM:SS format).
        Useful for quickly scanning through detected spawning bouts without watching full video.

        Saves:
            *_predicted_spawning_summary.csv: Table of spawning event timings
        """
        if self.pose_df is None:
            return
        outfile_path = str(self.video_path).replace('.mp4', f'{self.param_suffix}_predicted_spawning_summary.csv')
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
        """
        Generate CSV summary of detected double occupancy events with timestamps.

        Creates human-readable table with event_id, start time, and stop time (HH:MM:SS format).
        Useful for analyzing when both fish are simultaneously in breeding pipe.

        Saves:
            *_predicted_double_occupancy_summary.csv: Table of double occupancy event timings
        """
        if self.pose_df is None:
            return
        outfile_path = str(self.video_path).replace('.mp4', f'{self.param_suffix}_predicted_double_occupancy_summary.csv')
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
    """
    Concatenate all clip-level feature CSVs in directory tree into single collated file.

    Recursively finds all *_clipfeatures.csv files, concatenates them with video names as index,
    sorts alphabetically, and saves to collated_clipfeatures.csv in parent directory.

    Args:
        parent_dir (str/Path): Root directory to search for clipfeature CSV files

    Saves:
        collated_clipfeatures.csv: Combined DataFrame with one row per video
    """
    parent_dir = Path(parent_dir)
    clipfeature_csv_paths = list(parent_dir.glob('**/*_clipfeatures.csv'))
    rows = []
    for csv_path in clipfeature_csv_paths:
        rows.append(pd.read_csv(str(csv_path), index_col=0))
    df = pd.concat(rows, axis=0)
    df = df.sort_index()
    df.to_csv(str(parent_dir / 'collated_clipfeatures.csv'))
    pd.concat(rows, axis=0)

def process_video(video_path, quivering_annotation_path=None, pose_h5_path=None, visualize=False, n_minutes=None, min_likelihood=0.5):
    """
    Extract behavioral features from single video file.

    Convenience function that initializes FeatureExtractor, runs feature extraction, and optionally
    generates visualization. Prints progress messages to console.

    Args:
        video_path (str/Path): Path to .mp4 video file
        quivering_annotation_path (str/Path, optional): Path to Excel file with manual annotations
        pose_h5_path (str/Path, optional): Path to DeepLabCut pose .h5 file
        visualize (bool): If True, generate annotated video visualization after extraction
        n_minutes (int, optional): If provided, only analyze last n_minutes of video
        min_likelihood (float): Minimum keypoint confidence threshold (0-1)

    Saves:
        *_framefeatures.csv, *_clipfeatures.csv, and optionally *_featurevis.mp4
    """
    video_path = Path(video_path)
    print(f'processing {video_path.stem}')
    if quivering_annotation_path is not None:
        print(f'using annotation file: {quivering_annotation_path}')
    if pose_h5_path is not None:
        print(f'using pose file: {pose_h5_path}')
    fe = FeatureExtractor(video_path, quivering_annotation_path, pose_h5_path, n_minutes=n_minutes, min_likelihood=min_likelihood)
    fe.extract_all_features()
    if visualize:
        print(f'generating visualization for {video_path.stem}')
        fe.visualize_features()

def process_all(parent_dir, quivering_annotation_path, visualize=False, n_minutes=None, min_likelihood=0.5):
    """
    Batch process all videos in directory tree.

    Recursively finds all *cropped.mp4 videos, matches them with corresponding pose .h5 files
    (searches for pattern *cropped*[0-9].h5), and processes each. Handles missing pose files gracefully.

    Args:
        parent_dir (str/Path): Root directory containing videos and pose files
        quivering_annotation_path (str/Path): Path to Excel file with manual annotations for all videos
        visualize (bool): If True, generate annotated video visualizations for all videos
        n_minutes (int, optional): If provided, only analyze last n_minutes of each video
        min_likelihood (float): Minimum keypoint confidence threshold (0-1)

    Saves:
        Feature CSVs for each video, plus visualizations if requested
    """
    parent_dir = Path(parent_dir)
    vid_paths = list(parent_dir.glob('**/*cropped.mp4'))
    for vp in vid_paths:
        try:
            pose_h5_path = list(vp.parent.glob(f'{vp.stem}*[0-9].h5'))[0]
        except IndexError:
            pose_h5_path = None
        process_video(vp, quivering_annotation_path, pose_h5_path, visualize=visualize, n_minutes=n_minutes, min_likelihood=min_likelihood)
    print('all videos processed')

def delete_outputs(parent_dir, keep_pose_data=True):
    """
    Clean up generated output files from directory tree.

    Removes feature extraction outputs, visualizations, and optionally pose estimation files.
    Useful for re-running pipeline with different parameters or freeing disk space.

    Args:
        parent_dir (str/Path): Root directory to clean
        keep_pose_data (bool): If True, keep *_full.pickle, *_meta.pickle, and *_full.mp4 pose files.
                               If False, delete those as well (requires re-running DLC inference)

    Deletes:
        Always: *_assemblies.pickle, *_el.h5, *_el.pickle, *_filtered.csv, *_filtered.h5,
                *_labeled.mp4, *_framefeatures.csv, *_clipfeatures.csv, *_featurevis.mp4, *_roi.png
        If keep_pose_data=False: *_full.pickle, *_meta.pickle, *_full.mp4
    """
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


