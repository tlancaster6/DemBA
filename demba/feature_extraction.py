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
import ast

idx = pd.IndexSlice
from matplotlib.animation import FuncAnimation

class FeatureExtractor:
    """
Extracts behavioral features from cichlid courtship videos with DeepLabCut pose estimation.

    Analyzes videos to detect and quantify spawning-related behaviors including mouthing events
    (nose-to-stripe4 proximity), double occupancy of breeding pipe, and spawning bouts. Integrates
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

    def __init__(self, trial_manager, quivering_annotation_path=None, mouthing_dist_mm=None, min_likelihood=None, n_minutes=None):
        """
        Initialize FeatureExtractor with video, pose data, and processing parameters.

        Args:
            trial_manager (TrialManager): TrialManager instance for the trial. Used to resolve
                video and pose H5 paths, and mark completion status.
            quivering_annotation_path (str/Path, optional): Path to Excel file with manual quivering annotations
            mouthing_dist_mm (float, optional): Maximum nose-to-stripe4 distance in mm to count as mouthing event (default: from config)
            min_likelihood (float, optional): Minimum keypoint confidence threshold (0-1) for pose filtering (default: from config)
            n_minutes (int, optional): If provided, only analyze the last n_minutes of video (default: from config)
        """
        from demba import config

        # Store TrialManager
        self.trial_manager = trial_manager

        # Get paths from TrialManager
        self.video_path = str(trial_manager.video_path())
        self.h5_path = str(trial_manager.filtered_h5_path())

        self.shortform_id = self._get_shortform_id()
        self.n_minutes = n_minutes if n_minutes is not None else config.DEFAULT_N_MINUTES
        self.quivering_annotation_path = None if quivering_annotation_path is None else str(quivering_annotation_path)
        self.file_stem = Path(self.video_path).stem
        self.mouthing_dist_mm = mouthing_dist_mm if mouthing_dist_mm is not None else config.DEFAULT_MOUTHING_DIST_MM
        min_likelihood = min_likelihood if min_likelihood is not None else config.DEFAULT_MIN_LIKELIHOOD

        # Create parameter suffix for unique file naming
        self.param_suffix = f"_mdist{self.mouthing_dist_mm}mm_likelihood{min_likelihood}"
        if n_minutes is not None:
            self.param_suffix += f"_last{n_minutes}min"

        self.framefeatures_path = self.video_path.replace('.mp4', f'{self.param_suffix}_framefeatures.csv')
        self.clipfeatures_path = self.video_path.replace('.mp4', f'{self.param_suffix}_clipfeatures.csv')

        self.pose_df, self.individuals, self.bodyparts = load_poses(self.h5_path, min_likelihood=min_likelihood)
        self.quivering_annotation_df = self._load_quivering_annotations()
        self.roi_x, self.roi_y, self.roi_r, self.frame_height, self.frame_width = self._estimate_roi()
        self.mouthing_dist_pixels = self._calc_mouthing_dist_pixels()

        # Initialize feature registries
        self._init_frame_feature_registry()
        self._init_clip_feature_registry()

    def _init_frame_feature_registry(self):
        """
        Initialize the feature registry with methods, dependencies, and requirements.

        The registry defines:
        - method: The extraction function to call
        - depends_on: List of feature names that must be computed first
        - requires_pose: Whether pose data is needed
        - requires_annotations: Whether quivering annotations are needed
        - method_kwargs: Function to build kwargs from previously computed results
        """
        self.frame_feature_registry = {
            # Presence detection features (sex-specific)
            'male_in_frame': {
                'method': lambda: self._detect_frame_presence('male'),
                'depends_on': [],
                'requires_pose': True,
                'requires_annotations': False,
            },
            'female_in_frame': {
                'method': lambda: self._detect_frame_presence('female'),
                'depends_on': [],
                'requires_pose': True,
                'requires_annotations': False,
            },
            'male_in_pipe': {
                'method': lambda: self._detect_pipe_presence('male'),
                'depends_on': [],
                'requires_pose': True,
                'requires_annotations': False,
            },
            'female_in_pipe': {
                'method': lambda: self._detect_pipe_presence('female'),
                'depends_on': [],
                'requires_pose': True,
                'requires_annotations': False,
            },
            # Mouthing distance features (sex-specific)
            'male_mouthing_dist': {
                'method': lambda: self._calc_interaction_distances('male'),
                'depends_on': [],
                'requires_pose': True,
                'requires_annotations': False,
            },
            'female_mouthing_dist': {
                'method': lambda: self._calc_interaction_distances('female'),
                'depends_on': [],
                'requires_pose': True,
                'requires_annotations': False,
            },
            # Mouthing event features (sex-specific)
            'male_mouthing_event_id': {
                'method': lambda dists=None: self._detect_mouthing_events('male', dists=dists),
                'depends_on': ['male_mouthing_dist'],
                'requires_pose': True,
                'requires_annotations': False,
                'method_kwargs': lambda results: {'dists': results['male_mouthing_dist']}
            },
            'female_mouthing_event_id': {
                'method': lambda dists=None: self._detect_mouthing_events('female', dists=dists),
                'depends_on': ['female_mouthing_dist'],
                'requires_pose': True,
                'requires_annotations': False,
                'method_kwargs': lambda results: {'dists': results['female_mouthing_dist']}
            },
            # Spawning event (combines both sex mouthing events)
            'spawning_event_id': {
                'method': self._detect_spawning_events,
                'depends_on': ['male_mouthing_event_id', 'female_mouthing_event_id'],
                'requires_pose': True,
                'requires_annotations': False,
                'method_kwargs': lambda results: {
                    'male_mouthing_event_ids': results['male_mouthing_event_id'],
                    'female_mouthing_event_ids': results['female_mouthing_event_id']
                }
            },
            # Double occupancy event
            'double_occupancy_event_id': {
                'method': self._detect_double_occupancy_events,
                'depends_on': [],
                'requires_pose': True,
                'requires_annotations': False,
            },
            # Quivering annotations (not sex-split at frame level, already properly named)
            'male_lead_quiver': {
                'method': lambda: self._map_quivering_annotations()[0],
                'depends_on': [],
                'requires_pose': False,
                'requires_annotations': True,
            },
            'male_circle_quiver': {
                'method': lambda: self._map_quivering_annotations()[1],
                'depends_on': [],
                'requires_pose': False,
                'requires_annotations': True,
            },
            'female_circle_quiver': {
                'method': lambda: self._map_quivering_annotations()[2],
                'depends_on': [],
                'requires_pose': False,
                'requires_annotations': True,
            },
        }

    def _init_clip_feature_registry(self):
        """
        Initialize the clip-level feature registry with methods and requirements.

        Clip features are aggregations over the entire video (or temporal window).
        Unlike frame features, they don't have dependencies on each other - they
        all depend on frame features having been extracted already.
        """
        self.clip_feature_registry = {
            # Quivering annotation features (sex-specific)
            'male_lead_quivering_fraction': {
                'method': lambda: self._calc_quivering_fraction('male', lead=True),
                'requires_pose': False,
                'requires_annotations': True,
            },
            'male_circle_quivering_fraction': {
                'method': lambda: self._calc_quivering_fraction('male', lead=False),
                'requires_pose': False,
                'requires_annotations': True,
            },
            'female_circle_quivering_fraction': {
                'method': lambda: self._calc_quivering_fraction('female', lead=False),
                'requires_pose': False,
                'requires_annotations': True,
            },
            'n_male_lead_quivers': {
                'method': lambda: self._calc_n_quivers('male', lead=True),
                'requires_pose': False,
                'requires_annotations': True,
            },
            'n_male_circle_quivers': {
                'method': lambda: self._calc_n_quivers('male', lead=False),
                'requires_pose': False,
                'requires_annotations': True,
            },
            'n_female_circle_quivers': {
                'method': lambda: self._calc_n_quivers('female', lead=False),
                'requires_pose': False,
                'requires_annotations': True,
            },
            # Mouthing event features (sex-specific)
            'n_male_mouthing_events': {
                'method': lambda: self._calc_n_mouthing_events('male'),
                'requires_pose': True,
                'requires_annotations': False,
            },
            'n_female_mouthing_events': {
                'method': lambda: self._calc_n_mouthing_events('female'),
                'requires_pose': True,
                'requires_annotations': False,
            },
            'male_mouthing_event_fraction': {
                'method': lambda: self._calc_mouthing_event_fraction('male'),
                'requires_pose': True,
                'requires_annotations': False,
            },
            'female_mouthing_event_fraction': {
                'method': lambda: self._calc_mouthing_event_fraction('female'),
                'requires_pose': True,
                'requires_annotations': False,
            },
            # ROI occupancy features (sex-specific)
            'male_roi_occupancy_fraction': {
                'method': lambda: self._calc_roi_occupancy_fraction('male'),
                'requires_pose': True,
                'requires_annotations': False,
            },
            'female_roi_occupancy_fraction': {
                'method': lambda: self._calc_roi_occupancy_fraction('female'),
                'requires_pose': True,
                'requires_annotations': False,
            },
            # Other pose-based event features
            'n_double_occupancy_events': {
                'method': self._calc_n_double_occupancy_events,
                'requires_pose': True,
                'requires_annotations': False,
            },
            'n_spawning_events': {
                'method': self._calc_n_spawning_events,
                'requires_pose': True,
                'requires_annotations': False,
            },
            'double_occupancy_event_fraction': {
                'method': self._calc_double_occupancy_event_fraction,
                'requires_pose': True,
                'requires_annotations': False,
            },
            'spawning_event_fraction': {
                'method': self._calc_spawning_event_fraction,
                'requires_pose': True,
                'requires_annotations': False,
            },
            # ROI metadata
            'roi_x': {
                'method': lambda: self.roi_x,
                'requires_pose': True,
                'requires_annotations': False,
            },
            'roi_y': {
                'method': lambda: self.roi_y,
                'requires_pose': True,
                'requires_annotations': False,
            },
            'roi_r': {
                'method': lambda: self.roi_r,
                'requires_pose': True,
                'requires_annotations': False,
            },
            # Features that require both pose and annotations (sex-specific)
            'male_mouthing_female_quivering_phi': {
                'method': lambda: self._calc_mouthing_quivering_phi('male'),
                'requires_pose': True,
                'requires_annotations': True,
            },
            'female_mouthing_male_quivering_phi': {
                'method': lambda: self._calc_mouthing_quivering_phi('female'),
                'requires_pose': True,
                'requires_annotations': True,
            },
            'male_mouthing_female_quivering_jaccard': {
                'method': lambda: self._calc_mouthing_quivering_jaccard('male'),
                'requires_pose': True,
                'requires_annotations': True,
            },
            'female_mouthing_male_quivering_jaccard': {
                'method': lambda: self._calc_mouthing_quivering_jaccard('female'),
                'requires_pose': True,
                'requires_annotations': True,
            },
        }

    def _topological_sort(self, feature_names):
        """
        Sort features by dependencies using depth-first search.

        Automatically includes all required dependencies, even if not explicitly requested.

        Args:
            feature_names (list): List of feature names to sort

        Returns:
            list: Features in execution order (dependencies first), including all transitive dependencies

        Raises:
            ValueError: If circular dependencies are detected
        """
        visited = set()
        visiting = set()  # Track nodes in current DFS path for cycle detection
        result = []

        def visit(name):
            if name in visited:
                return
            if name in visiting:
                raise ValueError(f"Circular dependency detected involving feature '{name}'")

            visiting.add(name)

            # Visit ALL dependencies first (not just those in feature_names)
            for dep in self.frame_feature_registry[name]['depends_on']:
                visit(dep)

            visiting.remove(name)
            visited.add(name)
            result.append(name)

        for name in feature_names:
            visit(name)

        return result

    def load_feature_csvs(self):
        """
        Load previously computed frame-level and clip-level feature CSVs from disk.

        Populates self.framefeatures_df and self.clipfeatures_df attributes. Use this to reload
        features without re-computing, enabling downstream analysis without full re-extraction.
        """
        self.framefeatures_df = pd.read_csv(self.framefeatures_path, index_col=0, low_memory=False)
        self.clipfeatures_df = pd.read_csv(self.clipfeatures_path, index_col=0, low_memory=False)

    def extract_framefeatures(self, features_to_extract=None, verbose=True):
        """
        Extract frame-level behavioral features from video using feature registry.

        Frame-level features (computed per frame):
            - male_in_frame, female_in_frame: Boolean presence detection per sex
            - male_in_pipe, female_in_pipe: Boolean ROI presence per sex
            - male_mouthing_dist, female_mouthing_dist: Nose-to-stripe4 distances per sex
            - male_mouthing_event_id, female_mouthing_event_id: Mouthing event IDs per sex
            - spawning_event_id: Combined spawning events from both sexes
            - double_occupancy_event_id: Both fish in pipe simultaneously
            - male_lead_quiver, male_circle_quiver, female_circle_quiver: Quivering event IDs from annotations

        Args:
            features_to_extract (list, optional): List of feature names to extract. If None, extracts all available features.
            verbose (bool): If True, print progress messages during extraction

        Saves framefeatures_df to *_framefeatures.csv.
        """
        # Determine which features to extract
        if features_to_extract is None:
            features_to_extract = list(self.frame_feature_registry.keys())

        # Filter based on data availability
        available_features = []
        for feature_name in features_to_extract:
            if feature_name not in self.frame_feature_registry:
                print(f"Warning: Unknown feature '{feature_name}' requested, skipping")
                continue

            spec = self.frame_feature_registry[feature_name]

            # Check requirements
            if spec.get('requires_pose', False) and self.pose_df is None:
                if verbose:
                    print(f"Skipping '{feature_name}': requires pose data")
                continue
            if spec.get('requires_annotations', False) and self.quivering_annotation_df is None:
                if verbose:
                    print(f"Skipping '{feature_name}': requires quivering annotations")
                continue

            available_features.append(feature_name)

        if verbose:
            print(f"Extracting {len(available_features)} frame-level features")

        # Topologically sort to respect dependencies
        execution_order = self._topological_sort(available_features)

        # Execute in order, caching results
        results = {}
        for feature_name in execution_order:
            spec = self.frame_feature_registry[feature_name]

            # Build kwargs from dependencies
            kwargs = {}
            if 'method_kwargs' in spec:
                kwargs = spec['method_kwargs'](results)

            # Execute and cache
            if verbose:
                print(f"  - Extracting {feature_name}...")
            results[feature_name] = spec['method'](**kwargs)

        # Combine into DataFrame
        framefeatures_df = pd.concat(list(results.values()), axis=1)

        # Apply temporal windowing if needed
        if self.n_minutes is not None:
            n_frames = self.n_minutes * 60 * VIDEO_FPS
            framefeatures_df = framefeatures_df.tail(n_frames)
            if verbose:
                print(f"Applied temporal window: last {self.n_minutes} minutes ({n_frames} frames)")

        self.framefeatures_df = framefeatures_df
        self.framefeatures_df.to_csv(self.framefeatures_path)
        if verbose:
            print(f"Frame features saved to {self.framefeatures_path}")

    def extract_clipfeatures(self, verbose=True):
        """
        Extract clip-level behavioral features (aggregated over entire video).

        Clip-level features (aggregated over video):
            - n_male_mouthing_events, n_female_mouthing_events: Event counts per sex
            - male_mouthing_event_fraction, female_mouthing_event_fraction: Time fractions per sex
            - male_roi_occupancy_fraction, female_roi_occupancy_fraction: ROI presence fractions per sex
            - n_spawning_events, n_double_occupancy_events: Event counts
            - spawning_event_fraction, double_occupancy_event_fraction: Time fractions
            - roi_x, roi_y, roi_r: Pipe location and size
            - n_*_quivers: Count of quivering events by type and sex
            - *_quivering_fraction: Fraction of time spent quivering by type and sex
            - *_mouthing_*_quivering_phi/jaccard: Co-occurrence metrics by sex

        Args:
            verbose (bool): If True, print progress messages during extraction

        Saves clipfeatures_df to *_clipfeatures.csv.

        Note: Requires framefeatures to be extracted first.
        """
        if not hasattr(self, 'framefeatures_df') or self.framefeatures_df is None:
            raise ValueError("Frame features must be extracted before clip features. Call extract_framefeatures() first.")

        # Extract clip-level features using registry
        if verbose:
            print("Extracting clip-level features...")

        # Filter features based on data availability
        available_clip_features = []
        for feature_name, spec in self.clip_feature_registry.items():
            if spec.get('requires_pose', False) and self.pose_df is None:
                continue
            if spec.get('requires_annotations', False) and self.quivering_annotation_df is None:
                continue
            available_clip_features.append(feature_name)

        # Execute and collect results
        clipfeatures_series = pd.Series(dtype=float)
        for feature_name in available_clip_features:
            spec = self.clip_feature_registry[feature_name]
            if verbose:
                print(f"  - Extracting {feature_name}...")
            clipfeatures_series[feature_name] = spec['method']()

        self.clipfeatures_df = pd.DataFrame(clipfeatures_series, columns=[self.file_stem]).T
        self.clipfeatures_df.to_csv(self.clipfeatures_path)
        if verbose:
            print(f"Clip features saved to {self.clipfeatures_path}")

    def extract_all_features(self, features_to_extract=None, verbose=True):
        """
        Extract both frame-level and clip-level behavioral features from video.

        This is a convenience method that calls extract_framefeatures() followed by extract_clipfeatures().
        See those methods for details on what features are extracted.

        Args:
            features_to_extract (list, optional): List of frame-level feature names to extract. If None, extracts all available.
            verbose (bool): If True, print progress messages during extraction

        Saves framefeatures_df to *_framefeatures.csv and clipfeatures_df to *_clipfeatures.csv.
        """
        self.extract_framefeatures(features_to_extract=features_to_extract, verbose=verbose)
        self.extract_clipfeatures(verbose=verbose)

        # Mark stage as complete
        self.trial_manager.mark_stage_complete('feature_extraction')

    def _get_shortform_id(self):
        """
        Parse video filename stem to generate shortform ID.

        Extracts split (behave/control) and group number from the video path stem,
        then formats as shortform ID: D + B/C + group_number

        Examples:
            - "bgrb_t007_9.25.24_DB11cropped" -> DB11 (shortform found in stem)
            - "BHVE_group8" -> DB8 (behave group 8)
            - "CTRL_group12" -> DC12 (control group 12)

        Returns:
            str: Shortform ID (e.g., "DB11", "DC5")
        """
        import re

        stem = Path(self.video_path).stem

        # Check if shortform format (D[B/C][0-9]+) appears anywhere in stem
        shortform_match = re.search(r'D([BC])(\d+)', stem)
        if shortform_match:
            return shortform_match.group(0)  # Return just the matched shortform ID

        # Parse verbose format (BHVE_groupN or CTRL_groupN)
        verbose_match = re.search(r'(BHVE|CTRL)_group(\d+)', stem)
        if verbose_match:
            split_str = verbose_match.group(1)
            group_num = verbose_match.group(2)

            # Map split string to letter
            split_letter = 'B' if split_str == 'BHVE' else 'C'

            # Format as shortform ID
            return f'D{split_letter}{group_num}'

        # If no pattern matched, raise error
        raise ValueError(f"Could not parse shortform ID from video stem: {stem}")


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
        if self.shortform_id and (self.shortform_id in sheet_names):
            return pd.read_excel(self.quivering_annotation_path, sheet_name=self.shortform_id, skiprows=1)
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

    def _calc_interaction_distances(self, mouthing_sex):
        """
        Calculate distance from specified sex's nose to the other sex's stripe4.

        Mouthing behavior involves one fish nibbling the anal fin (near stripe4) of the other. Computes Euclidean
        distance between the mouthing individual's nose and the receiving individual's stripe4.

        Args:
            mouthing_sex (str): 'male' or 'female' - which sex is doing the mouthing

        Returns:
            pd.Series: Per-frame distance in pixels, name='male_mouthing_dist' or 'female_mouthing_dist'
        """
        # Map sex to individual ID (individual1 = male, individual2 = female)
        mouther_individual = 'individual1' if mouthing_sex == 'male' else 'individual2'
        receiver_individual = 'individual2' if mouthing_sex == 'male' else 'individual1'

        # Use explicit x,y column selection to avoid column ordering issues
        nose_coords = self.pose_df.loc[:, (mouther_individual, 'nose', ['x', 'y'])].values
        stripe4_coords = self.pose_df.loc[:, (receiver_individual, 'stripe4', ['x', 'y'])].values

        dists = nose_coords - stripe4_coords
        dist_series = pd.Series(np.hypot(dists[:, 0], dists[:, 1]),
                               index=self.pose_df.index,
                               name=f'{mouthing_sex}_mouthing_dist')
        return dist_series

    def _cluster_temporal_events(self, candidate_frames, reference_index, event_name, eps, min_samples):
        """
        Generic temporal clustering helper for detecting behavioral events.

        Uses DBSCAN1D to group temporally proximate frames into events, then fills gaps
        within each event to handle brief tracking failures.

        Args:
            candidate_frames (np.ndarray): Frame indices where event condition is met
            reference_index (pd.Index): Full video frame index for reindexing
            event_name (str): Name for the resulting event_id series
            eps (int): Maximum gap in frames to consider same event (DBSCAN epsilon)
            min_samples (int): Minimum frames required to qualify as event

        Returns:
            pd.Series: Per-frame event ID (>=0 during events, -1 otherwise)
        """
        labels = DBSCAN1D(eps, min_samples).fit_predict(candidate_frames)
        event_ids = pd.Series(data=labels, index=candidate_frames).reindex(reference_index, fill_value=-1)

        # Fill gaps within each event
        for eid in event_ids.unique():
            if eid >= 0:
                start_idx = event_ids[event_ids == eid].index.min()
                end_idx = event_ids[event_ids == eid].index.max()
                event_ids.loc[start_idx:end_idx] = eid

        event_ids.name = event_name
        return event_ids

    def _detect_mouthing_events(self, sex, dists=None, eps=None, min_samples=None):
        """
        Detect mouthing events for a specific sex using temporal clustering of sub-threshold nose-stripe4 distances.

        Uses DBSCAN1D to group nearby frames where distance < mouthing_dist_pixels. Fills gaps
        within events to handle brief tracking failures. Events must span at least min_samples frames.

        Args:
            sex (str): 'male' or 'female' - which sex is doing the mouthing
            dists (pd.Series, optional): Pre-computed interaction distances. If None, computes them.
            eps (int, optional): Maximum gap in frames to consider same event (DBSCAN epsilon) (default: from config)
            min_samples (int, optional): Minimum frames required to qualify as event (default: from config)

        Returns:
            pd.Series: Per-frame event ID (>=0 during events, -1 otherwise), name='male_mouthing_event_id' or 'female_mouthing_event_id'
        """
        from demba import config
        if eps is None:
            eps = config.DEFAULT_MOUTHING_EPS
        if min_samples is None:
            min_samples = config.DEFAULT_MOUTHING_MIN_SAMPLES

        if dists is None:
            dists = self._calc_interaction_distances(sex)

        subthresh_frames = dists.loc[dists < self.mouthing_dist_pixels].index.values
        event_name = f'{sex}_mouthing_event_id'
        return self._cluster_temporal_events(subthresh_frames, dists.index, event_name, eps, min_samples)

    def _detect_double_occupancy_events(self, eps=None, min_samples=None):
        """
        Detect sustained double occupancy of breeding pipe using temporal clustering.

        Both fish entering pipe together is prerequisite for spawning. Uses DBSCAN1D to identify
        continuous bouts where both male and female are inside pipe ROI. Fills gaps and requires minimum duration.

        Args:
            eps (int, optional): Maximum gap in frames to consider same event (default: from config)
            min_samples (int, optional): Minimum frames required to qualify as event (default: from config)

        Returns:
            pd.Series: Per-frame event ID (>=0 during events, -1 otherwise), name='double_occupancy_event_id'
        """
        from demba import config
        if eps is None:
            eps = config.DEFAULT_DOUBLE_OCCUPANCY_EPS
        if min_samples is None:
            min_samples = config.DEFAULT_DOUBLE_OCCUPANCY_MIN_SAMPLES

        # Check presence for both sexes
        male_in_pipe = self._detect_pipe_presence('male')
        female_in_pipe = self._detect_pipe_presence('female')

        # Double occupancy occurs when both are present
        double_occupancy = male_in_pipe & female_in_pipe
        double_occupancy_frames = double_occupancy[double_occupancy].index.values

        return self._cluster_temporal_events(double_occupancy_frames, male_in_pipe.index, 'double_occupancy_event_id', eps, min_samples)

    def _detect_spawning_events(self, male_mouthing_event_ids=None, female_mouthing_event_ids=None, eps=None, min_samples=None):
        """
        Detect spawning events as clusters of mouthing events (from either sex) in temporal proximity.

        Spawning consists of multiple mouthing bouts in quick succession. Uses DBSCAN1D on start/end
        frames of mouthing events from both sexes to identify temporal clusters indicating spawning bouts.

        Args:
            male_mouthing_event_ids (pd.Series, optional): Pre-computed male mouthing events. If None, computes them.
            female_mouthing_event_ids (pd.Series, optional): Pre-computed female mouthing events. If None, computes them.
            eps (int, optional): Maximum gap in frames between mouthing events in same spawning bout (default: from config)
            min_samples (int, optional): Minimum mouthing event endpoints required to qualify as spawning (default: from config)

        Returns:
            pd.Series: Per-frame event ID (>=0 during events, -1 otherwise), name='spawning_event_id'
        """
        from demba import config
        if eps is None:
            eps = config.DEFAULT_SPAWNING_EPS
        if min_samples is None:
            min_samples = config.DEFAULT_SPAWNING_MIN_SAMPLES

        if male_mouthing_event_ids is None:
            male_mouthing_event_ids = self._detect_mouthing_events('male')
        if female_mouthing_event_ids is None:
            female_mouthing_event_ids = self._detect_mouthing_events('female')

        # Get start and end frames of each mouthing event for both sexes
        all_mouthing_frames = []

        # Process male mouthing events
        male_events = male_mouthing_event_ids[male_mouthing_event_ids >= 0]
        if len(male_events) > 0:
            male_start = male_events.reset_index().groupby('male_mouthing_event_id').first()['index'].values
            male_end = male_events.reset_index().groupby('male_mouthing_event_id').last()['index'].values
            all_mouthing_frames.extend([male_start, male_end])

        # Process female mouthing events
        female_events = female_mouthing_event_ids[female_mouthing_event_ids >= 0]
        if len(female_events) > 0:
            female_start = female_events.reset_index().groupby('female_mouthing_event_id').first()['index'].values
            female_end = female_events.reset_index().groupby('female_mouthing_event_id').last()['index'].values
            all_mouthing_frames.extend([female_start, female_end])

        if not all_mouthing_frames:
            # No mouthing events found, return empty series
            return pd.Series(-1, index=male_mouthing_event_ids.index, name='spawning_event_id')

        combined_mouthing_event_frames = np.concatenate(all_mouthing_frames)

        return self._cluster_temporal_events(combined_mouthing_event_frames, male_mouthing_event_ids.index, 'spawning_event_id', eps, min_samples)

    def _detect_frame_presence(self, sex):
        """
        Detect whether a specific sex is present in frame based on any visible keypoint.

        Individual is considered present if ANY of their bodyparts has valid (non-NaN) pose estimate
        after likelihood filtering.

        Args:
            sex (str): 'male' or 'female'

        Returns:
            pd.Series: Per-frame boolean indicating presence, name='male_in_frame' or 'female_in_frame'
        """
        # Map sex to individual ID (individual1 = male, individual2 = female)
        individual = 'individual1' if sex == 'male' else 'individual2'
        presence = self.pose_df.loc[:, (individual, slice(None), slice(None))].notna().any(axis=1)
        presence.name = f'{sex}_in_frame'
        return presence

    def _detect_pipe_presence(self, sex):
        """
        Detect whether a specific sex is inside breeding pipe ROI based on centroid position.

        Uses centroid of all valid keypoints to determine if fish is inside circular pipe ROI.
        Computes Euclidean distance from centroid to ROI center and checks if <= roi_r.
        More robust than using single keypoint.

        Args:
            sex (str): 'male' or 'female'

        Returns:
            pd.Series: Per-frame boolean indicating presence in pipe, name='male_in_pipe' or 'female_in_pipe'
        """
        # Map sex to individual ID (individual1 = male, individual2 = female)
        individual = 'individual1' if sex == 'male' else 'individual2'

        # Get x,y coordinates for all bodyparts of this individual
        x_coords = self.pose_df.loc[:, (individual, slice(None), 'x')]
        y_coords = self.pose_df.loc[:, (individual, slice(None), 'y')]

        # Calculate centroid of valid keypoints (using nanmean to ignore missing data)
        centroid_x = x_coords.mean(axis=1)
        centroid_y = y_coords.mean(axis=1)

        # Calculate distance from centroid to ROI center
        dist_to_roi = np.sqrt((centroid_x - self.roi_x)**2 + (centroid_y - self.roi_y)**2)

        # Check if within ROI radius
        in_pipe = dist_to_roi <= self.roi_r
        in_pipe.name = f'{sex}_in_pipe'
        return in_pipe

    def _calc_n_mouthing_events(self, sex):
        """
        Count total number of distinct mouthing events for a specific sex in video.

        Args:
            sex (str): 'male' or 'female'

        Returns:
            int: Number of unique mouthing event IDs for that sex
        """
        col_name = f'{sex}_mouthing_event_id'
        event_ids = self.framefeatures_df[self.framefeatures_df[col_name] >= 0][col_name]
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

    def _calc_mouthing_event_fraction(self, sex):
        """
        Calculate fraction of video time spent in mouthing events for a specific sex.

        Args:
            sex (str): 'male' or 'female'

        Returns:
            float: Proportion of frames with mouthing_event_id >= 0 for that sex (range 0-1)
        """
        col_name = f'{sex}_mouthing_event_id'
        n_mouthing_frames = len(self.framefeatures_df[self.framefeatures_df[col_name] >= 0])
        mouthing_fraction = n_mouthing_frames / len(self.framefeatures_df)
        return mouthing_fraction

    def _calc_roi_occupancy_fraction(self, sex):
        """
        Calculate fraction of time a specific sex is inside breeding pipe ROI.

        Args:
            sex (str): 'male' or 'female'

        Returns:
            float: Fraction of frames where specified sex is in pipe (range 0-1)
        """
        col_name = f'{sex}_in_pipe'
        in_pipe_frames = self.framefeatures_df[col_name].sum()
        fraction = in_pipe_frames / len(self.framefeatures_df)
        return fraction

    def _calc_quivering_fraction(self, sex, lead=False):
        """
        Calculate fraction of video time spent in specified quivering behavior.

        Args:
            sex (str): 'male' or 'female'
            lead (bool): If False, calculate circle quivering fraction. If True, calculate lead quivering fraction.
                        Note: Female lead quivering does not exist.

        Returns:
            float: Proportion of frames with quivering event ID >= 0

        Raises:
            ValueError: If lead=True and sex='female' (invalid combination)
        """
        if lead and sex == 'female':
            raise ValueError("Female lead quivering does not exist")

        if lead:
            col_name = 'male_lead_quiver'
        else:
            col_name = f'{sex}_circle_quiver'

        n_quivering_frames = len(self.framefeatures_df[self.framefeatures_df[col_name] >= 0])
        quivering_fraction = n_quivering_frames / len(self.framefeatures_df)
        return quivering_fraction

    def _calc_n_quivers(self, sex, lead=False):
        """
        Count number of distinct quivering events of specified type.

        Args:
            sex (str): 'male' or 'female'
            lead (bool): If False, count circle quivering events. If True, count lead quivering events.
                        Note: Female lead quivering does not exist.

        Returns:
            int: Number of unique quivering event IDs

        Raises:
            ValueError: If lead=True and sex='female' (invalid combination)
        """
        if lead and sex == 'female':
            raise ValueError("Female lead quivering does not exist")

        if lead:
            col_name = 'male_lead_quiver'
        else:
            col_name = f'{sex}_circle_quiver'

        event_ids = self.framefeatures_df[self.framefeatures_df[col_name] >= 0][col_name]
        n_events = len(event_ids.unique())
        return n_events

    def _calc_mouthing_quivering_phi(self, mouthing_sex):
        """
        Calculate Phi coefficient (correlation) between mouthing by one sex and circle quivering by the other.

        Measures association between automated mouthing detection and manual quivering annotations.
        Phi coefficient ranges from -1 (perfect negative correlation) to +1 (perfect positive correlation).

        Args:
            mouthing_sex (str): 'male' or 'female' - which sex is doing the mouthing

        Returns:
            float: Phi coefficient between specified sex's mouthing and opposite sex's circle quivering
        """
        mouthing_col = f'{mouthing_sex}_mouthing_event_id'
        quiver_col = 'female_circle_quiver' if mouthing_sex == 'male' else 'male_circle_quiver'

        mouthing_events = self.framefeatures_df[mouthing_col] >= 0
        quivers = self.framefeatures_df[quiver_col] >= 0
        phi = phi_coefficient(mouthing_events, quivers)
        return phi

    def _calc_mouthing_quivering_jaccard(self, mouthing_sex):
        """
        Calculate Jaccard index (overlap) between mouthing by one sex and circle quivering by the other.

        Measures temporal overlap as: (intersection) / (union). Jaccard index ranges from
        0 (no overlap) to 1 (perfect overlap) between mouthing and quivering frames.

        Args:
            mouthing_sex (str): 'male' or 'female' - which sex is doing the mouthing

        Returns:
            float: Jaccard index between specified sex's mouthing and opposite sex's circle quivering
        """
        mouthing_col = f'{mouthing_sex}_mouthing_event_id'
        quiver_col = 'female_circle_quiver' if mouthing_sex == 'male' else 'male_circle_quiver'

        mouthing_events = self.framefeatures_df[mouthing_col] >= 0
        quivers = self.framefeatures_df[quiver_col] >= 0
        jaccard = jaccard_index(mouthing_events, quivers)
        return jaccard

    def _clean_metadata_string(self, meta_str):
        """
        Extract quivering type from annotation metadata string.

        Args:
            meta_str (str): Raw metadata string containing TEMPORAL-SEGMENTS field

        Returns:
            str: Quivering type ('male-lead-quiver', 'male-circle-quiver', or 'female-circle-quiver')

        Raises:
            ValueError: If metadata string cannot be safely parsed
        """
        try:
            meta_dict = ast.literal_eval(meta_str)
            return meta_dict["TEMPORAL-SEGMENTS"]
        except (ValueError, SyntaxError, KeyError) as e:
            raise ValueError(f"Failed to parse metadata string: {meta_str}. Error: {e}")
        
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
        ref_df["temporal_segment_start"] = (pd.to_numeric(ref_df["temporal_segment_start"], errors='coerce') * VIDEO_FPS).round()
        ref_df["temporal_segment_end"] = (pd.to_numeric(ref_df["temporal_segment_end"], errors='coerce') * VIDEO_FPS).round()
        ref_df["metadata"] = (ref_df["metadata"]).apply(self._clean_metadata_string)

        # Determine max frame from pose data if available, otherwise use a large default
        if self.pose_df is not None:
            max_frame_count = len(self.pose_df)
        else:
            # Use max annotation time as fallback
            max_annotation_frame = int(ref_df["temporal_segment_end"].max())
            max_frame_count = max_annotation_frame + 1000  # Add buffer

        male_lead_quiver = pd.Series(False, index=pd.RangeIndex(0, max_frame_count), name='male_lead_quiver')
        male_circle_quiver = pd.Series(False, index=pd.RangeIndex(0, max_frame_count), name='male_circle_quiver')
        female_circle_quiver = pd.Series(False, index=pd.RangeIndex(0, max_frame_count), name='female_circle_quiver')
        MIN_FRAME = 0
        MAX_FRAME = max_frame_count - 1
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


def process_video(trial_manager, quivering_annotation_path=None, visualize=False, n_minutes=None, min_likelihood=None):
    """
    Extract behavioral features from single video file.

    Convenience function that initializes FeatureExtractor, runs feature extraction, and optionally
    generates visualization. Prints progress messages to console.

    Args:
        trial_manager (TrialManager): TrialManager instance for the trial. Used to resolve video
            and pose H5 paths, and mark completion status.
        quivering_annotation_path (str/Path, optional): Path to Excel file with manual annotations
        visualize (bool): If True, generate annotated video visualization after extraction
        n_minutes (int, optional): If provided, only analyze last n_minutes of video (default: from config)
        min_likelihood (float, optional): Minimum keypoint confidence threshold (0-1) (default: from config)

    Saves:
        *_framefeatures.csv, *_clipfeatures.csv, and optionally *_featurevis.mp4
    """
    video_path = trial_manager.video_path()
    print(f'processing {video_path.stem}')
    if quivering_annotation_path is not None:
        print(f'using annotation file: {quivering_annotation_path}')
    print(f'using pose file: {trial_manager.filtered_h5_path()}')

    fe = FeatureExtractor(trial_manager, quivering_annotation_path, n_minutes=n_minutes, min_likelihood=min_likelihood)
    fe.extract_all_features()
    if visualize:
        if list(video_path.parent.glob('*_featurevis.mp4')):
            print('feature visualization already exists. Skipping')
        else:
            print(f'generating visualization for {video_path.stem}')
            fe.visualize_features()
