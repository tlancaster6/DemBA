"""
DemBA - DeepLabCut-augmented Multi-animal Behavioral Analysis

A pipeline for pose estimation, identity correction, and behavioral feature extraction
from multi-animal videos using DeepLabCut.
"""

# Core pipeline modules
from .pose_estimation import estimate_pose
from .identity_correction import (
    PatchExtractor, CoOccupancyDetector, TripletDataset, SimpleCNN, TripletLoss,
    train_encoder, extract_all_embeddings, cluster_and_assign_ids,
    reassign_tracklet_ids, visualize_embeddings, main, prepare_id_correction, complete_id_correction
)
from .tracklet_stitching import stitch_by_identity
from .filtering import filter_predictions
from .feature_extraction import (
    FeatureExtractor, process_video
)
from .analysis import Plotter, trial_sort_key
from .visualization import create_labeled_video

# Utility modules
from . import utils
from .utils.dlc import load_bboxes, load_poses, load_tracklets, parse_full_pickle_path, parse_trial_name
from .utils.roi import estimate_roi, estimate_roi_hough, crop_video_to_roi
from .utils.metrics import phi_coefficient, jaccard_index

# Configuration
from . import config

__version__ = '0.1.0'

__all__ = [
    # Pose estimation
    'estimate_pose',
    # Identity correction
    'PatchExtractor',
    'CoOccupancyDetector',
    'TripletDataset',
    'SimpleCNN',
    'TripletLoss',
    'train_encoder',
    'extract_all_embeddings',
    'cluster_and_assign_ids',
    'reassign_tracklet_ids',
    'visualize_embeddings',
    # Tracklet stitching
    'stitch_by_identity',
    # Filtering
    'filter_predictions',
    # Feature extraction
    'FeatureExtractor',
    'process_video',
    # Analysis
    'Plotter',
    'trial_sort_key',
    # Visualization
    'create_labeled_video',
    # Utilities
    'utils',
    'config',
    'load_bboxes',
    'load_poses',
    'load_tracklets',
    'parse_full_pickle_path',
    'parse_trial_name',
    'estimate_roi',
    'estimate_roi_hough',
    'crop_video_to_roi',
    'phi_coefficient',
    'jaccard_index',
]
