"""Configuration constants for DemBA pipeline."""

# Video and ROI constants
ROI_RADIUS_MM = 79.375  # Breeding pipe radius in millimeters
VIDEO_FPS = 30  # Frames per second for video analysis

# Default parameters for pose estimation
DEFAULT_SHUFFLE = 1
DEFAULT_N_FISH = 2
DEFAULT_TRACK_METHOD = 'ellipse'

# Default parameters for feature extraction
DEFAULT_MOUTHING_DIST_MM = 10  # Nose-to-genital distance threshold for mouthing detection
DEFAULT_MIN_LIKELIHOOD = 0.5  # Minimum keypoint confidence threshold

# Default parameters for ID correction
DEFAULT_PATCH_SIZE = 128
DEFAULT_PADDING = 10
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_MIN_SILHOUETTE = 0.2

# Default parameters for tracklet stitching
DEFAULT_MIN_TRACKLET_LENGTH = 10
