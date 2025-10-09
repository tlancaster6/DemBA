"""Configuration constants for DemBA pipeline."""

# Video and ROI constants
ROI_RADIUS_MM = 79.375  # Breeding pipe radius in millimeters
VIDEO_FPS = 30  # Frames per second for video analysis

# Default parameters for pose estimation
DEFAULT_SHUFFLE = 1
DEFAULT_N_FISH = 2
DEFAULT_TRACK_METHOD = 'ellipse'

# Default parameters for feature extraction
DEFAULT_MOUTHING_DIST_MM = 10  # Nose-to-genital distance threshold for mouthing detection (mm)
DEFAULT_MIN_LIKELIHOOD = 0.5  # Minimum keypoint confidence threshold
DEFAULT_N_MINUTES = None  # Time restriction in minutes (None = analyze entire video)
DEFAULT_VISUALIZE_FLAG = True # whether to create a video visualization of feature extraction outputs

# Default parameters for ID correction
DEFAULT_PATCH_SIZE = 128
DEFAULT_PADDING = 10
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_MIN_SILHOUETTE = 0.2
DEFAULT_MIN_TRACKLET_LENGTH = 60
DEFAULT_MIN_OVERLAP_FRAMES = 30  # Minimum co-occupancy frames for tracklet pair inclusion
DEFAULT_MIN_KEYPOINTS = 5  # Minimum valid keypoints for bbox calculation

# Default parameters for ID correction training
DEFAULT_ID_N_EPOCHS = 200
DEFAULT_ID_BATCH_SIZE = 32
DEFAULT_ID_LEARNING_RATE = 0.001
DEFAULT_ID_DEVICE = 'cuda'
DEFAULT_ID_SAMPLES_PER_EPOCH = 1000  # Training samples per epoch
DEFAULT_ID_EMBEDDING_DIM = 128  # CNN embedding dimension
DEFAULT_ID_TRIPLET_MARGIN = 1.0  # Triplet loss margin
DEFAULT_ID_CACHE_FRAME_STRIDE = 5  # Sample every Nth frame for patch cache (reduces memory)

# Default parameters for ID correction interactive mapping
DEFAULT_ID_N_SEGMENTS = 3  # Number of trajectory segments to show per cluster
DEFAULT_ID_SEGMENT_DURATION_SEC = 3  # Duration of each segment in seconds

# Default parameters for filtering
DEFAULT_FILTER_TYPE = 'median'
DEFAULT_FILTER_WINDOW_LENGTH = 5
DEFAULT_FILTER_P_BOUND = 0.001
DEFAULT_FILTER_AR_DEGREE = 3
DEFAULT_FILTER_MA_DEGREE = 1
DEFAULT_FILTER_ALPHA = 0.01

# Default parameters for visualization
DEFAULT_VIZ_FILTERED = True
DEFAULT_VIZ_GRID_WIDTH = 8
DEFAULT_VIZ_GRID_HEIGHT = 6

# Default parameters for analysis
DEFAULT_ANALYSIS_BIN_WIDTH = 1800  # Bin width in frames for heatmaps (1800 frames = 60s at 30fps)

# Default parameters for behavioral event detection (DBSCAN clustering)
DEFAULT_MOUTHING_EPS = 5  # Maximum gap in frames for mouthing events
DEFAULT_MOUTHING_MIN_SAMPLES = 10  # Minimum frames required for mouthing event
DEFAULT_DOUBLE_OCCUPANCY_EPS = 30  # Maximum gap in frames for double occupancy events
DEFAULT_DOUBLE_OCCUPANCY_MIN_SAMPLES = 30  # Minimum frames required for double occupancy event
DEFAULT_SPAWNING_EPS = 300  # Maximum gap in frames between mouthing events in spawning bout
DEFAULT_SPAWNING_MIN_SAMPLES = 6  # Minimum mouthing event endpoints required for spawning
