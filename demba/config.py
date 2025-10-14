"""Configuration constants for DemBA pipeline."""

# =============================================================================
# SHARED PARAMETERS (used across multiple pipeline stages)
# =============================================================================
ROI_RADIUS_MM = 79.375  # Breeding pipe radius in millimeters
VIDEO_FPS = 30  # Frames per second for video analysis
DEFAULT_SHUFFLE = 1  # Default DeepLabCut shuffle to use
DEFAULT_TRAINING_FRACTION = 0.95  # Default DeepLabCut training fraction for scorer name generation
DEFAULT_N_FISH = 2  # Default number of fish in the video
DEFAULT_TRACK_METHOD = 'ellipse'  # Tracking method for DeepLabCut multi-animal pose estimation


# =============================================================================
# 1. POSE ESTIMATION PARAMETERS
# =============================================================================
# (Uses shared parameters: DEFAULT_SHUFFLE, DEFAULT_N_FISH, DEFAULT_TRACK_METHOD)


# =============================================================================
# 2. ID CORRECTION PARAMETERS
# =============================================================================
# Patch extraction
DEFAULT_PATCH_SIZE = 128  # Input dimensions (w and h) for the ID model. Patches will be resized to this dimension
DEFAULT_PADDING = 10  # Padding (in original image pixels) to add around the keypoint-based bbox approximation
DEFAULT_CONF_THRESHOLD = 0.25  # Minimum keypoint confidence for inclusion in bbox calculation
DEFAULT_MIN_KEYPOINTS = 5  # Minimum valid keypoints for bbox calculation

# Tracklet filtering
DEFAULT_MIN_TRACKLET_LENGTH = 60  # Minimum tracklet length (frames) for inclusion in training set
DEFAULT_MIN_OVERLAP_FRAMES = 30  # Minimum co-occupancy frames for tracklet pair inclusion in training set

# Model training
DEFAULT_ID_N_EPOCHS = 200  # Default number of training epochs. Early stopping will usually terminate training much earlier
DEFAULT_ID_BATCH_SIZE = 32  # Default batch size for Re-ID CNN
DEFAULT_ID_LEARNING_RATE = 0.001  # Default initial learning rate. LR will auto-reduce on plateau
DEFAULT_ID_DEVICE = 'cuda'  # Default device. GPU acceleration (cuda) recommended
DEFAULT_ID_SAMPLES_PER_EPOCH = 1000  # Training samples per epoch
DEFAULT_ID_EMBEDDING_DIM = 128  # CNN embedding dimension
DEFAULT_ID_TRIPLET_MARGIN = 1.0  # Triplet loss margin
DEFAULT_ID_CACHE_FRAME_STRIDE = 5  # Sample every Nth frame for patch cache (reduces memory)
DEFAULT_ID_NUM_WORKERS = 0  # Number of DataLoader worker processes. Set to 0 on Windows to avoid multiprocessing overhead

# Clustering and assignment
DEFAULT_MIN_SILHOUETTE = 0.2  # Minimum silhouette score for an embedded point to be associated with a specific ID

# Interactive mapping
DEFAULT_ID_N_SEGMENTS = 3  # Number of trajectory segments to show per cluster
DEFAULT_ID_SEGMENT_DURATION_SEC = 3  # Duration of each segment in seconds


# =============================================================================
# 3. TRACKLET STITCHING PARAMETERS
# =============================================================================
DEFAULT_STITCH_MIN_LENGTH = 10  # Minimum tracklet length to include in stitching
DEFAULT_STITCH_N_TRACKS = 2  # Number of individuals/tracks to reconstruct
DEFAULT_MIN_CONJOINED_RUN_LENGTH = 40  # Minimum consecutive frames of same ID to count as "real" identity run
DEFAULT_SPLIT_CONJOINED = True  # Whether to split tracklets that switch between tracking different individuals


# =============================================================================
# 4. FILTERING PARAMETERS
# =============================================================================
DEFAULT_FILTER_TYPE = 'median'  # Type of filter to apply to pose predictions
DEFAULT_FILTER_WINDOW_LENGTH = 5  # Window length for temporal filtering (in frames)
DEFAULT_FILTER_P_BOUND = 0.001  # P-value bound for statistical filtering
DEFAULT_FILTER_AR_DEGREE = 3  # Autoregressive degree for ARMA filtering
DEFAULT_FILTER_MA_DEGREE = 1  # Moving average degree for ARMA filtering
DEFAULT_FILTER_ALPHA = 0.01  # Significance level for filtering


# =============================================================================
# 5. FEATURE EXTRACTION PARAMETERS
# =============================================================================
# Basic feature parameters
DEFAULT_MOUTHING_DIST_MM = 10  # Nose-to-stripe4 distance threshold for mouthing detection (mm)
DEFAULT_MIN_LIKELIHOOD = 0.5  # Minimum keypoint confidence threshold
DEFAULT_N_MINUTES = None  # Time restriction in minutes (None = analyze entire video)
DEFAULT_VISUALIZE_FLAG = False  # Whether to create a video visualization of feature extraction outputs. Can be slow

# Behavioral event detection (DBSCAN clustering)
DEFAULT_MOUTHING_EPS = 5  # Maximum gap in frames for mouthing events
DEFAULT_MOUTHING_MIN_SAMPLES = 10  # Minimum frames required for mouthing event
DEFAULT_DOUBLE_OCCUPANCY_EPS = 30  # Maximum gap in frames for double occupancy events
DEFAULT_DOUBLE_OCCUPANCY_MIN_SAMPLES = 30  # Minimum frames required for double occupancy event
DEFAULT_SPAWNING_EPS = 300  # Maximum gap in frames between mouthing events in spawning bout
DEFAULT_SPAWNING_MIN_SAMPLES = 6  # Minimum mouthing event endpoints required for spawning


# =============================================================================
# 6. VISUALIZATION PARAMETERS
# =============================================================================
DEFAULT_VIZ_FILTERED = True  # Whether to visualize filtered or raw pose predictions
DEFAULT_VIZ_GRID_WIDTH = 8  # Grid width for identity consistency visualization
DEFAULT_VIZ_GRID_HEIGHT = 6  # Grid height for identity consistency visualization


# =============================================================================
# 7. ANALYSIS PARAMETERS
# =============================================================================
DEFAULT_ANALYSIS_BIN_WIDTH = 1800  # Bin width in frames for heatmaps (1800 frames = 60s at 30fps)
