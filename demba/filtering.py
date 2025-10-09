"""Temporal filtering of pose predictions."""

import DeepLabCut.deeplabcut as dlc


def filter_predictions(config_path, video_path, shuffle=None):
    """
    Apply temporal filtering to pose predictions to smooth trajectories and remove outliers.

    Uses median filtering to smooth pose estimates across time, helping to remove
    tracking jitter and outliers while preserving true motion.

    Parameters
    ----------
    config_path : str or Path
        Full path to DeepLabCut config.yaml file
    video_path : str or Path
        Full path to video file
    shuffle : int, optional
        Integer specifying shuffle index of training dataset (default: from config)

    Returns
    -------
    None
        Saves filtered predictions to *_filtered.h5 and *_filtered.csv files

    Notes
    -----
    Source: DeepLabCut/post_processing/filtering.py
    """
    from demba import config as demba_config
    if shuffle is None:
        shuffle = demba_config.DEFAULT_SHUFFLE

    dlc.filterpredictions(
        config_path,  # Full path of the config.yaml file
        str(video_path),  # Full path of the video to filter predictions for
        shuffle=shuffle,  # Integer specifying shuffle index of training dataset
        filtertype=demba_config.DEFAULT_FILTER_TYPE,  # Filter type: 'arima', 'median', or 'spline'
        windowlength=demba_config.DEFAULT_FILTER_WINDOW_LENGTH,  # Window size for median filter (should be odd)
        p_bound=demba_config.DEFAULT_FILTER_P_BOUND,  # Likelihood threshold for ARIMA missing data detection
        ARdegree=demba_config.DEFAULT_FILTER_AR_DEGREE,  # Autoregressive degree for ARIMA model
        MAdegree=demba_config.DEFAULT_FILTER_MA_DEGREE,  # Moving average degree for ARIMA model
        alpha=demba_config.DEFAULT_FILTER_ALPHA,  # Significance level for ARIMA outlier detection
        save_as_csv=True,  # Save filtered predictions as .csv file
        track_method=demba_config.DEFAULT_TRACK_METHOD  # Tracking method used to generate predictions
    )
