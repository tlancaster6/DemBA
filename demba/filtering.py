"""Temporal filtering of pose predictions."""

import DeepLabCut.deeplabcut as dlc


def filter_predictions(config_path, video_path, shuffle=1):
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
        Integer specifying shuffle index of training dataset (default: 1)

    Returns
    -------
    None
        Saves filtered predictions to *_filtered.h5 and *_filtered.csv files

    Notes
    -----
    Source: DeepLabCut/post_processing/filtering.py
    """
    dlc.filterpredictions(
        config_path,  # Full path of the config.yaml file
        str(video_path),  # Full path of the video to filter predictions for
        shuffle=shuffle,  # Integer specifying shuffle index of training dataset
        filtertype="median",  # Filter type: 'arima', 'median', or 'spline'
        windowlength=5,  # Window size for median filter (should be odd)
        p_bound=0.001,  # Likelihood threshold for ARIMA missing data detection
        ARdegree=3,  # Autoregressive degree for ARIMA model
        MAdegree=1,  # Moving average degree for ARIMA model
        alpha=0.01,  # Significance level for ARIMA outlier detection
        save_as_csv=True,  # Save filtered predictions as .csv file
        track_method='ellipse'  # Tracking method used to generate predictions
    )
