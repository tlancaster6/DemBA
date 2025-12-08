"""Temporal filtering of pose predictions."""

import DeepLabCut.deeplabcut as dlc


def filter_predictions(trial_manager):
    """
    Apply temporal filtering to pose predictions to smooth trajectories and remove outliers.

    Uses median filtering to smooth pose estimates across time, helping to remove
    tracking jitter and outliers while preserving true motion.

    Parameters
    ----------
    trial_manager : TrialManager
        TrialManager instance for the trial. Used to resolve config, video paths
        and mark completion status.

    Returns
    -------
    None
        Saves filtered predictions to *_filtered.h5 and *_filtered.csv files

    Notes
    -----
    Source: DeepLabCut/post_processing/filtering.py
    """
    from demba import config as demba_config

    # Get paths from TrialManager
    config_path = trial_manager.config_path
    video_path = trial_manager.video_path()
    shuffle = trial_manager.shuffle

    dlc.filterpredictions(
        str(config_path),  # Full path of the config.yaml file
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

    # Mark stage as complete
    trial_manager.mark_stage_complete('filtering')
