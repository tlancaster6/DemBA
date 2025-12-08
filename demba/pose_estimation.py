"""Pose estimation pipeline stage."""

from pathlib import Path
import DeepLabCut.deeplabcut as dlc
import DeepLabCut.deeplabcut.pose_estimation_pytorch as pep
from .file_manager import TrialManager

print(dlc.__file__)


def estimate_pose(trial_manager: TrialManager, n_fish=None, force_rerun=False):
    """
    Run pose estimation and generate tracklets for a video.

    Parameters
    ----------
    trial_manager : TrialManager
        TrialManager instance for the trial
    n_fish : int, optional
        Number of individuals to track (default: from config)
    force_rerun : bool, optional
        Whether to forcibly re-run pose estimation, even if pose output already exists

    Returns
    -------
    None
        Saves pose predictions to *_full.pickle and tracklets to *_el.pickle
        Marks 'pose_estimation' stage as complete in registry
    """
    from . import config as demba_config

    # Load defaults
    if n_fish is None:
        n_fish = demba_config.DEFAULT_N_FISH
    track_method = demba_config.DEFAULT_TRACK_METHOD

    config_path = trial_manager.config_path
    video_path = trial_manager.video_path()
    shuffle = trial_manager.shuffle

    # Check if already completed
    if not force_rerun and trial_manager.is_stage_complete('pose_estimation'):
        print(f'Pose estimation already completed for {video_path.name}. Use force_rerun=True to re-run.')
        return

    # Run pose estimation if needed
    full_pickle = trial_manager.full_pickle_path()
    if not force_rerun and full_pickle.exists():
        print(f'Pose already extracted for {video_path.name}. Skipping.')
    else:
        print(f'\n\nEstimating pose for {video_path.name}')
        # Source: DeepLabCut/compat.py
        pep.analyze_videos(
            str(config_path),  # Full path of the config.yaml file
            [str(video_path)],  # List of strings containing full paths to videos for analysis
            videotype="",  # Video extension filter (empty = all common extensions)
            shuffle=shuffle,  # Integer specifying shuffle index of training dataset
            save_as_csv=True,  # Save predictions in .csv file format
            robust_nframes=False,  # Robustly evaluate video frame count (slower but robust against mild video corruption)
            n_tracks=n_fish,  # Number of tracks for multi-animal tracking
            animal_names=[f'individual{i+1}' for i in range(n_fish)],  # List of animal names for multi-animal projects
            auto_track=False,  # Leave this as False since we want to use the below code for better control
            overwrite=True,
            save_as_df=True,  # Save the pose predictions (pre-tracking) as an h5 file
            detector_batch_size=4
        )

    # Generate tracklets if needed
    el_pickle = trial_manager.el_pickle_path()
    if not force_rerun and el_pickle.exists():
        print(f'Tracklets already extracted for {video_path.name}. Skipping.')
    else:
        print('Generating tracklets')
        # Source: DeepLabCut/compat.py
        dlc.convert_detections2tracklets(
            str(config_path),  # Full path of the config.yaml file
            [str(video_path)],  # List of strings containing full paths to videos
            videotype="",  # Video extension filter (empty = all common extensions)
            shuffle=shuffle,  # Integer specifying shuffle index of training dataset
            overwrite=True,  # Overwrite existing tracklet files
            ignore_bodyparts=None,  # Body parts to ignore during tracking
            track_method=track_method  # Tracking method: 'box', 'skeleton', or 'ellipse'
        )

    # Mark stage as complete
    trial_manager.mark_stage_complete('pose_estimation')
    print(f'Pose estimation complete for {video_path.name}')

