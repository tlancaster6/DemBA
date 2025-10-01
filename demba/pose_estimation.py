"""Pose estimation pipeline stage."""

from pathlib import Path
import DeepLabCut.deeplabcut as dlc
import DeepLabCut.deeplabcut.pose_estimation_pytorch as pep

print(dlc.__file__)

def estimate_pose(config_path, video_path, shuffle=1, n_fish=2, force_rerun=False):
    """
    Run pose estimation and generate tracklets for a video.

    Parameters
    ----------
    config_path : str or Path
        Full path to DeepLabCut config.yaml file
    video_path : str or Path
        Full path to video file
    shuffle : int, optional
        Shuffle index of training dataset (default: 1)
    n_fish : int, optional
        Number of individuals to track (default: 2)
    force_rerun : bool, optional
        Whether to forcibly re-run pose estimation, even if pose output already exists

    Returns
    -------
    None
        Saves pose predictions to *_full.pickle and tracklets to *_el.pickle
    """
    track_method = 'ellipse'
    video_path = Path(video_path)
    if not force_rerun:
        if list(video_path.parent.glob('*_full.pickle')):
            print(f'{video_path.name} already analyzed, skipping')
            return

    print(f'\n\nestimating pose for {video_path.name}')
    # Source: DeepLabCut/compat.py
    pep.analyze_videos(
        config_path,  # Full path of the config.yaml file
        [str(video_path)],  # List of strings containing full paths to videos for analysis
        videotype="",  # Video extension filter (empty = all common extensions)
        shuffle=shuffle,  # Integer specifying shuffle index of training dataset
        save_as_csv=True,  # Save predictions in .csv file format
        robust_nframes=False,  # Robustly evaluate video frame count (slower but robust against mild video corruption)
        n_tracks=n_fish,  # Number of tracks for multi-animal tracking
        animal_names=[f'individual{i+1}' for i in range(n_fish)],  # List of animal names for multi-animal projects
        auto_track=False, # leave this as False since we want to use the below code for better control
        overwrite=True,
        save_as_df=True, # save the pose predictions (pre-tracking) as an h5 file
        detector_batch_size=4
    )

    print('generating tracklets')
    # Source: DeepLabCut/compat.py
    dlc.convert_detections2tracklets(
        config_path,  # Full path of the config.yaml file
        [str(video_path)],  # List of strings containing full paths to videos
        videotype="",  # Video extension filter (empty = all common extensions)
        shuffle=shuffle,  # Integer specifying shuffle index of training dataset
        overwrite=True,  # Overwrite existing tracklet files
        ignore_bodyparts=None,  # Body parts to ignore during tracking
        track_method=track_method  # Tracking method: 'box', 'skeleton', or 'ellipse'
    )

