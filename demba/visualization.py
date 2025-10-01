"""Visualization functions for pose estimation and behavioral features."""

import DeepLabCut.deeplabcut as dlc


def create_labeled_video(config_path, video_path, shuffle=1, filtered=True):
    """
    Create video with pose estimation overlays showing tracked keypoints and skeletons.

    Generates a video with colored markers for each tracked individual and their
    keypoints, along with skeleton connections between body parts.

    Parameters
    ----------
    config_path : str or Path
        Full path to DeepLabCut config.yaml file
    video_path : str or Path
        Full path to video file
    shuffle : int, optional
        Integer specifying shuffle index of training dataset (default: 1)
    filtered : bool, optional
        Whether to use filtered predictions if available (default: True)

    Returns
    -------
    None
        Saves labeled video to *_labeled.mp4 file

    Notes
    -----
    Source: DeepLabCut/utils/make_labeled_video.py
    Colors are assigned by individual for multi-animal tracking.
    """
    print('generating trajectory visualization')
    dlc.create_labeled_video(
        config_path,  # Full path of the config.yaml file
        [str(video_path)],  # List of strings containing full paths to videos
        shuffle=shuffle,  # Integer specifying shuffle index of training dataset
        filtered=filtered,  # Use filtered predictions (if available)
        fastmode=True,  # Fast mode for video creation (less accurate)
        codec="mp4v",  # Video codec for output video
        draw_skeleton=True,  # Draw skeleton connections between body parts
        color_by="individual",  # Color scheme: 'bodypart' or 'individual'
        track_method='ellipse'
    )
