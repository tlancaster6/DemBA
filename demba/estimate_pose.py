import deeplabcut as dlc
from pathlib import Path
import pandas as pd

idx = pd.IndexSlice

def estimate_pose(config_path, video_path, shuffle=1, n_fish=2, visualize=True, debug_visualize=False, skip_tracking=False, transreid=True):
    track_method = 'ellipse'
    video_path = Path(video_path)
    print(f'estimating pose for {video_path.name}')
    # Source: DeepLabCut/compat.py
    dlc.analyze_videos(
        config_path,  # Full path of the config.yaml file
        [str(video_path)],  # List of strings containing full paths to videos for analysis
        videotype="",  # Video extension filter (empty = all common extensions)
        shuffle=shuffle,  # Integer specifying shuffle index of training dataset
        save_as_csv=True,  # Save predictions in .csv file format
        robust_nframes=False,  # Robustly evaluate video frame count (slower but robust against mild video corruption)
        n_tracks=n_fish,  # Number of tracks for multi-animal tracking
        animal_names=[f'individual{i+1}' for i in range(n_fish)]  # List of animal names for multi-animal projects
    )
    if debug_visualize:
        # Source: DeepLabCut/utils/make_labeled_video.py
        dlc.create_video_with_all_detections(
            config_path,  # Full path of the config.yaml file
            [str(video_path)],  # List of strings containing full paths to videos
            videotype="",  # Video extension filter (empty = all common extensions)
            shuffle=shuffle,  # Integer specifying shuffle index of training dataset
            confidence_to_alpha=True,  # Map confidence values to alpha transparency
        )
    if skip_tracking:
        return

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

    if transreid:
        print('training and deploying REID transformer')
        dlc.transformer_reID(
            config_path,
            [str(video_path)],
            shuffle=shuffle,
            videotype="mp4",
            n_tracks=n_fish,
            track_method=track_method,
            n_triplets=5000,
            train_epochs=100
        )
        track_method='transformer'
    else:
        print('stiching tracklets')
        try:
            # Source: DeepLabCut/refine_training_dataset/stitch.py
            dlc.stitch_tracklets(
                config_path,  # Full path of the config.yaml file
                [str(video_path)],  # List of paths to videos for tracklet stitching
                videotype="",  # Video extension filter (empty = all common extensions)
                shuffle=shuffle,  # Integer specifying shuffle index of training dataset
                n_tracks=n_fish,  # Expected number of tracks/animals
                animal_names= [f'individual{i+1}' for i in range(n_fish)],  # List of animal names for identification
                min_length=10,  # Minimum tracklet length to consider
                split_tracklets=False,  # Whether to split tracklets at gaps
                prestitch_residuals=True,  # Compute residuals before stitching
                max_gap=None,  # Maximum gap size to stitch across. If None, determined at runtime
                weight_func=None,  # Function to weight tracklet connections. If None, defaults to the Tracklet.distance_to() method, which basically calculates the Euclidean head-to-tail distance between two tracklets. If using a custom function, must take two tracklets as arguments and return a scalar that is inversely proportional to the likelihood that two tracklets belong to the same track
                track_method=track_method,  # Tracking method: 'box', 'skeleton', or 'ellipse'
                save_as_csv=True # Also save results as a csv
            )
        except Exception as e:
            print(f'stitching failed for {video_path.name} with exception: {e}')
            return
    if debug_visualize:
        print('generating unfiltered trajectory visualization')
        dlc.create_labeled_video(
            config_path,  # Full path of the config.yaml file
            [str(video_path)],  # List of strings containing full paths to videos
            shuffle=shuffle,  # Integer specifying shuffle index of training dataset
            filtered=False,  # Use filtered predictions (if available)
            codec="mp4v",  # Video codec for output video
            draw_skeleton=True,  # Draw skeleton connections between body parts
            color_by="individual",  # Color scheme: 'bodypart' or 'individual'
            track_method=track_method
        )

    # Source: DeepLabCut/post_processing/filtering.py
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
        track_method=track_method  # Tracking method used to generate predictions
    )
    if visualize:
        print('generating filtered trajectory visualization')
        # Source: DeepLabCut/utils/make_labeled_video.py
        dlc.create_labeled_video(
            config_path,  # Full path of the config.yaml file
            [str(video_path)],  # List of strings containing full paths to videos
            shuffle=shuffle,  # Integer specifying shuffle index of training dataset
            filtered=True,  # Use filtered predictions (if available)
            fastmode=True,  # Fast mode for video creation (less accurate)
            codec="mp4v",  # Video codec for output video
            draw_skeleton=True,  # Draw skeleton connections between body parts
            color_by="individual",  # Color scheme: 'bodypart' or 'individual'
            track_method=track_method
        )
    print(f'analyzed {video_path.name} successfully')
