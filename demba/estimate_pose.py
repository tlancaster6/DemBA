import deeplabcut as dlc
from pathlib import Path
import pandas as pd

idx = pd.IndexSlice

def estimate_pose(config_path, video_path, shuffle=1, n_fish=2, visualize=True, skip_tracking=False):
    video_path = Path(video_path)
    print(f'estimating pose for {video_path.name}')
    # Source: DeepLabCut/compat.py
    dlc.analyze_videos(
        config_path,  # Full path of the config.yaml file
        [str(video_path)],  # List of strings containing full paths to videos for analysis
        videotype="",  # Video extension filter (empty = all common extensions)
        shuffle=shuffle,  # Integer specifying shuffle index of training dataset
        trainingsetindex=0,  # Integer specifying which TrainingsetFraction to use
        gputouse=None,  # only applicable to TF backend
        save_as_csv=True,  # Save predictions in .csv file format
        in_random_order=True,  # Whether to analyze videos in random order
        destfolder=None,  # Destination folder for analysis data (None = video path)
        batchsize=None,  # only applicable to TF backend
        cropping=None,  # Cropping coordinates as [x1, x2, y1, y2]
        TFGPUinference=True,  # only applicable to TF backend
        dynamic=(False, 0.5, 10),  # Dynamic cropping: (state, det_threshold, margin). Probably not applicable to multi-animal?
        modelprefix="",  # Directory containing deeplabcut models. leave blank for auto-detection
        robust_nframes=False,  # Robustly evaluate video frame count (slower but robust against mild video corruption)
        allow_growth=False,  # only applicable to TF backend
        use_shelve=False,  # Use persistent database-like object for data storage
        auto_track=False,  # Automatically perform tracking and stitching. Leave as False here because we stitch separately for more control
        n_tracks=n_fish,  # Number of tracks for multi-animal tracking
        animal_names=[f'individual{i+1}' for i in range(n_fish)]  # List of animal names for multi-animal projects
    )
    if visualize:
        # Source: DeepLabCut/utils/make_labeled_video.py
        dlc.create_video_with_all_detections(
            config_path,  # Full path of the config.yaml file
            [str(video_path)],  # List of strings containing full paths to videos
            videotype="",  # Video extension filter (empty = all common extensions)
            shuffle=shuffle,  # Integer specifying shuffle index of training dataset
            trainingsetindex=0,  # Integer specifying which TrainingsetFraction to use
            displayedbodyparts="all",  # Body parts to display ("all" or list of parts)
            cropping=None,  # Cropping coordinates as [x1, x2, y1, y2]
            destfolder=None,  # Destination folder for output (None = video path)
            modelprefix="",  # Directory containing deeplabcut models
            confidence_to_alpha=True,  # Map confidence values to alpha transparency
            plot_bboxes=True  # Plot bounding boxes for detections
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
        trainingsetindex=0,  # Integer specifying which TrainingsetFraction to use
        overwrite=True,  # Overwrite existing tracklet files
        destfolder=None,  # Destination folder for output (None = video path)
        ignore_bodyparts=None,  # Body parts to ignore during tracking
        inferencecfg=None,  # Override Inference configuration dictionary
        modelprefix="",  # Override Directory containing deeplabcut models
        greedy=False,  # only applicable to TF backend. Use greedy tracklet assembly
        calibrate=False,  # only applicable to TF backend. Calibrate tracklet assembly (requires minimal missing data)
        window_size=0,  # only applicable to TF backend. Window size for during tracklet assembly
        identity_only=False,  # Track identity only (no pose). requires a model trained on ID's
        track_method="ellipse"  # Tracking method: 'box', 'skeleton', or 'ellipse'
    )
    print('stiching tracklets')
    try:
        # Source: DeepLabCut/refine_training_dataset/stitch.py
        dlc.stitch_tracklets(
            config_path,  # Full path of the config.yaml file
            [video_path],  # List of paths to videos for tracklet stitching
            videotype="",  # Video extension filter (empty = all common extensions)
            shuffle=shuffle,  # Integer specifying shuffle index of training dataset
            trainingsetindex=0,  # Integer specifying which TrainingsetFraction to use
            n_tracks=n_fish,  # Expected number of tracks/animals
            animal_names= [f'individual{i+1}' for i in range(n_fish)],  # List of animal names for identification
            min_length=10,  # Minimum tracklet length to consider
            split_tracklets=False,  # Whether to split tracklets at gaps
            prestitch_residuals=True,  # Compute residuals before stitching
            max_gap=None,  # Maximum gap size to stitch across. If None, determined at runtime
            weight_func=None,  # Function to weight tracklet connections. If None, defaults to the Tracklet.distance_to() method, which basically calculates the Euclidean head-to-tail distance between two tracklets. If using a custom function, must take two tracklets as arguments and return a scalar that is inversely proportional to the likelihood that two tracklets belong to the same track
            destfolder=None,  # Destination folder for output (None = video path)
            modelprefix="",  # Override Directory containing deeplabcut models
            track_method="ellipse",  # Tracking method: 'box', 'skeleton', or 'ellipse'
            save_as_csv=True # Also save results as a csv
        )
    except Exception as e:
        print(f'stitching failed for {video_path.name} with exception: {e}')
        return
    # Source: DeepLabCut/post_processing/filtering.py
    dlc.filterpredictions(
        config_path,  # Full path of the config.yaml file
        str(video_path),  # Full path of the video to filter predictions for
        videotype="",  # Video extension filter (empty = all common extensions)
        shuffle=1,  # Integer specifying shuffle index of training dataset
        trainingsetindex=0,  # Integer specifying which TrainingsetFraction to use
        filtertype="median",  # Filter type: 'arima', 'median', or 'spline'
        windowlength=5,  # Window size for median filter (should be odd)
        p_bound=0.001,  # Likelihood threshold for ARIMA missing data detection
        ARdegree=3,  # Autoregressive degree for ARIMA model
        MAdegree=1,  # Moving average degree for ARIMA model
        alpha=0.01,  # Significance level for ARIMA outlier detection
        save_as_csv=True,  # Save filtered predictions as .csv file
        destfolder=None,  # Destination folder for output (None = video path)
        modelprefix="",  # Directory containing deeplabcut models
        track_method=""  # Tracking method used to generate predictions
    )
    print(f'analyzed {video_path.name} successfully')
    if visualize:
        print('running visualization')
        # Source: DeepLabCut/utils/make_labeled_video.py
        dlc.create_labeled_video(
            config_path,  # Full path of the config.yaml file
            [str(video_path)],  # List of strings containing full paths to videos
            videotype="",  # Video extension filter (empty = all common extensions)
            shuffle=1,  # Integer specifying shuffle index of training dataset
            trainingsetindex=0,  # Integer specifying which TrainingsetFraction to use
            filtered=True,  # Use filtered predictions (if available)
            fastmode=True,  # Fast mode for video creation (less accurate)
            save_frames=False,  # Save individual labeled frames
            keypoints_only=False,  # Display keypoints only (no skeleton/trails)
            Frames2plot=None,  # List of frame indices to plot (None = all frames)
            displayedbodyparts="all",  # Body parts to display ("all" or list of parts)
            displayedindividuals="all",  # Individuals to display ("all" or list)
            codec="mp4v",  # Video codec for output video
            outputframerate=None,  # Output video framerate (None = same as input)
            destfolder=None,  # Destination folder for output (None = video path)
            draw_skeleton=False,  # Draw skeleton connections between body parts
            trailpoints=0,  # Number of previous points to show as trail (0 = none)
            displaycropped=False,  # Display cropped frames
            color_by="individual",  # Color scheme: 'bodypart' or 'individual'
            modelprefix=""  # Directory containing deeplabcut models
        )
