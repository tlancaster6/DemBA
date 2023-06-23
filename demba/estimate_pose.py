import deeplabcut as dlc
from pathlib import Path


def analyze_video(config_path, video_path, n_fish=3):
    dlc.analyze_videos(config_path, [video_path], auto_track=False)
    dlc.convert_detections2tracklets(config_path, [video_path], track_method='ellipse')
    dlc.stitch_tracklets(config_path, [video_path], split_tracklets=False, n_tracks=n_fish)
    fill_gaps(config_path, video_path)
    dlc.filterpredictions(config_path, video_path)


def fill_gaps(config_path, video_path):
    tracklet_manager = dlc.refine_training_dataset.tracklets.TrackletManager(config_path, max_gap=5)
    tracklet_manager.load_tracklets_from_hdf(str(next(Path(video_path).parent.glob('*_el.h5'))))
    tracklet_manager.save()
