import deeplabcut as dlc
from pathlib import Path
import pandas as pd

idx = pd.IndexSlice

def estimate_pose(config_path, video_path, n_fish=None, visualize=False):
    video_path = str(video_path)
    dlc.analyze_videos(config_path, [video_path], auto_track=False, robust_nframes=True)
    dlc.convert_detections2tracklets(config_path, [video_path], track_method='ellipse')
    if n_fish is None:
        n_fish = 3
        while n_fish > 0:
            try:
                print(f'attempting stitching with n_tracks={n_fish}')
                dlc.stitch_tracklets(config_path, [video_path], n_tracks=n_fish)
                break
            except ValueError as e:
                print(f'failed to stitch tracklets with n_fish={n_fish}')
                print(e)
                n_fish -= 1
        if n_fish == 0:
            print('stitching failed')
            return
    else:
        dlc.stitch_tracklets(config_path, [video_path], n_tracks=n_fish)
    dlc.filterpredictions(config_path, video_path)
    print(f'analyzed {Path(video_path).name} successfully')
    if visualize:
        print('running visualization')
        dlc.create_labeled_video(config_path, [video_path], color_by='individual', filtered=True)


def delete_outputs(target_dir, keep_pose_data=True):
    target_dir = Path(target_dir)
    targets = ['*_assemblies.pickle', '*_el.h5', '*_el.pickle', '*_filtered.csv', '*_filtered.h5', '*_labeled.mp4',
               '*_framefeatures.csv', '*_clipfeatures.csv', '*_featurevis.mp4', '*_roi.png']
    if not keep_pose_data:
        targets.extend(['*_full.pickle', '*_meta.pickle', '*_full.mp4'])
    for target in targets:
        if list(target_dir.glob(target)):
            list(target_dir.glob(target))[0].unlink()
