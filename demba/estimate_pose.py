import deeplabcut as dlc
from pathlib import Path
from demba.utils import dlc_utils, gen_utils
import pandas as pd
import cv2
idx = pd.IndexSlice


def analyze_video(config_path, video_path, n_fish=3):
    dlc.analyze_videos(config_path, [video_path], auto_track=False)
    dlc.convert_detections2tracklets(config_path, [video_path], track_method='ellipse')
    while n_fish > 0:
        try:
            dlc.stitch_tracklets(config_path, [video_path], split_tracklets=False, n_tracks=n_fish)
            break
        except ValueError:
            print(f'failed to stitch tracklets with n_fish={n_fish}')
            n_fish -= 1
    if n_fish == 0:
        print('stitching failed')
        return
    fill_gaps(config_path, video_path)
    dlc.filterpredictions(config_path, video_path)
    fix_individual_names(video_path)

def fix_individual_names(video_path):
    h5_path = str(next(Path(video_path).parent.glob('*_filtered.h5')))
    csv_path = h5_path.replace('.h5', '.csv')
    df = pd.read_hdf(h5_path)
    df.rename(columns={'ind1': 'individual1', 'ind2': 'individual2', 'ind3': 'individual3'}, inplace=True)
    df.to_csv(csv_path)
    df.to_hdf(h5_path, "df_with_missing", format="table", mode="w")


def fill_gaps(config_path, video_path):
    tracklet_manager = dlc.refine_training_dataset.tracklets.TrackletManager(config_path, max_gap=5)
    tracklet_manager.load_tracklets_from_hdf(str(next(Path(video_path).parent.glob('*_el.h5'))))
    tracklet_manager.save()
#
# def custom_restich(video_path):
#     y_max, x_max = gen_utils.get_frame_size(video_path)
#     df = dlc_utils.h5_to_df(dlc_utils.locate_filtered_h5(video_path), drop_likelihoods=True)
#     deltas = df.groupby('individuals', axis=1).notnull().any(axis=1).astype(int).diff()
#     starts = deltas[deltas.isin([1]).any(axis=1)].index
#     stops = [x-1 for x in list(deltas[deltas.isin([-1]).any(axis=1)].index)]
#     for start_idx, stop_idx in list(zip(starts, stops):
#
#         if any([(stop_idx - i) < 5 for i in starts]):
#
#
#
#     deltas = df.groupby('individuals', axis=1).any().astype(int).diff()
#     drops = deltas[deltas.isin([-1]).any(axis=1)]
#     pickups = deltas[deltas.isin([1]).any(axis=1)]
#     for drop_idx in drops.index:
#         merge_candidates = [i for i in pickups.index if abs(drop_idx - i) < 5]
#         for mc in merge_candidates:
#             merge_window = df.iloc[min(mc,drop_idx-1):max(mc, drop_idx)+1]
#
#
#
#
#
#     return df, y_max, x_max
#
# def
#
# vid = r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\testclip\testclip.mp4"
# config = r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\config.yaml"