from demba.utils import dlc_utils
import pandas as pd
import numpy as np
idx = pd.IndexSlice


class Track:

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.start, self.stop = self.data.index.min(), self.data.index.max()
        self.individual = data.columns.get_level_values(0)[0]
        self.data = self.data.droplevel('individuals', axis=1)


def load_pose_data(vid_path):
    return dlc_utils.h5_to_df(dlc_utils.locate_filtered_h5(vid_path), drop_likelihoods=True)


def isolate_tracks(pose_df):
    individuals = list(pose_df.columns.levels[0])
    tracks = []
    for individual in individuals:
        sub_df = pose_df.loc[:, idx[individual, :, :]]
        deltas = sub_df.notnull().any(axis=1).astype(int).diff()
        starts = list(deltas[deltas == 1].index)
        stops = list(deltas[deltas == -1].index)
        if sub_df.iloc[0].notnull().any():
            starts = [sub_df.iloc[0].name] + starts
        if sub_df.iloc[-1].notnull().any():
            stops = stops + [sub_df.iloc[-1].name + 1]
        for start, stop in list(zip(starts, stops)):
            tracks.append(Track(sub_df.iloc[start:stop]))
    tracks = sorted(tracks, key=lambda x: x.start)
    return tracks


def summarize_tracks(tracks):
    print(f'number of tracks: {len(tracks)}')
    track_lengths = [len(t.data)/30 for t in tracks]
    print(f'mean track length: {np.mean(track_lengths)} sec')
    print(f'max track length: {np.max(track_lengths)} sec')
    print(f'min track length: {np.min(track_lengths)} sec')
    print(f'median track length: {np.median(track_lengths)} sec')
