import pandas as pd
from pathlib import Path
idx = pd.IndexSlice

def locate_filtered_h5(video_path):
    return str(next(Path(video_path).parent.glob(Path(video_path).stem + '*_filtered.h5')))

def h5_to_df(h5_path, drop_likelihoods=False):
    df = pd.read_hdf(h5_path)
    df = df[df.columns.get_level_values(0)[0]]
    if drop_likelihoods:
        df = df.loc[:, idx[:, :, ('x', 'y')]]
        df.sort_index(axis=1, inplace=True)
    return df

def calc_rmse_per_bp(path_to_error_csv):
    """the input csv is produced after running 'evaluate_network' in the 'evaluation-results' directory. It will be
    called 'dist_x.csv', where x is the number of training iterations"""
    df = pd.read_csv(path_to_error_csv, index_col=0, header=[1, 2, 3])
    df = df.loc[..., idx[:, :, 'rmse']].droplevel(2, axis=1)
    df.columns = df.columns.get_level_values(1)
    df = df.melt()
    return df.groupby('body_parts').agg(['count', 'mean'])

# class Track:
#
#     def __init__(self, data: pd.DataFrame):
#         self.data = data
#         self.start, self.stop = self.data.index.min(), self.data.index.max()
#         self.individual = data.columns.get_level_values(0)[0]
#         self.data = self.data.droplevel('individuals', axis=1)
#
#     def merge_in(self, new_track):
#         self.data = pd.concat((self.data, new_track.data), axis=1).mean(axis=1)
#         self.start, self.stop = self.data.index.min(), self.data.index.max()
#
#
# class TrackManager:
#
#     def __init__(self, video_path):
#         self.video_path = video_path
#         self.original_df = h5_to_df(locate_filtered_h5(video_path), drop_likelihoods=True)
#         self.individuals = list(self.original_df.columns.levels[0])
#         self._extract_tracks()
#
#     def _extract_tracks(self):
#         self.tracks = []
#         for individual in self.individuals:
#             sub_df = self.original_df.loc[:, idx[individual, :, :]]
#             deltas = sub_df.notnull().any(axis=1).astype(int).diff()
#             starts = list(deltas[deltas == 1].index)
#             stops = list(deltas[deltas == -1].index)
#             if sub_df.iloc[0].notnull().any():
#                 starts = [sub_df.iloc[0].name] + starts
#             if sub_df.iloc[-1].notnull().any():
#                 stops = stops + [sub_df.iloc[-1].name + 1]
#             for start, stop in list(zip(starts, stops)):
#                 self.tracks.append(Track(sub_df.iloc[start:stop]))
#         self.tracks = sorted(self.tracks, key=lambda x: x.start)
#
#     def stitch_tracks(self, max_overlap=5):
#         new_tracks = []
#         for i, track1 in enumerate(self.tracks):
#             for track2 in self.tracks[i+1:]:
#             track2_candidates = [t for t in self.tracks if (track1.stop - t.start) < max_overlap]
#             for track2 in self.tracks:
#                 if (:
#
#         pass
#
#     def reduce_ids(self, min_gap=5):
#         pass
#

