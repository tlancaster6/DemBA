import pickle
import pandas as pd
from DeepLabCut import deeplabcut as dlc
idx = pd.IndexSlice

def load_bboxes(full_pickle_path):
    with open(full_pickle_path, 'rb') as handle:
        full_data = pickle.load(handle)
    bbox_dict = {}
    for frame_key, frame_data_dict in full_data.items():
        if 'frame' not in frame_key:
            # skip if the key doesn't contain "frame"
            continue
        bbox_dict[frame_key] = {key: frame_data_dict[key] for key in ['bboxes', 'bbox_scores']}
    return bbox_dict

def load_poses(pose_h5_path):
    pose_df = pd.read_hdf(pose_h5_path)
    scorer = pose_df.columns.get_level_values(0)[0]
    pose_df = pose_df.loc[:, scorer]
    pose_df = pose_df.loc[:, idx[:, :, ('x', 'y', 'likelihood')]]
    pose_df.columns = pose_df.columns.remove_unused_levels()
    pose_df = pose_df.sort_index(axis=1)
    individuals, bodyparts = [list(pose_df.columns.levels[i]) for i in [0, 1]]
    return pose_df, individuals, bodyparts
