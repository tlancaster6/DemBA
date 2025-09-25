import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import re
idx = pd.IndexSlice

def load_bboxes(full_pickle_path):
    """
    load bounding boxes from the "..._full.picle" file
    """
    with open(full_pickle_path, 'rb') as handle:
        full_data = pickle.load(handle)
    bbox_dict = {}
    for frame_key, frame_data_dict in full_data.items():
        if 'frame' not in frame_key:
            # skip if the key doesn't contain "frame"
            continue
        bbox_dict[frame_key] = {key: frame_data_dict[key] for key in ['bboxes', 'bbox_scores']}
    return bbox_dict

def load_poses(pose_h5_path, min_likelihood=0.1):
    if pose_h5_path is None:
        return None, None, None
    pose_df = pd.read_hdf(pose_h5_path)
    scorer = pose_df.columns.get_level_values(0)[0]
    pose_df = pose_df.loc[:, scorer]
    pose_df = pose_df.loc[:, idx[:, :, ('x', 'y', 'likelihood')]]
    pose_df.columns = pose_df.columns.remove_unused_levels()
    pose_df = pose_df.sort_index(axis=1)
    individuals, bodyparts = [list(pose_df.columns.levels[i]) for i in [0, 1]]

    # Filter low likelihood poses
    likelihood_cols = pose_df.columns[pose_df.columns.get_level_values(-1) == 'likelihood']
    low_likelihood_mask = pose_df[likelihood_cols] < min_likelihood

    for col in likelihood_cols:
        individual, bodypart = col[0], col[1]
        x_col = (individual, bodypart, 'x')
        y_col = (individual, bodypart, 'y')

        mask = low_likelihood_mask[col]
        pose_df.loc[mask, col] = np.nan
        pose_df.loc[mask, x_col] = np.nan
        pose_df.loc[mask, y_col] = np.nan

    return pose_df, individuals, bodyparts

def parse_full_pickle_path(full_pickle_path):
    """
    parse the _full.pickle file path into its constituent parts
    """
    full_pickle_path = Path(full_pickle_path)
    full_pickle_name = full_pickle_path.name
    parse_results = {}
    parse_results['video_stem'] = full_pickle_name.split('DLC_')[0]
    parse_results['dlc_scorer'] = full_pickle_name.split(parse_results['video_stem'])[1].split('_full')[0]
    parse_results['shuffle'] = int(full_pickle_name.split('shuffle')[1][0])
    parse_results['parent_dir'] = str(full_pickle_path.parent)
    return parse_results

def parse_trial_name(name):
    name = str(name).upper()
    for pattern, is_control in [(r'DB(\d+)', False), (r'DC(\d+)', True),
                               (r'BHVE.*?GROUP.*?(\d+)', False), (r'CTRL.*?GROUP.*?(\d+)', True)]:
        if m := re.search(pattern, name):
            return (int(m.group(1)), int(is_control))
    return (float('inf'), 2)