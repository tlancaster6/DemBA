import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import re
from DeepLabCut.deeplabcut.refine_training_dataset.stitch import Tracklet

idx = pd.IndexSlice

def load_bboxes(full_pickle_path):
    """
    load bounding boxes (and scores) from the "..._full.pickle" file
    :param full_pickle_path: path to "_full.pickle" file
    :return: dictionary containing bounding boxes and likelihoods for each frame
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
    """
    load pose data from an h5 file, strip some junk out of the multi-index column names, and (optionally) filter low
    confidence poses (replace with nan)
    :param pose_h5_path: path to pose h5 file
    :param min_likelihood: for poses with likelihoods before this threshold, replace the coordinates and likelihood with nan
    :return: dataframe of poses along with a list of individuals and a list of body-parts
    """
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

def load_tracklets(tracklet_pickle_path):

    def get_frame_ind(s):
        if isinstance(s, str):
            return int(re.findall(r"\d+", s)[0])
        return s

    with open(tracklet_pickle_path, 'rb') as handle:
        data = pickle.load(handle)

    tracklets = []
    header = data.pop("header", None)
    for k, dict_ in data.items():
        try:
            inds, data = zip(*[(get_frame_ind(k), v) for k, v in dict_.items()])
        except ValueError:
            continue
        inds = np.asarray(inds)
        data = np.asarray(data)
        try:
            nrows, ncols = data.shape
            # Detect if data has identity column (4 features) or not (3 features)
            n_features = 4 if ncols % 4 == 0 else 3
            data = data.reshape((nrows, ncols // n_features, n_features))
        except ValueError:
            pass
        tracklets.append(Tracklet(data, inds))
    return tracklets, header


def parse_full_pickle_path(full_pickle_path):
    """
    parse the _full.pickle file path into its constituent parts
    :param full_pickle_path: path to _full.pickle file
    :return: dictionary of parsing results
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
    """
    parse a trial name to infer the group number and whether it is a behave or control trial
    :param name: trial name
    :return: the group number (int) and whether it is a control trial (bool)
    """
    name = str(name).upper()
    for pattern, is_control in [(r'DB(\d+)', False), (r'DC(\d+)', True),
                               (r'BHVE.*?GROUP.*?(\d+)', False), (r'CTRL.*?GROUP.*?(\d+)', True)]:
        if m := re.search(pattern, name):
            return int(m.group(1)), int(is_control)
    return np.nan, np.nan


def get_identity_confidence(tracklet):
    """
    Calculate confidence in tracklet identity assignment based on mode frequency.

    The identity of a tracklet is determined by the mode of predicted identities
    across all frames. This function returns the proportion of frames that agree
    with the mode identity.

    Parameters
    ----------
    tracklet : Tracklet
        Tracklet object with data shape (nframes, nbodyparts, 3 or 4)
        where the 4th column (if present) contains identity predictions

    Returns
    -------
    confidence : float
        Mode frequency ratio in range [0, 1], where 1.0 means all frames
        agree on the identity. Returns np.nan if no identity data available.

    Examples
    --------
    confidence = 0.85  # 85% of frames agree on the assigned identity
    confidence = 1.0   # Perfect agreement across all frames
    confidence = 0.6   # Only 60% agreement - potentially unreliable
    """
    # Check if identity data exists (4th column)
    if tracklet.data.shape[-1] < 4:
        return np.nan

    # Extract identity predictions: shape (nframes, nbodyparts)
    identity_data = tracklet.data[..., 3]

    # Flatten and remove NaN values
    identity_predictions = identity_data.flatten()
    identity_predictions = identity_predictions[~np.isnan(identity_predictions)]

    # Return NaN if no valid predictions
    if len(identity_predictions) == 0:
        return np.nan

    # Get the mode (assigned identity)
    mode_identity = tracklet.identity

    # Calculate proportion of predictions matching the mode
    mode_count = np.sum(identity_predictions == mode_identity)
    total_count = len(identity_predictions)

    confidence = mode_count / total_count

    return confidence
