"""Stratified tracklet sampling for annotation."""

import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path

from demba import config
from demba.utils.dlc import load_tracklets


def detect_tracklet_context(tracklet, all_tracklets, overlap_threshold=None):
    """
    Determine if tracklet occurs in solo or duo context.

    Parameters
    ----------
    tracklet : Tracklet
        The tracklet to classify
    all_tracklets : list of Tracklet
        All tracklets in the video
    overlap_threshold : float, optional
        Minimum fraction of frames with co-occurrence to classify as 'duo'.
        Defaults to config.DEFAULT_EVAL_CONTEXT_OVERLAP_THRESHOLD

    Returns
    -------
    str : 'solo' or 'duo'

    Algorithm
    ---------
    1. For each other tracklet, calculate frame overlap
    2. Count frames where ANY other tracklet is present
    3. If overlap_fraction >= threshold: 'duo', else 'solo'
    """
    if overlap_threshold is None:
        overlap_threshold = config.DEFAULT_EVAL_CONTEXT_OVERLAP_THRESHOLD

    # Get frame set for this tracklet
    tracklet_frames = set(tracklet.inds)

    # Count overlapping frames with other tracklets
    overlapping_frames = set()

    for other in all_tracklets:
        if other is tracklet:
            continue

        # Find intersection
        other_frames = set(other.inds)
        overlap = tracklet_frames & other_frames
        overlapping_frames.update(overlap)

    # Calculate overlap fraction
    overlap_fraction = len(overlapping_frames) / len(tracklet_frames)

    return 'duo' if overlap_fraction >= overlap_threshold else 'solo'


def calculate_tracklet_silhouettes(embeddings_path):
    """
    Calculate mean silhouette score for each tracklet.

    Parameters
    ----------
    embeddings_path : Path
        Path to id_correction/embeddings.pkl file

    Returns
    -------
    dict : {tracklet_id: mean_silhouette}
    """
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)

    # Group silhouettes by tracklet
    tracklet_sils = defaultdict(list)
    for entry in embeddings:
        tracklet_idx = entry['tracklet_idx']
        sil_score = entry['silhouette_score']
        tracklet_sils[tracklet_idx].append(sil_score)

    # Calculate means
    mean_sils = {
        t_idx: np.mean(scores)
        for t_idx, scores in tracklet_sils.items()
    }

    return mean_sils


def get_predicted_identity(tracklet):
    """
    Extract the predicted identity for a tracklet.

    Parameters
    ----------
    tracklet : Tracklet
        Tracklet object from load_tracklets()

    Returns
    -------
    str or None : 'm', 'f', or None if no valid assignment
    """
    # Use the tracklet's identity property (uses mode internally)
    identity_value = tracklet.identity

    # Map to 'm' or 'f'
    if identity_value == 0:
        return 'm'
    elif identity_value == 1:
        return 'f'
    else:
        return None  # -1 or other unassigned value


def stratified_sample(tracklets, n_samples, duo_weight=None, solo_weight=None,
                     random_seed=None):
    """
    Perform stratified sampling with context weighting.

    Parameters
    ----------
    tracklets : list of dict
        Tracklet metadata with 'stratum' and 'context' keys
    n_samples : int
        Total number to sample
    duo_weight : float, optional
        Weight for duo context. Defaults to config.DEFAULT_EVAL_DUO_WEIGHT
    solo_weight : float, optional
        Weight for solo context. Defaults to config.DEFAULT_EVAL_SOLO_WEIGHT
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    list of dict : Sampled tracklets
    """
    if duo_weight is None:
        duo_weight = config.DEFAULT_EVAL_DUO_WEIGHT
    if solo_weight is None:
        solo_weight = config.DEFAULT_EVAL_SOLO_WEIGHT
    if random_seed is not None:
        np.random.seed(random_seed)

    # Group by stratum
    strata = defaultdict(list)
    for t in tracklets:
        strata[t['stratum']].append(t)

    # Calculate target samples per context
    n_duo_target = int(n_samples * duo_weight)
    n_solo_target = n_samples - n_duo_target

    # Get duo and solo strata
    duo_strata = {k: v for k, v in strata.items() if 'duo' in k}
    solo_strata = {k: v for k, v in strata.items() if 'solo' in k}

    # Sample from duo strata (distribute evenly across low/med/high)
    duo_samples = []
    if len(duo_strata) > 0:
        n_per_duo_stratum = max(1, n_duo_target // len(duo_strata))
        for stratum_name, stratum_tracklets in duo_strata.items():
            n_to_sample = min(n_per_duo_stratum, len(stratum_tracklets))
            if n_to_sample > 0:
                sampled = np.random.choice(
                    len(stratum_tracklets),
                    size=n_to_sample,
                    replace=False
                )
                duo_samples.extend([stratum_tracklets[i] for i in sampled])

    # Sample from solo strata
    solo_samples = []
    if len(solo_strata) > 0:
        n_per_solo_stratum = max(1, n_solo_target // len(solo_strata))
        for stratum_name, stratum_tracklets in solo_strata.items():
            n_to_sample = min(n_per_solo_stratum, len(stratum_tracklets))
            if n_to_sample > 0:
                sampled = np.random.choice(
                    len(stratum_tracklets),
                    size=n_to_sample,
                    replace=False
                )
                solo_samples.extend([stratum_tracklets[i] for i in sampled])

    # Combine
    all_samples = duo_samples + solo_samples

    # If we're short, sample more from largest strata
    if len(all_samples) < n_samples:
        remaining = n_samples - len(all_samples)
        largest_stratum = max(strata.values(), key=len)
        already_sampled = set(t['tracklet_idx'] for t in all_samples)
        available = [t for t in largest_stratum
                    if t['tracklet_idx'] not in already_sampled]

        if len(available) > 0:
            n_to_sample = min(remaining, len(available))
            sampled_indices = np.random.choice(
                len(available),
                size=n_to_sample,
                replace=False
            )
            additional = [available[i] for i in sampled_indices]
            all_samples.extend(additional)

    return all_samples[:n_samples]  # Trim to exact size


def sample_tracklets_stratified(
    trial_manager,
    n_samples=None,
    min_length=None,
    context_weights=None,
    random_seed=None
):
    """
    Sample tracklets using stratified sampling.

    Parameters
    ----------
    trial_manager : TrialManager
        Manages paths for a single trial
    n_samples : int, optional
        Number of tracklets to sample. Defaults to config.DEFAULT_EVAL_N_SAMPLES
    min_length : int, optional
        Minimum tracklet length (frames) for inclusion.
        Defaults to config.DEFAULT_EVAL_MIN_TRACKLET_LENGTH
    context_weights : dict, optional
        Sampling weights for 'duo' and 'solo' contexts.
        Defaults to {'duo': config.DEFAULT_EVAL_DUO_WEIGHT,
                     'solo': config.DEFAULT_EVAL_SOLO_WEIGHT}
    random_seed : int, optional
        Random seed for reproducibility. Defaults to config.DEFAULT_EVAL_RANDOM_SEED

    Returns
    -------
    list of dict
        Each entry contains:
        {
            'tracklet_idx': int,
            'length': int,
            'mean_silhouette': float,
            'predicted_id': str ('m' or 'f'),
            'context': str ('solo' or 'duo'),
            'confidence_stratum': str ('low', 'medium', 'high'),
            'length_stratum': str ('short', 'long'),
            'stratum': str (e.g., 'low_duo'),
            'start_frame': int,
            'end_frame': int
        }

    Algorithm
    ---------
    1. Load tracklets from pickle
    2. Load embeddings for silhouette scores
    3. Filter to tracklets >= min_length
    4. For each tracklet:
       - Calculate length
       - Calculate mean silhouette score
       - Determine context (solo vs duo)
       - Extract predicted identity
    5. Calculate percentiles (p20, p80) for confidence stratification
    6. Assign each tracklet to a stratum:
       - Length: short (<median), long (>=median)
       - Confidence: low (<p20), medium (p20-p80), high (>=p80)
       - Context: solo, duo
    7. Weighted stratified sampling
    8. Return sampled tracklet metadata
    """
    # Set defaults
    if n_samples is None:
        n_samples = config.DEFAULT_EVAL_N_SAMPLES
    if min_length is None:
        min_length = config.DEFAULT_EVAL_MIN_TRACKLET_LENGTH
    if random_seed is None:
        random_seed = config.DEFAULT_EVAL_RANDOM_SEED
    if context_weights is None:
        context_weights = {
            'duo': config.DEFAULT_EVAL_DUO_WEIGHT,
            'solo': config.DEFAULT_EVAL_SOLO_WEIGHT
        }

    # Paths
    tracklet_path = trial_manager.el_pickle_path()
    embeddings_path = trial_manager.id_correction_dir() / 'embeddings.pkl'

    # Load tracklets
    tracklets, header = load_tracklets(tracklet_path)

    # Load embeddings and calculate mean silhouettes
    mean_sils = calculate_tracklet_silhouettes(embeddings_path)

    # Filter by length and build metadata
    valid_tracklets = []

    for t_idx, tracklet in enumerate(tracklets):
        length = len(tracklet)

        if length < min_length:
            continue

        # Get mean silhouette (may be None if no embeddings)
        mean_sil = mean_sils.get(t_idx)
        if mean_sil is None:
            continue  # Skip tracklets without embeddings

        # Get predicted identity
        predicted_id = get_predicted_identity(tracklet)
        if predicted_id is None:
            continue  # Skip tracklets without identity

        # Determine context (solo vs duo)
        context = detect_tracklet_context(tracklet, tracklets)

        valid_tracklets.append({
            'tracklet_idx': t_idx,
            'length': length,
            'mean_silhouette': mean_sil,
            'predicted_id': predicted_id,
            'context': context,
            'start_frame': int(tracklet.start),
            'end_frame': int(tracklet.end)
        })

    if len(valid_tracklets) == 0:
        raise ValueError(
            f"No tracklets meet criteria (>= {min_length} frames with embeddings and identity)\n"
            f"Total tracklets: {len(tracklets)}\n"
            f"Try lowering min_length or check ID correction output"
        )

    # Calculate percentiles for confidence stratification (per-video adaptive)
    sils = np.array([t['mean_silhouette'] for t in valid_tracklets])
    p20 = np.percentile(sils, 20)
    p80 = np.percentile(sils, 80)

    # Calculate median for length stratification
    lengths = np.array([t['length'] for t in valid_tracklets])
    median_length = np.median(lengths)

    # Assign strata
    for t in valid_tracklets:
        # Confidence stratum
        if t['mean_silhouette'] < p20:
            conf_stratum = 'low'
        elif t['mean_silhouette'] < p80:
            conf_stratum = 'medium'
        else:
            conf_stratum = 'high'

        # Length stratum
        if t['length'] < median_length:
            length_stratum = 'short'
        else:
            length_stratum = 'long'

        t['confidence_stratum'] = conf_stratum
        t['length_stratum'] = length_stratum
        t['stratum'] = f"{conf_stratum}_{t['context']}"

    # Stratified sampling
    samples = stratified_sample(
        valid_tracklets,
        n_samples,
        duo_weight=context_weights['duo'],
        solo_weight=context_weights['solo'],
        random_seed=random_seed
    )

    return samples
