"""
Identity-preserving tracklet stitching.

This module provides stitching that strictly enforces identity constraints
by partitioning tracklets by ID before stitching.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from DeepLabCut.deeplabcut.refine_training_dataset.stitch import TrackletStitcher, Tracklet
from demba.utils.dlc import load_tracklets, split_conjoined_tracklets
from demba.config import (
    DEFAULT_STITCH_N_TRACKS,
    DEFAULT_STITCH_MIN_LENGTH,
    DEFAULT_MIN_CONJOINED_RUN_LENGTH,
    DEFAULT_SPLIT_CONJOINED
)


def calculate_track_purity(stitched_track, assigned_id):
    """
    Calculate track purity: proportion of frames matching the assigned ID.

    Parameters
    ----------
    stitched_track : Tracklet
        The stitched track composed of multiple tracklets
    assigned_id : int
        The ID that was assigned to this track

    Returns
    -------
    float
        Track purity (0.0 to 1.0), or -1 if identity data is not available
    """
    # Check if identity data exists (4th dimension)
    if stitched_track.data.shape[-1] < 4:
        return -1

    # Get all identity values across all frames and bodyparts
    # Shape is (n_frames, n_bodyparts, 4) where [:, :, 3] is identity
    identities = stitched_track.data[:, :, 3]

    # Count frames where the identity matches the assigned ID
    # For each frame, check if any bodypart has the matching identity
    matching_frames = 0
    total_frames = len(stitched_track)

    for frame_identities in identities:
        # Check if this frame's identities match the assigned ID
        # Use the mode (most common) identity in this frame
        frame_ids = frame_identities[~np.isnan(frame_identities)]
        if len(frame_ids) > 0:
            # Get the most common identity in this frame
            unique, counts = np.unique(frame_ids, return_counts=True)
            frame_mode_id = unique[np.argmax(counts)]
            if frame_mode_id == assigned_id:
                matching_frames += 1

    if total_frames == 0:
        return 0.0

    return matching_frames / total_frames


def stitch_by_identity(trial_manager,
                       n_tracks=None,
                       min_length=None,
                       animal_names=None,
                       split_conjoined=None,
                       min_conjoined_run_length=None):
    """
    Stitch tracklets with strict identity preservation.

    Strategy:
    1. Load tracklets with identity labels
    2. Split conjoined tracklets (tracklets that switch between tracking different fish)
    3. Partition tracklets by dominant identity (0, 1, or -1)
    4. Stitch each identity group independently
    5. Assign unidentified tracklets (-1) to closest track
    6. Write final tracks to H5

    Parameters
    ----------
    trial_manager : TrialManager
        TrialManager instance for the trial. Used to resolve tracklet and output paths
        and mark completion status.
    n_tracks : int, optional
        Number of individuals/tracks (should equal number of unique non--1 IDs).
        If None, uses DEFAULT_STITCH_N_TRACKS from config.
    min_length : int, optional
        Minimum tracklet length to include in stitching.
        If None, uses DEFAULT_STITCH_MIN_LENGTH from config.
    animal_names : list of str, optional
        Names for individuals. If None, uses ['individual1', 'individual2', ...]
    split_conjoined : bool, optional
        Whether to split tracklets that switch between tracking different individuals.
        If None, uses DEFAULT_SPLIT_CONJOINED from config.
    min_conjoined_run_length : int, optional
        Minimum consecutive frames of same ID to count as a "real" identity run
        when detecting conjoined tracklets. If None, uses DEFAULT_MIN_CONJOINED_RUN_LENGTH from config.

    Returns
    -------
    dict
        Mapping of identity ID to animal name
    """
    # Get paths from TrialManager
    tracklet_pickle_path = trial_manager.el_pickle_path()
    output_h5_path = trial_manager.stitched_h5_path()

    # Apply config defaults
    if n_tracks is None:
        n_tracks = DEFAULT_STITCH_N_TRACKS
    if min_length is None:
        min_length = DEFAULT_STITCH_MIN_LENGTH
    if split_conjoined is None:
        split_conjoined = DEFAULT_SPLIT_CONJOINED
    if min_conjoined_run_length is None:
        min_conjoined_run_length = DEFAULT_MIN_CONJOINED_RUN_LENGTH

    # Load tracklets
    print(f"Loading tracklets from {tracklet_pickle_path}...")
    tracklets, header = load_tracklets(tracklet_pickle_path)
    print(f"Loaded {len(tracklets)} tracklets")

    # Split conjoined tracklets
    if split_conjoined:
        print(f"\nSplitting conjoined tracklets (min_run_length={min_conjoined_run_length})...")
        tracklets, n_split = split_conjoined_tracklets(
            tracklets,
            min_run_length=min_conjoined_run_length,
            min_tracklet_length=min_length
        )
        if n_split > 0:
            print(f"  Split {n_split} conjoined tracklets")
            print(f"  Total tracklets after splitting: {len(tracklets)}")
        else:
            print(f"  No conjoined tracklets found")

    # Partition by identity
    print("\nPartitioning tracklets by identity...")
    tracklets_by_id = {i: [] for i in range(n_tracks)}
    tracklets_by_id[-1] = []  # For unassigned

    id_counts = {i: 0 for i in range(n_tracks)}
    id_counts[-1] = 0

    for t in tracklets:
        identity = t.identity
        if identity in tracklets_by_id:
            tracklets_by_id[identity].append(t)
            id_counts[identity] += 1

    print("Identity distribution:")
    for identity, count in sorted(id_counts.items()):
        if identity == -1:
            label = "unassigned"
        else:
            label = f"ID {identity}"
        pct = 100 * count / len(tracklets) if len(tracklets) > 0 else 0
        print(f"  {label}: {count} tracklets ({pct:.1f}%)")

    # Check if we have tracklets for each expected identity
    valid_ids = [i for i in range(n_tracks) if id_counts[i] > 0]

    if len(valid_ids) < n_tracks:
        print(f"\nWARNING: Only found {len(valid_ids)} identities but expected {n_tracks}")
        print(f"Valid IDs: {valid_ids}")
        n_tracks = len(valid_ids)

    # Stitch each identity group separately
    print(f"\nStitching {n_tracks} identity groups independently...")
    stitched_tracks = {}

    for identity in valid_ids:
        identity_tracklets = tracklets_by_id[identity]

        if not identity_tracklets:
            print(f"  ID {identity}: No tracklets (skipping)")
            continue

        print(f"  ID {identity}: Stitching {len(identity_tracklets)} tracklets...")

        # Sort tracklets by start time
        identity_tracklets_sorted = sorted(identity_tracklets, key=lambda t: t.start)

        # Simple approach: concatenate all tracklets for this identity
        # This avoids the graph optimization that causes "black magic" failures
        combined_track = identity_tracklets_sorted[0]
        frames_added = len(combined_track)

        for t in identity_tracklets_sorted[1:]:
            if len(t) >= min_length:
                combined_track = combined_track + t
                frames_added += len(t)

        stitched_tracks[identity] = combined_track
        print(f"    -> Combined {len(identity_tracklets)} tracklets into track with {frames_added:,} frames")

    # Handle unassigned tracklets (ID=-1)
    if tracklets_by_id[-1]:
        n_unassigned = len(tracklets_by_id[-1])
        frames_unassigned = sum(len(t) for t in tracklets_by_id[-1])
        total_frames = sum(len(t) for t in tracklets)
        pct_unassigned = 100 * frames_unassigned / total_frames if total_frames > 0 else 0
        print(f"\nDiscarding {n_unassigned} unidentified tracklets ({frames_unassigned} frames, {pct_unassigned:.1f}% of data)")

    # Create animal names
    if animal_names is None:
        animal_names = [f"individual{i+1}" for i in range(n_tracks)]

    # Map identities to animal names
    # Sort by identity ID to ensure consistent ordering
    id_to_name = {}
    for i, identity in enumerate(sorted(stitched_tracks.keys())):
        id_to_name[identity] = animal_names[i] if i < len(animal_names) else f"individual{i+1}"

    print(f"\nIdentity to animal name mapping:")
    for identity, name in sorted(id_to_name.items()):
        print(f"  ID {identity} -> {name}")

    # Write tracks to H5
    print(f"\nWriting tracks to {output_h5_path}...")

    # Create a temporary stitcher just to use its write_tracks method
    # We'll replace its tracks with our identity-separated tracks
    temp_stitcher = TrackletStitcher(
        list(stitched_tracks.values()),
        n_tracks=len(stitched_tracks),
        min_length=min_length
    )
    temp_stitcher.tracks = [stitched_tracks[id_] for id_ in sorted(stitched_tracks.keys())]
    temp_stitcher.header = header
    temp_stitcher.filename = str(tracklet_pickle_path)

    # Write using DeepLabCut's method
    temp_stitcher.write_tracks(
        output_name=str(output_h5_path),
        animal_names=[id_to_name[id_] for id_ in sorted(stitched_tracks.keys())],
        suffix="",
        save_as_csv=True
    )

    print("Identity-preserving stitching complete!")

    # Generate stitching summary
    print("\nGenerating stitching summary...")

    # Calculate track purity for each stitched track
    track_purities = {}
    for identity, track in stitched_tracks.items():
        purity = calculate_track_purity(track, identity)
        track_purities[identity] = purity

    # Calculate total tracklets and percent per ID
    total_tracklets = len(tracklets)
    tracklet_percentages = {}
    for identity in sorted(id_counts.keys()):
        if identity in stitched_tracks or identity == -1:
            pct = 100 * id_counts[identity] / total_tracklets if total_tracklets > 0 else 0
            tracklet_percentages[identity] = pct

    # Get unassigned statistics
    n_unassigned_tracklets = len(tracklets_by_id[-1])
    n_unassigned_frames = sum(len(t) for t in tracklets_by_id[-1])

    # Create summary content
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("TRACKLET STITCHING SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Trial: {trial_manager.trial_dir.name}")
    summary_lines.append("")

    summary_lines.append("-" * 70)
    summary_lines.append("STITCHING PARAMETERS")
    summary_lines.append("-" * 70)
    summary_lines.append(f"Number of tracks (n_tracks):              {n_tracks}")
    summary_lines.append(f"Minimum tracklet length (min_length):     {min_length}")
    summary_lines.append(f"Split conjoined tracklets:                {split_conjoined}")
    summary_lines.append(f"Min conjoined run length:                 {min_conjoined_run_length}")
    summary_lines.append("")

    summary_lines.append("-" * 70)
    summary_lines.append("CONJOINED TRACKLET SPLITTING")
    summary_lines.append("-" * 70)
    if split_conjoined:
        summary_lines.append(f"Number of conjoined tracklets split:      {n_split}")
    else:
        summary_lines.append("Conjoined tracklet splitting was disabled")
    summary_lines.append("")

    summary_lines.append("-" * 70)
    summary_lines.append("TRACKLET DISTRIBUTION BY ID")
    summary_lines.append("-" * 70)
    summary_lines.append(f"Total tracklets processed:                {total_tracklets}")
    summary_lines.append("")
    for identity in sorted([k for k in id_counts.keys() if k != -1]):
        count = id_counts[identity]
        pct = tracklet_percentages.get(identity, 0)
        summary_lines.append(f"  ID {identity}:  {count:4d} tracklets ({pct:5.1f}%)")
    summary_lines.append("")

    summary_lines.append("-" * 70)
    summary_lines.append("UNASSIGNED TRACKLETS (DISCARDED)")
    summary_lines.append("-" * 70)
    summary_lines.append(f"Number of unassigned tracklets:           {n_unassigned_tracklets}")
    summary_lines.append(f"Number of unassigned frames:              {n_unassigned_frames}")
    if total_tracklets > 0:
        unassigned_pct = tracklet_percentages.get(-1, 0)
        summary_lines.append(f"Percent of tracklets unassigned:          {unassigned_pct:.1f}%")
    summary_lines.append("")

    summary_lines.append("-" * 70)
    summary_lines.append("TRACK PURITY (by ID)")
    summary_lines.append("-" * 70)
    summary_lines.append("Track purity = proportion of frames matching assigned ID")
    summary_lines.append("")
    for identity in sorted(stitched_tracks.keys()):
        purity = track_purities[identity]
        if purity >= 0:
            summary_lines.append(f"  ID {identity}:  {purity:.3f} ({purity*100:.1f}%)")
        else:
            summary_lines.append(f"  ID {identity}:  N/A (identity data not available)")

    # Calculate average purity
    valid_purities = [p for p in track_purities.values() if p >= 0]
    if valid_purities:
        avg_purity = np.mean(valid_purities)
        summary_lines.append("")
        summary_lines.append(f"Average track purity:                     {avg_purity:.3f} ({avg_purity*100:.1f}%)")
    summary_lines.append("")
    summary_lines.append("=" * 70)

    # Write summary to file
    summary_dir = trial_manager.id_correction_dir()
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / 'stitching_summary.txt'

    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f"Summary written to: {summary_path}")

    # Mark stage as complete
    trial_manager.mark_stage_complete('tracklet_stitching')

    return id_to_name


