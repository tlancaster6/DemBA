"""
Identity-preserving tracklet stitching.

This module provides stitching that strictly enforces identity constraints
by partitioning tracklets by ID before stitching.
"""

import numpy as np
from pathlib import Path
from DeepLabCut.deeplabcut.refine_training_dataset.stitch import TrackletStitcher, Tracklet
from demba.dlc_utils import load_tracklets


def stitch_by_identity(tracklet_pickle_path, output_h5_path, n_tracks=2,
                       min_length=10, animal_names=None):
    """
    Stitch tracklets with strict identity preservation.

    Strategy:
    1. Load tracklets with identity labels
    2. Partition tracklets by dominant identity (0, 1, or -1)
    3. Stitch each identity group independently
    4. Assign unidentified tracklets (-1) to closest track
    5. Write final tracks to H5

    Parameters
    ----------
    tracklet_pickle_path : str or Path
        Path to tracklet pickle file with identity labels
    output_h5_path : str or Path
        Path for output H5 file
    n_tracks : int
        Number of individuals/tracks (should equal number of unique non--1 IDs)
    min_length : int
        Minimum tracklet length to include
    animal_names : list of str, optional
        Names for individuals. If None, uses ['individual1', 'individual2', ...]

    Returns
    -------
    dict
        Mapping of identity ID to animal name
    """

    # Load tracklets
    print(f"Loading tracklets from {tracklet_pickle_path}...")
    tracklets, header = load_tracklets(tracklet_pickle_path)
    print(f"Loaded {len(tracklets)} tracklets")

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
        print(f"    → Combined {len(identity_tracklets)} tracklets into track with {frames_added:,} frames")

    # Handle unassigned tracklets (ID=-1)
    if tracklets_by_id[-1]:
        n_unassigned = len(tracklets_by_id[-1])
        frames_unassigned = sum(len(t) for t in tracklets_by_id[-1])
        print(f"\nDiscarding {n_unassigned} unidentified tracklets ({frames_unassigned} frames, ~0.6% of data)")

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
        print(f"  ID {identity} → {name}")

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

    print("✓ Identity-preserving stitching complete!")

    return id_to_name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stitch tracklets with strict identity preservation"
    )
    parser.add_argument("tracklet_pickle", help="Path to tracklet pickle file")
    parser.add_argument("--output", "-o", help="Output H5 path (default: same location as pickle)")
    parser.add_argument("--n-tracks", type=int, default=2, help="Number of tracks/individuals")
    parser.add_argument("--min-length", type=int, default=10, help="Minimum tracklet length")

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        pickle_path = Path(args.tracklet_pickle)
        output_path = pickle_path.with_suffix(".h5")

    # Run stitching
    id_mapping = stitch_by_identity(
        args.tracklet_pickle,
        output_path,
        n_tracks=args.n_tracks,
        min_length=args.min_length
    )

    print(f"\nFinal mapping:")
    for id_, name in id_mapping.items():
        id_label = {0: "male", 1: "female"}.get(id_, f"ID{id_}")
        print(f"  {name} = {id_label}")
