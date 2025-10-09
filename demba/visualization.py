"""Visualization functions for pose estimation and behavioral features."""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import DeepLabCut.deeplabcut as dlc
from demba.identity_correction import PatchExtractor
from demba.utils.dlc import load_tracklets, get_identity_confidence


def create_labeled_video(config_path, video_path, shuffle=None, filtered=None):
    """
    Create video with pose estimation overlays showing tracked keypoints and skeletons.

    Generates a video with colored markers for each tracked individual and their
    keypoints, along with skeleton connections between body parts.

    Parameters
    ----------
    config_path : str or Path
        Full path to DeepLabCut config.yaml file
    video_path : str or Path
        Full path to video file
    shuffle : int, optional
        Integer specifying shuffle index of training dataset (default: from config)
    filtered : bool, optional
        Whether to use filtered predictions if available (default: from config)

    Returns
    -------
    None
        Saves labeled video to *_labeled.mp4 file

    Notes
    -----
    Source: DeepLabCut/utils/make_labeled_video.py
    Colors are assigned by individual for multi-animal tracking.
    """
    from demba import config
    if shuffle is None:
        shuffle = config.DEFAULT_SHUFFLE
    if filtered is None:
        filtered = config.DEFAULT_VIZ_FILTERED

    print('generating trajectory visualization')
    dlc.create_labeled_video(
        config_path,  # Full path of the config.yaml file
        [str(video_path)],  # List of strings containing full paths to videos
        shuffle=shuffle,  # Integer specifying shuffle index of training dataset
        filtered=filtered,  # Use filtered predictions (if available)
        fastmode=True,  # Fast mode for video creation (less accurate)
        codec="mp4v",  # Video codec for output video
        draw_skeleton=True,  # Draw skeleton connections between body parts
        color_by="individual",  # Color scheme: 'bodypart' or 'individual'
        track_method=config.DEFAULT_TRACK_METHOD
    )


def create_identity_consistency_grids(
    tracklet_pickle_path,
    video_path,
    grid_width=None,
    grid_height=None,
    patch_size=None,
    padding=None,
    conf_threshold=None
):
    """
    Create visual grids showing sample patches from each identity.

    One patch is shown per tracklet to verify visual consistency within
    each assigned identity. Saves to the id_correction directory.

    Parameters
    ----------
    tracklet_pickle_path : str or Path
        Path to *_el.pickle file with identity assignments
    video_path : str or Path
        Path to video file for extracting visual patches
    grid_width : int, optional
        Number of patches across (default: from config)
    grid_height : int, optional
        Number of patches down (default: from config)
    patch_size : int, optional
        Size of extracted patches (default: from config)
    padding : int, optional
        Padding around keypoints when extracting patches (default: from config)
    conf_threshold : float, optional
        Minimum confidence threshold for keypoints (default: from config)

    Returns
    -------
    output_path : Path
        Path to the saved visualization
    """
    from demba import config
    if grid_width is None:
        grid_width = config.DEFAULT_VIZ_GRID_WIDTH
    if grid_height is None:
        grid_height = config.DEFAULT_VIZ_GRID_HEIGHT
    if patch_size is None:
        patch_size = config.DEFAULT_PATCH_SIZE
    if padding is None:
        padding = config.DEFAULT_PADDING
    if conf_threshold is None:
        conf_threshold = config.DEFAULT_CONF_THRESHOLD

    tracklet_path = Path(tracklet_pickle_path)
    video_path = Path(video_path)

    # Validate paths
    if not tracklet_path.exists():
        raise FileNotFoundError(f"Tracklet pickle not found: {tracklet_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load tracklets using utils function
    print(f"Loading tracklets from {tracklet_path}...")
    tracklets, header = load_tracklets(tracklet_path)

    # Group tracklets by identity
    tracklets_by_identity = {}
    for tracklet in tracklets:
        identity = tracklet.identity
        if identity not in tracklets_by_identity:
            tracklets_by_identity[identity] = []
        tracklets_by_identity[identity].append(tracklet)

    print(f"Found {len(tracklets)} tracklets across {len(tracklets_by_identity)} identities")
    for identity, tlist in tracklets_by_identity.items():
        print(f"  Identity {identity}: {len(tlist)} tracklets")

    # Filter to identities 0 and 1
    if 0 not in tracklets_by_identity or 1 not in tracklets_by_identity:
        raise ValueError("Expected identities 0 and 1 in tracklets")
    # Initialize patch extractor
    print(f"Opening video: {video_path}")
    patch_extractor = PatchExtractor(
        video_path,
        patch_size=patch_size,
        padding=padding,
        conf_threshold=conf_threshold
    )
    patch_extractor.open_video()

    # Extract sample patches for each identity
    patches_by_identity = {}
    n_patches_per_identity = grid_width * grid_height

    try:
        for identity in [0, 1]:
            print(f"\nExtracting patches for Identity {identity}...")
            tracklet_list = tracklets_by_identity[identity]

            # Sample tracklets evenly
            if len(tracklet_list) <= n_patches_per_identity:
                sampled_tracklets = tracklet_list
            else:
                # Sample evenly across the list
                indices = np.linspace(0, len(tracklet_list) - 1, n_patches_per_identity, dtype=int)
                sampled_tracklets = [tracklet_list[i] for i in indices]

            patches = []
            confidences = []
            for tracklet in sampled_tracklets:
                # Extract patch from middle of tracklet
                mid_frame_local = len(tracklet) // 2
                mid_frame_global = tracklet.inds[mid_frame_local]

                # Get keypoints: data has shape (nframes, nbodyparts, 3 or 4)
                # where last dimension is [x, y, likelihood] or [x, y, likelihood, identity]
                keypoints = tracklet.data[mid_frame_local]  # Shape: (nbodyparts, 3 or 4)
                patch = patch_extractor.extract_patch(mid_frame_global, keypoints)

                if patch is not None:
                    patches.append(patch)
                    # Calculate identity confidence for this tracklet
                    confidence = get_identity_confidence(tracklet)
                    confidences.append(confidence)

                if len(patches) >= n_patches_per_identity:
                    break

            patches_by_identity[identity] = patches
            tracklets_by_identity[identity] = {'patches': patches, 'confidences': confidences}
            print(f"  Extracted {len(patches)} patches")

    finally:
        patch_extractor.close_video()

    # Determine output directory
    output_dir = video_path.parent / 'id_correction'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and save separate figure for each identity
    # Figure aspect ratio 4:3
    fig_width = 12
    fig_height = 9  # 4:3 aspect ratio

    colors = {0: '#1f77b4', 1: '#ff7f0e'}  # Blue and orange
    output_paths = []

    for identity in [0, 1]:
        patches = tracklets_by_identity[identity]['patches']
        confidences = tracklets_by_identity[identity]['confidences']

        # Create figure for this identity
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

        # Set title
        ax.set_title(
            f'Identity {identity} - Visual Consistency Check ({len(patches)} tracklets)',
            fontsize=16,
            fontweight='bold',
            color=colors[identity],
            pad=20
        )

        # Turn off axis
        ax.axis('off')

        # Create grid of patches
        if len(patches) == 0:
            ax.text(0.5, 0.5, 'No patches extracted',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14)
        else:
            # Calculate grid dimensions
            n_cols = grid_width
            n_rows = grid_height

            # Create inset axes for each patch
            for idx, (patch, confidence) in enumerate(zip(patches[:grid_width * grid_height],
                                                           confidences[:grid_width * grid_height])):
                row = idx // n_cols
                col = idx % n_cols

                # Calculate position in axes coordinates (0-1)
                # Leave small margins
                margin = 0.01
                cell_width = (1.0 - 2 * margin) / n_cols
                cell_height = (1.0 - 2 * margin) / n_rows

                x_pos = margin + col * cell_width
                y_pos = 1.0 - margin - (row + 1) * cell_height

                # Create inset axes - make slightly smaller than cell for spacing
                inset_margin = 0.02
                inset = ax.inset_axes([
                    x_pos + inset_margin,
                    y_pos + inset_margin,
                    cell_width - 2 * inset_margin,
                    cell_height - 2 * inset_margin
                ])

                # Display patch (convert BGR to RGB)
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                inset.imshow(patch_rgb)
                inset.axis('off')

                # Add confidence text below the patch
                if not np.isnan(confidence):
                    # Color code by confidence level
                    if confidence >= 0.9:
                        conf_color = 'green'
                    elif confidence >= 0.7:
                        conf_color = 'orange'
                    else:
                        conf_color = 'red'

                    inset.text(0.5, -0.1, f'{confidence:.2f}',
                             ha='center', va='top',
                             transform=inset.transAxes,
                             fontsize=8, fontweight='bold',
                             color=conf_color)

        plt.tight_layout()

        # Save figure
        output_path = output_dir / f'identity_{identity}_consistency_check.pdf'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved Identity {identity} visualization to {output_path}")
        output_paths.append(output_path)

    print(f"\nSaved {len(output_paths)} visualizations to {output_dir}")
    return output_paths
