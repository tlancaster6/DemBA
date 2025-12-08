"""General utilities for DemBA pipeline.

This module contains helper functions for batch mode processing and other
general utilities that don't fit into more specific modules.
"""

import pickle
from pathlib import Path


def save_batch_metadata(prep_data, output_dir):
    """Save lightweight metadata for batch mode between phases.

    Stores only paths and small objects. Avoids redundant storage of
    tracklets, embeddings, and models which are already saved to disk.

    Parameters
    ----------
    prep_data : dict
        Dictionary returned from train_id_model() containing tracklets,
        embeddings, model, etc.
    output_dir : Path
        ID correction output directory (typically trial_dir/id_correction)

    Returns
    -------
    metadata_path : Path
        Path to saved metadata file
    """
    metadata = {
        'tracklet_path': str(prep_data['tracklet_path']),
        'video_path': str(prep_data['video_path']),
        'output_dir': str(prep_data['output_dir']),
        'kmeans': prep_data['kmeans'],
        'embeddings_path': str(output_dir / 'embeddings.pkl'),
        'model_path': str(output_dir / 'encoder_model.pth'),
        'comparison_video_path': str(prep_data['comparison_video_path']),
    }
    metadata_path = output_dir / 'batch_prep_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    return metadata_path


def load_batch_metadata(output_dir):
    """Load batch metadata from output directory.

    Parameters
    ----------
    output_dir : Path
        ID correction output directory

    Returns
    -------
    metadata : dict or None
        Metadata dictionary, or None if file doesn't exist
    """
    metadata_path = output_dir / 'batch_prep_metadata.pkl'
    if not metadata_path.exists():
        return None
    with open(metadata_path, 'rb') as f:
        return pickle.load(f)


def save_batch_cluster_mapping(cluster_mapping, output_dir):
    """Save cluster mapping for batch mode.

    Parameters
    ----------
    cluster_mapping : dict
        Maps cluster ID to semantic label (e.g., {0: 'male', 1: 'female'})
    output_dir : Path
        ID correction output directory

    Returns
    -------
    mapping_path : Path
        Path to saved mapping file
    """
    mapping_path = output_dir / 'batch_cluster_mapping.pkl'
    with open(mapping_path, 'wb') as f:
        pickle.dump(cluster_mapping, f)
    return mapping_path


def load_batch_cluster_mapping(output_dir):
    """Load cluster mapping from output directory.

    Parameters
    ----------
    output_dir : Path
        ID correction output directory

    Returns
    -------
    cluster_mapping : dict or None
        Cluster mapping dictionary, or None if file doesn't exist
    """
    mapping_path = output_dir / 'batch_cluster_mapping.pkl'
    if not mapping_path.exists():
        return None
    with open(mapping_path, 'rb') as f:
        return pickle.load(f)


def reconstruct_prep_data(metadata, patch_size, padding, conf_threshold, device):
    """Reconstruct prep_data from lightweight metadata by loading from disk.

    Loads tracklets, embeddings, and model from their saved locations.
    Recreates patch_extractor and model objects.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary from load_batch_metadata()
    patch_size : int
        Patch size for PatchExtractor
    padding : int
        Padding for PatchExtractor
    conf_threshold : float
        Confidence threshold for PatchExtractor
    device : str
        Device for model ('cuda' or 'cpu')

    Returns
    -------
    prep_data : dict
        Reconstructed prep_data dictionary matching train_id_model() output
    """
    import torch
    from demba.utils.dlc import load_tracklets
    from demba.identity_correction import SimpleCNN, PatchExtractor

    # Load already-saved data from disk
    tracklets, header = load_tracklets(Path(metadata['tracklet_path']))

    with open(metadata['embeddings_path'], 'rb') as f:
        embeddings = pickle.load(f)

    # Recreate model and load weights
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN()
    model.load_state_dict(torch.load(metadata['model_path'], map_location=device_obj))
    model.to(device_obj)
    model.eval()

    # Recreate patch extractor
    patch_extractor = PatchExtractor(
        Path(metadata['video_path']),
        patch_size=patch_size,
        padding=padding,
        conf_threshold=conf_threshold
    )

    prep_data = {
        'tracklets': tracklets,
        'header': header,
        'embeddings': embeddings,
        'kmeans': metadata['kmeans'],
        'patch_extractor': patch_extractor,
        'model': model,
        'tracklet_path': Path(metadata['tracklet_path']),
        'video_path': Path(metadata['video_path']),
        'output_dir': Path(metadata['output_dir']),
        'comparison_video_path': Path(metadata['comparison_video_path'])
    }
    return prep_data
