"""
Triplet loss-based ID correction for DeepLabCut tracklets.

This script trains a per-video CNN classifier using triplet loss to ensure consistent
identity assignment throughout video sequences. It uses co-occupancy frames (where both
animals are present) to automatically generate training data.
"""

import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
from demba.utils.dlc import load_tracklets
from demba import config


class PatchExtractor:
    """Extract RGB patches from video frames based on keypoint locations."""

    def __init__(self, video_path, patch_size=None, padding=None, conf_threshold=None, min_pts=None):
        """
        Parameters
        ----------
        video_path : str or Path
            Path to video file
        patch_size : int, optional
            Target size for patches (square) (default: from config)
        padding : int, optional
            Fixed padding width around keypoints in pixels (default: from config)
        conf_threshold : float, optional
            Minimum confidence threshold for including keypoints in bbox calculation (default: from config)
        min_pts: int, optional
            Minimum number of valid keypoints for performing bbox calculation (default: 5)
        """
        self.video_path = Path(video_path)
        self.patch_size = patch_size if patch_size is not None else config.DEFAULT_PATCH_SIZE
        self.padding = padding if padding is not None else config.DEFAULT_PADDING
        self.conf_threshold = conf_threshold if conf_threshold is not None else config.DEFAULT_CONF_THRESHOLD
        self.min_pts = min_pts if min_pts is not None else config.DEFAULT_MIN_KEYPOINTS
        self.cap = None

    def open_video(self):
        """Open video capture."""
        if self.cap is None:
            self.cap = cv2.VideoCapture(str(self.video_path))
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video: {self.video_path}")

    def close_video(self):
        """Close video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def compute_bbox(self, keypoints):
        """
        Compute bounding box from keypoint data.

        Parameters
        ----------
        keypoints : ndarray
            Shape (n_bodyparts, 3) with columns [x, y, confidence]

        Returns
        -------
        bbox : tuple
            (min_x, min_y, max_x, max_y) or None if no valid keypoints
        """
        # Filter by confidence and check for NaN
        valid_mask = (keypoints[:, 2] >= self.conf_threshold) & ~np.isnan(keypoints[:, 0])

        if valid_mask.sum() < self.min_pts:
            return None

        valid_kpts = keypoints[valid_mask, :2]

        min_x = np.nanmin(valid_kpts[:, 0])
        min_y = np.nanmin(valid_kpts[:, 1])
        max_x = np.nanmax(valid_kpts[:, 0])
        max_y = np.nanmax(valid_kpts[:, 1])

        # Check if still NaN after filtering
        if np.isnan(min_x) or np.isnan(min_y) or np.isnan(max_x) or np.isnan(max_y):
            return None

        # Add padding
        min_x = max(0, min_x - self.padding)
        min_y = max(0, min_y - self.padding)
        max_x = max_x + self.padding
        max_y = max_y + self.padding

        # Ensure that, after padding, the narrower dimension is at least 4x the padding width. Useful in cases where
        # most keypoints can fall along a single line, and that line happens to be parallel to the x or y axis,
        # resulting in an overly-narrow crop
        width = max_x - min_x
        height = max_y - min_y
        min_dimension = 4 * self.padding

        if width < min_dimension:
            # Expand width symmetrically
            deficit = min_dimension - width
            min_x = max(0, min_x - deficit / 2)
            max_x = max_x + deficit / 2

        if height < min_dimension:
            # Expand height symmetrically
            deficit = min_dimension - height
            min_y = max(0, min_y - deficit / 2)
            max_y = max_y + deficit / 2

        return (int(min_x), int(min_y), int(max_x), int(max_y))

    def extract_patch(self, frame_idx, keypoints):
        """
        Extract and resize patch from video frame.

        Parameters
        ----------
        frame_idx : int
            Frame index to extract
        keypoints : ndarray
            Shape (n_bodyparts, 3) with columns [x, y, confidence]

        Returns
        -------
        patch : ndarray
            RGB patch of shape (patch_size, patch_size, 3) or None if extraction fails
        """
        # Ensure video is open (for multiprocessing workers)
        self.open_video()

        bbox = self.compute_bbox(keypoints)
        if bbox is None:
            return None

        min_x, min_y, max_x, max_y = bbox

        # Read frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Clip bbox to frame boundaries
        h, w = frame.shape[:2]
        min_x = max(0, min(min_x, w))
        min_y = max(0, min(min_y, h))
        max_x = max(0, min(max_x, w))
        max_y = max(0, min(max_y, h))

        if max_x <= min_x or max_y <= min_y:
            return None

        # Extract crop
        crop = frame[min_y:max_y, min_x:max_x]

        # Resize maintaining aspect ratio
        crop_h, crop_w = crop.shape[:2]
        if crop_h > crop_w:
            new_h = self.patch_size
            new_w = int(crop_w * self.patch_size / crop_h)
        else:
            new_w = self.patch_size
            new_h = int(crop_h * self.patch_size / crop_w)

        resized = cv2.resize(crop, (new_w, new_h))

        # Pad to square
        patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        y_offset = (self.patch_size - new_h) // 2
        x_offset = (self.patch_size - new_w) // 2
        patch[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return patch


class CoOccupancyDetector:
    """Detect frames where both individuals are present."""

    def __init__(self, tracklets, min_conf=0.5):
        """
        Parameters
        ----------
        tracklets : list of Tracklet
            List of tracklet objects
        min_conf : float
            Minimum average confidence for a detection to be considered valid
        """
        self.tracklets = tracklets
        self.min_conf = min_conf

    def find_co_occupancy_frames(self, min_overlap_frames=10):
        """
        Find all frames where exactly two tracklets overlap.

        Optionally filters to only return tracklet pairs with significant overlap.

        Parameters
        ----------
        min_overlap_frames : int
            Minimum number of co-occupancy frames required for a tracklet pair to be included.
            Set to 0 to include all co-occupancy frames. Default: 10

        Returns
        -------
        co_occupancy : list of dict
            Each dict contains:
                - 'frame': frame index
                - 'tracklet1': index of first tracklet
                - 'tracklet2': index of second tracklet
                - 'idx1': index within tracklet1.inds
                - 'idx2': index within tracklet2.inds
        """
        from collections import defaultdict

        co_occupancy = []

        # Build frame-to-tracklets mapping
        frame_to_tracklets = {}
        for t_idx, tracklet in enumerate(self.tracklets):
            for local_idx, frame in enumerate(tracklet.inds):
                # Check if this detection has sufficient confidence
                mean_conf = np.nanmean(tracklet.data[local_idx, :, 2])
                if mean_conf >= self.min_conf:
                    if frame not in frame_to_tracklets:
                        frame_to_tracklets[frame] = []
                    frame_to_tracklets[frame].append((t_idx, local_idx))

        # Find frames with exactly 2 tracklets
        for frame, tracklet_list in frame_to_tracklets.items():
            if len(tracklet_list) == 2:
                (t1_idx, local1), (t2_idx, local2) = tracklet_list
                co_occupancy.append({
                    'frame': frame,
                    'tracklet1': t1_idx,
                    'tracklet2': t2_idx,
                    'idx1': local1,
                    'idx2': local2
                })

        # Filter by minimum overlap if specified
        if min_overlap_frames > 0:
            # Count co-occupancy frames per tracklet pair
            pair_overlap_counts = defaultdict(int)
            pair_co_occurrences = defaultdict(list)
            for co_occ in co_occupancy:
                t1, t2 = co_occ['tracklet1'], co_occ['tracklet2']
                pair_key = tuple(sorted([t1, t2]))
                pair_overlap_counts[pair_key] += 1
                pair_co_occurrences[pair_key].append(co_occ)

            # Filter to only pairs with significant overlap
            significant_pairs = {pair for pair, count in pair_overlap_counts.items() if count > min_overlap_frames}
            filtered_co_occupancy = []
            for pair in significant_pairs:
                filtered_co_occupancy.extend(pair_co_occurrences[pair])

            print(f"Co-occupancy filtering: {len(significant_pairs)} tracklet pairs with >{min_overlap_frames} frames of overlap")
            print(f"  Filtered to {len(filtered_co_occupancy)} co-occupancy frames (from {len(co_occupancy)} total)")

            return filtered_co_occupancy

        return co_occupancy


class TripletDataset(Dataset):
    """Dataset for triplet loss training."""

    def __init__(self, tracklets, co_occupancy_frames, patch_extractor, samples_per_epoch=None,
                 min_tracklet_length=10, patch_cache=None):
        """
        Parameters
        ----------
        tracklets : list of Tracklet
            List of tracklet objects
        co_occupancy_frames : list of dict
            Co-occupancy frame information from CoOccupancyDetector
        patch_extractor : PatchExtractor
            Patch extraction object
        samples_per_epoch : int, optional
            Number of triplets to generate per epoch (default: from config)
        min_tracklet_length : int
            Minimum tracklet length to include in training (default: 10)
        patch_cache : dict, optional
            Pre-extracted patch cache mapping (tracklet_idx, local_idx) -> patch array
        """
        self.tracklets = tracklets
        self.co_occupancy_frames = co_occupancy_frames
        self.patch_extractor = patch_extractor
        self.samples_per_epoch = samples_per_epoch if samples_per_epoch is not None else config.DEFAULT_ID_SAMPLES_PER_EPOCH
        self.min_tracklet_length = min_tracklet_length
        self.patch_cache = patch_cache if patch_cache is not None else {}

        # Build index for fast lookup
        self._build_tracklet_index()

    def _build_tracklet_index(self):
        """Build index mapping tracklet idx to frame indices, filtering out short tracklets."""
        self.tracklet_frame_map = {}
        n_filtered = 0
        for t_idx, tracklet in enumerate(self.tracklets):
            if len(tracklet) >= self.min_tracklet_length:
                self.tracklet_frame_map[t_idx] = {
                    'frames': tracklet.inds,
                    'data': tracklet.data
                }
            else:
                n_filtered += 1

        if n_filtered > 0:
            print(f"Filtered out {n_filtered} tracklets shorter than {self.min_tracklet_length} frames from training")

    def _find_nearest_cached_frame(self, tracklet_idx, local_idx):
        """
        Find the nearest cached frame index for a tracklet.

        If the exact local_idx is not in cache, search for the nearest cached frame
        within the same tracklet.

        Parameters
        ----------
        tracklet_idx : int
            Tracklet index
        local_idx : int
            Desired local frame index within tracklet

        Returns
        -------
        nearest_idx : int or None
            Nearest cached local frame index, or None if no cached frames found
        """
        # Check if exact match is in cache
        if (tracklet_idx, local_idx) in self.patch_cache:
            return local_idx

        # Find all cached indices for this tracklet
        tracklet_length = len(self.tracklets[tracklet_idx])
        cached_indices = [i for i in range(tracklet_length)
                         if (tracklet_idx, i) in self.patch_cache]

        if not cached_indices:
            return None

        # Find nearest index
        nearest_idx = min(cached_indices, key=lambda i: abs(i - local_idx))
        return nearest_idx

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        """
        Generate a triplet: (anchor, positive, negative).

        Anchor and positive come from same tracklet (temporal neighbors).
        Negative comes from different tracklet in co-occupancy frame.
        """
        max_attempts = 10
        for _ in range(max_attempts):
            # Sample a co-occupancy frame
            co_occ = np.random.choice(self.co_occupancy_frames)

            # Randomly pick which tracklet is anchor
            if np.random.rand() < 0.5:
                anchor_t_idx = co_occ['tracklet1']
                neg_t_idx = co_occ['tracklet2']
                anchor_local_idx = co_occ['idx1']
                neg_local_idx = co_occ['idx2']
            else:
                anchor_t_idx = co_occ['tracklet2']
                neg_t_idx = co_occ['tracklet1']
                anchor_local_idx = co_occ['idx2']
                neg_local_idx = co_occ['idx1']

            # Skip if anchor or negative tracklet was filtered out
            if anchor_t_idx not in self.tracklet_frame_map or neg_t_idx not in self.tracklet_frame_map:
                continue

            anchor_tracklet = self.tracklets[anchor_t_idx]
            anchor_frame = co_occ['frame']

            # Sample positive from same tracklet (different frame)
            available_frames = [i for i in range(len(anchor_tracklet.inds))
                              if i != anchor_local_idx]
            if not available_frames:
                continue

            pos_local_idx = np.random.choice(available_frames)
            pos_frame = anchor_tracklet.inds[pos_local_idx]

            # Extract patches (from cache if available, otherwise find nearest cached frame)
            # For sparse cache, prefer nearest cached frame over on-the-fly extraction
            anchor_nearest_idx = self._find_nearest_cached_frame(anchor_t_idx, anchor_local_idx)
            pos_nearest_idx = self._find_nearest_cached_frame(anchor_t_idx, pos_local_idx)
            neg_nearest_idx = self._find_nearest_cached_frame(neg_t_idx, neg_local_idx)

            # Skip if we can't find cached patches for all three
            if anchor_nearest_idx is None or pos_nearest_idx is None or neg_nearest_idx is None:
                continue

            # Skip if anchor and positive map to the same cached frame (violates triplet requirement)
            if anchor_nearest_idx == pos_nearest_idx:
                continue

            anchor_patch = self.patch_cache[(anchor_t_idx, anchor_nearest_idx)]
            pos_patch = self.patch_cache[(anchor_t_idx, pos_nearest_idx)]
            neg_patch = self.patch_cache[(neg_t_idx, neg_nearest_idx)]

            if anchor_patch is not None and pos_patch is not None and neg_patch is not None:
                # Convert to tensors and normalize with augmentation
                # Each sample gets independent augmentation for better generalization
                anchor_tensor = self._preprocess(anchor_patch, augment=True)
                pos_tensor = self._preprocess(pos_patch, augment=True)
                neg_tensor = self._preprocess(neg_patch, augment=True)

                return anchor_tensor, pos_tensor, neg_tensor

        # If all attempts fail, return zeros (will be filtered out)
        empty = torch.zeros((3, self.patch_extractor.patch_size, self.patch_extractor.patch_size))
        return empty, empty, empty

    def _preprocess(self, patch, augment=True):
        """Convert patch to tensor and normalize with optional augmentation.

        Parameters
        ----------
        patch : ndarray
            RGB patch of shape (H, W, 3)
        augment : bool
            Whether to apply data augmentation (default: True)

        Returns
        -------
        patch_tensor : torch.Tensor
            Normalized tensor of shape (3, H, W)
        """
        # Convert BGR to RGB
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        patch_norm = patch_rgb.astype(np.float32) / 255.0
        # Convert to CHW format
        patch_tensor = torch.from_numpy(patch_norm).permute(2, 0, 1)

        # Apply augmentation during training
        if augment:
            # Random rotation in 90-degree increments (0, 90, 180, 270 degrees)
            k = np.random.randint(0, 4)  # Number of 90-degree rotations
            if k > 0:
                # torch.rot90 rotates in the plane of the last two dimensions
                patch_tensor = torch.rot90(patch_tensor, k=k, dims=[1, 2])

        return patch_tensor


class SimpleCNN(nn.Module):
    """Simple CNN encoder for embedding extraction."""

    def __init__(self, embedding_dim=None):
        if embedding_dim is None:
            embedding_dim = config.DEFAULT_ID_EMBEDDING_DIM
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 -> 64

            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            # Conv block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8
        )

        self.embedder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.embedder(x)
        # L2 normalize embeddings
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class TripletLoss(nn.Module):
    """Triplet loss with margin."""

    def __init__(self, margin=None):
        if margin is None:
            margin = config.DEFAULT_ID_TRIPLET_MARGIN
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


def build_patch_cache(tracklets, co_occupancy_frames, patch_extractor, cache_path, frame_stride=None):
    """
    Pre-extract and cache patches for all detections needed during training.

    Saves cache to disk as a pickle file. If cache file already exists, loads from disk.

    Caches patches for:
    1. All detections in co-occupancy frames (anchor/negative candidates)
    2. All detections in tracklets that appear in co-occupancy frames (positive candidates)

    Parameters
    ----------
    tracklets : list of Tracklet
        List of tracklet objects
    co_occupancy_frames : list of dict
        Co-occupancy frame information
    patch_extractor : PatchExtractor
        Patch extraction object
    cache_path : Path or str
        Path to save/load cache file
    frame_stride : int, optional
        Sample every Nth frame from tracklets (default: from config)
        Higher values reduce cache size but may reduce training diversity

    Returns
    -------
    patch_cache : dict
        Dictionary mapping (tracklet_idx, local_idx) -> patch array (uint8)
    """
    import pickle
    from pathlib import Path

    cache_path = Path(cache_path)

    if frame_stride is None:
        frame_stride = config.DEFAULT_ID_CACHE_FRAME_STRIDE

    # Load from disk if exists
    if cache_path.exists():
        print(f"Loading patch cache from {cache_path}...")
        with open(cache_path, 'rb') as f:
            patch_cache = pickle.load(f)
        print(f"Loaded {len(patch_cache)} patches from cache")
        return patch_cache

    # Build cache
    print(f"Building patch cache (frame_stride={frame_stride})...")
    patch_extractor.open_video()

    # Collect all (tracklet_idx, local_idx) pairs we need
    # Cache frames from tracklets involved in the provided co-occupancy frames, sampled with stride
    tracklets_in_co_occ = set()
    for co_occ in co_occupancy_frames:
        tracklets_in_co_occ.add(co_occ['tracklet1'])
        tracklets_in_co_occ.add(co_occ['tracklet2'])

    needed_patches = set()
    for t_idx in tracklets_in_co_occ:
        tracklet = tracklets[t_idx]
        # Sample every frame_stride frames
        for local_idx in range(0, len(tracklet), frame_stride):
            needed_patches.add((t_idx, local_idx))

    # Calculate percentage of all possible patches
    total_possible_patches = sum(len(tracklet) for tracklet in tracklets)
    cache_percentage = (len(needed_patches) / total_possible_patches) * 100
    print(f"Extracting {len(needed_patches)} unique patches ({cache_percentage:.1f}% of all {total_possible_patches} possible patches)...")

    # Extract all patches
    patch_cache = {}
    failed_count = 0

    for t_idx, local_idx in tqdm(needed_patches, desc="Caching patches"):
        tracklet = tracklets[t_idx]
        frame = tracklet.inds[local_idx]
        keypoints = tracklet.data[local_idx]

        patch = patch_extractor.extract_patch(frame, keypoints)
        if patch is not None:
            patch_cache[(t_idx, local_idx)] = patch
        else:
            failed_count += 1

    patch_extractor.close_video()

    # Calculate cache size
    if patch_cache:
        sample_patch = next(iter(patch_cache.values()))
        bytes_per_patch = sample_patch.nbytes
        total_mb = (len(patch_cache) * bytes_per_patch) / (1024**2)
        print(f"Cache built: {len(patch_cache)} patches, {total_mb:.1f} MB")
        if failed_count > 0:
            print(f"  ({failed_count} patches failed to extract)")

    # Save to disk
    print(f"Saving cache to {cache_path}...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(patch_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cache saved ({cache_path.stat().st_size / (1024**2):.1f} MB on disk)")

    return patch_cache


def train_encoder(tracklets, co_occupancy_frames, patch_extractor, output_dir,
                 n_epochs=200, batch_size=32, lr=0.001, device='cuda', min_tracklet_length=60,
                 frame_stride=5, warmup_epochs=10):
    """
    Train CNN encoder with triplet loss.

    Parameters
    ----------
    tracklets : list of Tracklet
        List of tracklet objects
    co_occupancy_frames : list of dict
        Co-occupancy frame information
    patch_extractor : PatchExtractor
        Patch extraction object
    output_dir : Path
        Directory to save model weights
    n_epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    lr : float
        Learning rate (target learning rate after warmup)
    device : str
        'cuda' or 'cpu'
    min_tracklet_length : int
        Minimum tracklet length to include in training (default: 10)
    frame_stride : int, optional
        Sample every Nth frame for patch cache (default: from config)
    warmup_epochs : int, optional
        Number of epochs for learning rate warmup (default: 10)

    Returns
    -------
    model : SimpleCNN
        Trained model
    """
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    criterion = TripletLoss()

    # Start with a small learning rate for warmup
    initial_lr = lr * 0.1  # Start at 10% of target learning rate
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    # Learning rate scheduler - reduces LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )

    # Build patch cache to speed up training
    cache_path = output_dir / "patch_cache.pkl"
    patch_cache = build_patch_cache(tracklets, co_occupancy_frames, patch_extractor, cache_path,
                                   frame_stride=frame_stride)

    dataset = TripletDataset(tracklets, co_occupancy_frames, patch_extractor,
                            samples_per_epoch=None, min_tracklet_length=min_tracklet_length,
                            patch_cache=patch_cache)

    # Use num_workers from config (0 on Windows to avoid multiprocessing overhead with DLC imports)
    num_workers = config.DEFAULT_ID_NUM_WORKERS
    persistent = num_workers > 0  # Only use persistent workers if we have workers
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=True, persistent_workers=persistent)

    # Training loop
    model.train()
    losses = []
    learning_rates = []

    # Early stopping parameters
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 20  # Stop if no improvement for 10 epochs
    min_delta = 1e-4  # Minimum change to qualify as improvement

    for epoch in range(n_epochs):
        # Warmup phase: linearly increase learning rate
        if epoch < warmup_epochs:
            # Linear warmup from initial_lr to target lr
            warmup_factor = (epoch + 1) / warmup_epochs
            current_warmup_lr = initial_lr + (lr - initial_lr) * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_warmup_lr
            print(f"Warmup phase: Epoch {epoch+1}/{warmup_epochs}, LR: {current_warmup_lr:.6f}")

        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")

        for anchor, pos, neg in pbar:
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            # Skip invalid batches
            if torch.sum(anchor) == 0:
                continue

            optimizer.zero_grad()

            anchor_emb = model(anchor)
            pos_emb = model(pos)
            neg_emb = model(neg)

            loss = criterion(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        losses.append(avg_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        # Only step the scheduler after warmup is complete
        if epoch >= warmup_epochs:
            scheduler.step(avg_loss)

        # Early stopping check
        if avg_loss < best_loss - min_delta:
            # Significant improvement
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  New best loss: {best_loss:.4f}")
        else:
            # No improvement
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epoch(s)")

            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best loss: {best_loss:.4f}")
                # Restore best model
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print("Restored best model weights")
                break

    # Video handles are managed by worker processes and will be cleaned up automatically

    # Save model
    model_path = output_dir / 'encoder_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot loss curve and learning rate
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Loss plot
    ax1.plot(losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Triplet Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Learning rate plot
    ax2.plot(learning_rates, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')  # Log scale for better visibility
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_loss.png', dpi=150)
    plt.close()

    return model


def extract_all_embeddings(model, tracklets, patch_extractor, device='cuda'):
    """
    Extract embeddings for all detections in all tracklets.

    Parameters
    ----------
    model : SimpleCNN
        Trained encoder model
    tracklets : list of Tracklet
        List of tracklet objects
    patch_extractor : PatchExtractor
        Patch extraction object
    device : str
        'cuda' or 'cpu'

    Returns
    -------
    embeddings : list of dict
        Each dict contains:
            - 'tracklet_idx': int
            - 'frame': int
            - 'local_idx': int
            - 'embedding': ndarray
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval()

    patch_extractor.open_video()

    embeddings = []

    with torch.no_grad():
        for t_idx, tracklet in enumerate(tqdm(tracklets, desc="Extracting embeddings")):
            for local_idx, frame in enumerate(tracklet.inds):
                keypoints = tracklet.data[local_idx]
                patch = patch_extractor.extract_patch(frame, keypoints)

                if patch is not None:
                    # Preprocess
                    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                    patch_norm = patch_rgb.astype(np.float32) / 255.0
                    patch_tensor = torch.from_numpy(patch_norm).permute(2, 0, 1)
                    patch_tensor = patch_tensor.unsqueeze(0).to(device)

                    # Extract embedding
                    embedding = model(patch_tensor).cpu().numpy()[0]

                    embeddings.append({
                        'tracklet_idx': t_idx,
                        'frame': frame,
                        'local_idx': local_idx,
                        'embedding': embedding
                    })

    patch_extractor.close_video()

    return embeddings


def cluster_and_assign_ids(embeddings, n_clusters=2):
    """
    Cluster embeddings into n_clusters groups and compute silhouette scores.

    Parameters
    ----------
    embeddings : list of dict
        Embedding information from extract_all_embeddings
    n_clusters : int
        Number of clusters (individuals)

    Returns
    -------
    embeddings : list of dict
        Input embeddings with added 'cluster' and 'silhouette_score' fields
    kmeans : KMeans
        Fitted KMeans model (returned for visualization)
    """
    from sklearn.metrics import silhouette_samples

    # Extract embedding vectors
    emb_vectors = np.array([e['embedding'] for e in embeddings])

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(emb_vectors)

    # Compute silhouette scores for each sample
    silhouette_scores = silhouette_samples(emb_vectors, clusters)

    # Add cluster labels and silhouette scores
    for i, (cluster_id, sil_score) in enumerate(zip(clusters, silhouette_scores)):
        embeddings[i]['cluster'] = int(cluster_id)
        embeddings[i]['silhouette_score'] = float(sil_score)

    # Print statistics
    print(f"Silhouette scores - Mean: {silhouette_scores.mean():.3f}, "
          f"Min: {silhouette_scores.min():.3f}, Max: {silhouette_scores.max():.3f}")
    print(f"Negative scores: {(silhouette_scores < 0).sum()} / {len(silhouette_scores)} "
          f"({100 * (silhouette_scores < 0).sum() / len(silhouette_scores):.1f}%)")

    return embeddings, kmeans


def compute_segment_co_occupancy_ratio(frame_start, frame_end, tracklet_idx, tracklets):
    """
    Compute the ratio of frames in a segment where 2 animals are in frame.

    Parameters
    ----------
    frame_start : int
        Starting frame index of the segment
    frame_end : int
        Ending frame index of the segment
    tracklet_idx : int
        Index of the tracklet being evaluated
    tracklets : list of Tracklet
        All tracklets in the video

    Returns
    -------
    float
        Ratio of frames with exactly 2 tracklets present (0.0 to 1.0)
    """
    # Build frame-to-tracklets mapping for the segment range
    frame_to_tracklets = {}
    for t_idx, tracklet in enumerate(tracklets):
        for frame in tracklet.inds:
            if frame_start <= frame <= frame_end:
                if frame not in frame_to_tracklets:
                    frame_to_tracklets[frame] = []
                frame_to_tracklets[frame].append(t_idx)

    # Count frames with exactly 2 tracklets
    segment_frames = set(range(frame_start, frame_end + 1))
    segment_frames_in_tracklet = set(tracklets[tracklet_idx].inds) & segment_frames

    if len(segment_frames_in_tracklet) == 0:
        return 0.0

    co_occupancy_count = sum(
        1 for frame in segment_frames_in_tracklet
        if frame in frame_to_tracklets and len(frame_to_tracklets[frame]) == 2
    )

    return co_occupancy_count / len(segment_frames_in_tracklet)


def prepare_cluster_comparison_video(embeddings, tracklets, patch_extractor,
                                    n_segments=None, segment_duration_sec=None, fps=None):
    """
    Create a video showing trajectory segments from each cluster (non-interactive preparation step).

    This function performs all the heavy video processing work upfront, before user interaction.
    It should be called during the model training phase (Phase 1 in batch mode).

    Parameters
    ----------
    embeddings : list of dict
        Embedding information with cluster labels
    tracklets : list of Tracklet
        List of tracklet objects
    patch_extractor : PatchExtractor
        Patch extraction object
    n_segments : int, optional
        Number of trajectory segments to show per cluster (default: from config)
    segment_duration_sec : float, optional
        Duration of each segment in seconds (default: from config)
    fps : int, optional
        Frames per second of output video (default: from config)

    Returns
    -------
    output_path : Path
        Path to the created comparison video
    """
    if n_segments is None:
        n_segments = config.DEFAULT_ID_N_SEGMENTS
    if segment_duration_sec is None:
        segment_duration_sec = config.DEFAULT_ID_SEGMENT_DURATION_SEC
    if fps is None:
        fps = config.VIDEO_FPS

    # Group embeddings by (tracklet_idx, cluster)
    tracklet_clusters = {}
    for emb in embeddings:
        t_idx = emb['tracklet_idx']
        cluster = emb['cluster']
        if t_idx not in tracklet_clusters:
            tracklet_clusters[t_idx] = []
        tracklet_clusters[t_idx].append(cluster)

    # Find tracklets predominantly belonging to each cluster
    cluster_tracklets = {0: [], 1: []}
    for t_idx, clusters in tracklet_clusters.items():
        if len(clusters) < 10:  # Skip very short tracklets
            continue
        # Count cluster assignments
        cluster_counts = {0: clusters.count(0), 1: clusters.count(1)}
        total = sum(cluster_counts.values())

        # Assign to cluster if >70% of frames belong to it
        for cluster_id in [0, 1]:
            if cluster_counts[cluster_id] / total > 0.7:
                tracklet = tracklets[t_idx]
                frame_start = tracklet.inds[0]
                frame_end = tracklet.inds[-1]

                # Compute co-occupancy ratio for this tracklet
                co_occupancy_ratio = compute_segment_co_occupancy_ratio(
                    frame_start, frame_end, t_idx, tracklets
                )

                cluster_tracklets[cluster_id].append({
                    'tracklet_idx': t_idx,
                    'purity': cluster_counts[cluster_id] / total,
                    'length': len(tracklets[t_idx]),
                    'co_occupancy_ratio': co_occupancy_ratio
                })

    # Compute composite score with equal weighting
    for cluster_id in [0, 1]:
        tracklet_list = cluster_tracklets[cluster_id]

        if not tracklet_list:
            continue

        # Normalize length to [0, 1] range
        max_length = max(x['length'] for x in tracklet_list)
        min_length = min(x['length'] for x in tracklet_list)
        length_range = max_length - min_length if max_length > min_length else 1

        # Compute composite score: equal weights for purity, co_occupancy, and normalized length
        for x in tracklet_list:
            norm_length = (x['length'] - min_length) / length_range
            x['score'] = (x['purity'] + x['co_occupancy_ratio'] + norm_length) / 3.0

        # Sort by composite score
        tracklet_list.sort(key=lambda x: x['score'], reverse=True)

    # Sample segments from top tracklets
    segment_frames = int(segment_duration_sec * fps)
    cluster_segments = {}

    for cluster_id in [0, 1]:
        segments = []
        for tracklet_info in cluster_tracklets[cluster_id][:n_segments]:
            t_idx = tracklet_info['tracklet_idx']
            tracklet = tracklets[t_idx]

            # Find a good continuous segment
            if len(tracklet) > segment_frames:
                # Sample from middle of tracklet
                start_idx = (len(tracklet) - segment_frames) // 2
                # Get actual frame indices (only include frames that exist in tracklet)
                segment_inds = tracklet.inds[start_idx:start_idx + segment_frames]
                frame_start = segment_inds[0]
                frame_end = segment_inds[-1]
            else:
                # Use entire tracklet
                frame_start = tracklet.inds[0]
                frame_end = tracklet.inds[-1]

            segments.append({
                'tracklet_idx': t_idx,
                'frame_start': frame_start,
                'frame_end': frame_end,
                'frame_indices': tracklet.inds[
                    np.where((tracklet.inds >= frame_start) & (tracklet.inds <= frame_end))[0]
                ]
            })

            if len(segments) >= n_segments:
                break

        cluster_segments[cluster_id] = segments

    # Create comparison video
    output_path = patch_extractor.video_path.parent / 'id_correction' / 'cluster_comparison.mp4'
    patch_extractor.open_video()

    # Get video properties
    cap = patch_extractor.cap
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create side-by-side video
    out_width = frame_width * 2
    out_height = frame_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))

    print(f"\nCreating cluster comparison video...")

    # Process each segment pair
    max_segments = max(len(cluster_segments[0]), len(cluster_segments[1]))

    for seg_idx in range(max_segments):
        print(f"  Segment {seg_idx + 1}/{max_segments}")

        # Get segments for both clusters (or None if exhausted)
        segs = {}
        for cluster_id in [0, 1]:
            if seg_idx < len(cluster_segments[cluster_id]):
                segs[cluster_id] = cluster_segments[cluster_id][seg_idx]
            else:
                segs[cluster_id] = None

        if all(s is None for s in segs.values()):
            continue

        # Get all unique frame indices across both segments
        all_frames = set()
        for cluster_id in [0, 1]:
            if segs[cluster_id] is not None:
                all_frames.update(segs[cluster_id]['frame_indices'])

        if not all_frames:
            continue

        # Process only frames that exist in at least one tracklet
        for frame_idx in sorted(all_frames):
            frames_to_draw = {}

            for cluster_id in [0, 1]:
                if segs[cluster_id] is None or frame_idx not in segs[cluster_id]['frame_indices']:
                    # Black frame with label
                    frames_to_draw[cluster_id] = np.zeros((frame_height, frame_width, 3),
                                                          dtype=np.uint8)
                else:
                    seg = segs[cluster_id]

                    # Read frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()

                    if not ret:
                        frames_to_draw[cluster_id] = np.zeros((frame_height, frame_width, 3),
                                                              dtype=np.uint8)
                    else:
                        # Get tracklet data for this frame
                        tracklet = tracklets[seg['tracklet_idx']]
                        local_idx = np.where(tracklet.inds == frame_idx)[0]

                        if len(local_idx) > 0:
                            keypoints = tracklet.data[local_idx[0]]

                            # Draw bounding box
                            bbox = patch_extractor.compute_bbox(keypoints)
                            if bbox is not None:
                                min_x, min_y, max_x, max_y = bbox
                                color = (0, 255, 0)  # Green
                                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)

                        frames_to_draw[cluster_id] = frame

            # Add labels
            for cluster_id in [0, 1]:
                frame = frames_to_draw[cluster_id]
                label = f"CLUSTER {cluster_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2
                thickness = 4

                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Draw background rectangle
                padding = 10
                cv2.rectangle(frame,
                            (10, 10),
                            (text_width + 20 + padding, text_height + 20 + padding),
                            (0, 0, 0), -1)

                # Draw text
                cv2.putText(frame, label, (20, text_height + 20),
                          font, font_scale, (255, 255, 255), thickness)

            # Concatenate side by side
            combined = np.hstack([frames_to_draw[0], frames_to_draw[1]])
            out.write(combined)

    out.release()
    patch_extractor.close_video()

    print(f"\nCluster comparison video saved to: {output_path}")

    return output_path


def interactive_cluster_mapping(output_path):
    """
    Prompt user to map clusters to biological sex (interactive input only).

    This function only handles the user input portion, making it fast and suitable
    for batched interaction. The video should already be created using
    prepare_cluster_comparison_video().

    Parameters
    ----------
    output_path : Path
        Path to the cluster comparison video created by prepare_cluster_comparison_video()

    Returns
    -------
    mapping : dict
        Maps cluster ID to semantic label (e.g., {0: 'male', 1: 'female'})
    """
    print("\n" + "=" * 60)
    print("Cluster Mapping")
    print("=" * 60)
    print(f"\nVideo location: {output_path}")
    print("Left panel = Cluster 0, Right panel = Cluster 1")
    print("Please review the video to identify which cluster is male/female.\n")

    mapping = {}
    for cluster_id in [0, 1]:
        while True:
            label = input(f"Cluster {cluster_id} is (male/female): ").strip().lower()
            if label in ['male', 'female']:
                mapping[cluster_id] = label
                break
            else:
                print("Invalid input. Please enter 'male' or 'female'.")

    return mapping


def reassign_tracklet_ids(tracklets, embeddings, cluster_mapping, output_dir,
                          original_pickle_path, header, min_silhouette=0.2):
    # Ensure paths are absolute
    original_pickle_path = Path(original_pickle_path).resolve()
    output_dir = Path(output_dir).resolve()
    """
    Reassign IDs in tracklets based on cluster assignments and save in original format.
    Backs up original file and replaces it with corrected version.

    Parameters
    ----------
    tracklets : list of Tracklet
        Original tracklets
    embeddings : list of dict
        Embedding information with cluster labels and silhouette scores
    cluster_mapping : dict
        Maps cluster ID to semantic label (male=0, female=1)
    output_dir : Path
        Directory to save backup
    original_pickle_path : Path
        Path to original pickle file (will be replaced)
    header : DataFrame
        Header from original pickle
    min_silhouette : float
        Minimum silhouette score to assign ID (otherwise keep as -1).
        Recommended: 0.2 (moderate confidence), 0.0 (any positive assignment),
        0.5 (high confidence only). Default: 0.2

    Returns
    -------
    corrected_data : dict
        Dictionary in original pickle format with updated IDs
    """
    import shutil
    from datetime import datetime

    # Map semantic labels to numeric IDs
    label_to_id = {'male': 0, 'female': 1}

    # Build mapping from (tracklet_idx, local_idx) to cluster and silhouette
    tracklet_frame_to_cluster = {}
    for emb in embeddings:
        key = (emb['tracklet_idx'], emb['local_idx'])
        tracklet_frame_to_cluster[key] = {
            'cluster': emb['cluster'],
            'silhouette': emb['silhouette_score']
        }

    # Create corrected data structure matching original format
    corrected_data = {'header': header}

    # Counters for statistics
    n_assigned = {'male': 0, 'female': 0}
    n_low_confidence = 0
    n_no_embedding = 0

    # Process each tracklet
    for t_idx, tracklet in enumerate(tracklets):
        tracklet_dict = {}

        for local_idx, frame in enumerate(tracklet.inds):
            # Copy original data (shape: n_bodyparts x 4, where last col is ID)
            frame_data = tracklet.data[local_idx].copy()

            # Ensure data has ID column
            if frame_data.shape[1] == 3:
                id_col = np.full((frame_data.shape[0], 1), -1.0)
                frame_data = np.hstack([frame_data, id_col])

            # Check if we have an embedding for this detection
            key = (t_idx, local_idx)
            if key in tracklet_frame_to_cluster:
                cluster_info = tracklet_frame_to_cluster[key]
                cluster = cluster_info['cluster']
                silhouette = cluster_info['silhouette']

                # Check silhouette score threshold
                if silhouette >= min_silhouette:
                    # Assign ID based on cluster
                    semantic_label = cluster_mapping[cluster]
                    numeric_id = label_to_id[semantic_label]
                    frame_data[:, 3] = numeric_id
                    n_assigned[semantic_label] += 1
                else:
                    # Low confidence - keep as -1
                    frame_data[:, 3] = -1.0
                    n_low_confidence += 1
            else:
                # No embedding - keep as -1
                frame_data[:, 3] = -1.0
                n_no_embedding += 1

            # Store with frame key
            frame_key = f"frame{frame:06d}"
            tracklet_dict[frame_key] = frame_data

        # Add tracklet to corrected data
        if tracklet_dict:
            corrected_data[t_idx] = tracklet_dict

    # Create backup of original file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use shorter backup filename to avoid Windows path length limits
    backup_filename = f"backup_{timestamp}{original_pickle_path.suffix}"
    backup_path = output_dir / backup_filename

    # Copy original to backup location
    shutil.copy2(original_pickle_path, backup_path)
    print(f"\nOriginal tracklets backed up to: {backup_path}")

    # Save corrected tracklets, replacing the original file
    with open(original_pickle_path, 'wb') as f:
        pickle.dump(corrected_data, f)

    print(f"Corrected tracklets saved to: {original_pickle_path}")

    # Print statistics
    total_detections = sum(len(t) for t in tracklets)
    n_total_assigned = sum(n_assigned.values())

    print(f"\nID Assignment Statistics:")
    print(f"  Male (ID=0):        {n_assigned['male']:6d} detections "
          f"({100 * n_assigned['male'] / total_detections:.1f}%)")
    print(f"  Female (ID=1):      {n_assigned['female']:6d} detections "
          f"({100 * n_assigned['female'] / total_detections:.1f}%)")
    print(f"  Low confidence (sillhouette < {min_silhouette}:     {n_low_confidence:6d} detections "
          f"({100 * n_low_confidence / total_detections:.1f}%)")
    print(f"  No embedding:       {n_no_embedding:6d} detections "
          f"({100 * n_no_embedding / total_detections:.1f}%)")
    print(f"  Total unassigned:   {n_low_confidence + n_no_embedding:6d} detections "
          f"({100 * (n_low_confidence + n_no_embedding) / total_detections:.1f}%)")
    print(f"  Total:              {total_detections:6d} detections")

    # Return statistics for summary report
    stats = {
        'n_male': n_assigned['male'],
        'n_female': n_assigned['female'],
        'n_low_confidence': n_low_confidence,
        'n_no_embedding': n_no_embedding,
        'total_detections': total_detections
    }

    return corrected_data, stats


def visualize_embeddings(embeddings, kmeans, cluster_mapping, output_dir, min_silhouette):
    """
    Create visualization of clustered embeddings.

    Parameters
    ----------
    embeddings : list of dict
        Embedding information with cluster labels and silhouette scores
    kmeans : KMeans
        Fitted KMeans model
    cluster_mapping : dict
        Cluster to label mapping
    output_dir : Path
        Output directory
    min_silhouette : float
        Silhouette threshold used for filtering
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Extract data
    emb_vectors = np.array([e['embedding'] for e in embeddings])
    clusters = np.array([e['cluster'] for e in embeddings])
    silhouettes = np.array([e['silhouette_score'] for e in embeddings])

    # Reduce to 2D for visualization
    print("Computing 2D projections (PCA and t-SNE)...")

    # PCA (fast, linear)
    pca = PCA(n_components=2, random_state=42)
    emb_pca = pca.fit_transform(emb_vectors)

    # t-SNE (slower, non-linear, better separation)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_tsne = tsne.fit_transform(emb_vectors)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Color maps
    colors = {0: '#1f77b4', 1: '#ff7f0e'}  # Blue and orange
    cluster_labels = {0: cluster_mapping.get(0, 'Cluster 0').upper(),
                     1: cluster_mapping.get(1, 'Cluster 1').upper()}

    # --- Plot 1: PCA colored by cluster ---
    ax = axes[0, 0]
    for cluster_id in [0, 1]:
        mask = clusters == cluster_id
        ax.scatter(emb_pca[mask, 0], emb_pca[mask, 1],
                  c=colors[cluster_id], label=cluster_labels[cluster_id],
                  alpha=0.6, s=20, edgecolors='none')

    # Plot centroids
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    for cluster_id in [0, 1]:
        ax.scatter(centroids_pca[cluster_id, 0], centroids_pca[cluster_id, 1],
                  c=colors[cluster_id], marker='X', s=300, edgecolors='black',
                  linewidths=2, zorder=10)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('PCA Projection - Colored by Cluster', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: t-SNE colored by cluster ---
    ax = axes[0, 1]
    for cluster_id in [0, 1]:
        mask = clusters == cluster_id
        ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
                  c=colors[cluster_id], label=cluster_labels[cluster_id],
                  alpha=0.6, s=20, edgecolors='none')

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE Projection - Colored by Cluster', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: PCA colored by silhouette score ---
    ax = axes[1, 0]
    scatter = ax.scatter(emb_pca[:, 0], emb_pca[:, 1],
                        c=silhouettes, cmap='RdYlGn', s=20,
                        alpha=0.7, edgecolors='none', vmin=-0.5, vmax=1.0)

    # Add threshold line in colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Silhouette Score', fontsize=11)
    cbar.ax.axhline(y=min_silhouette, color='red', linewidth=2, linestyle='--')
    cbar.ax.text(0.5, min_silhouette, f' Threshold={min_silhouette}',
                va='center', ha='left', fontsize=9, color='red', fontweight='bold')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('PCA - Colored by Silhouette Score', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Silhouette score distribution ---
    ax = axes[1, 1]

    # Histogram for each cluster
    for cluster_id in [0, 1]:
        mask = clusters == cluster_id
        cluster_sil = silhouettes[mask]
        ax.hist(cluster_sil, bins=50, alpha=0.6, color=colors[cluster_id],
               label=cluster_labels[cluster_id], edgecolor='black', linewidth=0.5)

    # Add threshold line
    ax.axvline(x=min_silhouette, color='red', linewidth=2, linestyle='--',
              label=f'Threshold = {min_silhouette}')

    # Add statistics text
    stats_text = (f"Mean: {silhouettes.mean():.3f}\n"
                 f"Std: {silhouettes.std():.3f}\n"
                 f"Below threshold: {(silhouettes < min_silhouette).sum()} "
                 f"({100*(silhouettes < min_silhouette).sum()/len(silhouettes):.1f}%)")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Silhouette Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Silhouette Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Overall title
    fig.suptitle('Embedding Clustering Visualization', fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    save_path = output_dir / 'embedding_visualization.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Embedding visualization saved to: {save_path}")


def save_summary_report(output_dir, tracklets, embeddings, cluster_mapping,
                       id_stats, min_silhouette, video_name):
    """
    Save detailed statistics to a text file.

    Parameters
    ----------
    output_dir : Path
        Output directory
    tracklets : list of Tracklet
        List of tracklet objects
    embeddings : list of dict
        Embedding information with silhouette scores
    cluster_mapping : dict
        Cluster to label mapping
    id_stats : dict
        ID assignment statistics
    min_silhouette : float
        Silhouette threshold used
    video_name : str
        Name of the video
    """
    # Compute tracklet statistics
    tracklet_lengths = [len(t) for t in tracklets]
    n_tracklets = len(tracklets)
    mean_length = np.mean(tracklet_lengths)
    std_length = np.std(tracklet_lengths)
    min_length = np.min(tracklet_lengths)
    max_length = np.max(tracklet_lengths)

    # Compute silhouette statistics by cluster
    cluster_silhouettes = {0: [], 1: []}
    for emb in embeddings:
        cluster_silhouettes[emb['cluster']].append(emb['silhouette_score'])

    all_silhouettes = [emb['silhouette_score'] for emb in embeddings]
    mean_silhouette = np.mean(all_silhouettes)
    std_silhouette = np.std(all_silhouettes)
    min_silhouette_score = np.min(all_silhouettes)
    max_silhouette_score = np.max(all_silhouettes)

    # Compute per-cluster silhouette statistics
    cluster_sil_stats = {}
    for cluster_id in [0, 1]:
        if cluster_silhouettes[cluster_id]:
            cluster_sil_stats[cluster_id] = {
                'mean': np.mean(cluster_silhouettes[cluster_id]),
                'std': np.std(cluster_silhouettes[cluster_id]),
                'min': np.min(cluster_silhouettes[cluster_id]),
                'max': np.max(cluster_silhouettes[cluster_id])
            }
        else:
            cluster_sil_stats[cluster_id] = {
                'mean': 0, 'std': 0, 'min': 0, 'max': 0
            }

    # Compute ID assignment ratios
    n_valid = id_stats['n_male'] + id_stats['n_female']
    n_invalid = id_stats['n_low_confidence'] + id_stats['n_no_embedding']
    total = id_stats['total_detections']

    valid_ratio = n_valid / total if total > 0 else 0
    invalid_ratio = n_invalid / total if total > 0 else 0
    male_female_ratio = id_stats['n_male'] / id_stats['n_female'] if id_stats['n_female'] > 0 else 0

    # Compute embeddings coverage
    n_embeddings = len(embeddings)
    embedding_coverage = n_embeddings / total if total > 0 else 0

    # Write report
    report_path = output_dir / 'summary_statistics.txt'

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ID CORRECTION SUMMARY STATISTICS\n")
        f.write("="*70 + "\n\n")

        f.write(f"Video: {video_name}\n")
        f.write(f"Date: {np.datetime64('now')}\n")
        f.write(f"Silhouette threshold: {min_silhouette}\n\n")

        # Tracklet statistics
        f.write("-"*70 + "\n")
        f.write("TRACKLET STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Number of tracklets:        {n_tracklets:8d}\n")
        f.write(f"Total detections:           {total:8d}\n")
        f.write(f"Average tracklet length:    {mean_length:8.2f} frames\n")
        f.write(f"Std tracklet length:        {std_length:8.2f} frames\n")
        f.write(f"Min tracklet length:        {min_length:8d} frames\n")
        f.write(f"Max tracklet length:        {max_length:8d} frames\n\n")

        # Embedding statistics
        f.write("-"*70 + "\n")
        f.write("EMBEDDING STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Detections with embeddings: {n_embeddings:8d} ({100*embedding_coverage:.1f}%)\n")
        f.write(f"Detections without:         {id_stats['n_no_embedding']:8d} "
                f"({100*id_stats['n_no_embedding']/total:.1f}%)\n\n")

        # Silhouette score statistics
        f.write("-"*70 + "\n")
        f.write("SILHOUETTE SCORE STATISTICS (All Embeddings)\n")
        f.write("-"*70 + "\n")
        f.write(f"Mean:                       {mean_silhouette:8.3f}\n")
        f.write(f"Std:                        {std_silhouette:8.3f}\n")
        f.write(f"Min:                        {min_silhouette_score:8.3f}\n")
        f.write(f"Max:                        {max_silhouette_score:8.3f}\n")
        f.write(f"Negative scores:            {sum(1 for s in all_silhouettes if s < 0):8d} "
                f"({100*sum(1 for s in all_silhouettes if s < 0)/len(all_silhouettes):.1f}%)\n\n")

        # Per-cluster silhouette statistics
        f.write("-"*70 + "\n")
        f.write("SILHOUETTE SCORES BY CLUSTER\n")
        f.write("-"*70 + "\n")
        for cluster_id in [0, 1]:
            label = cluster_mapping[cluster_id]
            stats = cluster_sil_stats[cluster_id]
            n_cluster = len(cluster_silhouettes[cluster_id])
            f.write(f"Cluster {cluster_id} ({label.upper()}):\n")
            f.write(f"  N detections:             {n_cluster:8d}\n")
            f.write(f"  Mean silhouette:          {stats['mean']:8.3f}\n")
            f.write(f"  Std silhouette:           {stats['std']:8.3f}\n")
            f.write(f"  Min silhouette:           {stats['min']:8.3f}\n")
            f.write(f"  Max silhouette:           {stats['max']:8.3f}\n\n")

        # ID assignment statistics
        f.write("-"*70 + "\n")
        f.write("ID ASSIGNMENT STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Male (ID=0):                {id_stats['n_male']:8d} "
                f"({100*id_stats['n_male']/total:.1f}%)\n")
        f.write(f"Female (ID=1):              {id_stats['n_female']:8d} "
                f"({100*id_stats['n_female']/total:.1f}%)\n")
        f.write(f"Low confidence (ID=-1):     {id_stats['n_low_confidence']:8d} "
                f"({100*id_stats['n_low_confidence']/total:.1f}%)\n")
        f.write(f"No embedding (ID=-1):       {id_stats['n_no_embedding']:8d} "
                f"({100*id_stats['n_no_embedding']/total:.1f}%)\n\n")

        f.write(f"Total valid (ID != -1):      {n_valid:8d} "
                f"({100*valid_ratio:.1f}%)\n")
        f.write(f"Total invalid (ID = -1):    {n_invalid:8d} "
                f"({100*invalid_ratio:.1f}%)\n")
        f.write(f"Valid/Invalid ratio:        {valid_ratio/invalid_ratio if invalid_ratio > 0 else float('inf'):8.2f}\n\n")

        f.write(f"Male/Female ratio:          {male_female_ratio:8.2f}\n\n")

        # Cluster mapping
        f.write("-"*70 + "\n")
        f.write("CLUSTER MAPPING\n")
        f.write("-"*70 + "\n")
        for cluster_id in sorted(cluster_mapping.keys()):
            f.write(f"Cluster {cluster_id} -> {cluster_mapping[cluster_id].upper()}\n")

        f.write("\n" + "="*70 + "\n")

    print(f"\nSummary statistics saved to: {report_path}")


def assign_male_only_ids(tracklets, header, tracklet_path, output_dir, n_co_occupancy_frames, min_co_occupancy_threshold):
    """
    Assign all detections to male (ID=0) when co-occupancy is insufficient for triplet training.

    This fallback is appropriate for trials where one sex (typically female) is rarely or never
    visible, making it impossible to train a reliable triplet loss model.

    Parameters
    ----------
    tracklets : list of Tracklet
        Original tracklets
    header : DataFrame
        Header from original pickle
    tracklet_path : Path
        Path to original pickle file (will be replaced)
    output_dir : Path
        Directory to save backup and summary report
    n_co_occupancy_frames : int
        Number of co-occupancy frames detected
    min_co_occupancy_threshold : int
        Threshold that triggered the fallback

    Returns
    -------
    None
    """
    import shutil
    from datetime import datetime

    print("\nApplying male-only ID assignment...")

    # Ensure paths are absolute
    tracklet_path = Path(tracklet_path).resolve()
    output_dir = Path(output_dir).resolve()

    # Create corrected data structure matching original format
    corrected_data = {'header': header}

    # Counter for statistics
    total_detections = 0

    # Process each tracklet - assign all to male (ID=0)
    for t_idx, tracklet in enumerate(tracklets):
        tracklet_dict = {}

        for local_idx, frame in enumerate(tracklet.inds):
            # Copy original data
            frame_data = tracklet.data[local_idx].copy()

            # Ensure data has ID column
            if frame_data.shape[1] == 3:
                id_col = np.full((frame_data.shape[0], 1), 0.0)  # All male
                frame_data = np.hstack([frame_data, id_col])
            else:
                # Set all to male
                frame_data[:, 3] = 0.0

            total_detections += 1

            # Store with frame key
            frame_key = f"frame{frame:06d}"
            tracklet_dict[frame_key] = frame_data

        # Add tracklet to corrected data
        if tracklet_dict:
            corrected_data[t_idx] = tracklet_dict

    # Create backup of original file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"backup_{timestamp}{tracklet_path.suffix}"
    backup_path = output_dir / backup_filename

    # Copy original to backup location
    shutil.copy2(tracklet_path, backup_path)
    print(f"Original tracklets backed up to: {backup_path}")

    # Save corrected tracklets, replacing the original file
    with open(tracklet_path, 'wb') as f:
        pickle.dump(corrected_data, f)

    print(f"Corrected tracklets saved to: {tracklet_path}")

    # Print statistics
    print(f"\nMale-Only ID Assignment Statistics:")
    print(f"  Male (ID=0):        {total_detections:6d} detections (100.0%)")
    print(f"  Female (ID=1):      {0:6d} detections (0.0%)")
    print(f"  Total:              {total_detections:6d} detections")

    # Generate summary report
    report_path = output_dir / 'summary_statistics.txt'

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ID CORRECTION SUMMARY - MALE-ONLY FALLBACK\n")
        f.write("="*70 + "\n\n")

        f.write(f"Video: {tracklet_path.stem}\n")
        f.write(f"Date: {np.datetime64('now')}\n\n")

        f.write("-"*70 + "\n")
        f.write("FALLBACK REASON\n")
        f.write("-"*70 + "\n")
        f.write(f"Co-occupancy frames detected: {n_co_occupancy_frames}\n")
        f.write(f"Minimum threshold required:   {min_co_occupancy_threshold}\n\n")
        f.write("Insufficient co-occupancy frames for reliable triplet loss training.\n")
        f.write("This typically occurs when one individual (usually female) is rarely\n")
        f.write("or never visible in the video (e.g., control trials).\n\n")

        f.write("-"*70 + "\n")
        f.write("FALLBACK STRATEGY\n")
        f.write("-"*70 + "\n")
        f.write("All detections assigned to male (ID=0).\n")
        f.write("Any female detections (if present) are labeled as male.\n\n")

        f.write("-"*70 + "\n")
        f.write("TRACKLET STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Number of tracklets:        {len(tracklets):8d}\n")
        f.write(f"Total detections:           {total_detections:8d}\n\n")

        f.write("-"*70 + "\n")
        f.write("ID ASSIGNMENT STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Male (ID=0):                {total_detections:8d} (100.0%)\n")
        f.write(f"Female (ID=1):              {0:8d} (0.0%)\n")
        f.write(f"Unassigned (ID=-1):         {0:8d} (0.0%)\n\n")

        f.write("="*70 + "\n")
        f.write("NOTE: This trial used male-only fallback due to insufficient\n")
        f.write("co-occupancy. Standard triplet loss training was not performed.\n")
        f.write("="*70 + "\n")

    print(f"\nSummary report saved to: {report_path}")
    print("\n" + "="*60)
    print("Male-only ID assignment complete!")
    print("="*60)


def train_id_model(tracklet_path, n_epochs=None, batch_size=None, lr=None,
                   patch_size=None, padding=None, conf_threshold=None,
                   device=None, force_retrain=False, min_tracklet_length=None,
                   min_overlap_frames=None, frame_stride=None, min_co_occupancy_frames=None):
    """
    Train ID correction model, extract embeddings, and perform clustering.
    This is substage 1 of ID correction: all non-interactive model training steps.

    Parameters
    ----------
    tracklet_path : str or Path
        Path to *el.pickle tracklet file
    n_epochs : int, optional
        Number of training epochs (default: from config)
    batch_size : int, optional
        Batch size (default: from config)
    lr : float, optional
        Learning rate (default: from config)
    patch_size : int, optional
        Patch size (default: from config)
    padding : int, optional
        Padding around keypoints in pixels (default: from config)
    conf_threshold : float, optional
        Confidence threshold (default: from config)
    device : str, optional
        Device to use: cuda or cpu (default: from config)
    force_retrain : bool, optional
        Force retraining even if model exists (default: False)
    min_tracklet_length : int, optional
        Minimum tracklet length to include in training (default: from config)
    min_overlap_frames : int, optional
        Minimum number of co-occupancy frames required for a tracklet pair to be included
        in training. Only tracklet pairs with >min_overlap_frames of overlap are used.
        Set to 0 to include all co-occupancy frames. (default: from config)
    frame_stride : int, optional
        Sample every Nth frame for patch cache (default: from config)
    min_co_occupancy_frames : int, optional
        Minimum total co-occupancy frames required to run triplet training. If below this
        threshold, falls back to male-only ID assignment (default: from config)

    Returns
    -------
    prep_data : dict
        Dictionary containing:
            - 'tracklets': list of Tracklet objects
            - 'header': DataFrame header
            - 'embeddings': list of embedding dicts with cluster labels
            - 'kmeans': fitted KMeans model
            - 'patch_extractor': PatchExtractor object
            - 'model': trained SimpleCNN model
            - 'tracklet_path': Path to tracklet file
            - 'video_path': Path to video file
            - 'output_dir': Path to output directory
    """
    # Load defaults from config
    if n_epochs is None:
        n_epochs = config.DEFAULT_ID_N_EPOCHS
    if batch_size is None:
        batch_size = config.DEFAULT_ID_BATCH_SIZE
    if lr is None:
        lr = config.DEFAULT_ID_LEARNING_RATE
    if patch_size is None:
        patch_size = config.DEFAULT_PATCH_SIZE
    if padding is None:
        padding = config.DEFAULT_PADDING
    if conf_threshold is None:
        conf_threshold = config.DEFAULT_CONF_THRESHOLD
    if device is None:
        device = config.DEFAULT_ID_DEVICE
    if min_tracklet_length is None:
        min_tracklet_length = config.DEFAULT_MIN_TRACKLET_LENGTH
    if min_overlap_frames is None:
        min_overlap_frames = config.DEFAULT_MIN_OVERLAP_FRAMES
    if min_co_occupancy_frames is None:
        min_co_occupancy_frames = config.DEFAULT_MIN_CO_OCCUPANCY_FRAMES

    # Setup paths
    tracklet_path = Path(tracklet_path)
    video_path = tracklet_path.parent / (tracklet_path.stem.split('DLC')[0] + '.mp4')
    output_dir = tracklet_path.parent / 'id_correction'
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / 'encoder_model.pth'

    print("="*60)
    print("Triplet Loss ID Correction")
    print("="*60)
    print(f"Tracklet file: {tracklet_path.name}")
    print(f"Video file: {video_path.name}")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")

    # Load tracklets
    print("Loading tracklets...")
    tracklets, header = load_tracklets(tracklet_path)
    print(f"Loaded {len(tracklets)} tracklets")

    # Check if tracklets already have ID assignments (i.e., already corrected)
    has_existing_ids = False
    existing_id_count = 0
    total_detections = sum(len(t) for t in tracklets)

    for tracklet in tracklets:
        for local_idx in range(len(tracklet)):
            frame_data = tracklet.data[local_idx]
            if frame_data.shape[1] == 4:  # Has ID column
                # Check if any IDs are not -1
                ids = frame_data[:, 3]
                non_default_ids = ids[(ids != -1) & (~np.isnan(ids))]
                if len(non_default_ids) > 0:
                    has_existing_ids = True
                    existing_id_count += len(non_default_ids)

    if has_existing_ids:
        print(f"\n{'!'*60}")
        print("WARNING: Input tracklets already contain ID assignments!")
        print(f"{'!'*60}")
        print(f"Found {existing_id_count} detections with assigned IDs (not -1)")
        print(f"This suggests the file may have been previously corrected.")
        print(f"Using existing IDs. If you want to re-run correction, restore from backup first.")

        # Check for backups
        backup_files = sorted(output_dir.glob(f"{tracklet_path.stem}_backup_*{tracklet_path.suffix}"))
        if backup_files:
            print(f"\nBackup files available in: {output_dir}")
            for i, backup in enumerate(backup_files[-3:], 1):  # Show last 3
                print(f"  - {backup.name}")

        print("Skipping identity correction.")
        return

    print()

    # Setup patch extractor
    patch_extractor = PatchExtractor(video_path,
                                     patch_size=patch_size,
                                     padding=padding,
                                     conf_threshold=conf_threshold)

    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Check if model exists
    if model_path.exists() and not force_retrain:
        print(f"Found existing model at {model_path}")
        print("Loading model... (use force_retrain=True to retrain)\n")
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        # Detect co-occupancy frames
        print("Detecting co-occupancy frames...")
        detector = CoOccupancyDetector(tracklets, min_conf=conf_threshold)
        co_occupancy_frames = detector.find_co_occupancy_frames(min_overlap_frames=min_overlap_frames)
        print(f"Found {len(co_occupancy_frames)} co-occupancy frames\n")

        # Check if we have enough co-occupancy frames for reliable training
        if len(co_occupancy_frames) < min_co_occupancy_frames:
            print("="*60)
            print("INSUFFICIENT CO-OCCUPANCY FOR ID TRAINING")
            print("="*60)
            print(f"Co-occupancy frames: {len(co_occupancy_frames)} (threshold: {min_co_occupancy_frames})\n")
            print("This trial has too few frames where both animals are visible")
            print("simultaneously. Training the triplet loss model would be unreliable.\n")
            print("Falling back to male-only ID assignment:")
            print("  - All detections will be assigned ID=0 (male)")
            print("  - Any female detections (if present) will be labeled as male")
            print("  - This is appropriate for control trials with minimal female presence\n")
            print("="*60 + "\n")

            # Apply male-only fallback
            assign_male_only_ids(tracklets, header, tracklet_path, output_dir,
                               len(co_occupancy_frames), min_co_occupancy_frames)

            # Return None to signal that standard pipeline should be skipped
            return None

        # Train encoder
        print(f"Sufficient co-occupancy detected. Proceeding with triplet loss training...\n")
        print("\nTraining encoder...")
        model = train_encoder(tracklets, co_occupancy_frames, patch_extractor, output_dir,
                             n_epochs=n_epochs, batch_size=batch_size,
                             lr=lr, device=device, min_tracklet_length=min_tracklet_length,
                             frame_stride=frame_stride)

    # Extract embeddings
    embeddings_path = output_dir / 'embeddings.pkl'
    if embeddings_path.exists() and not force_retrain:
        print(f"\nFound existing embeddings at {embeddings_path}")
        print("Loading embeddings... (use force_retrain=True to recompute)\n")
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded {len(embeddings)} embeddings\n")
    else:
        print("\nExtracting embeddings for all detections...")
        embeddings = extract_all_embeddings(model, tracklets, patch_extractor, device=device)
        print(f"Extracted {len(embeddings)} embeddings")

        # Cache embeddings
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings cached to {embeddings_path}\n")

    # Cluster
    print("Clustering embeddings...")
    embeddings, kmeans = cluster_and_assign_ids(embeddings, n_clusters=2)
    print()

    # Update cached embeddings with cluster information
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)

    # Prepare cluster comparison video (non-interactive, done upfront)
    print("Creating cluster comparison video...")
    video_path_out = prepare_cluster_comparison_video(embeddings, tracklets, patch_extractor)
    print()

    # Return all data needed for the interactive step
    prep_data = {
        'tracklets': tracklets,
        'header': header,
        'embeddings': embeddings,
        'kmeans': kmeans,
        'patch_extractor': patch_extractor,
        'model': model,
        'tracklet_path': tracklet_path,
        'video_path': video_path,
        'output_dir': output_dir,
        'comparison_video_path': video_path_out
    }

    print("Model training complete!")
    print("Call map_clusters_to_sex() with the returned data to run the interactive mapping.\n")

    return prep_data


def map_clusters_to_sex(prep_data):
    """
    Interactively map clusters to biological sex (male/female).
    This is substage 2 of ID correction: the interactive portion requiring user input.

    The comparison video should already be created by train_id_model(). This function
    only prompts the user for input, making it very fast and suitable for batch mode.

    Parameters
    ----------
    prep_data : dict
        Dictionary returned from train_id_model() containing:
            - 'tracklets': list of Tracklet objects
            - 'header': DataFrame header
            - 'embeddings': list of embedding dicts with cluster labels
            - 'kmeans': fitted KMeans model
            - 'patch_extractor': PatchExtractor object
            - 'model': trained SimpleCNN model
            - 'tracklet_path': Path to tracklet file
            - 'video_path': Path to video file
            - 'output_dir': Path to output directory
            - 'comparison_video_path': Path to comparison video

    Returns
    -------
    cluster_mapping : dict
        Maps cluster ID to semantic label (e.g., {0: 'male', 1: 'female'})
    """
    # Unpack preparation data
    tracklet_path = prep_data['tracklet_path']
    video_path = prep_data['video_path']
    comparison_video_path = prep_data['comparison_video_path']

    print("="*60)
    print("Interactive ID Correction - Cluster Mapping")
    print("="*60)
    print(f"Tracklet file: {tracklet_path.name}")
    print(f"Video file: {video_path.name}")
    print("="*60 + "\n")

    # Interactive mapping (just user input, video already created)
    print("Mapping clusters to individuals...")
    cluster_mapping = interactive_cluster_mapping(comparison_video_path)
    print(f"Cluster mapping: {cluster_mapping}\n")

    return cluster_mapping


def assign_corrected_ids(prep_data, cluster_mapping, min_silhouette=None):
    """
    Assign corrected IDs to tracklets and generate final outputs.
    This is substage 3 of ID correction: ID reassignment, visualization, and reporting.

    Parameters
    ----------
    prep_data : dict
        Dictionary returned from train_id_model() containing:
            - 'tracklets': list of Tracklet objects
            - 'header': DataFrame header
            - 'embeddings': list of embedding dicts with cluster labels
            - 'kmeans': fitted KMeans model
            - 'patch_extractor': PatchExtractor object
            - 'model': trained SimpleCNN model
            - 'tracklet_path': Path to tracklet file
            - 'video_path': Path to video file
            - 'output_dir': Path to output directory
    cluster_mapping : dict
        Maps cluster ID to semantic label from map_clusters_to_sex()
    min_silhouette : float, optional
        Minimum silhouette score to assign ID (default: from config).
        Range: -1 to 1. Recommended: 0.0 (lenient), 0.2 (moderate), 0.5 (strict)

    Returns
    -------
    corrected_data : dict
        Dictionary in original pickle format with updated IDs
    id_stats : dict
        ID assignment statistics
    """
    # Load defaults from config
    if min_silhouette is None:
        min_silhouette = config.DEFAULT_MIN_SILHOUETTE

    # Unpack preparation data
    tracklets = prep_data['tracklets']
    header = prep_data['header']
    embeddings = prep_data['embeddings']
    kmeans = prep_data['kmeans']
    tracklet_path = prep_data['tracklet_path']
    video_path = prep_data['video_path']
    output_dir = prep_data['output_dir']

    print("="*60)
    print("ID Assignment and Finalization")
    print("="*60)

    # Visualize embeddings
    print("Creating embedding visualization...")
    visualize_embeddings(embeddings, kmeans, cluster_mapping, output_dir, min_silhouette)

    # Reassign IDs and save
    print("Reassigning IDs and saving corrected tracklets...")
    corrected_tracklets, id_stats = reassign_tracklet_ids(tracklets, embeddings, cluster_mapping,
                                                          output_dir, tracklet_path, header,
                                                          min_silhouette=min_silhouette)

    # Generate summary report
    save_summary_report(output_dir, tracklets, embeddings, cluster_mapping,
                       id_stats, min_silhouette, video_path.stem)

    print("\n" + "="*60)
    print("ID correction complete!")
    print("="*60)

    return corrected_tracklets, id_stats


def main(trial_manager, n_epochs=None, batch_size=None, lr=None, patch_size=None,
         padding=None, conf_threshold=None, device=None, force_retrain=False,
         min_silhouette=None, min_tracklet_length=None, min_overlap_frames=None,
         frame_stride=None, min_co_occupancy_frames=None):
    """
    Triplet loss-based ID correction for DeepLabCut tracklets.

    This function runs the complete three-substage pipeline:
      1. train_id_model(): Train CNN, extract embeddings, cluster (non-interactive)
      2. map_clusters_to_sex(): Interactive cluster-to-sex mapping (interactive)
      3. assign_corrected_ids(): ID reassignment, visualization, reporting (non-interactive)

    Parameters
    ----------
    trial_manager : TrialManager
        TrialManager instance for the trial. Used to resolve tracklet paths and mark
        completion status.
    n_epochs : int, optional
        Number of training epochs (default: from config)
    batch_size : int, optional
        Batch size (default: from config)
    lr : float, optional
        Learning rate (default: from config)
    patch_size : int, optional
        Patch size (default: from config)
    padding : int, optional
        Padding around keypoints in pixels (default: from config)
    conf_threshold : float, optional
        Confidence threshold (default: from config)
    device : str, optional
        Device to use: cuda or cpu (default: from config)
    force_retrain : bool, optional
        Force retraining even if model exists (default: False)
    min_silhouette : float, optional
        Minimum silhouette score to assign ID (default: from config).
        Range: -1 to 1. Recommended: 0.0 (lenient), 0.2 (moderate), 0.5 (strict)
    min_tracklet_length : int, optional
        Minimum tracklet length to include in training (default: from config)
    min_overlap_frames : int, optional
        Minimum number of co-occupancy frames required for a tracklet pair to be included
        in training. Only tracklet pairs with >min_overlap_frames of overlap are used.
        Set to 0 to include all co-occupancy frames. (default: from config)
    frame_stride : int, optional
        Sample every Nth frame for patch cache (default: from config)
    min_co_occupancy_frames : int, optional
        Minimum total co-occupancy frames required to run triplet training. If below this
        threshold, falls back to male-only ID assignment (default: from config)

    Returns
    -------
    corrected_data : dict
        Dictionary in original pickle format with updated IDs
    id_stats : dict
        ID assignment statistics
    """
    # Get tracklet path from TrialManager
    tracklet_path = trial_manager.el_pickle_path()
    # Substage 1: Train model, extract embeddings, and cluster
    prep_data = train_id_model(
        tracklet_path=tracklet_path,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        patch_size=patch_size,
        padding=padding,
        conf_threshold=conf_threshold,
        device=device,
        force_retrain=force_retrain,
        min_tracklet_length=min_tracklet_length,
        min_overlap_frames=min_overlap_frames,
        frame_stride=frame_stride,
        min_co_occupancy_frames=min_co_occupancy_frames
    )

    if prep_data is None:
        # Training was skipped (e.g., already corrected or insufficient co-occupancy)
        # Mark stage as complete even for fallback cases
        trial_manager.mark_stage_complete('identity_correction')
        return None, None

    # Substage 2: Interactive cluster mapping
    cluster_mapping = map_clusters_to_sex(prep_data)

    # Substage 3: Assign IDs and finalize
    corrected_data, id_stats = assign_corrected_ids(
        prep_data=prep_data,
        cluster_mapping=cluster_mapping,
        min_silhouette=min_silhouette
    )

    # Mark stage as complete
    trial_manager.mark_stage_complete('identity_correction')

    return corrected_data, id_stats