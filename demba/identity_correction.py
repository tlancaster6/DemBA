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


class PatchExtractor:
    """Extract RGB patches from video frames based on keypoint locations."""

    def __init__(self, video_path, patch_size=128, padding=10, conf_threshold=0.5):
        """
        Parameters
        ----------
        video_path : str or Path
            Path to video file
        patch_size : int
            Target size for patches (square)
        padding : int
            Fixed padding width around keypoints in pixels
        conf_threshold : float
            Minimum confidence threshold for including keypoints in bbox calculation
        """
        self.video_path = Path(video_path)
        self.patch_size = patch_size
        self.padding = padding
        self.conf_threshold = conf_threshold
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
        valid_mask = (keypoints[:, 2] >= self.conf_threshold) & \
                     (~np.isnan(keypoints[:, 0])) & \
                     (~np.isnan(keypoints[:, 1])) & \
                     (~np.isnan(keypoints[:, 2]))

        if not valid_mask.any():
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

    def find_co_occupancy_frames(self):
        """
        Find all frames where exactly two tracklets overlap.

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

        return co_occupancy


class TripletDataset(Dataset):
    """Dataset for triplet loss training."""

    def __init__(self, tracklets, co_occupancy_frames, patch_extractor, samples_per_epoch=1000):
        """
        Parameters
        ----------
        tracklets : list of Tracklet
            List of tracklet objects
        co_occupancy_frames : list of dict
            Co-occupancy frame information from CoOccupancyDetector
        patch_extractor : PatchExtractor
            Patch extraction object
        samples_per_epoch : int
            Number of triplets to generate per epoch
        """
        self.tracklets = tracklets
        self.co_occupancy_frames = co_occupancy_frames
        self.patch_extractor = patch_extractor
        self.samples_per_epoch = samples_per_epoch

        # Build index for fast lookup
        self._build_tracklet_index()

    def _build_tracklet_index(self):
        """Build index mapping tracklet idx to frame indices."""
        self.tracklet_frame_map = {}
        for t_idx, tracklet in enumerate(self.tracklets):
            self.tracklet_frame_map[t_idx] = {
                'frames': tracklet.inds,
                'data': tracklet.data
            }

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

            anchor_tracklet = self.tracklets[anchor_t_idx]
            anchor_frame = co_occ['frame']

            # Sample positive from same tracklet (different frame)
            available_frames = [i for i in range(len(anchor_tracklet.inds))
                              if i != anchor_local_idx]
            if not available_frames:
                continue

            pos_local_idx = np.random.choice(available_frames)
            pos_frame = anchor_tracklet.inds[pos_local_idx]

            # Extract patches
            anchor_kpts = anchor_tracklet.data[anchor_local_idx]
            pos_kpts = anchor_tracklet.data[pos_local_idx]
            neg_kpts = self.tracklets[neg_t_idx].data[neg_local_idx]

            anchor_patch = self.patch_extractor.extract_patch(anchor_frame, anchor_kpts)
            pos_patch = self.patch_extractor.extract_patch(pos_frame, pos_kpts)
            neg_patch = self.patch_extractor.extract_patch(co_occ['frame'], neg_kpts)

            if anchor_patch is not None and pos_patch is not None and neg_patch is not None:
                # Convert to tensors and normalize
                anchor_tensor = self._preprocess(anchor_patch)
                pos_tensor = self._preprocess(pos_patch)
                neg_tensor = self._preprocess(neg_patch)

                return anchor_tensor, pos_tensor, neg_tensor

        # If all attempts fail, return zeros (will be filtered out)
        empty = torch.zeros((3, self.patch_extractor.patch_size, self.patch_extractor.patch_size))
        return empty, empty, empty

    def _preprocess(self, patch):
        """Convert patch to tensor and normalize."""
        # Convert BGR to RGB
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        patch_norm = patch_rgb.astype(np.float32) / 255.0
        # Convert to CHW format
        patch_tensor = torch.from_numpy(patch_norm).permute(2, 0, 1)
        return patch_tensor


class SimpleCNN(nn.Module):
    """Simple CNN encoder for embedding extraction."""

    def __init__(self, embedding_dim=128):
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

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


def train_encoder(tracklets, co_occupancy_frames, patch_extractor, output_dir,
                 n_epochs=50, batch_size=32, lr=0.001, device='cuda'):
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
        Learning rate
    device : str
        'cuda' or 'cpu'

    Returns
    -------
    model : SimpleCNN
        Trained model
    """
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    criterion = TripletLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TripletDataset(tracklets, co_occupancy_frames, patch_extractor,
                            samples_per_epoch=1000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Open video
    patch_extractor.open_video()

    # Training loop
    model.train()
    losses = []

    for epoch in range(n_epochs):
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
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    # Close video
    patch_extractor.close_video()

    # Save model
    model_path = output_dir / 'encoder_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Triplet Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(output_dir / 'training_loss.png')
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


def interactive_cluster_mapping(embeddings, tracklets, patch_extractor,
                               n_segments=3, segment_duration_sec=3, fps=30):
    """
    Create a video showing trajectory segments from each cluster for user to map to male/female.

    Parameters
    ----------
    embeddings : list of dict
        Embedding information with cluster labels
    tracklets : list of Tracklet
        List of tracklet objects
    patch_extractor : PatchExtractor
        Patch extraction object
    n_segments : int
        Number of trajectory segments to show per cluster
    segment_duration_sec : float
        Duration of each segment in seconds
    fps : int
        Frames per second of output video

    Returns
    -------
    mapping : dict
        Maps cluster ID to semantic label (e.g., {0: 'male', 1: 'female'})
    """
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
                cluster_tracklets[cluster_id].append({
                    'tracklet_idx': t_idx,
                    'purity': cluster_counts[cluster_id] / total,
                    'length': len(tracklets[t_idx])
                })

    # Sort by purity and length
    for cluster_id in [0, 1]:
        cluster_tracklets[cluster_id].sort(
            key=lambda x: (x['purity'], x['length']),
            reverse=True
        )

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
    print("Please review the video to identify which cluster is male/female.\n")

    # Get user input
    print("=" * 60)
    print("Cluster Mapping")
    print("=" * 60)
    print(f"\nVideo saved to: {output_path}")
    print("Left panel = Cluster 0, Right panel = Cluster 1\n")

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
    backup_filename = f"{original_pickle_path.stem}_backup_{timestamp}{original_pickle_path.suffix}"
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
    print(f"  Low confidence:     {n_low_confidence:6d} detections "
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


def prepare_id_correction(tracklet_path, n_epochs=50, batch_size=32, lr=0.001,
                          patch_size=128, padding=10, conf_threshold=0.5,
                          device='cuda', force_retrain=False):
    """
    Prepare ID correction by training model, extracting embeddings, and clustering.
    This function runs all non-interactive steps up to (but not including) the
    interactive cluster mapping.

    Parameters
    ----------
    tracklet_path : str or Path
        Path to *el.pickle tracklet file
    n_epochs : int, optional
        Number of training epochs (default: 50)
    batch_size : int, optional
        Batch size (default: 32)
    lr : float, optional
        Learning rate (default: 0.001)
    patch_size : int, optional
        Patch size (default: 128)
    padding : int, optional
        Padding around keypoints in pixels (default: 10)
    conf_threshold : float, optional
        Confidence threshold (default: 0.5)
    device : str, optional
        Device to use: cuda or cpu (default: cuda)
    force_retrain : bool, optional
        Force retraining even if model exists (default: False)

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
        print(f"\nOptions:")
        print(f"  1. Continue anyway (will re-run ID correction)")
        print(f"  2. Restore from backup (if available in id_correction folder)")
        print(f"  3. Abort")

        # Check for backups
        backup_files = sorted(output_dir.glob(f"{tracklet_path.stem}_backup_*{tracklet_path.suffix}"))
        if backup_files:
            print(f"\nFound {len(backup_files)} backup file(s):")
            for i, backup in enumerate(backup_files[-3:], 1):  # Show last 3
                print(f"  {i}. {backup.name}")

        response = input("\nContinue? (y/n): ").strip().lower()
        if response != 'y':
            print("Aborted by user.")
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
        co_occupancy_frames = detector.find_co_occupancy_frames()
        print(f"Found {len(co_occupancy_frames)} co-occupancy frames\n")

        if len(co_occupancy_frames) < 100:
            print("WARNING: Very few co-occupancy frames found. Results may be unreliable.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return

        # Train encoder
        print("\nTraining encoder...")
        model = train_encoder(tracklets, co_occupancy_frames, patch_extractor, output_dir,
                             n_epochs=n_epochs, batch_size=batch_size,
                             lr=lr, device=device)

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
        'output_dir': output_dir
    }

    print("Preparation complete!")
    print("Call complete_id_correction() with the returned data to run the interactive mapping.\n")

    return prep_data


def complete_id_correction(prep_data, min_silhouette=0.2):
    """
    Complete ID correction by running the interactive cluster mapping and saving results.
    This function runs the interactive portion that requires user input.

    Parameters
    ----------
    prep_data : dict
        Dictionary returned from prepare_id_correction() containing:
            - 'tracklets': list of Tracklet objects
            - 'header': DataFrame header
            - 'embeddings': list of embedding dicts with cluster labels
            - 'kmeans': fitted KMeans model
            - 'patch_extractor': PatchExtractor object
            - 'model': trained SimpleCNN model
            - 'tracklet_path': Path to tracklet file
            - 'video_path': Path to video file
            - 'output_dir': Path to output directory
    min_silhouette : float, optional
        Minimum silhouette score to assign ID (default: 0.2).
        Range: -1 to 1. Recommended: 0.0 (lenient), 0.2 (moderate), 0.5 (strict)

    Returns
    -------
    corrected_data : dict
        Dictionary in original pickle format with updated IDs
    id_stats : dict
        ID assignment statistics
    """
    # Unpack preparation data
    tracklets = prep_data['tracklets']
    header = prep_data['header']
    embeddings = prep_data['embeddings']
    kmeans = prep_data['kmeans']
    patch_extractor = prep_data['patch_extractor']
    model = prep_data['model']
    tracklet_path = prep_data['tracklet_path']
    video_path = prep_data['video_path']
    output_dir = prep_data['output_dir']

    print("="*60)
    print("Interactive ID Correction - Cluster Mapping")
    print("="*60)
    print(f"Tracklet file: {tracklet_path.name}")
    print(f"Video file: {video_path.name}")
    print("="*60 + "\n")

    # Interactive mapping
    print("Mapping clusters to individuals...")
    cluster_mapping = interactive_cluster_mapping(embeddings, tracklets, patch_extractor)
    print(f"Cluster mapping: {cluster_mapping}\n")

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


def main(tracklet_path, n_epochs=50, batch_size=32, lr=0.001, patch_size=128,
         padding=10, conf_threshold=0.5, device='cuda', force_retrain=False,
         min_silhouette=0.2):
    """
    Triplet loss-based ID correction for DeepLabCut tracklets.

    This function runs the complete pipeline: preparation (training, embedding extraction,
    clustering) followed by interactive cluster mapping and ID reassignment.

    Parameters
    ----------
    tracklet_path : str or Path
        Path to *el.pickle tracklet file
    n_epochs : int, optional
        Number of training epochs (default: 50)
    batch_size : int, optional
        Batch size (default: 32)
    lr : float, optional
        Learning rate (default: 0.001)
    patch_size : int, optional
        Patch size (default: 128)
    padding : int, optional
        Padding around keypoints in pixels (default: 10)
    conf_threshold : float, optional
        Confidence threshold (default: 0.5)
    device : str, optional
        Device to use: cuda or cpu (default: cuda)
    force_retrain : bool, optional
        Force retraining even if model exists (default: False)
    min_silhouette : float, optional
        Minimum silhouette score to assign ID (default: 0.2).
        Range: -1 to 1. Recommended: 0.0 (lenient), 0.2 (moderate), 0.5 (strict)

    Returns
    -------
    corrected_data : dict
        Dictionary in original pickle format with updated IDs
    id_stats : dict
        ID assignment statistics
    """
    # Run preparation phase
    prep_data = prepare_id_correction(
        tracklet_path=tracklet_path,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        patch_size=patch_size,
        padding=padding,
        conf_threshold=conf_threshold,
        device=device,
        force_retrain=force_retrain
    )

    if prep_data is None:
        # User aborted during preparation
        return None, None

    # Run interactive completion phase
    corrected_data, id_stats = complete_id_correction(
        prep_data=prep_data,
        min_silhouette=min_silhouette
    )

    return corrected_data, id_stats