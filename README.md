# DemBA

**DeepLabCut Multi-animal Tracking with Improved re-ID**

A Python pipeline for pose estimation, identity correction, tracklet stitching, and behavioral feature extraction from multi-animal videos using DeepLabCut.
Originally developed for analysis of cichlid (fish) behavioral videos in the Streelman and McGrath labs at Georgia Tech.

## Features

- **Pose Estimation**: Multi-animal pose tracking using DeepLabCut
- **Identity Correction**: Per-video triplet loss CNN for consistent identity assignment across tracklets
- **Tracklet Stitching**: Combine short tracklets into continuous identity tracks
- **Temporal Filtering**: Smooth pose trajectories over time using DeepLabCut
- **Feature Extraction**: Automated behavioral feature detection and quantification
- **Visualization**: Create labeled videos with pose overlays (DeepLabCut) and other visualizations
- **Statistical Analysis**: Generate boxplots, correlation plots, and heatmaps for feature

## Status and Scope

This repository was developed for cichlid behavioral analysis and prioritizes applicability to that specific use case over generalizability. Many components (particularly feature extraction and analysis) are specialized for our experimental setup and behaviors of interest.

However, the **CNN-based identity correction** and **hard-partition tracklet stitching** approaches may be broadly useful and could potentially be integrated into DeepLabCut's official library:

**CNN-based ReID vs. DLC's Transformer ReID:**
- Learns from RGB image patches rather than pose backbone features
- May perform better when pre-trained features don't capture visual differences
- Includes silhouette-based quality filtering for uncertainty quantification
- Simpler architecture (CNN vs. Transformer), faster training
- Both enable train-once-per-species, reID-per-video workflow

**CNN-based ReID vs. DLC's Supervised Identity Tracking:**
- No identity annotation required (unsupervised clustering)
- generalized pose model + specialized ID models, rather than a specialized pose+ID model for each video
- Better for experiments where individuals have similar body plans across videos, but the exact individuals in each video vary

**Hard-partition stitching** (vs. DLC's soft-constraint graph optimization):
- Strictly enforces identity boundaries by partitioning tracklets before stitching
- Simpler and more robust when identity preservation is critical
- Avoids the complexity and occasional failures (common on very long videos) of global graph optimization

See [Comparison with DeepLabCut's Transformer ReID](#comparison-with-deeplabcuts-transformer-reid) for detailed technical differences.

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)

### Install from source

```bash
git clone https://github.com/yourusername/DemBA.git
cd DemBA
pip install -e .
```

### Dependencies

Core dependencies include:
- numpy
- pandas
- opencv-python
- torch
- scikit-learn
- matplotlib
- tqdm
- deeplabcut

## Quick Start

### Recommended Project Structure

DemBA expects a specific project structure for optimal functionality. While some operations may work with non-standard layouts, **using the recommended structure is strongly advised** for full pipeline compatibility:

```
MyProject/
├── config.yaml                    # DeepLabCut project config
└── Analysis_<date>/               # Analysis directory (can be any name)
    ├── Videos/                    # Required: Trial video directories
    │   ├── trial1/               # Each trial in its own directory
    │   │   ├── trial1.mp4        # Video file (name matches directory)
    │   │   ├── trial1_roi.png    # ROI image (optional)
    │   │   └── ...               # Pipeline outputs created here
    │   ├── trial2/
    │   │   ├── trial2.mp4
    │   │   └── ...
    │   └── ...
    └── Annotations/               # Optional: Manual annotations
        └── quivering_annotations.xlsx  # Auto-detected by batch mode
```

**Key Requirements:**
- `config.yaml` must be in or above the Analysis directory
- Each trial must be in its own subdirectory under `Videos/`
- Video filename must match the directory name (e.g., `trial1/trial1.mp4`)
- Annotations directory is optional but auto-detected if present

### Command Line Interface

DemBA provides a comprehensive CLI with modular stages, full pipeline mode, and batch mode.

```bash
# Run full pipeline on a single trial
python main.py full --video Videos/trial1/trial1.mp4 --dlc-config config.yaml

# Run batch mode on multiple trials (recommended for projects with many videos)
python main.py batch --project-dir /path/to/MyProject/Analysis_<date>

# Run individual stages
python main.py pose --video Videos/trial1/trial1.mp4 --dlc-config config.yaml
python main.py id-correction --video Videos/trial1/trial1.mp4 --dlc-config config.yaml
python main.py stitch --video Videos/trial1/trial1.mp4 --dlc-config config.yaml
python main.py filter --video Videos/trial1/trial1.mp4 --dlc-config config.yaml
python main.py features --video Videos/trial1/trial1.mp4 --dlc-config config.yaml
python main.py visualize --video Videos/trial1/trial1.mp4 --dlc-config config.yaml
python main.py analyze --project-dir /path/to/MyProject/Analysis_<date>
```

### Python API

```python
import demba

# Pose estimation
demba.estimate_pose(
    config_path='config.yaml',
    video_path='trial1.mp4',
    n_fish=2
)

# Identity correction
demba.prepare_id_correction('trial1_el.pickle')
demba.complete_id_correction('trial1_el.pickle')

# Tracklet stitching
demba.stitch_by_identity(
    tracklet_pickle_path='trial1_el.pickle',
    output_h5_path='trial1_el.h5',
    n_tracks=2
)

# Feature extraction
demba.process_video(
    video_path='trial1.mp4',
    pose_h5_path='trial1_el.h5',
    visualize=True
)
```

## Pipeline Stages

DemBA provides 8 modular stages that can be run individually or combined via `full` (single trial) or `batch` (multiple trials) modes.

### 1. Pose Estimation
Runs standard DeepLabCut multi-animal pose estimation (dlc.analyze_videos) and simple SORT-style stitching 
(dlc.convert_detections2tracklets). 

```bash
python main.py pose --video trial1.mp4 --dlc-config config.yaml --n-fish 2
```

### 2. Identity Correction

**The Challenge**: Multi-animal pose tracking produces identity swaps when individuals occlude each other, come into close proximity, or leave and return to the field of view

**DemBA's Approach**: A per-video triplet loss CNN that learns to distinguish individuals using the pose tracking data itself—no manual annotations required beyond the pose annotations used to train the DLC pose estimation model

**How It Works**:
1. **Automatic Training Data Generation**: Identifies "co-occupancy frames" where both animals are simultaneously detected, providing anchor-negative pairs for contrastive learning.
2. **Visual Embedding Learning**: Trains a CNN to extract visual embeddings from image patches around each detection, using triplet loss to separate embeddings by identity.
3. **Unsupervised Clustering**: Applies KMeans clustering to embeddings across all detections, grouping them into two identity clusters.
4. **Interactive Mapping**: Creates a comparison video showing representative segments from each cluster. User labels which cluster corresponds to each individual.
5. **Quality-Filtered Assignment**: Assigns identity labels using silhouette scores to filter low-confidence predictions.

**Usage**:
```bash
# Two-phase approach (recommended for large videos)
python main.py id-correction --tracklet-pickle trial1_el.pickle --n-epochs 150

# Or call phases separately for more control
# Phase 1: Train model and extract embeddings (non-interactive, can be cached)
demba.prepare_id_correction('trial1_el.pickle')
# Phase 2: Interactive cluster mapping and ID assignment
demba.complete_id_correction('trial1_el.pickle')
```

**Key Parameters**:
- `--n-epochs`: Training epochs (default: 200, but early stopping typically terminates much earlier)
- `--min-silhouette`: Quality threshold for ID assignment (default: 0.2, range: -1 to 1)
- `--frame-stride`: Sampling density for patch cache (default: 5, lower = more data)
- `--min-overlap-frames`: Minimum co-occupancy duration for training pairs (default: 30)

See [Identity Correction: Technical Details](#identity-correction-technical-details) for in-depth explanation.

### 3. Tracklet Stitching
Combines short tracklets into continuous tracks based on learned identities.

```bash
python main.py stitch --tracklet-pickle trial1_el.pickle --output-h5 trial1_el.h5 --n-tracks 2
```

### 4. Temporal Filtering
Applies standard DeepLabCut temporal smoothing to reduce jitter in pose predictions.

```bash
python main.py filter --video trial1.mp4 --dlc-config config.yaml
```

### 5. Feature Extraction
Extracts behavioral features from the pose data. This module is specialized for our specific behaviors of interest,
and would need modification for more generalized applications. 

```bash
python main.py features --video trial1.mp4 --pose-h5 trial1_el.h5 --visualize
```

### 6. Visualization
Creates labeled videos with skeleton overlays and identity labels (using dlc.create_labeled_video)

```bash
python main.py visualize --video trial1.mp4 --dlc-config config.yaml
```

### 7. Analysis
Generates statistical plots and correlation analyses across all trials in a project.

```bash
python main.py analyze --project-dir /path/to/MyProject/Analysis
```

### 8. Batch Mode

**The Challenge**: When processing many videos, running the full pipeline sequentially requires you to be present for the interactive ID assignment step of each video, which interrupts the workflow.

**Batch Mode Solution**: Separates the pipeline into three phases:
1. **Phase 1 (Preparation)**: Runs pose estimation and ID model training for all videos (non-interactive, can run overnight)
2. **Phase 2 (Interactive)**: Interactive cluster mapping for all videos in one sitting
3. **Phase 3 (Finalization)**: Completes remaining pipeline stages for all videos (non-interactive)

This allows you to complete all interactive tasks at once, then let the rest run unattended.

**Usage**:
```bash
# Process entire project with one command
python main.py batch --project-dir /path/to/MyProject/Analysis

# Optional: manually specify annotations file
python main.py batch --project-dir /path/to/MyProject/Analysis \
    --quivering-annotations custom_annotations.xlsx
```

**Features**:
- Automatically discovers all trials in `Videos/` directory
- Auto-detects config file and annotations
- Tracks completion status per trial - can resume if interrupted
- Skips already-completed stages automatically
- Runs project-level analysis at the end

**When to Use**:
- ✅ Multiple videos to process (>3 trials)
- ✅ Want to minimize interactive time
- ✅ Processing overnight or on remote server
- ❌ Single trial (use `full` mode instead)
- ❌ Need fine control over individual stages

## Identity Correction: Technical Details

Identity correction addresses a common challenge in multi-animal behavioral analysis: maintaining consistent identity labels across long videos despite occlusions, close proximity, and visual similarity between individuals.

### The Problem

In my experience working with long cichlid videos (3.5 hours), DeepLabCut's standard tracking pipeline often failed when it came to reID and stitching short tracklets into long trajectories. Multiple approaches were attempted:

1. **Standard multi-animal tracking with SORT-style tracklet generation and graph-based tracklet stitching**: 
   - Produced identity swaps throughout the video (as expected for videos where all animals can leave the frame)
   - Often failed during graph-based stitching (possibly due to video length and large spatiotemporal distances between individual tracklets when animals leave frame)
2. **DLC's Transformer ReID + graph-based stitching**: Did not achieve ReID rates much above random, possibly because:
   - The pose estimation backbone features didn't capture the subtle visual differences between the two fish (distinguishable only to a skilled annotator)
   - Graph optimization still struggled, and the addition of often incorrect pose data likely did more harm than good
3. **Identity-aware pose estimation model** (training with `identity=True` after manually annotating male/female separately): Also unsuccessful
   - Compared to our ID-agnostic pose model (trained on ~850 images representing dozens of unique individuals) the supervised ID model (trained on 30 images from a single 2-individual trial) performed poorly at pose estimation
   - Poor pose estimation results obscured any possible underlying ID success
   - This method might work given larger training sets, but annotating 100+ images every time we run a new trial was not feasible for us.

The core challenge: maintaining consistent identity assignment across very long videos where visually similar animals frequently occlude each other and leave/return to the frame, while also minimizing manual annotation requirements

### The Solution: Self-Supervised Triplet Learning

DemBA uses a per-video CNN trained with triplet loss to learn discriminative visual features for each individual, then clusters these features to assign consistent identities. The system generates its own training data from the tracklet structure itself—no manual labeling required.

### Detailed Pipeline

#### 1. Co-Occupancy Detection
The system identifies frames where exactly two tracklets overlap—when both animals are visible but tracked separately. These "co-occupancy frames" are useful for contrastive learning because they contain:
- Same-individual examples (frames within a tracklet)
- Different-individual examples (frames from overlapping tracklets)

Parameters:
- `min_conf`: Minimum detection confidence (default: 0.5)
- `min_overlap_frames`: Minimum overlap duration to consider a tracklet pair valid (default: 10 frames)

#### 2. Patch Extraction
For each detection, extract a bounding box around all detected keypoints:
- Compute bbox from high-confidence keypoints (threshold: 0.25)
- Add padding (default: 10 pixels)
- Ensure minimum dimensions (4× padding) to avoid degenerate crops
- Resize to square patches (default: 128×128) maintaining aspect ratio
- Pad to square with black pixels if needed

This produces consistent visual representations centered on each animal regardless of pose or orientation.

#### 3. Triplet Generation
For each training iteration, sample a triplet:
```
Anchor:   Detection from Tracklet A at time t₁ (co-occupancy frame)
Positive: Detection from Tracklet A at time t₂ (different time, same animal)
Negative: Detection from Tracklet B at time t₁ (same time, different animal)
```

The co-occupancy constraint ensures anchor and negative show different individuals, while temporal continuity within tracklets ensures anchor and positive show the same individual.

Data augmentation: Random 90° rotations during preprocessing.

#### 4. CNN Architecture (SimpleCNN)
A compact and simple convolutional encoder optimized for embedding extraction:
```
Input: 128×128×3 RGB patch
→ Conv Block 1: 32 filters, BatchNorm, ReLU, MaxPool (→64×64)
→ Conv Block 2: 64 filters, BatchNorm, ReLU, MaxPool (→32×32)
→ Conv Block 3: 128 filters, BatchNorm, ReLU, MaxPool (→16×16)
→ Conv Block 4: 256 filters, BatchNorm, ReLU, MaxPool (→8×8)
→ Flatten + FC(512) + ReLU + Dropout(0.5)
→ FC(embedding_dim) [default: 128]
→ L2 Normalization
Output: 128-dimensional unit vector
```

**Architecture Visualization:**
```
                           SimpleCNN Architecture

┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: 128×128×3                            │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Conv2d (3 → 32)       │  kernel=3, padding=1
                    │   BatchNorm2d(32)       │
                    │   ReLU                  │
                    │   MaxPool2d(2×2)        │
                    └────────────┬────────────┘
                                 │ 64×64×32
                    ┌────────────▼────────────┐
                    │   Conv2d (32 → 64)      │  kernel=3, padding=1
                    │   BatchNorm2d(64)       │
                    │   ReLU                  │
                    │   MaxPool2d(2×2)        │
                    └────────────┬────────────┘
                                 │ 32×32×64
                    ┌────────────▼────────────┐
                    │   Conv2d (64 → 128)     │  kernel=3, padding=1
                    │   BatchNorm2d(128)      │
                    │   ReLU                  │
                    │   MaxPool2d(2×2)        │
                    └────────────┬────────────┘
                                 │ 16×16×128
                    ┌────────────▼────────────┐
                    │   Conv2d (128 → 256)    │  kernel=3, padding=1
                    │   BatchNorm2d(256)      │
                    │   ReLU                  │
                    │   MaxPool2d(2×2)        │
                    └────────────┬────────────┘
                                 │ 8×8×256
                    ┌────────────▼────────────┐
                    │   Flatten               │
                    └────────────┬────────────┘
                                 │ 16,384 features
                    ┌────────────▼────────────┐
                    │   Linear(16384 → 512)   │
                    │   ReLU                  │
                    │   Dropout(p=0.5)        │
                    └────────────┬────────────┘
                                 │ 512 features
                    ┌────────────▼────────────┐
                    │   Linear(512 → 128)     │
                    └────────────┬────────────┘
                                 │ 128 features
                    ┌────────────▼────────────┐
                    │   L2 Normalize          │
                    └────────────┬────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│               OUTPUT: 128-dim unit vector embedding                 │
└─────────────────────────────────────────────────────────────────────┘

Total Parameters: ~8.9M
Input: RGB image patches (128×128×3)
Output: L2-normalized embeddings (128-dim)
Loss: Triplet loss with margin=1.0
```

The L2 normalization ensures embeddings lie on a hypersphere, making cosine/Euclidean distances equivalent and improving clustering.

#### 5. Triplet Loss Training
The loss function:
```
L = max(0, ||f(anchor) - f(positive)||² - ||f(anchor) - f(negative)||² + margin)
```

This encourages:
- Small distances between same-individual pairs
- Large distances between different-individual pairs
- A margin of separation (default: 1.0)

Training features:
- Optimizer: Adam (lr=0.001)
- Learning rate scheduling: ReduceLROnPlateau (factor=0.5, patience=3)
- Early stopping: Stops if no improvement for 10 epochs
- Batch size: 32
- Epochs: 200 (but early stopping typically terminates much earlier)
- Samples per epoch: 1000 triplets

**Patch Caching**: To speed up training, patches are pre-extracted and cached with sparse sampling (every `frame_stride` frames, default: 5). This reduces cache size significantly while maintaining training diversity. Cache is saved to disk and reused across runs.

#### 6. Embedding Extraction
After training, extract embeddings for all detections in all tracklets (not just training samples). This produces an embedding for every detection in the video.

#### 7. KMeans Clustering
Cluster all embeddings into k=2 groups (assuming 2 animals):
```python
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)
```

**Silhouette Score**: For each detection, compute its silhouette score:
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
where:
- `a(i)`: Mean distance to other points in same cluster
- `b(i)`: Mean distance to nearest different cluster

Range: -1 (wrong cluster) to +1 (perfect cluster fit)

This provides a confidence metric for each ID assignment.

#### 8. Interactive Cluster Mapping
Since clusters are unlabeled (we don't know if cluster 0 is male or female), the system creates a side-by-side comparison video:
- Left panel: Segments from Cluster 0
- Right panel: Segments from Cluster 1
- Shows n_segments (default: 3) representative tracklets from each cluster
- Each segment is segment_duration_sec (default: 3) seconds long
- Only uses tracklets with >70% purity (most frames in one cluster)

User watches the video and provides mapping:
```
Cluster 0 is (male/female): male
Cluster 1 is (male/female): female
```

This interactive step establishes the cluster→identity mapping for the entire video, and is the only manual input required for the entire pipeline.

#### 9. ID Reassignment with Quality Filtering
Assign numeric IDs based on:
1. Cluster membership (from KMeans)
2. User mapping (cluster → semantic label → numeric ID)
3. Silhouette threshold (default: 0.2)

```python
if silhouette_score >= min_silhouette:
    ID = label_to_id[cluster_mapping[cluster]]  # 0 for male, 1 for female
else:
    ID = -1  # Low confidence, keep unassigned
```

The original tracklet file is backed up with timestamp, then replaced with corrected IDs.

### Output Files

Generated in `<video_dir>/id_correction/`:
- `encoder_model.pth`: Trained CNN weights
- `patch_cache.pkl`: Pre-extracted image patches (~100-500 MB)
- `embeddings.pkl`: Extracted embeddings for all detections
- `training_loss.png`: Loss curve and learning rate schedule
- `cluster_comparison.mp4`: Interactive comparison video
- `embedding_visualization.png`: PCA/t-SNE projections with silhouette scores
- `summary_statistics.txt`: Comprehensive statistics report
- `backup_<timestamp>.pickle`: Backup of original tracklets

### Quality Metrics

**Silhouette Score Interpretation**:
- `s > 0.5`: Strong confidence, well-separated
- `0.2 < s ≤ 0.5`: Moderate confidence (default threshold: 0.2)
- `0 < s ≤ 0.2`: Weak confidence, near cluster boundary
- `s < 0`: Likely misclassified

**Typical Results** (for well-separated individuals):
- Mean silhouette: 0.4-0.7
- Assigned IDs: 85-95% of detections
- Low confidence: 5-10% of detections
- Failed extractions: <5% of detections

**Tuning `min_silhouette`**:
- `0.0`: Lenient, maximize coverage (use if individuals are very distinct)
- `0.2`: Moderate, balanced precision/recall (recommended default)
- `0.5`: Strict, maximize precision (use if identities are critical)

### Best Practices

**When Identity Correction Works Best**:
- Individuals with visual differences (size, color, pattern)
- Videos with substantial co-occupancy (both animals visible together)
- Good pose estimation quality (high confidence keypoints)
- Consistent lighting and camera angle

**Troubleshooting**:
- **Low silhouette scores** (<0.3 mean): Individuals may be too similar visually. Try decreasing `frame_stride` to get more diverse training samples, or lowering `min_silhouette` to accept more assignments.
- **Few co-occupancy frames** (<100): Check if animals actually overlap in video. May need to adjust `min_conf` threshold.
- **Poor clustering separation**: Individuals may be visually identical. Consider using behavioral features (location preferences, movement patterns) for post-hoc ID refinement.
- **High training loss**: Increase `n_epochs` or adjust `lr` for more stable training.

**Computational Requirements**:
- GPU strongly recommended, especially for long videos
- default batch-size may need to be lowered for older GPU's

### Comparison with DeepLabCut's Transformer ReID

DemBA's identity correction was inspired by DeepLabCut's unsupervised ReID module but differs in key aspects of both the learning approach and how identity information is used during tracklet stitching.

#### Learning Approach

**DeepLabCut Transformer ReID:**
- **Architecture**: Transformer-based (4 attention heads, 4 blocks, 768-dim) with MLP output to 128-dim embeddings
- **Input features**: 2,048-dimensional features from the last layer of the pre-trained pose estimation CNN backbone
- **Feature type**: High-level "keypoint embeddings" containing visual information around each keypoint
- **Training data**: Triplets sampled from co-occupancy frames (anchor-positive from same tracklet, anchor-negative from different tracklets)
- **Output**: 128-dimensional appearance embeddings per detection

**DemBA Identity Correction:**
- **Architecture**: Compact CNN (4 conv blocks, 256 filters max) with fully connected layers to 128-dim embeddings
- **Input features**: Raw RGB image patches (128×128) extracted from bounding boxes around all keypoints
- **Feature type**: Full visual appearance of the animal (color, texture, patterns)
- **Training data**: Same triplet strategy from co-occupancy frames, with sparse patch caching for efficiency
- **Output**: 128-dimensional L2-normalized embeddings per detection

**Key Differences:**
- DLC leverages features from your existing ID-agnostic pose estimation model; DemBA learns appearance features from scratch per video
- DLC uses keypoint-level features; DemBA uses whole-animal appearance
- DemBA includes silhouette scores for quality filtering; DLC uses cosine similarity directly

#### Stitching Strategy

The more significant difference lies in how identity information is used during tracklet stitching:

**DeepLabCut Approach (Soft Constraint):**
- Builds a single graph containing all tracklets
- Uses ReID embeddings as an **additional cost term** in the graph optimization
- Identity similarity provides a **soft bias**: edges between same-identity tracklets get lower weights (0.01× base cost) while different-identity edges get higher weights (1.0× base cost)
- The global optimization can still connect different identities if spatial/temporal costs are favorable
- Relies on the graph optimization to balance identity, distance, motion, and temporal constraints

**DemBA Approach (Hard Constraint):**
- **Identifies and splits conjoined tracklets**: Detects tracklets that switch between tracking different individuals and splits them at identity boundaries (based on runs of consecutive frames with same ID)
- **Partitions tracklets by identity before stitching**: tracklets with ID=0 go into one group, ID=1 into another, ID=-1 (unassigned) are discarded
- Each identity group is stitched **independently** using simple temporal concatenation
- **No graph optimization across identities**: identity boundaries are strictly enforced
- Discards low-confidence detections (silhouette < threshold) rather than attempting to stitch them
- Much simpler stitching: just concatenates tracklets of the same ID in temporal order

**Code Comparison:**

DLC's soft constraint (from `stitch.py:1239-1244`):
```python
with_id = any(tracklet.identity != -1 for tracklet in stitcher)
if with_id and weight_func is None:
    def weight_func(t1, t2):
        w = 0.01 if t1.identity == t2.identity else 1
        return w * stitcher.calculate_edge_weight(t1, t2)
```

DemBA's hard partition (from `tracklet_stitching.py:93-151`):
```python
# Partition by identity
tracklets_by_id = {i: [] for i in range(n_tracks)}
tracklets_by_id[-1] = []  # For unassigned

for t in tracklets:
    identity = t.identity
    if identity in tracklets_by_id:
        tracklets_by_id[identity].append(t)

# Stitch each identity group independently
for identity in valid_ids:
    identity_tracklets = tracklets_by_id[identity]
    combined_track = identity_tracklets_sorted[0]
    for t in identity_tracklets_sorted[1:]:
        if len(t) >= min_length:
            combined_track = combined_track + t
```

## Configuration

Default parameters can be found in `demba/config.py`:

```python
DEFAULT_N_FISH = 2
DEFAULT_MIN_LIKELIHOOD = 0.5
DEFAULT_PATCH_SIZE = 128
DEFAULT_MOUTHING_DIST_MM = 10
VIDEO_FPS = 30
```

Parameters can be overridden via command-line arguments.

## Code Structure

```
DemBA/
├── demba/                      # Main package
│   ├── pose_estimation.py      # DeepLabCut integration
│   ├── identity_correction.py  # Identity correction pipeline
│   ├── tracklet_stitching.py   # Tracklet combination
│   ├── filtering.py            # Temporal filtering
│   ├── feature_extraction.py   # Behavioral feature detection
│   ├── visualization.py        # Video visualization
│   ├── analysis.py             # Statistical analysis
│   ├── file_manager.py         # TrialManager & ProjectManager
│   ├── config.py               # Configuration constants
│   └── utils/                  # Utility functions
│       ├── dlc.py              # DeepLabCut helpers
│       ├── roi.py              # ROI estimation
│       ├── gen_utils.py        # Batch mode utilities
│       └── metrics.py          # Evaluation metrics
├── main.py                     # CLI entry point
├── setup.py                    # Package installation
└── README.md                   # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. I'm especially open to help/suggestions regarding feature integration into the official DLC library

## License

This project is licensed under the GNU Lesser General Public License v3.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use DemBA in your research, please cite:

```bibtex
@software{demba2023,
  title = {DemBA: DeepLabCut-augmented Multi-animal Behavioral Analysis},
  author = {Lancaster, Tucker},
  year = {2023},
  url = {https://github.com/tlancaster6/DemBA}
}
```

## Acknowledgments

- Built around, and heavily inspired by, [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)
- Data and biological inspiration courtesy of Kathryn Leatherbury, Streelman Lab, Georgia Tech
- 

