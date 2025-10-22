# DemBA Repository Orientation Guide

**Last Updated:** 2025-10-22
**Purpose:** Quick orientation for AI assistants to minimize token usage and expedite inquiries

---

## Quick Overview

**Project Name:** DemBA (DeepLabCut Multi-animal Behavioral Analysis with Improved re-ID)
**Primary Use Case:** Cichlid fish behavioral video analysis (Streelman & McGrath labs, Georgia Tech)
**Core Technology:** Python pipeline built on top of DeepLabCut for multi-animal pose tracking

**Key Innovation:** CNN-based per-video identity correction using triplet loss, combined with hard-partition tracklet stitching to maintain consistent identity assignment across long videos (tested on 3.5-hour videos).

---

## Repository Structure

### Core Package: `demba/`

```
demba/
├── __init__.py                # Package exports and version (v0.1.0)
├── config.py                  # All pipeline configuration constants
├── file_manager.py            # TrialManager & ProjectManager classes for file path management
├── pose_estimation.py         # Stage 1: DeepLabCut pose estimation wrapper
├── identity_correction.py     # Stage 2: CNN triplet-loss ID correction (CORE MODULE)
├── tracklet_stitching.py      # Stage 3: Hard-partition stitching by identity
├── filtering.py               # Stage 4: Temporal filtering wrapper
├── feature_extraction.py      # Stage 5: Behavioral feature detection (cichlid-specific)
├── visualization.py           # Stage 6: Labeled video creation
├── analysis.py                # Stage 7: Statistical plots and analysis
└── utils/
    ├── dlc.py                 # DeepLabCut file loading utilities
    ├── roi.py                 # ROI estimation and video cropping
    ├── metrics.py             # Evaluation metrics (phi, jaccard)
    └── gen_utils.py           # Batch mode utilities (metadata save/load)
```

### Entry Points

- **`main.py`**: CLI entry point with argparse for all pipeline stages
- **`setup.py`**: Package installation configuration

### External Dependencies

- **`DeepLabCut/`**: External library (EXCLUDE from analysis, treat as black box)
- **`.claude/`**: AI assistant configuration (not committed)

---

## Architecture: File Management System

### TrialManager Class (`file_manager.py`)

**Purpose:** Manages all file paths and completion status for a single trial/video.

**Key Responsibilities:**
- Constructs exact file paths using DeepLabCut scorer naming conventions
- Tracks pipeline completion status via `.demba_registry.json`
- Provides methods to check which stages are complete

**Important Attributes:**
- `trial_dir`: Path to trial directory (e.g., `Videos/trial1/`)
- `video_stem`: Video filename without extension
- `scorer_name`: DLC scorer string (e.g., `DLC_Resnet50_..._shuffle1_..._el`)
- `config_path`: Path to DeepLabCut config.yaml

**Key Methods:**
```python
tm.video_path()                    # Videos/trial1/trial1.mp4
tm.el_pickle_path()                # trial1<scorer>_el.pickle (tracklets)
tm.h5_path()                       # trial1<scorer>.h5 (pose data)
tm.stitched_h5_path()              # trial1<scorer>_el.h5 (stitched tracks)
tm.filtered_h5_path()              # trial1<scorer>_el_filtered.h5
tm.id_correction_dir()             # trial1/id_correction/
tm.is_stage_complete('stage_name') # Check completion status
tm.mark_stage_complete('stage_name') # Mark stage as done
```

**Valid Pipeline Stages:**
1. `pose_estimation`
2. `identity_correction`
3. `tracklet_stitching`
4. `filtering`
5. `feature_extraction`
6. `visualization`

### ProjectManager Class (`file_manager.py`)

**Purpose:** Manages project-level operations across multiple trials.

**Key Responsibilities:**
- Discovers all trial directories in `Videos/`
- Provides batch access to TrialManager instances
- Filters trials by completion status
- Auto-detects quivering annotations

**Expected Project Structure:**
```
MyProject/
├── config.yaml                    # DeepLabCut project config
└── Analysis_<date>/               # Analysis directory
    ├── Videos/                    # Required: Trial video directories
    │   ├── trial1/               # Each trial in its own directory
    │   │   ├── trial1.mp4        # Video file (name matches directory)
    │   │   ├── trial1_roi.png    # ROI image (optional)
    │   │   └── ...               # Pipeline outputs created here
    │   ├── trial2/
    │   └── ...
    └── Annotations/               # Optional: Manual annotations
        └── quivering_annotations.xlsx  # Auto-detected by batch mode
```

**Key Methods:**
```python
pm.list_trial_dirs()               # List all trial directories
pm.get_trial_manager('trial1')     # Get TrialManager for specific trial
pm.get_all_trial_managers()        # Get all TrialManagers
pm.get_incomplete_trials('stage')  # Filter by completion status
pm.find_quivering_annotations()    # Auto-detect annotations file
```

---

## Pipeline Stages (Execution Order)

### 1. Pose Estimation (`pose_estimation.py`)
- **What:** Runs DeepLabCut multi-animal pose estimation
- **Input:** Video file, config.yaml
- **Output:**
  - `*_full.pickle`: Raw detections
  - `*_el.pickle`: SORT-style tracklets (identity-unaware)
- **Functions:** `estimate_pose(trial_manager, n_fish, force_rerun)`

### 2. Identity Correction (`identity_correction.py`) ⭐ CORE MODULE
- **What:** Trains CNN with triplet loss to assign consistent identities
- **Input:** `*_el.pickle` (tracklets from pose estimation)
- **Output:**
  - `id_correction/encoder_model.pth`: Trained CNN
  - `id_correction/patch_cache.pkl`: Pre-extracted image patches
  - `id_correction/embeddings.pkl`: Per-detection embeddings
  - `id_correction/cluster_comparison.mp4`: Interactive mapping video
  - `*_el.pickle` (updated with corrected IDs)

**Two-Phase Approach:**
1. **Preparation Phase (non-interactive):**
   - `train_id_model()`: Train CNN, extract embeddings, cluster
   - Saves metadata to `batch_metadata.json`

2. **Interactive Phase:**
   - `map_clusters_to_sex()`: User maps clusters to identities
   - Saves mapping to `cluster_mapping.json`

3. **Finalization Phase (non-interactive):**
   - `assign_corrected_ids()`: Apply mapping to tracklets

**Key Classes:**
- `PatchExtractor`: Extract RGB patches from video based on keypoints
- `CoOccupancyDetector`: Find frames where both animals are visible
- `TripletDataset`: Generate triplets for contrastive learning
- `SimpleCNN`: 4-layer CNN encoder (outputs 128-dim embeddings)
- `TripletLoss`: Contrastive loss with margin

**Architecture Details:**
- Input: 128×128 RGB patches
- CNN: 4 conv blocks (32→64→128→256 filters) + FC(512) + FC(128)
- Loss: Triplet loss with margin=1.0
- Training: Adam optimizer, early stopping, LR scheduling
- Clustering: KMeans (k=2) with silhouette score filtering
- Default threshold: silhouette ≥ 0.2 for ID assignment

### 3. Tracklet Stitching (`tracklet_stitching.py`)
- **What:** Combines short tracklets into continuous tracks (hard partition by identity)
- **Input:** `*_el.pickle` (with corrected IDs)
- **Output:**
  - `*_el.h5`: Stitched pose data in DLC format
  - `*_el.csv`: CSV version
- **Key Feature:** Splits "conjoined" tracklets (those that switch identities) before stitching
- **Functions:** `stitch_by_identity(trial_manager, n_tracks, min_length, animal_names)`

### 4. Temporal Filtering (`filtering.py`)
- **What:** Applies DeepLabCut's temporal smoothing to reduce jitter
- **Input:** `*_el.h5`
- **Output:** `*_el_filtered.h5`, `*_el_filtered.csv`
- **Functions:** `filter_predictions(trial_manager)`

### 5. Feature Extraction (`feature_extraction.py`)
- **What:** Extracts behavioral features (CICHLID-SPECIFIC)
- **Input:** Video + filtered pose data + optional annotations
- **Output:**
  - `*_framefeatures.csv`: Per-frame features
  - `*_clipfeatures.csv`: Per-clip aggregated features
  - `*_featurevis.mp4`: Visualization video (if enabled)
- **Key Features:** Mouthing events, double occupancy, spawning bouts, position tracking
- **Functions:** `process_video(trial_manager, quivering_annotation_path, visualize, n_minutes, min_likelihood)`

### 6. Visualization (`visualization.py`)
- **What:** Creates labeled videos with skeleton overlays
- **Input:** Video + filtered pose data + config
- **Output:**
  - `*_labeled.mp4`: DLC labeled video
  - Identity consistency grids
- **Functions:** `create_labeled_video(trial_manager)`, `create_identity_consistency_grids(trial_manager)`

### 7. Analysis (`analysis.py`)
- **What:** Statistical analysis across all trials (PROJECT-LEVEL)
- **Input:** All trial feature CSVs
- **Output:** Boxplots, correlation plots, heatmaps
- **Functions:** `Plotter` class with methods for each plot type

---

## Command-Line Interface (`main.py`)

### Execution Modes

**Individual Stages:**
```bash
python main.py pose --video Videos/trial1/trial1.mp4 --dlc-config config.yaml
python main.py id-correction --video Videos/trial1/trial1.mp4 --dlc-config config.yaml
python main.py stitch --video Videos/trial1/trial1.mp4 --dlc-config config.yaml
python main.py filter --video Videos/trial1/trial1.mp4 --dlc-config config.yaml
python main.py features --video Videos/trial1/trial1.mp4 --dlc-config config.yaml
python main.py visualize --video Videos/trial1/trial1.mp4 --dlc-config config.yaml
python main.py analyze --project-dir /path/to/Analysis
```

**Full Pipeline (Single Trial):**
```bash
python main.py full --video Videos/trial1/trial1.mp4 --dlc-config config.yaml
```

**Batch Mode (Multiple Trials):**
```bash
python main.py batch --project-dir /path/to/Analysis
```

**Batch Mode Flow:**
- **Phase 1 (Preparation):** Pose + ID training (non-interactive, can run overnight)
- **Phase 2 (Interactive):** Cluster mapping for all trials (batched user input)
- **Phase 3 (Finalization):** ID assignment + remaining pipeline (non-interactive)

**Key Benefit:** All interactive steps done at once, rest runs unattended.

---

## Configuration System (`config.py`)

### Key Constants

**Shared Parameters:**
- `VIDEO_FPS = 30`
- `DEFAULT_N_FISH = 2`
- `DEFAULT_SHUFFLE = 1`
- `ROI_RADIUS_MM = 79.375` (breeding pipe radius)

**ID Correction Parameters (Most Important):**
- `DEFAULT_PATCH_SIZE = 128`
- `DEFAULT_PADDING = 10`
- `DEFAULT_ID_N_EPOCHS = 200` (early stopping usually terminates earlier)
- `DEFAULT_ID_BATCH_SIZE = 32`
- `DEFAULT_MIN_SILHOUETTE = 0.2` (quality threshold for ID assignment)
- `DEFAULT_ID_CACHE_FRAME_STRIDE = 5` (sample every Nth frame for caching)
- `DEFAULT_MIN_TRACKLET_LENGTH = 60`
- `DEFAULT_MIN_OVERLAP_FRAMES = 30`

**Feature Extraction Parameters:**
- `DEFAULT_MOUTHING_DIST_MM = 10`
- `DEFAULT_MIN_LIKELIHOOD = 0.5`

---

## Common Workflows

### Troubleshooting ID Correction

**Problem:** Low silhouette scores (<0.3 mean)
- **Cause:** Individuals too visually similar
- **Solutions:**
  1. Decrease `frame_stride` for more diverse training samples
  2. Lower `min_silhouette` threshold to accept more assignments
  3. Check if individuals actually have visual differences

**Problem:** Few co-occupancy frames (<100)
- **Cause:** Animals rarely visible together
- **Solutions:**
  1. Adjust `conf_threshold` (lower to include more detections)
  2. Verify both animals are in the video
  3. Check video quality

**Problem:** Poor clustering separation
- **Cause:** Individuals visually identical
- **Solutions:**
  1. Consider using behavioral features for post-hoc refinement
  2. Use location preferences or movement patterns
  3. May need different approach than visual appearance

### File Path Resolution

**Given:** Video at `Videos/trial1/trial1.mp4` and `config.yaml`

**Expected Files:**
1. After pose: `Videos/trial1/trial1<scorer>_el.pickle`
2. After ID: `Videos/trial1/id_correction/encoder_model.pth`
3. After stitching: `Videos/trial1/trial1<scorer>_el.h5`
4. After filtering: `Videos/trial1/trial1<scorer>_el_filtered.h5`
5. After features: `Videos/trial1/trial1_mdist10mm_likelihood0.5_framefeatures.csv`

**Scorer Format:** `DLC_<backbone>_<project><date>shuffle<n>_detector_<snapshot>_snapshot_<snapshot>`

---

## Python API Usage

```python
import demba

# Create trial manager
from demba.file_manager import TrialManager
tm = TrialManager(
    trial_dir='Videos/trial1',
    config_path='config.yaml'
)

# Run individual stages
demba.estimate_pose(trial_manager=tm, n_fish=2)
demba.identity_correction.train_id_model(tracklet_path=tm.el_pickle_path())
cluster_mapping = demba.identity_correction.map_clusters_to_sex(prep_data)
demba.stitch_by_identity(trial_manager=tm, n_tracks=2)
demba.filter_predictions(trial_manager=tm)
demba.process_video(trial_manager=tm, visualize=False)
```

---

## Development Context

### Git Status (Snapshot at Conversation Start)
- **Current Branch:** `tucker_dev`
- **Main Branch:** `main`
- **Modified Files:**
  - `demba/tracklet_stitching.py`
  - `main.py`
- **Untracked:**
  - `.claude/` (AI config)
  - `main_old.py` (backup)

### Recent Commits (Last 5)
1. `7dd44d0` - Fixing missed pickle update between steps for batch operation
2. `399afed` - Reweighting clip choice function for interactive ID
3. `7a6ebdf` - Fixing tkinter thread safety violation
4. `89d1b25` - Adding batch mode
5. `41e7b82` - Adding batch mode

---

## Key Design Patterns

### 1. Hard-Partition Stitching (vs. DLC's Soft-Constraint)
- **DLC Approach:** Graph optimization with ReID as soft bias
- **DemBA Approach:**
  1. Split tracklets at identity boundaries
  2. Partition by identity before stitching
  3. Stitch each identity group independently
  4. No cross-identity graph optimization

**Benefit:** Stricter identity enforcement, simpler, more robust on long videos

### 2. Batch Mode Architecture
**Challenge:** Interactive step blocks automation

**Solution:** Three-phase separation
1. **Preparation:** All non-interactive work (pose + training)
2. **Interactive:** All user input at once (cluster mapping)
3. **Finalization:** All remaining work (stitching + features)

**Implementation:** Uses metadata caching system in `utils/gen_utils.py`

### 3. Registry-Based Completion Tracking
- Each trial has `.demba_registry.json` tracking stage completion
- Enables resume-on-failure for batch processing
- Prevents redundant computation

---

## Feature Extraction Specifics (Cichlid-Focused)

### Behavioral Features Detected
1. **Mouthing Events:** Male nose near female lateral line stripe
   - Threshold: `DEFAULT_MOUTHING_DIST_MM = 10mm`
   - Clustering: DBSCAN with `eps=5`, `min_samples=10`

2. **Double Occupancy:** Both fish in ROI simultaneously
   - Clustering: DBSCAN with `eps=30`, `min_samples=30`

3. **Spawning Bouts:** Clusters of mouthing events
   - Clustering: DBSCAN on mouthing endpoints with `eps=300`, `min_samples=6`

4. **Position Tracking:** Coordinates relative to ROI center

**Output Files:**
- `framefeatures.csv`: Per-frame boolean flags and continuous values
- `clipfeatures.csv`: Aggregated statistics per behavioral bout

---

## Important Notes for AI Assistants

### When Answering Questions

1. **File Path Questions:** Always reference `TrialManager` or `ProjectManager` methods
2. **Pipeline Stage Questions:** Check `VALID_STAGES` list and execution order
3. **Configuration Questions:** Reference `config.py` constants
4. **ID Correction Questions:** This is the core innovation, explain thoroughly
5. **Batch Mode Questions:** Emphasize three-phase structure

### Common User Intents

**"How do I run X?"**
→ Provide CLI command + mention corresponding function in API

**"Where is file Y located?"**
→ Reference TrialManager method that constructs the path

**"Pipeline stage Z failed"**
→ Check if it's ID correction (most complex), provide troubleshooting steps

**"How does identity correction work?"**
→ Explain triplet loss, co-occupancy detection, clustering, interactive mapping

**"What's different from DeepLabCut?"**
→ Highlight: (1) CNN vs Transformer ReID, (2) Hard-partition vs soft-constraint stitching, (3) Unsupervised vs supervised ID

### Token-Saving Strategies

1. **Don't re-read README.md** unless user asks about high-level concepts
2. **Use this guide** for file paths, stage order, and architecture
3. **Reference config.py** for parameter defaults instead of searching code
4. **Use file_manager.py** for understanding file naming conventions
5. **Focus on identity_correction.py** only when ID-related questions arise

---

## Quick Reference: File Extensions

- `.mp4`: Video files
- `.pickle`: DLC tracklet data (serialized Tracklet objects)
- `.h5`: Pose data in HDF5 format (DLC standard)
- `.csv`: Pose data in CSV format
- `.pth`: PyTorch model weights
- `.pkl`: Cached patch data or embeddings
- `.json`: Metadata (registry, cluster mapping, batch metadata)
- `.png`: Images (ROI, visualizations)
- `.xlsx`: Excel annotations (manual behavioral annotations)

---

## Conclusion

This guide provides the essential orientation for understanding the DemBA codebase. For detailed algorithmic questions, refer to specific module docstrings. For usage questions, consult the CLI commands and API examples above.

**Core Takeaway:** DemBA is a 7-stage pipeline where identity correction (Stage 2) is the key innovation, using CNN triplet loss and hard-partition stitching to maintain consistent identities across long multi-animal videos.
