# DemBA

**DeepLabCut-augmented Multi-animal Behavioral Analysis**

A Python pipeline for pose estimation, identity correction, tracklet stitching, and behavioral feature extraction from multi-animal videos using DeepLabCut.

## Features

- **Pose Estimation**: Multi-animal pose tracking using DeepLabCut
- **Identity Correction**: Deep learning-based identity assignment using triplet loss and clustering
- **Tracklet Stitching**: Combine short tracklets into continuous identity tracks
- **Temporal Filtering**: Smooth pose trajectories over time
- **Feature Extraction**: Automated behavioral feature detection and quantification
- **Visualization**: Create labeled videos with pose overlays
- **Statistical Analysis**: Generate boxplots, correlation plots, and heatmaps

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
- DeepLabCut (included in the repository)

## Quick Start

### Command Line Interface

DemBA provides a comprehensive CLI with modular stages and a full pipeline mode.

```bash
# Run full pipeline end-to-end
python main.py full --video data/trial1.mp4 --dlc-config config.yaml

# Run individual stages
python main.py pose --video data/trial1.mp4 --dlc-config config.yaml
python main.py id-correction --tracklet-pickle data/trial1_el.pickle
python main.py stitch --tracklet-pickle data/trial1_el.pickle --output-h5 data/trial1_el.h5
python main.py filter --video data/trial1.mp4 --dlc-config config.yaml
python main.py features --video data/trial1.mp4 --pose-h5 data/trial1_el.h5
python main.py visualize --video data/trial1.mp4 --dlc-config config.yaml
python main.py analyze --parent-dir data/ --plots all
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

### 1. Pose Estimation
Runs DeepLabCut multi-animal tracking on input videos.

```bash
python main.py pose --video trial1.mp4 --dlc-config config.yaml --n-fish 2
```

### 2. Identity Correction
Uses a CNN with triplet loss to learn visual embeddings and correct identity swaps.

```bash
python main.py id-correction --tracklet-pickle trial1_el.pickle --n-epochs 50
```

### 3. Tracklet Stitching
Combines short tracklets into continuous tracks based on learned identities.

```bash
python main.py stitch --tracklet-pickle trial1_el.pickle --output-h5 trial1_el.h5 --n-tracks 2
```

### 4. Temporal Filtering
Applies temporal smoothing to reduce jitter in pose predictions.

```bash
python main.py filter --video trial1.mp4 --dlc-config config.yaml
```

### 5. Feature Extraction
Extracts behavioral features such as proximity, motion, and orientation.

```bash
python main.py features --video trial1.mp4 --pose-h5 trial1_el.h5 --visualize
```

### 6. Visualization
Creates labeled videos with skeleton overlays and identity labels.

```bash
python main.py visualize --video trial1.mp4 --dlc-config config.yaml
```

### 7. Analysis
Generates statistical plots and correlation analyses.

```bash
python main.py analyze --parent-dir data/ --plots boxplots correlation heatmaps
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

## Project Structure

```
DemBA/
├── demba/                      # Main package
│   ├── pose_estimation.py      # DeepLabCut integration
│   ├── identity_correction.py  # Identity correction pipeline
│   ├── tracklet_stitching.py   # Tracklet combination
│   ├── filtering.py            # Temporal filtering
│   ├── feature_extraction.py   # Behavioral feature detection
│   ├── visualization.py        # Video visualization
│   ├── config.py               # Configuration constants
│   └── utils/                  # Utility functions
│       ├── dlc.py              # DeepLabCut helpers
│       ├── roi.py              # ROI estimation
│       └── metrics.py          # Evaluation metrics
├── main.py                     # CLI entry point
├── setup.py                    # Package installation
└── README.md                   # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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

- Built on [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) for pose estimation
- Uses triplet loss and deep metric learning for identity tracking
- Developed for automated behavioral analysis of fish social interactions
