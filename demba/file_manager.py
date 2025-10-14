"""File management system for DemBA pipeline.

This module provides two main classes for managing file paths and tracking
analysis completion status:
- ProjectManager: Project-level file management
- TrialManager: Trial-level file management

The system uses DeepLabCut's config and scorer name generation to construct
exact file paths without wildcard matching.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from DeepLabCut.deeplabcut.core.config import read_config_as_dict
from DeepLabCut.deeplabcut.utils.auxiliaryfunctions import get_scorer_name


def get_dlc_scorer_name(config_path: Path, shuffle: int = 1,
                       training_fraction: float = 0.95) -> str:
    """
    Get the DeepLabCut scorer name for file path construction.

    This wraps DeepLabCut's get_scorer_name function to generate the scorer
    string used in DLC output filenames (e.g.,
    'DLC_Resnet50_demasoni_singlenucSep11shuffle3_detector_best-130_snapshot_best-130').

    Parameters
    ----------
    config_path : Path
        Path to DeepLabCut project config.yaml
    shuffle : int, optional
        DLC shuffle number (default: 1)
    training_fraction : float, optional
        DLC training fraction (default: 0.95)

    Returns
    -------
    scorer_name : str
        The scorer string for file naming

    Examples
    --------
    >>> config_path = Path('projects/myproject/config.yaml')
    >>> scorer = get_dlc_scorer_name(config_path, shuffle=3, training_fraction=0.95)
    >>> scorer
    'DLC_Resnet50_myprojectSep11shuffle3_detector_best-130_snapshot_best-130'
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config
    cfg = read_config_as_dict(str(config_path))

    # Get snapshot iteration from config (default to 'unknown' if not specified)
    # This matches the actual files which have 'best-130' or similar
    trainingsiterations = 'unknown'
    if 'snapshotindex' in cfg:
        snap_idx = cfg['snapshotindex']
        if snap_idx != -1:
            trainingsiterations = f'best-{snap_idx}'

    # Get scorer name
    scorer, _ = get_scorer_name(
        cfg,
        shuffle=shuffle,
        trainFraction=training_fraction,
        trainingsiterations=trainingsiterations,
        modelprefix='',
    )

    return scorer


class TrialManager:
    """
    Manages files and completion status for a single trial/video.

    This class provides:
    - Centralized file path construction for all pipeline stages
    - Completion status tracking via a registry file
    - Methods to check which pipeline stages have been completed

    Parameters
    ----------
    trial_dir : str or Path
        Path to the trial directory (e.g., 'Videos/video_name/')
    config_path : str or Path
        Path to DeepLabCut project config.yaml
    shuffle : int, optional
        DLC shuffle number (default: from demba.config)
    training_fraction : float, optional
        DLC training fraction (default: from demba.config)

    Attributes
    ----------
    trial_dir : Path
        Path to trial directory
    video_stem : str
        Video filename without extension
    scorer_name : str
        DLC scorer string for file naming
    config_path : Path
        Path to project config.yaml

    Examples
    --------
    >>> tm = TrialManager('Videos/myvideo/', 'config.yaml')
    >>> tm.video_path()
    WindowsPath('Videos/myvideo/myvideo.mp4')
    >>> tm.el_pickle_path()
    WindowsPath('Videos/myvideo/myvideoRLC_Resnet50_...shuffle1_..._el.pickle')
    >>> tm.is_stage_complete('pose_estimation')
    True
    """

    # Valid pipeline stage names
    VALID_STAGES = [
        'pose_estimation',
        'identity_correction',
        'tracklet_stitching',
        'filtering',
        'feature_extraction',
        'visualization',
    ]

    def __init__(self, trial_dir: Path, config_path: Path,
                 shuffle: Optional[int] = None,
                 training_fraction: Optional[float] = None):
        """Initialize TrialManager for a single trial."""
        from demba import config as demba_config

        self.trial_dir = Path(trial_dir).resolve()
        self.config_path = Path(config_path).resolve()

        if not self.trial_dir.exists():
            raise FileNotFoundError(f"Trial directory not found: {self.trial_dir}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        # Get video stem from directory name
        self.video_stem = self.trial_dir.name

        # Load defaults from config
        if shuffle is None:
            shuffle = demba_config.DEFAULT_SHUFFLE
        if training_fraction is None:
            training_fraction = demba_config.DEFAULT_TRAINING_FRACTION

        self.shuffle = shuffle
        self.training_fraction = training_fraction

        # Get DLC scorer name
        self.scorer_name = get_dlc_scorer_name(
            self.config_path,
            shuffle=shuffle,
            training_fraction=training_fraction
        )

        # Registry file path
        self._registry_path = self.trial_dir / '.demba_registry.json'

        # Load existing registry or create empty one
        self._registry = self._load_registry()

    # =========================================================================
    # File Path Methods
    # =========================================================================

    def video_path(self) -> Path:
        """Return path to video file."""
        return self.trial_dir / f"{self.video_stem}.mp4"

    def roi_path(self) -> Path:
        """Return path to ROI image."""
        return self.trial_dir / f"{self.video_stem}_roi.png"

    def full_pickle_path(self) -> Path:
        """Return path to full tracklet pickle file."""
        return self.trial_dir / f"{self.video_stem}{self.scorer_name}_full.pickle"

    def el_pickle_path(self) -> Path:
        """Return path to ellipse-tracked tracklet pickle file."""
        return self.trial_dir / f"{self.video_stem}{self.scorer_name}_el.pickle"

    def h5_path(self) -> Path:
        """Return path to pose estimation H5 file."""
        return self.trial_dir / f"{self.video_stem}{self.scorer_name}.h5"

    def csv_path(self) -> Path:
        """Return path to pose estimation CSV file."""
        return self.trial_dir / f"{self.video_stem}{self.scorer_name}.csv"

    def stitched_h5_path(self) -> Path:
        """Return path to stitched tracklets H5 file."""
        return self.trial_dir / f"{self.video_stem}{self.scorer_name}_el.h5"

    def stitched_csv_path(self) -> Path:
        """Return path to stitched tracklets CSV file."""
        return self.trial_dir / f"{self.video_stem}{self.scorer_name}_el.csv"

    def filtered_h5_path(self) -> Path:
        """Return path to filtered tracklets H5 file."""
        return self.trial_dir / f"{self.video_stem}{self.scorer_name}_el_filtered.h5"

    def filtered_csv_path(self) -> Path:
        """Return path to filtered tracklets CSV file."""
        return self.trial_dir / f"{self.video_stem}{self.scorer_name}_el_filtered.csv"

    def framefeatures_path(self, mouthing_dist_mm: float = 10,
                          likelihood: float = 0.5) -> Path:
        """
        Return path to frame features CSV file.

        Parameters
        ----------
        mouthing_dist_mm : float, optional
            Mouthing distance threshold in mm (default: 10)
        likelihood : float, optional
            Minimum keypoint likelihood (default: 0.5)
        """
        return self.trial_dir / (
            f"{self.video_stem}_mdist{mouthing_dist_mm:.0f}mm_"
            f"likelihood{likelihood}_framefeatures.csv"
        )

    def clipfeatures_path(self, mouthing_dist_mm: float = 10,
                         likelihood: float = 0.5) -> Path:
        """
        Return path to clip features CSV file.

        Parameters
        ----------
        mouthing_dist_mm : float, optional
            Mouthing distance threshold in mm (default: 10)
        likelihood : float, optional
            Minimum keypoint likelihood (default: 0.5)
        """
        return self.trial_dir / (
            f"{self.video_stem}_mdist{mouthing_dist_mm:.0f}mm_"
            f"likelihood{likelihood}_clipfeatures.csv"
        )

    def labeled_video_path(self, filtered: bool = True,
                          id_threshold: float = 0.25) -> Path:
        """
        Return path to labeled video file.

        Parameters
        ----------
        filtered : bool, optional
            Whether using filtered data (default: True)
        id_threshold : float, optional
            ID confidence threshold (default: 0.25)
        """
        filter_str = '_filtered' if filtered else ''
        id_str = f"_id_p{int(id_threshold * 100)}"
        return self.trial_dir / (
            f"{self.video_stem}{self.scorer_name}_el{filter_str}{id_str}_labeled.mp4"
        )

    def feature_vis_video_path(self, mouthing_dist_mm: float = 10,
                               likelihood: float = 0.5) -> Path:
        """
        Return path to feature visualization video.

        Parameters
        ----------
        mouthing_dist_mm : float, optional
            Mouthing distance threshold in mm (default: 10)
        likelihood : float, optional
            Minimum keypoint likelihood (default: 0.5)
        """
        return self.trial_dir / (
            f"{self.video_stem}_mdist{mouthing_dist_mm:.0f}mm_"
            f"likelihood{likelihood}_featurevis.mp4"
        )

    def id_correction_dir(self) -> Path:
        """Return path to ID correction output directory."""
        return self.trial_dir / 'id_correction'

    # =========================================================================
    # Registry Methods
    # =========================================================================

    def _load_registry(self) -> Dict:
        """Load completion registry from disk."""
        if self._registry_path.exists():
            with open(self._registry_path, 'r') as f:
                return json.load(f)
        else:
            # Initialize empty registry
            return {stage: {'completed': False, 'timestamp': None}
                   for stage in self.VALID_STAGES}

    def _save_registry(self):
        """Save completion registry to disk."""
        with open(self._registry_path, 'w') as f:
            json.dump(self._registry, f, indent=2)

    def mark_stage_complete(self, stage_name: str):
        """
        Mark a pipeline stage as complete.

        Parameters
        ----------
        stage_name : str
            Name of the stage (e.g., 'pose_estimation')

        Raises
        ------
        ValueError
            If stage_name is not a valid stage
        """
        if stage_name not in self.VALID_STAGES:
            raise ValueError(
                f"Invalid stage name '{stage_name}'. "
                f"Valid stages: {', '.join(self.VALID_STAGES)}"
            )

        self._registry[stage_name] = {
            'completed': True,
            'timestamp': datetime.now().isoformat()
        }
        self._save_registry()

    def is_stage_complete(self, stage_name: str) -> bool:
        """
        Check if a pipeline stage has been completed.

        Parameters
        ----------
        stage_name : str
            Name of the stage to check

        Returns
        -------
        bool
            True if stage is complete, False otherwise
        """
        if stage_name not in self.VALID_STAGES:
            raise ValueError(
                f"Invalid stage name '{stage_name}'. "
                f"Valid stages: {', '.join(self.VALID_STAGES)}"
            )

        return self._registry[stage_name]['completed']

    def get_completion_status(self) -> Dict[str, bool]:
        """
        Get completion status for all pipeline stages.

        Returns
        -------
        dict
            Dictionary mapping stage names to completion status (bool)
        """
        return {stage: info['completed']
                for stage, info in self._registry.items()}

    def reset_stage(self, stage_name: str):
        """
        Reset a stage to incomplete status.

        Parameters
        ----------
        stage_name : str
            Name of the stage to reset
        """
        if stage_name not in self.VALID_STAGES:
            raise ValueError(
                f"Invalid stage name '{stage_name}'. "
                f"Valid stages: {', '.join(self.VALID_STAGES)}"
            )

        self._registry[stage_name] = {
            'completed': False,
            'timestamp': None
        }
        self._save_registry()

    def __repr__(self):
        """String representation."""
        return f"TrialManager('{self.trial_dir.name}')"


class ProjectManager:
    """
    Manages project-level files and multiple trials.

    This class provides:
    - Discovery of all trial directories in a project
    - Batch access to TrialManager instances
    - Filtering trials by completion status
    - Project-level file management

    Parameters
    ----------
    project_dir : str or Path
        Path to project Analysis directory
    shuffle : int, optional
        DLC shuffle number (default: from demba.config)
    training_fraction : float, optional
        DLC training fraction (default: from demba.config)

    Attributes
    ----------
    project_dir : Path
        Path to project Analysis directory
    config_path : Path
        Path to project config.yaml
    videos_dir : Path
        Path to Videos/ subdirectory
    shuffle : int
        DLC shuffle number
    training_fraction : float
        DLC training fraction

    Examples
    --------
    >>> pm = ProjectManager('projects/myproject/Analysis')
    >>> trial_dirs = pm.list_trial_dirs()
    >>> len(trial_dirs)
    10
    >>> incomplete = pm.get_incomplete_trials('pose_estimation')
    >>> for tm in incomplete:
    ...     print(tm.video_stem)
    """

    def __init__(self, project_dir: Path, shuffle: Optional[int] = None,
                 training_fraction: Optional[float] = None):
        """Initialize ProjectManager for a project."""
        from demba import config as demba_config

        self.project_dir = Path(project_dir).resolve()

        if not self.project_dir.exists():
            raise FileNotFoundError(
                f"Project directory not found: {self.project_dir}"
            )

        # Find config.yaml (should be in parent directory)
        self.config_path = self._find_config()

        # Videos directory
        self.videos_dir = self.project_dir / 'Videos'
        if not self.videos_dir.exists():
            raise FileNotFoundError(
                f"Videos directory not found: {self.videos_dir}"
            )

        # Load defaults from config
        if shuffle is None:
            shuffle = demba_config.DEFAULT_SHUFFLE
        if training_fraction is None:
            training_fraction = demba_config.DEFAULT_TRAINING_FRACTION

        self.shuffle = shuffle
        self.training_fraction = training_fraction

    def _find_config(self) -> Path:
        """Find config.yaml by searching parent directories."""
        current = self.project_dir
        for _ in range(3):  # Search up to 3 levels
            config_path = current / 'config.yaml'
            if config_path.exists():
                return config_path
            current = current.parent

        raise FileNotFoundError(
            f"Could not find config.yaml in or above {self.project_dir}"
        )

    def list_trial_dirs(self) -> List[Path]:
        """
        List all trial directories in the Videos/ folder.

        Returns
        -------
        List[Path]
            List of paths to trial directories
        """
        # A trial directory should contain a .mp4 file with matching name
        trial_dirs = []
        for item in self.videos_dir.iterdir():
            if item.is_dir():
                # Check if there's a video file with the same name
                video_file = item / f"{item.name}.mp4"
                if video_file.exists():
                    trial_dirs.append(item)

        return sorted(trial_dirs)

    def get_trial_manager(self, trial_name: str) -> TrialManager:
        """
        Get a TrialManager for a specific trial.

        Parameters
        ----------
        trial_name : str
            Name of the trial directory

        Returns
        -------
        TrialManager
            TrialManager instance for the trial
        """
        trial_dir = self.videos_dir / trial_name
        if not trial_dir.exists():
            raise FileNotFoundError(f"Trial directory not found: {trial_dir}")

        return TrialManager(
            trial_dir,
            self.config_path,
            shuffle=self.shuffle,
            training_fraction=self.training_fraction
        )

    def get_all_trial_managers(self) -> List[TrialManager]:
        """
        Get TrialManager instances for all trials.

        Returns
        -------
        List[TrialManager]
            List of TrialManager instances
        """
        trial_dirs = self.list_trial_dirs()
        return [
            TrialManager(
                trial_dir,
                self.config_path,
                shuffle=self.shuffle,
                training_fraction=self.training_fraction
            )
            for trial_dir in trial_dirs
        ]

    def get_completed_trials(self, stage_name: str) -> List[TrialManager]:
        """
        Get all trials that have completed a specific stage.

        Parameters
        ----------
        stage_name : str
            Name of the pipeline stage

        Returns
        -------
        List[TrialManager]
            List of TrialManagers for completed trials
        """
        all_trials = self.get_all_trial_managers()
        return [tm for tm in all_trials if tm.is_stage_complete(stage_name)]

    def get_incomplete_trials(self, stage_name: str) -> List[TrialManager]:
        """
        Get all trials that have NOT completed a specific stage.

        Parameters
        ----------
        stage_name : str
            Name of the pipeline stage

        Returns
        -------
        List[TrialManager]
            List of TrialManagers for incomplete trials
        """
        all_trials = self.get_all_trial_managers()
        return [tm for tm in all_trials if not tm.is_stage_complete(stage_name)]

    def print_completion_summary(self):
        """Print a summary table of completion status for all trials."""
        all_trials = self.get_all_trial_managers()

        if not all_trials:
            print("No trials found in project.")
            return

        # Header
        stages = TrialManager.VALID_STAGES
        header = f"{'Trial':<30} " + " ".join(f"{s[:4]:>4}" for s in stages)
        print(header)
        print("-" * len(header))

        # Rows
        for tm in all_trials:
            status = tm.get_completion_status()
            status_str = " ".join(
                "  ✓ " if status[s] else "  - " for s in stages
            )
            trial_name = tm.video_stem[:30]  # Truncate if too long
            print(f"{trial_name:<30} {status_str}")

        # Summary counts
        print("\n" + "=" * len(header))
        print("Summary:")
        for stage in stages:
            completed = sum(1 for tm in all_trials if tm.is_stage_complete(stage))
            total = len(all_trials)
            print(f"  {stage:<25}: {completed}/{total} trials")

    def __repr__(self):
        """String representation."""
        return f"ProjectManager('{self.project_dir}')"
