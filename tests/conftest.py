"""Shared pytest fixtures for DemBA tests."""

import pytest
import yaml
from pathlib import Path
from demba.file_manager import TrialManager, ProjectManager


@pytest.fixture
def mock_dlc_config(tmp_path):
    """Create a mock DeepLabCut config.yaml file.

    Returns
    -------
    Path
        Path to the created config.yaml file
    """
    config_path = tmp_path / "config.yaml"
    config_data = {
        'Task': 'test_task',
        'scorer': 'TestScorer',
        'date': 'Jan01',
        'TrainingFraction': [0.95],
        'iteration': 0,
        'default_net_type': 'resnet_50',
        'snapshotindex': -1,
        'project_path': str(tmp_path),  # Required by DLC
    }
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def mock_trial_structure(tmp_path, mock_dlc_config):
    """Create a complete mock trial directory structure.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'trial_dir': Path to trial directory
        - 'video_path': Path to mock video file
        - 'config_path': Path to config.yaml
    """
    # Create trial directory
    trial_dir = tmp_path / "Videos" / "test_trial"
    trial_dir.mkdir(parents=True)

    # Create mock video file
    video_path = trial_dir / "test_trial.mp4"
    video_path.touch()

    return {
        'trial_dir': trial_dir,
        'video_path': video_path,
        'config_path': mock_dlc_config
    }


@pytest.fixture
def mock_trial_manager(mock_trial_structure):
    """Create a TrialManager instance with mock data.

    Returns
    -------
    TrialManager or None
        TrialManager instance, or None if DLC is not installed
    """
    try:
        tm = TrialManager(
            trial_dir=mock_trial_structure['trial_dir'],
            config_path=mock_trial_structure['config_path'],
            shuffle=1,
            training_fraction=0.95
        )
        return tm
    except ImportError:
        pytest.skip("DeepLabCut not installed")


@pytest.fixture
def mock_project_structure(tmp_path):
    """Create a complete mock project directory structure.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'project_dir': Path to project Analysis directory
        - 'videos_dir': Path to Videos directory
        - 'config_path': Path to config.yaml
        - 'trial_dirs': List of trial directory paths
    """
    # Create project structure
    project_dir = tmp_path / "test_project" / "Analysis"
    videos_dir = project_dir / "Videos"
    videos_dir.mkdir(parents=True)

    # Create annotations directory
    annotations_dir = project_dir / "Annotations"
    annotations_dir.mkdir()

    # Create config in parent directory (as ProjectManager expects)
    config_path = tmp_path / "test_project" / "config.yaml"
    config_data = {
        'Task': 'test_task',
        'scorer': 'TestScorer',
        'date': 'Jan01',
        'TrainingFraction': [0.95],
        'iteration': 0,
        'default_net_type': 'resnet_50',
        'snapshotindex': -1,
        'project_path': str(tmp_path / "test_project"),  # Required by DLC
    }
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)

    # Create mock trial directories
    trial_dirs = []
    for i in range(3):
        trial_dir = videos_dir / f"trial_{i}"
        trial_dir.mkdir()
        video_file = trial_dir / f"trial_{i}.mp4"
        video_file.touch()
        trial_dirs.append(trial_dir)

    return {
        'project_dir': project_dir,
        'videos_dir': videos_dir,
        'annotations_dir': annotations_dir,
        'config_path': config_path,
        'trial_dirs': trial_dirs
    }


@pytest.fixture
def mock_project_manager(mock_project_structure):
    """Create a ProjectManager instance with mock data.

    Returns
    -------
    ProjectManager or None
        ProjectManager instance, or None if DLC is not installed
    """
    try:
        pm = ProjectManager(mock_project_structure['project_dir'])
        return pm
    except ImportError:
        pytest.skip("DeepLabCut not installed")


@pytest.fixture
def test_video_path():
    """Get path to the test video resource.

    Returns
    -------
    Path
        Path to test_clip.mp4 in tests/resources/
    """
    resources_dir = Path(__file__).parent / "resources"
    video_path = resources_dir / "test_clip.mp4"

    if not video_path.exists():
        pytest.skip(f"Test video not found: {video_path}")

    return video_path


@pytest.fixture(scope="session")
def test_resources_dir():
    """Get path to the test resources directory.

    Returns
    -------
    Path
        Path to tests/resources/ directory
    """
    return Path(__file__).parent / "resources"
