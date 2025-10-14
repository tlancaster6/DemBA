"""Tests for demba.file_manager module."""

import pytest
import json
from pathlib import Path
from demba.file_manager import TrialManager, ProjectManager


class TestTrialManager:
    """Tests for TrialManager class."""

    @pytest.fixture
    def trial_manager(self, tmp_path):
        """Create a TrialManager instance for testing."""
        # Create mock directory structure
        trial_dir = tmp_path / "Videos" / "test_video"
        trial_dir.mkdir(parents=True)

        # Create mock video file
        video_file = trial_dir / "test_video.mp4"
        video_file.touch()

        # Create mock config.yaml
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
            import yaml
            yaml.dump(config_data, f)

        # Note: This will fail without actual DLC installation
        # For now, we'll test the parts that don't require DLC
        return trial_dir, config_path

    def test_file_paths(self, trial_manager):
        """Test that file paths are constructed correctly."""
        trial_dir, config_path = trial_manager

        # Skip if DLC not available
        try:
            tm = TrialManager(trial_dir, config_path, shuffle=1, training_fraction=0.95)
        except ImportError:
            pytest.skip("DeepLabCut not installed")

        # Test basic paths
        assert tm.video_path() == trial_dir / "test_video.mp4"
        assert tm.roi_path() == trial_dir / "test_video_roi.png"

        # Test that scorer_name is in the path
        el_pickle = tm.el_pickle_path()
        assert el_pickle.parent == trial_dir
        assert el_pickle.name.startswith("test_video")
        assert el_pickle.name.endswith("_el.pickle")

    def test_registry_operations(self, trial_manager):
        """Test completion registry operations."""
        trial_dir, config_path = trial_manager

        try:
            tm = TrialManager(trial_dir, config_path)
        except ImportError:
            pytest.skip("DeepLabCut not installed")

        # Initially, all stages should be incomplete
        assert not tm.is_stage_complete('pose_estimation')
        assert not tm.is_stage_complete('identity_correction')

        # Mark a stage complete
        tm.mark_stage_complete('pose_estimation')
        assert tm.is_stage_complete('pose_estimation')
        assert not tm.is_stage_complete('identity_correction')

        # Check registry file was created
        registry_path = trial_dir / '.demba_registry.json'
        assert registry_path.exists()

        # Check registry contents
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        assert registry['pose_estimation']['completed'] is True
        assert registry['identity_correction']['completed'] is False

        # Reset a stage
        tm.reset_stage('pose_estimation')
        assert not tm.is_stage_complete('pose_estimation')

    def test_invalid_stage_name(self, trial_manager):
        """Test that invalid stage names raise ValueError."""
        trial_dir, config_path = trial_manager

        try:
            tm = TrialManager(trial_dir, config_path)
        except ImportError:
            pytest.skip("DeepLabCut not installed")

        with pytest.raises(ValueError, match="Invalid stage name"):
            tm.mark_stage_complete('invalid_stage')

        with pytest.raises(ValueError, match="Invalid stage name"):
            tm.is_stage_complete('invalid_stage')

    def test_feature_paths_with_parameters(self, trial_manager):
        """Test feature extraction paths with custom parameters."""
        trial_dir, config_path = trial_manager

        try:
            tm = TrialManager(trial_dir, config_path)
        except ImportError:
            pytest.skip("DeepLabCut not installed")

        # Test with default parameters
        frame_features = tm.framefeatures_path()
        assert "mdist10mm_likelihood0.5_framefeatures.csv" in str(frame_features)

        # Test with custom parameters
        frame_features = tm.framefeatures_path(mouthing_dist_mm=15, likelihood=0.7)
        assert "mdist15mm_likelihood0.7_framefeatures.csv" in str(frame_features)

        clip_features = tm.clipfeatures_path(mouthing_dist_mm=15, likelihood=0.7)
        assert "mdist15mm_likelihood0.7_clipfeatures.csv" in str(clip_features)


class TestProjectManager:
    """Tests for ProjectManager class."""

    @pytest.fixture
    def project_manager(self, tmp_path):
        """Create a ProjectManager test environment."""
        # Create project structure
        project_dir = tmp_path / "test_project" / "Analysis"
        videos_dir = project_dir / "Videos"
        videos_dir.mkdir(parents=True)

        # Create mock config
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
            import yaml
            yaml.dump(config_data, f)

        # Create mock trial directories
        for i in range(3):
            trial_dir = videos_dir / f"trial_{i}"
            trial_dir.mkdir()
            video_file = trial_dir / f"trial_{i}.mp4"
            video_file.touch()

        return project_dir

    def test_list_trial_dirs(self, project_manager):
        """Test listing trial directories."""
        try:
            pm = ProjectManager(project_manager)
        except ImportError:
            pytest.skip("DeepLabCut not installed")

        trial_dirs = pm.list_trial_dirs()
        assert len(trial_dirs) == 3
        assert all(d.name.startswith("trial_") for d in trial_dirs)

    def test_get_trial_manager(self, project_manager):
        """Test getting a specific TrialManager."""
        try:
            pm = ProjectManager(project_manager)
        except ImportError:
            pytest.skip("DeepLabCut not installed")

        tm = pm.get_trial_manager("trial_0")
        assert tm.video_stem == "trial_0"
        assert tm.trial_dir.name == "trial_0"

    def test_completion_filtering(self, project_manager):
        """Test filtering trials by completion status."""
        try:
            pm = ProjectManager(project_manager)
        except ImportError:
            pytest.skip("DeepLabCut not installed")

        all_trials = pm.get_all_trial_managers()
        assert len(all_trials) == 3

        # Mark some trials complete
        all_trials[0].mark_stage_complete('pose_estimation')
        all_trials[1].mark_stage_complete('pose_estimation')

        # Test filtering
        completed = pm.get_completed_trials('pose_estimation')
        incomplete = pm.get_incomplete_trials('pose_estimation')

        assert len(completed) == 2
        assert len(incomplete) == 1
        assert completed[0].video_stem in ['trial_0', 'trial_1']
        assert incomplete[0].video_stem == 'trial_2'

    def test_config_not_found(self, tmp_path):
        """Test error when config.yaml is not found."""
        project_dir = tmp_path / "no_config" / "Analysis"
        videos_dir = project_dir / "Videos"
        videos_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Could not find config.yaml"):
            ProjectManager(project_dir)


def test_get_dlc_scorer_name():
    """Test DLC scorer name generation."""
    from demba.file_manager import get_dlc_scorer_name

    # This test requires actual DLC installation and valid config
    # For now, we just test that the function exists and has correct signature
    assert callable(get_dlc_scorer_name)

    # Test with non-existent config
    with pytest.raises(FileNotFoundError):
        get_dlc_scorer_name(Path("nonexistent_config.yaml"))
