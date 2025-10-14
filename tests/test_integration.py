"""Integration tests for DemBA pipeline with TrialManager/ProjectManager.

These tests verify that pipeline modules have the correct function signatures
and accept TrialManager/ProjectManager instances. They do NOT require a full
DLC project setup - they focus on signature checking and basic integration.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from demba.file_manager import TrialManager, ProjectManager


@pytest.fixture
def simple_mock_trial_manager():
    """Create a simple mock TrialManager without requiring DLC setup."""
    mock_tm = Mock(spec=TrialManager)
    mock_tm.trial_dir = Path("/mock/trial")
    mock_tm.config_path = Path("/mock/config.yaml")
    mock_tm.video_stem = "test_video"
    mock_tm.shuffle = 1
    mock_tm.scorer_name = "TestScorerJan01"
    mock_tm.video_path.return_value = Path("/mock/trial/test_video.mp4")
    mock_tm.el_pickle_path.return_value = Path("/mock/trial/test_video_el.pickle")
    mock_tm.full_pickle_path.return_value = Path("/mock/trial/test_video_full.pickle")
    mock_tm.stitched_h5_path.return_value = Path("/mock/trial/test_video_stitched.h5")
    mock_tm.filtered_h5_path.return_value = Path("/mock/trial/test_video_filtered.h5")
    mock_tm.is_stage_complete.return_value = False
    return mock_tm


@pytest.fixture
def simple_mock_project_manager():
    """Create a simple mock ProjectManager without requiring DLC setup."""
    mock_pm = Mock(spec=ProjectManager)
    mock_pm.project_dir = Path("/mock/project/Analysis")
    mock_pm.config_path = Path("/mock/project/config.yaml")
    mock_pm.videos_dir = Path("/mock/project/Analysis/Videos")
    mock_pm.annotations_dir = Path("/mock/project/Analysis/Annotations")
    mock_pm.list_trial_dirs.return_value = [
        Path("/mock/project/Analysis/Videos/trial1"),
        Path("/mock/project/Analysis/Videos/trial2"),
    ]
    return mock_pm


class TestPoseEstimationIntegration:
    """Test pose_estimation.py integration with TrialManager."""

    def test_estimate_pose_signature(self):
        """Test that estimate_pose has correct signature accepting TrialManager."""
        from demba.pose_estimation import estimate_pose
        import inspect

        sig = inspect.signature(estimate_pose)
        params = list(sig.parameters.keys())

        # First parameter should be trial_manager
        assert params[0] == 'trial_manager'
        # Should NOT have config_path or video_path as parameters
        assert 'config_path' not in params
        assert 'video_path' not in params


class TestIdentityCorrectionIntegration:
    """Test identity_correction.py integration with TrialManager."""

    def test_id_correction_signature(self):
        """Test that identity_correction.main has correct signature."""
        from demba.identity_correction import main as id_correction_main
        import inspect

        sig = inspect.signature(id_correction_main)
        params = list(sig.parameters.keys())

        # First parameter should be trial_manager
        assert params[0] == 'trial_manager'
        # Should NOT have tracklet_path as parameter
        assert 'tracklet_path' not in params


class TestTrackletStitchingIntegration:
    """Test tracklet_stitching.py integration with TrialManager."""

    def test_stitch_signature(self):
        """Test that stitch_by_identity has correct signature."""
        from demba.tracklet_stitching import stitch_by_identity
        import inspect

        sig = inspect.signature(stitch_by_identity)
        params = list(sig.parameters.keys())

        # First parameter should be trial_manager
        assert params[0] == 'trial_manager'
        # Should NOT have path parameters
        assert 'tracklet_pickle_path' not in params
        assert 'output_h5_path' not in params


class TestFilteringIntegration:
    """Test filtering.py integration with TrialManager."""

    def test_filter_signature(self):
        """Test that filter_predictions has correct signature."""
        from demba.filtering import filter_predictions
        import inspect

        sig = inspect.signature(filter_predictions)
        params = list(sig.parameters.keys())

        # First parameter should be trial_manager
        assert params[0] == 'trial_manager'
        # Should NOT have path parameters
        assert 'config_path' not in params
        assert 'video_path' not in params


class TestVisualizationIntegration:
    """Test visualization.py integration with TrialManager."""

    def test_create_labeled_video_signature(self):
        """Test that create_labeled_video has correct signature."""
        from demba.visualization import create_labeled_video
        import inspect

        sig = inspect.signature(create_labeled_video)
        params = list(sig.parameters.keys())

        # First parameter should be trial_manager
        assert params[0] == 'trial_manager'
        # Should NOT have path parameters
        assert 'config_path' not in params
        assert 'video_path' not in params

    def test_create_identity_grids_signature(self):
        """Test that create_identity_consistency_grids has correct signature."""
        from demba.visualization import create_identity_consistency_grids
        import inspect

        sig = inspect.signature(create_identity_consistency_grids)
        params = list(sig.parameters.keys())

        # First parameter should be trial_manager
        assert params[0] == 'trial_manager'
        # Should NOT have path parameters
        assert 'tracklet_pickle_path' not in params
        assert 'video_path' not in params


class TestFeatureExtractionIntegration:
    """Test feature_extraction.py integration with TrialManager."""

    def test_feature_extractor_signature(self):
        """Test that FeatureExtractor has correct signature."""
        from demba.feature_extraction import FeatureExtractor
        import inspect

        sig = inspect.signature(FeatureExtractor.__init__)
        params = list(sig.parameters.keys())

        # First parameter (after self) should be trial_manager
        assert params[1] == 'trial_manager'
        # Should NOT have path parameters
        assert 'video_path' not in params
        assert 'pose_h5_path' not in params

    def test_process_video_signature(self):
        """Test that process_video has correct signature."""
        from demba.feature_extraction import process_video
        import inspect

        sig = inspect.signature(process_video)
        params = list(sig.parameters.keys())

        # First parameter should be trial_manager
        assert params[0] == 'trial_manager'
        # Should NOT have path parameters
        assert 'video_path' not in params
        assert 'pose_h5_path' not in params


class TestAnalysisIntegration:
    """Test analysis.py integration with ProjectManager."""

    def test_plotter_signature(self):
        """Test that Plotter has correct signature."""
        from demba.analysis import Plotter
        import inspect

        sig = inspect.signature(Plotter.__init__)
        params = list(sig.parameters.keys())

        # First parameter (after self) should be project_manager
        assert params[1] == 'project_manager'
        # Should NOT have parent_dir parameter
        assert 'parent_dir' not in params


class TestMainPyIntegration:
    """Test main.py command functions with TrialManager."""

    def test_cmd_functions_exist(self):
        """Test that all cmd_* functions exist and are callable."""
        from main import (cmd_pose, cmd_id_correction, cmd_stitch,
                        cmd_filter, cmd_features, cmd_visualize, cmd_analyze)

        # Verify functions exist and are callable
        assert callable(cmd_pose)
        assert callable(cmd_id_correction)
        assert callable(cmd_stitch)
        assert callable(cmd_filter)
        assert callable(cmd_features)
        assert callable(cmd_visualize)
        assert callable(cmd_analyze)

    def test_get_trial_manager_exists(self):
        """Test that get_trial_manager helper exists."""
        from main import get_trial_manager
        assert callable(get_trial_manager)
