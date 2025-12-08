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


@pytest.mark.slow
@pytest.mark.e2e
def test_full_pipeline_end_to_end(e2e_dlc_config, e2e_test_video_dir):
    """Run the complete DemBA pipeline end-to-end on test video.

    This test runs all pipeline stages on a 60-second test clip:
    1. Pose estimation
    2. Identity correction
    3. Tracklet stitching
    4. Filtering
    5. Feature extraction
    6. Visualization

    All outputs are written to tests/BHVE_group9_316800-318599/ directory.
    """
    from argparse import Namespace
    from main import cmd_full
    from demba import config
    import shutil

    # Video filename
    video_filename = "BHVE_group9_316800-318599.mp4"

    # Mock input() for interactive cluster mapping during ID correction
    # Returns 'male' for cluster 0, 'female' for cluster 1
    with patch('builtins.input', side_effect=['male', 'female']):
        try:
            # Create args namespace mimicking command-line arguments
            args = Namespace(
                video=e2e_test_video_dir / video_filename,
                dlc_config=e2e_dlc_config,
                parent_dir=None,  # Skip analysis stage
                quivering_annotations=None,
                shuffle=3,  # Match the model we have
                n_fish=2,
                n_tracks=2,
                min_length=config.DEFAULT_MIN_TRACKLET_LENGTH,
                animal_names=None,
                min_likelihood=config.DEFAULT_MIN_LIKELIHOOD,
                n_minutes=None,  # Process full video
                mouthing_dist_mm=config.DEFAULT_MOUTHING_DIST_MM,
                bin_width=config.DEFAULT_ANALYSIS_BIN_WIDTH,
                force=True,  # Force rerun
                visualize=False,  # Skip video rendering for speed
                plots=['all'],
                n_epochs=10,  # Quick training
                batch_size=config.DEFAULT_ID_BATCH_SIZE,
                lr=config.DEFAULT_ID_LEARNING_RATE,
                patch_size=config.DEFAULT_PATCH_SIZE,
                padding=config.DEFAULT_PADDING,
                conf_threshold=config.DEFAULT_CONF_THRESHOLD,
                device='cuda',  # CI-friendly
                force_retrain=True,
                min_silhouette=config.DEFAULT_MIN_SILHOUETTE,
                cache_frame_stride=10,  # Reduce memory usage
            )

            # Run full pipeline
            cmd_full(args)

            # Create TrialManager to check expected outputs
            tm = TrialManager(
                trial_dir=e2e_test_video_dir,
                config_path=e2e_dlc_config,
                shuffle=3,
                training_fraction=0.95
            )

            # Assert all expected output files exist
            assert tm.el_pickle_path().exists(), "Pose estimation pickle not created"
            assert tm.stitched_h5_path().exists(), "Stitched H5 not created"
            assert tm.stitched_csv_path().exists(), "Stitched CSV not created"
            assert tm.filtered_h5_path().exists(), "Filtered H5 not created"
            assert tm.filtered_csv_path().exists(), "Filtered CSV not created"
            assert tm.framefeatures_path().exists(), "Frame features not created"
            assert tm.clipfeatures_path().exists(), "Clip features not created"

            # Check ID correction directory exists
            assert tm.id_correction_dir().exists(), "ID correction directory not created"

            # Basic sanity checks
            assert tm.el_pickle_path().stat().st_size > 0, "Pickle file is empty"
            assert tm.stitched_h5_path().stat().st_size > 0, "Stitched H5 is empty"
            assert tm.filtered_h5_path().stat().st_size > 0, "Filtered H5 is empty"

            print(f"\nAll pipeline stages completed successfully!")
            print(f"Output files located in: {e2e_test_video_dir}")
            input('End to end test complete. Press enter when you are ready to exit testing and delete the output files')

        finally:
            # Cleanup: Delete all generated files except the original test video
            print(f"\nCleaning up test outputs...")
            for item in e2e_test_video_dir.iterdir():
                if item.name != video_filename:
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                            print(f"  Removed directory: {item.name}")
                        else:
                            item.unlink()
                            print(f"  Removed file: {item.name}")
                    except Exception as e:
                        print(f"  Warning: Could not remove {item.name}: {e}")
            print("Cleanup complete.")
