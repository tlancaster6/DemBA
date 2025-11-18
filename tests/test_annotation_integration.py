"""
Integration test for tracklet annotation system.

Tests end-to-end clip generation and package creation on a single video.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demba.file_manager import TrialManager
from demba.utils.tracklet_sampler import sample_tracklets_stratified
from demba.utils.clip_generator import generate_tracklet_clip
from demba.utils.annotation_package import check_required_files


def test_single_video_annotation_package():
    """
    Integration test on single reference video.
    Tests end-to-end tracklet sampling and clip generation.
    """
    print("=" * 70)
    print("INTEGRATION TEST: Single Video Annotation Package")
    print("=" * 70)

    # Test parameters - UPDATE THESE PATHS FOR YOUR SYSTEM
    TEST_VIDEO_DIR = Path("projects/demasoni_singlenuc-tucker-2025-09-11/Analysis102725/Videos/bgrb_t015_10.29.24_DB14cropped")
    TEST_DLC_CONFIG = Path("projects/demasoni_singlenuc-tucker-2025-09-11/config.yaml")
    TEST_OUTPUT = Path("/tmp/demba_test_annotation_package")
    TEST_N_SAMPLES = 5  # Small for fast testing

    # Check if paths exist
    if not TEST_VIDEO_DIR.exists():
        print(f"ERROR: Test video directory not found: {TEST_VIDEO_DIR}")
        print("Please update TEST_VIDEO_DIR in the test file to point to a valid video directory")
        return False

    if not TEST_DLC_CONFIG.exists():
        print(f"ERROR: Test DLC config not found: {TEST_DLC_CONFIG}")
        print("Please update TEST_DLC_CONFIG in the test file to point to a valid config.yaml")
        return False

    print(f"\nTest video: {TEST_VIDEO_DIR.name}")
    print(f"Output: {TEST_OUTPUT}\n")

    # Create trial manager
    print("Step 1: Creating TrialManager...")
    tm = TrialManager(
        trial_dir=TEST_VIDEO_DIR,
        config_path=TEST_DLC_CONFIG
    )
    print(f"  ✓ TrialManager created")

    # Check required files
    print("\nStep 2: Checking required files...")
    all_present, missing = check_required_files(tm)

    if not all_present:
        print(f"  ✗ Missing files: {', '.join(missing)}")
        print("\nRequired files:")
        print(f"  - Video file: {tm.video_path()}")
        print(f"  - Tracklet pickle: {tm.el_pickle_path()}")
        print(f"  - Embeddings: {tm.id_correction_dir() / 'embeddings.pkl'}")
        print("\nPlease run the pipeline (pose estimation, ID correction, stitching) first.")
        return False

    print(f"  ✓ All required files present")

    # Sample tracklets
    print(f"\nStep 3: Sampling {TEST_N_SAMPLES} tracklets...")
    try:
        samples = sample_tracklets_stratified(
            tm,
            n_samples=TEST_N_SAMPLES,
            min_length=60
        )
        print(f"  ✓ Sampled {len(samples)} tracklets")

        # Print sample statistics
        contexts = [s['context'] for s in samples]
        conf_strata = [s['confidence_stratum'] for s in samples]
        print(f"    - Solo: {contexts.count('solo')}, Duo: {contexts.count('duo')}")
        print(f"    - Low: {conf_strata.count('low')}, Med: {conf_strata.count('medium')}, High: {conf_strata.count('high')}")

    except Exception as e:
        print(f"  ✗ Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Generate clips
    print(f"\nStep 4: Generating {len(samples)} video clips...")
    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    clip_dir = TEST_OUTPUT / 'clips'
    clip_dir.mkdir(exist_ok=True)

    successful_clips = 0
    failed_clips = 0

    for i, sample in enumerate(samples):
        clip_id = f"test_clip_{i+1:03d}"
        output_path = clip_dir / f"{clip_id}.mp4"

        try:
            metadata = generate_tracklet_clip(
                tm,
                tracklet_idx=sample['tracklet_idx'],
                clip_id=clip_id,
                output_path=output_path
            )

            if not output_path.exists():
                print(f"  ✗ {clip_id}: File not created")
                failed_clips += 1
                continue

            print(f"  ✓ {clip_id}: {metadata['duration_sec']:.1f}s ({metadata['n_frames']} frames)")
            successful_clips += 1

        except Exception as e:
            print(f"  ✗ {clip_id}: {e}")
            failed_clips += 1
            continue

    print(f"\nClip generation complete: {successful_clips} succeeded, {failed_clips} failed")

    # Validate a clip
    if successful_clips > 0:
        print("\nStep 5: Validating first clip...")
        import cv2
        first_clip = clip_dir / "test_clip_001.mp4"

        if first_clip.exists():
            cap = cv2.VideoCapture(str(first_clip))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                print(f"  ✓ Clip properties:")
                print(f"    - Resolution: {width}x{height}")
                print(f"    - FPS: {fps}")
                print(f"    - Frames: {frame_count}")
            else:
                print(f"  ✗ Cannot open clip for validation")
                return False
        else:
            print(f"  ✗ First clip not found")
            return False

    # Final summary
    print("\n" + "=" * 70)
    if successful_clips == len(samples) and failed_clips == 0:
        print("INTEGRATION TEST PASSED")
        print("=" * 70)
        print(f"\nAll {successful_clips} clips generated successfully")
        print(f"Output directory: {TEST_OUTPUT}")
        return True
    else:
        print("INTEGRATION TEST COMPLETED WITH ERRORS")
        print("=" * 70)
        print(f"\nSuccessful: {successful_clips}/{len(samples)}")
        print(f"Failed: {failed_clips}/{len(samples)}")
        print(f"Output directory: {TEST_OUTPUT}")
        return False


if __name__ == '__main__':
    success = test_single_video_annotation_package()
    sys.exit(0 if success else 1)