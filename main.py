#!/usr/bin/env python3
"""
DemBA - DeepLabCut-augmented Multi-animal Behavioral Analysis

Main entrypoint for the analysis pipeline.
"""

import argparse
import sys
from pathlib import Path

# Set matplotlib backend to non-GUI before any matplotlib imports
# This prevents tkinter-related threading issues in batch/headless mode
import matplotlib
matplotlib.use('Agg')

import demba
from demba import config
from demba.file_manager import TrialManager, ProjectManager


def get_trial_manager(args):
    """Create TrialManager from args. Requires video and dlc_config."""
    if not hasattr(args, 'video') or not hasattr(args, 'dlc_config'):
        raise ValueError("Both --video and --dlc-config are required to infer file paths")

    if args.video is None or args.dlc_config is None:
        raise ValueError("Both --video and --dlc-config are required to infer file paths")

    return TrialManager(
        trial_dir=args.video.parent,
        config_path=args.dlc_config,
        shuffle=args.shuffle,
        training_fraction=config.DEFAULT_TRAINING_FRACTION
    )


def cmd_pose(args):
    """Run pose estimation on video(s)."""
    from demba.pose_estimation import estimate_pose

    tm = get_trial_manager(args)
    print(f"Running pose estimation on: {tm.video_path()}")
    estimate_pose(
        trial_manager=tm,
        n_fish=args.n_fish,
        force_rerun=args.force
    )
    print("Pose estimation complete")


def cmd_id_correction(args):
    """Run identity correction on tracklets."""
    from demba.identity_correction import main as id_correction_main

    tm = get_trial_manager(args)
    print(f"Running identity correction on: {tm.el_pickle_path()}")
    id_correction_main(
        trial_manager=tm,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patch_size=args.patch_size,
        padding=args.padding,
        conf_threshold=args.conf_threshold,
        device=args.device,
        force_retrain=args.force_retrain,
        min_silhouette=args.min_silhouette,
        frame_stride=args.cache_frame_stride
    )
    print("Identity correction complete")


def cmd_stitch(args):
    """Run tracklet stitching."""
    from demba.tracklet_stitching import stitch_by_identity

    tm = get_trial_manager(args)
    print(f"Stitching tracklets: {tm.el_pickle_path()}")
    stitch_by_identity(
        trial_manager=tm,
        n_tracks=args.n_tracks,
        min_length=args.min_length,
        animal_names=args.animal_names
    )
    print("Tracklet stitching complete")


def cmd_filter(args):
    """Run temporal filtering on pose predictions."""
    from demba.filtering import filter_predictions

    tm = get_trial_manager(args)
    print(f"Filtering predictions for: {tm.video_path()}")
    filter_predictions(trial_manager=tm)
    print("Filtering complete")


def cmd_features(args):
    """Extract behavioral features from video(s)."""
    from demba.feature_extraction import process_video

    tm = get_trial_manager(args)
    print(f"Processing video: {tm.video_path()}")
    process_video(
        trial_manager=tm,
        quivering_annotation_path=args.quivering_annotations,
        visualize=args.visualize,
        n_minutes=args.n_minutes,
        min_likelihood=args.min_likelihood
    )
    print("Feature extraction complete")


def cmd_visualize(args):
    """Create labeled video with pose overlays."""
    from demba.visualization import create_labeled_video, create_identity_consistency_grids

    tm = get_trial_manager(args)
    print(f"Creating labeled video for: {tm.video_path()}")
    create_labeled_video(trial_manager=tm)

    print(f"Creating ID consistency grids for {tm.video_path()}")
    create_identity_consistency_grids(trial_manager=tm)

    print("Visualization complete")


def cmd_analyze(args):
    """Run statistical analysis and generate plots."""
    from demba.analysis import Plotter

    pm = ProjectManager(args.project_dir)
    print(f"Analyzing data in: {pm.project_dir}")
    plotter = Plotter(
        project_manager=pm,
        mouthing_dist_mm=args.mouthing_dist_mm,
        min_likelihood=args.min_likelihood,
        n_minutes=args.n_minutes
    )
    if  'all' in args.plots or 'boxplots' in args.plots:
        print("  Generating boxplots...")
        plotter.generate_clipfeature_boxplots()

    if 'all' in args.plots or 'correlation' in args.plots:
        print("  Generating correlation plots...")
        plotter.generate_auto_manual_correlation_plots()
        plotter.generate_cross_sex_correlation_plots()

    if 'all' in args.plots or 'heatmaps' in args.plots:
        print("  Generating heatmaps...")
        plotter.generate_event_timeseries_heatmaps(bin_width_frames=args.bin_width)

    print("Analysis complete")


def cmd_full(args):
    """Run the complete pipeline end-to-end."""
    print("="*60)
    print("Running full DemBA pipeline")
    print("="*60)

    # Initialize TrialManager for file path management
    trial_manager = get_trial_manager(args)

    print(f"\nTrial: {trial_manager.video_stem}")
    print(f"Scorer: {trial_manager.scorer_name[:60]}...")

    # Display current completion status
    status = trial_manager.get_completion_status()
    print("\nPipeline stage status:")
    for stage, completed in status.items():
        symbol = "DONE" if completed else "TODO"
        print(f"  [{symbol}] {stage}")
    print()

    # Step 1: Pose estimation
    print("\n[1/7] Running pose estimation...")
    cmd_pose(args)

    # Verify pose estimation produced expected files
    if not trial_manager.el_pickle_path().exists():
        raise FileNotFoundError(
            f"Pose estimation did not produce expected file: {trial_manager.el_pickle_path()}"
        )

    # Step 2: Identity correction
    print("\n[2/7] Running identity correction...")
    cmd_id_correction(args)

    # Step 3: Tracklet stitching
    print("\n[3/7] Running tracklet stitching...")
    cmd_stitch(args)

    # Step 4: Filtering
    print("\n[4/7] Running temporal filtering...")
    cmd_filter(args)

    # Step 5: Feature extraction
    print("\n[5/7] Extracting features...")
    cmd_features(args)

    # Step 6: Visualization
    print("\n[6/7] Creating labeled video...")
    cmd_visualize(args)

    # Step 7: Analysis
    if args.project_dir:
        print("\n[7/7] Running analysis...")
        cmd_analyze(args)
    else:
        print("\n[7/7] Skipping analysis (no project directory specified)")

    print("\n" + "="*60)
    print("Full pipeline complete!")
    print("="*60)

    # Display final completion status
    status = trial_manager.get_completion_status()
    completed_stages = sum(1 for s in status.values() if s)
    print(f"\nCompleted {completed_stages}/{len(status)} pipeline stages")
    print("="*60)


def cmd_batch(args):
    """Run batch pipeline on multiple trials with batched interactive ID assignment."""
    from demba.utils.gen_utils import (
        save_batch_metadata, load_batch_metadata,
        save_batch_cluster_mapping, load_batch_cluster_mapping,
        reconstruct_prep_data
    )
    from demba.identity_correction import train_id_model, map_clusters_to_sex, assign_corrected_ids

    print("="*60)
    print("Batch Mode - Multi-Trial Pipeline")
    print("="*60)

    # Initialize ProjectManager
    pm = ProjectManager(
        project_dir=args.project_dir,
        shuffle=args.shuffle,
        training_fraction=config.DEFAULT_TRAINING_FRACTION
    )

    print(f"Project directory: {pm.project_dir}")
    print(f"Config file: {pm.config_path}")
    print(f"Videos directory: {pm.videos_dir}")

    # Find quivering annotations if not provided
    quivering_annotations = args.quivering_annotations
    if quivering_annotations is None:
        quivering_annotations = pm.find_quivering_annotations()
        if quivering_annotations:
            print(f"Found annotations: {quivering_annotations}")
        else:
            print("No quivering annotations found")

    # Get all trial directories
    trial_dirs = pm.list_trial_dirs()

    if not trial_dirs:
        raise ValueError(f"No valid trial directories found in {pm.videos_dir}")

    print(f"Found {len(trial_dirs)} trials to process")
    for td in trial_dirs:
        print(f"  - {td.name}")
    print()

    # =========================================================================
    # PHASE 1: Preparation (non-interactive)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 1: Preparation (Pose + ID Model Training)")
    print("="*60 + "\n")

    phase1_completed = []
    phase1_failed = []

    for i, trial_dir in enumerate(trial_dirs, 1):
        print(f"\n[{i}/{len(trial_dirs)}] Processing {trial_dir.name}...")

        try:
            # Create TrialManager using ProjectManager's config
            tm = TrialManager(
                trial_dir=trial_dir,
                config_path=pm.config_path,
                shuffle=args.shuffle,
                training_fraction=config.DEFAULT_TRAINING_FRACTION
            )

            # Check if pose estimation is complete
            if not tm.is_stage_complete('pose_estimation'):
                print(f"  Running pose estimation...")
                # Create a copy of args with video path set
                import argparse as ap
                args_copy = ap.Namespace(**vars(args))
                args_copy.video = tm.video_path()
                args_copy.dlc_config = pm.config_path
                cmd_pose(args_copy)
            else:
                print(f"  Pose estimation already complete")

            # Check if ID model training already done
            id_correction_dir = tm.id_correction_dir()
            id_correction_dir.mkdir(exist_ok=True)
            metadata = load_batch_metadata(id_correction_dir)

            if metadata is not None:
                print(f"  ID model training already complete")
            else:
                print(f"  Training ID model...")
                prep_data = train_id_model(
                    tracklet_path=tm.el_pickle_path(),
                    n_epochs=args.n_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    patch_size=args.patch_size,
                    padding=args.padding,
                    conf_threshold=args.conf_threshold,
                    device=args.device,
                    force_retrain=args.force_retrain,
                    min_tracklet_length=args.min_length,
                    frame_stride=args.cache_frame_stride
                )

                if prep_data is None:
                    raise ValueError("ID training returned None (may have been skipped)")

                # Save lightweight metadata
                save_batch_metadata(prep_data, id_correction_dir)
                print(f"  Metadata saved")

            phase1_completed.append(trial_dir)

        except Exception as e:
            print(f"  ERROR: {e}")
            phase1_failed.append((trial_dir, str(e)))
            continue

    print(f"\nPhase 1 Summary: {len(phase1_completed)}/{len(trial_dirs)} trials completed")
    if phase1_failed:
        print(f"  Failed trials:")
        for trial_dir, error in phase1_failed:
            print(f"    - {trial_dir.name}: {error}")

    if not phase1_completed:
        print("\nNo trials completed Phase 1. Exiting.")
        return

    # =========================================================================
    # PHASE 2: Interactive ID Assignment (batched)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: Interactive Cluster Mapping (All Trials)")
    print("="*60 + "\n")

    phase2_completed = []

    for i, trial_dir in enumerate(phase1_completed, 1):
        print(f"\n[{i}/{len(phase1_completed)}] Mapping clusters for {trial_dir.name}...")

        try:
            tm = TrialManager(
                trial_dir=trial_dir,
                config_path=pm.config_path,
                shuffle=args.shuffle,
                training_fraction=config.DEFAULT_TRAINING_FRACTION
            )
            id_correction_dir = tm.id_correction_dir()

            # Check if mapping already exists
            cluster_mapping = load_batch_cluster_mapping(id_correction_dir)
            if cluster_mapping is not None:
                print(f"  Cluster mapping already exists: {cluster_mapping}")
                phase2_completed.append(trial_dir)
                continue

            # Load metadata and reconstruct prep_data
            metadata = load_batch_metadata(id_correction_dir)
            prep_data = reconstruct_prep_data(
                metadata,
                patch_size=args.patch_size,
                padding=args.padding,
                conf_threshold=args.conf_threshold,
                device=args.device
            )

            # Interactive mapping (user input)
            cluster_mapping = map_clusters_to_sex(prep_data)

            # Save mapping
            save_batch_cluster_mapping(cluster_mapping, id_correction_dir)
            print(f"  Mapping saved: {cluster_mapping}")

            phase2_completed.append(trial_dir)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\nPhase 2 Summary: {len(phase2_completed)}/{len(phase1_completed)} trials completed")

    if not phase2_completed:
        print("\nNo trials completed Phase 2. Exiting.")
        return

    # =========================================================================
    # PHASE 3: Finalization (non-interactive)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 3: Finalization (ID Assignment + Remaining Pipeline)")
    print("="*60 + "\n")

    for i, trial_dir in enumerate(phase2_completed, 1):
        print(f"\n[{i}/{len(phase2_completed)}] Finalizing {trial_dir.name}...")

        try:
            tm = TrialManager(
                trial_dir=trial_dir,
                config_path=pm.config_path,
                shuffle=args.shuffle,
                training_fraction=config.DEFAULT_TRAINING_FRACTION
            )
            id_correction_dir = tm.id_correction_dir()

            # Load metadata and reconstruct prep_data
            metadata = load_batch_metadata(id_correction_dir)
            prep_data = reconstruct_prep_data(
                metadata,
                patch_size=args.patch_size,
                padding=args.padding,
                conf_threshold=args.conf_threshold,
                device=args.device
            )

            # Load cluster mapping
            cluster_mapping = load_batch_cluster_mapping(id_correction_dir)

            # Assign corrected IDs
            if not tm.is_stage_complete('identity_correction'):
                print(f"  Assigning corrected IDs...")
                assign_corrected_ids(prep_data, cluster_mapping, min_silhouette=args.min_silhouette)
            else:
                print(f"  ID assignment already complete")

            # Run remaining pipeline stages
            import argparse as ap
            args_copy = ap.Namespace(**vars(args))
            args_copy.video = tm.video_path()
            args_copy.dlc_config = pm.config_path
            args_copy.quivering_annotations = quivering_annotations

            if not tm.is_stage_complete('tracklet_stitching'):
                print(f"  Stitching tracklets...")
                cmd_stitch(args_copy)
            else:
                print(f"  Stitching already complete")

            if not tm.is_stage_complete('filtering'):
                print(f"  Filtering...")
                cmd_filter(args_copy)
            else:
                print(f"  Filtering already complete")

            if not tm.is_stage_complete('feature_extraction'):
                print(f"  Extracting features...")
                cmd_features(args_copy)
            else:
                print(f"  Feature extraction already complete")

            if not tm.is_stage_complete('visualization'):
                print(f"  Creating visualizations...")
                cmd_visualize(args_copy)
            else:
                print(f"  Visualization already complete")

            print(f"  {trial_dir.name} complete!")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # =========================================================================
    # Analysis (always run since we have ProjectManager)
    # =========================================================================
    print("\n" + "="*60)
    print("Running Project-Level Analysis")
    print("="*60 + "\n")
    cmd_analyze(args)

    print("\n" + "="*60)
    print("Batch Mode Complete!")
    print("="*60)


def cmd_create_annotation_package(args):
    """Create annotation package for tracklet evaluation."""
    from demba.utils.annotation_package import build_annotation_package

    print("="*60)
    print("Creating Annotation Package")
    print("="*60)

    # Initialize ProjectManager
    pm = ProjectManager(
        project_dir=args.project_dir,
        shuffle=args.shuffle if hasattr(args, 'shuffle') else config.DEFAULT_SHUFFLE,
        training_fraction=config.DEFAULT_TRAINING_FRACTION
    )

    print(f"Project directory: {pm.project_dir}")
    print(f"Output directory: {args.output}\n")

    # Build annotation package
    context_weights = {
        'duo': args.duo_weight,
        'solo': 1.0 - args.duo_weight
    }

    summary = build_annotation_package(
        project_manager=pm,
        output_dir=args.output,
        n_samples_per_video=args.n_samples,
        min_tracklet_length=args.min_length,
        context_weights=context_weights,
        random_seed=args.random_seed
    )

    print("\nAnnotation package creation complete!")


def cmd_evaluate_annotations(args):
    """Evaluate tracklet annotations and generate metrics."""
    from demba.evaluate_annotations import (
        load_and_validate_annotations,
        calculate_accuracy_metrics,
        generate_evaluation_report
    )

    print("="*60)
    print("Evaluating Tracklet Annotations")
    print("="*60)

    print(f"Annotations: {args.annotations}")
    print(f"Metadata: {args.metadata}")
    print(f"Output: {args.output}\n")

    # Load and validate
    print("Loading and validating annotations...")
    annotations_df = load_and_validate_annotations(
        annotation_csv_path=args.annotations,
        metadata_json_path=args.metadata
    )
    # Store annotation file path for report
    annotations_df.attrs['annotation_file'] = str(args.annotations)

    print(f"  ✓ Loaded {len(annotations_df)} annotations\n")

    # Calculate metrics
    print("Calculating accuracy metrics...")
    metrics = calculate_accuracy_metrics(annotations_df)
    print(f"  ✓ Overall accuracy: {metrics['overall']['accuracy']*100:.1f}%")
    print(f"  ✓ Frame-weighted accuracy: {metrics['overall']['frame_weighted_accuracy']*100:.1f}%\n")

    # Generate report
    print("Generating evaluation report...")
    generate_evaluation_report(
        metrics=metrics,
        annotations_df=annotations_df,
        output_dir=args.output,
        create_plots=args.create_plots
    )

    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="DemBA - DeepLabCut-augmented Multi-animal Behavioral Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline stages (in order):
  1. pose          - Pose estimation from video
  2. id-correction - Correct identity swaps in tracklets
  3. stitch        - Combine tracklets into continuous tracks
  4. filter        - Temporal filtering to smooth trajectories
  5. features      - Extract behavioral features
  6. visualize     - Create labeled videos
  7. analyze       - Generate statistical plots
  8. full          - Run complete pipeline end-to-end
  9. batch         - Run pipeline on multiple trials with batched ID assignment

Examples:
  # Run pose estimation
  python main.py pose --video Videos/trial1/trial1.mp4 --dlc-config config.yaml

  # Run identity correction (all paths inferred from video + config)
  python main.py id-correction --video Videos/trial1/trial1.mp4 --dlc-config config.yaml

  # Run full pipeline on single trial
  python main.py full --video Videos/trial1/trial1.mp4 --dlc-config config.yaml

  # Run batch mode on multiple trials (interactive ID assignment done in one sitting)
  python main.py batch --project-dir /path/to/project/Analysis

  # Run analysis on project
  python main.py analyze --project-dir /path/to/project
        """
    )

    parser.add_argument('--version', action='version', version=f'DemBA {demba.__version__}')

    subparsers = parser.add_subparsers(dest='command', help='Pipeline stage to run')
    subparsers.required = True

    # ========== POSE ESTIMATION ==========
    pose_parser = subparsers.add_parser('pose', help='Run pose estimation')
    pose_parser.add_argument('--video', required=True, type=Path, help='Path to video file')
    pose_parser.add_argument('--dlc-config', required=True, type=Path, help='Path to DeepLabCut config.yaml')
    pose_parser.add_argument('--shuffle', type=int, default=config.DEFAULT_SHUFFLE, help='Shuffle index')
    pose_parser.add_argument('--n-fish', type=int, default=config.DEFAULT_N_FISH, help='Number of individuals')
    pose_parser.add_argument('--force', action='store_true', help='Force re-run even if output exists')
    pose_parser.set_defaults(func=cmd_pose)

    # ========== IDENTITY CORRECTION ==========
    id_parser = subparsers.add_parser('id-correction', help='Run identity correction')
    id_parser.add_argument('--video', required=True, type=Path, help='Path to video file')
    id_parser.add_argument('--dlc-config', required=True, type=Path, help='Path to DeepLabCut config.yaml')
    id_parser.add_argument('--shuffle', type=int, default=config.DEFAULT_SHUFFLE, help='Shuffle index')
    id_parser.add_argument('--n-epochs', type=int, default=config.DEFAULT_ID_N_EPOCHS, help='Training epochs')
    id_parser.add_argument('--batch-size', type=int, default=config.DEFAULT_ID_BATCH_SIZE, help='Batch size')
    id_parser.add_argument('--lr', type=float, default=config.DEFAULT_ID_LEARNING_RATE, help='Learning rate')
    id_parser.add_argument('--patch-size', type=int, default=config.DEFAULT_PATCH_SIZE, help='Patch size')
    id_parser.add_argument('--padding', type=int, default=config.DEFAULT_PADDING, help='Padding around keypoints')
    id_parser.add_argument('--conf-threshold', type=float, default=config.DEFAULT_CONF_THRESHOLD, help='Confidence threshold')
    id_parser.add_argument('--device', choices=['cuda', 'cpu'], default=config.DEFAULT_ID_DEVICE, help='Device to use')
    id_parser.add_argument('--force-retrain', action='store_true', help='Force model retraining')
    id_parser.add_argument('--min-silhouette', type=float, default=config.DEFAULT_MIN_SILHOUETTE, help='Minimum silhouette score')
    id_parser.add_argument('--cache-frame-stride', type=int, default=config.DEFAULT_ID_CACHE_FRAME_STRIDE,
                          help='Sample every Nth frame for patch cache (higher = less memory, default: 5)')
    id_parser.set_defaults(func=cmd_id_correction)

    # ========== TRACKLET STITCHING ==========
    stitch_parser = subparsers.add_parser('stitch', help='Stitch tracklets')
    stitch_parser.add_argument('--video', required=True, type=Path, help='Path to video file')
    stitch_parser.add_argument('--dlc-config', required=True, type=Path, help='Path to DeepLabCut config.yaml')
    stitch_parser.add_argument('--shuffle', type=int, default=config.DEFAULT_SHUFFLE, help='Shuffle index')
    stitch_parser.add_argument('--n-tracks', type=int, default=config.DEFAULT_N_FISH, help='Number of tracks')
    stitch_parser.add_argument('--min-length', type=int, default=config.DEFAULT_STITCH_MIN_LENGTH, help='Minimum tracklet length')
    stitch_parser.add_argument('--animal-names', nargs='+', help='Animal names (e.g., individual1 individual2)')
    stitch_parser.set_defaults(func=cmd_stitch)

    # ========== FILTERING ==========
    filter_parser = subparsers.add_parser('filter', help='Run temporal filtering')
    filter_parser.add_argument('--video', required=True, type=Path, help='Path to video file')
    filter_parser.add_argument('--dlc-config', required=True, type=Path, help='Path to DeepLabCut config.yaml')
    filter_parser.add_argument('--shuffle', type=int, default=config.DEFAULT_SHUFFLE, help='Shuffle index')
    filter_parser.set_defaults(func=cmd_filter)

    # ========== FEATURE EXTRACTION ==========
    features_parser = subparsers.add_parser('features', help='Extract behavioral features')
    features_parser.add_argument('--video', required=True, type=Path, help='Path to video file')
    features_parser.add_argument('--dlc-config', required=True, type=Path, help='Path to DeepLabCut config.yaml')
    features_parser.add_argument('--shuffle', type=int, default=config.DEFAULT_SHUFFLE, help='Shuffle index')
    features_parser.add_argument('--quivering-annotations', type=Path, help='Path to quivering annotations Excel file')
    features_parser.add_argument('--visualize', type=bool, default=config.DEFAULT_VISUALIZE_FLAG, help='Generate feature visualizations')
    features_parser.add_argument('--n-minutes', type=int, default=config.DEFAULT_N_MINUTES, help='Only analyze last N minutes')
    features_parser.add_argument('--min-likelihood', type=float, default=config.DEFAULT_MIN_LIKELIHOOD, help='Minimum keypoint likelihood')
    features_parser.set_defaults(func=cmd_features)

    # ========== VISUALIZATION ==========
    viz_parser = subparsers.add_parser('visualize', help='Create labeled video')
    viz_parser.add_argument('--video', required=True, type=Path, help='Path to video file')
    viz_parser.add_argument('--dlc-config', required=True, type=Path, help='Path to DeepLabCut config.yaml')
    viz_parser.add_argument('--shuffle', type=int, default=config.DEFAULT_SHUFFLE, help='Shuffle index')
    viz_parser.set_defaults(func=cmd_visualize)

    # ========== ANALYSIS ==========
    analyze_parser = subparsers.add_parser('analyze', help='Run statistical analysis')
    analyze_parser.add_argument('--project-dir', required=True, type=Path, help='Project directory containing Videos and Annotations')
    analyze_parser.add_argument('--plots', nargs='+', choices=['boxplots', 'correlation', 'heatmaps', 'all'], default='all', help='Types of plots to generate')
    analyze_parser.add_argument('--mouthing-dist-mm', type=float, default=config.DEFAULT_MOUTHING_DIST_MM, help='Mouthing distance threshold (mm)')
    analyze_parser.add_argument('--min-likelihood', type=float, default=config.DEFAULT_MIN_LIKELIHOOD, help='Minimum keypoint likelihood')
    analyze_parser.add_argument('--n-minutes', type=int, default=config.DEFAULT_N_MINUTES, help='Time restriction in minutes')
    analyze_parser.add_argument('--bin-width', type=int, default=config.DEFAULT_ANALYSIS_BIN_WIDTH, help='Bin width in frames for heatmaps')
    analyze_parser.set_defaults(func=cmd_analyze)

    # ========== FULL PIPELINE ==========
    full_parser = subparsers.add_parser('full', help='Run complete pipeline')
    full_parser.add_argument('--video', required=True, type=Path, help='Path to video file')
    full_parser.add_argument('--dlc-config', required=True, type=Path, help='Path to DeepLabCut config.yaml')
    full_parser.add_argument('--project-dir', type=Path, help='Project directory for analysis stage')
    full_parser.add_argument('--quivering-annotations', type=Path, help='Path to quivering annotations')

    # Parameters (use defaults from config)
    full_parser.add_argument('--shuffle', type=int, default=config.DEFAULT_SHUFFLE)
    full_parser.add_argument('--n-fish', type=int, default=config.DEFAULT_N_FISH)
    full_parser.add_argument('--n-tracks', type=int, default=config.DEFAULT_N_FISH)
    full_parser.add_argument('--min-length', type=int, default=config.DEFAULT_STITCH_MIN_LENGTH)
    full_parser.add_argument('--animal-names', nargs='+')
    full_parser.add_argument('--min-likelihood', type=float, default=config.DEFAULT_MIN_LIKELIHOOD)
    full_parser.add_argument('--n-minutes', type=int, default=config.DEFAULT_N_MINUTES)
    full_parser.add_argument('--mouthing-dist-mm', type=float, default=config.DEFAULT_MOUTHING_DIST_MM)
    full_parser.add_argument('--bin-width', type=int, default=config.DEFAULT_ANALYSIS_BIN_WIDTH)
    full_parser.add_argument('--force', action='store_true')
    full_parser.add_argument('--visualize', type=bool, default=config.DEFAULT_VISUALIZE_FLAG)
    full_parser.add_argument('--plots', nargs='+', choices=['boxplots', 'correlation', 'heatmaps', 'all'], default=['all'])

    # ID correction parameters
    full_parser.add_argument('--n-epochs', type=int, default=config.DEFAULT_ID_N_EPOCHS)
    full_parser.add_argument('--batch-size', type=int, default=config.DEFAULT_ID_BATCH_SIZE)
    full_parser.add_argument('--lr', type=float, default=config.DEFAULT_ID_LEARNING_RATE)
    full_parser.add_argument('--patch-size', type=int, default=config.DEFAULT_PATCH_SIZE)
    full_parser.add_argument('--padding', type=int, default=config.DEFAULT_PADDING)
    full_parser.add_argument('--conf-threshold', type=float, default=config.DEFAULT_CONF_THRESHOLD)
    full_parser.add_argument('--device', choices=['cuda', 'cpu'], default=config.DEFAULT_ID_DEVICE)
    full_parser.add_argument('--force-retrain', action='store_true')
    full_parser.add_argument('--min-silhouette', type=float, default=config.DEFAULT_MIN_SILHOUETTE)
    full_parser.add_argument('--cache-frame-stride', type=int, default=config.DEFAULT_ID_CACHE_FRAME_STRIDE,
                            help='Sample every Nth frame for patch cache (higher = less memory, default: 5)')

    full_parser.set_defaults(func=cmd_full)

    # ========== BATCH MODE ==========
    batch_parser = subparsers.add_parser('batch',
        help='Run pipeline on multiple trials with batched interactive ID assignment')

    # Required arguments
    batch_parser.add_argument('--project-dir', required=True, type=Path,
        help='Project directory (contains Videos/ and Annotations/ subdirs)')

    # Optional arguments
    batch_parser.add_argument('--quivering-annotations', type=Path,
        help='Path to quivering annotations Excel file (auto-detected if not provided)')

    # Pipeline parameters (same as full mode, all with config defaults)
    batch_parser.add_argument('--shuffle', type=int, default=config.DEFAULT_SHUFFLE)
    batch_parser.add_argument('--n-fish', type=int, default=config.DEFAULT_N_FISH)
    batch_parser.add_argument('--n-tracks', type=int, default=config.DEFAULT_N_FISH)
    batch_parser.add_argument('--min-length', type=int, default=config.DEFAULT_STITCH_MIN_LENGTH)
    batch_parser.add_argument('--animal-names', nargs='+')
    batch_parser.add_argument('--min-likelihood', type=float, default=config.DEFAULT_MIN_LIKELIHOOD)
    batch_parser.add_argument('--n-minutes', type=int, default=config.DEFAULT_N_MINUTES)
    batch_parser.add_argument('--mouthing-dist-mm', type=float, default=config.DEFAULT_MOUTHING_DIST_MM)
    batch_parser.add_argument('--bin-width', type=int, default=config.DEFAULT_ANALYSIS_BIN_WIDTH)
    batch_parser.add_argument('--force', action='store_true')
    batch_parser.add_argument('--visualize', type=bool, default=config.DEFAULT_VISUALIZE_FLAG)
    batch_parser.add_argument('--plots', nargs='+',
        choices=['boxplots', 'correlation', 'heatmaps', 'all'], default=['all'])

    # ID correction parameters
    batch_parser.add_argument('--n-epochs', type=int, default=config.DEFAULT_ID_N_EPOCHS)
    batch_parser.add_argument('--batch-size', type=int, default=config.DEFAULT_ID_BATCH_SIZE)
    batch_parser.add_argument('--lr', type=float, default=config.DEFAULT_ID_LEARNING_RATE)
    batch_parser.add_argument('--patch-size', type=int, default=config.DEFAULT_PATCH_SIZE)
    batch_parser.add_argument('--padding', type=int, default=config.DEFAULT_PADDING)
    batch_parser.add_argument('--conf-threshold', type=float, default=config.DEFAULT_CONF_THRESHOLD)
    batch_parser.add_argument('--device', choices=['cuda', 'cpu'], default=config.DEFAULT_ID_DEVICE)
    batch_parser.add_argument('--force-retrain', action='store_true')
    batch_parser.add_argument('--min-silhouette', type=float, default=config.DEFAULT_MIN_SILHOUETTE)
    batch_parser.add_argument('--cache-frame-stride', type=int, default=config.DEFAULT_ID_CACHE_FRAME_STRIDE,
        help='Sample every Nth frame for patch cache (higher = less memory, default: 5)')

    batch_parser.set_defaults(func=cmd_batch)

    # ========== CREATE ANNOTATION PACKAGE ==========
    annot_package_parser = subparsers.add_parser(
        'create-annotation-package',
        help='Create annotation package for tracklet evaluation'
    )
    annot_package_parser.add_argument('--project-dir', required=True, type=Path,
        help='Project directory (contains Videos/ subdirectory)')
    annot_package_parser.add_argument('--output', type=Path, default='annotation_package',
        help='Output directory for annotation package (default: annotation_package)')
    annot_package_parser.add_argument('--n-samples', type=int, default=config.DEFAULT_EVAL_N_SAMPLES,
        help=f'Number of tracklets to sample per video (default: {config.DEFAULT_EVAL_N_SAMPLES})')
    annot_package_parser.add_argument('--min-length', type=int, default=config.DEFAULT_EVAL_MIN_TRACKLET_LENGTH,
        help=f'Minimum tracklet length in frames (default: {config.DEFAULT_EVAL_MIN_TRACKLET_LENGTH})')
    annot_package_parser.add_argument('--duo-weight', type=float, default=config.DEFAULT_EVAL_DUO_WEIGHT,
        help=f'Sampling weight for duo contexts (default: {config.DEFAULT_EVAL_DUO_WEIGHT})')
    annot_package_parser.add_argument('--random-seed', type=int, default=config.DEFAULT_EVAL_RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {config.DEFAULT_EVAL_RANDOM_SEED})')
    annot_package_parser.add_argument('--shuffle', type=int, default=config.DEFAULT_SHUFFLE,
        help='DeepLabCut shuffle index')
    annot_package_parser.set_defaults(func=cmd_create_annotation_package)

    # ========== EVALUATE ANNOTATIONS ==========
    eval_annot_parser = subparsers.add_parser(
        'evaluate-annotations',
        help='Evaluate tracklet annotations and generate metrics'
    )
    eval_annot_parser.add_argument('--annotations', required=True, type=Path,
        help='Path to completed annotation_sheet.csv')
    eval_annot_parser.add_argument('--metadata', required=True, type=Path,
        help='Path to clip_metadata.json')
    eval_annot_parser.add_argument('--output', type=Path, default='evaluation_results',
        help='Output directory for evaluation results (default: evaluation_results)')
    eval_annot_parser.add_argument('--create-plots', action='store_true', default=True,
        help='Generate plots (requires matplotlib, default: True)')
    eval_annot_parser.add_argument('--no-plots', dest='create_plots', action='store_false',
        help='Skip plot generation')
    eval_annot_parser.set_defaults(func=cmd_evaluate_annotations)

    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Validate paths exist where required
    if hasattr(args, 'video') and args.video:
        if not args.video.exists():
            print(f"Error: Video file not found: {args.video}", file=sys.stderr)
            sys.exit(1)

    if hasattr(args, 'dlc_config') and args.dlc_config:
        if not args.dlc_config.exists():
            print(f"Error: DLC config not found: {args.dlc_config}", file=sys.stderr)
            sys.exit(1)

    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
