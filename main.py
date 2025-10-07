#!/usr/bin/env python3
"""
DemBA - DeepLabCut-augmented Multi-animal Behavioral Analysis

Main entrypoint for the analysis pipeline.
"""

import argparse
import sys
from pathlib import Path

import demba
from demba import config


def cmd_pose(args):
    """Run pose estimation on video(s)."""
    from demba.pose_estimation import estimate_pose

    print(f"Running pose estimation on: {args.video}")
    estimate_pose(
        config_path=args.dlc_config,
        video_path=args.video,
        shuffle=args.shuffle,
        n_fish=args.n_fish,
        force_rerun=args.force
    )
    print(" Pose estimation complete")


def cmd_id_correction(args):
    """Run identity correction on tracklets."""
    from demba.identity_correction import main as id_correction_main

    print(f"Running identity correction on: {args.tracklet_pickle}")
    id_correction_main(
        tracklet_path=args.tracklet_pickle,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patch_size=args.patch_size,
        padding=args.padding,
        conf_threshold=args.conf_threshold,
        device=args.device,
        force_retrain=args.force_retrain,
        min_silhouette=args.min_silhouette
    )
    print(" Identity correction complete")


def cmd_stitch(args):
    """Run tracklet stitching."""
    from demba.tracklet_stitching import stitch_by_identity

    print(f"Stitching tracklets: {args.tracklet_pickle}")
    stitch_by_identity(
        tracklet_pickle_path=args.tracklet_pickle,
        output_h5_path=args.output_h5,
        n_tracks=args.n_tracks,
        min_length=args.min_length,
        animal_names=args.animal_names
    )
    print(" Tracklet stitching complete")


def cmd_filter(args):
    """Run temporal filtering on pose predictions."""
    from demba.filtering import filter_predictions

    print(f"Filtering predictions for: {args.video}")
    filter_predictions(
        config_path=args.dlc_config,
        video_path=args.video,
        shuffle=args.shuffle
    )
    print(" Filtering complete")


def cmd_features(args):
    """Extract behavioral features from video(s)."""
    from demba.feature_extraction import process_video

    print(f"Processing video: {args.video}")
    process_video(
        video_path=args.video,
        quivering_annotation_path=args.quivering_annotations,
        pose_h5_path=args.pose_h5,
        visualize=args.visualize,
        n_minutes=args.n_minutes,
        min_likelihood=args.min_likelihood
    )
    print(" Feature extraction complete")


def cmd_visualize(args):
    """Create labeled video with pose overlays."""
    from demba.visualization import create_labeled_video
    print(f"Creating labeled video for: {args.video}")
    create_labeled_video(
        config_path=args.dlc_config.resolve(),
        video_path=args.video.resolve(),
        shuffle=args.shuffle
    )
    print(" Visualization complete")


def cmd_analyze(args):
    """Run statistical analysis and generate plots."""
    from demba.analysis import Plotter

    print(f"Analyzing data in: {args.parent_dir}")
    plotter = Plotter(
        parent_dir=args.parent_dir,
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

    if 'all' in args.plots or 'heatmaps' in args.plots:
        print("  Generating heatmaps...")
        plotter.generate_event_timeseries_heatmaps(bin_width_frames=args.bin_width)

    print(" Analysis complete")


def cmd_full(args):
    """Run the complete pipeline end-to-end."""
    print("="*60)
    print("Running full DemBA pipeline")
    print("="*60)

    # Step 1: Pose estimation
    print("\n[1/7] Running pose estimation...")
    cmd_pose(args)

    # Infer file paths from pose estimation output
    video_dir = args.video.parent
    full_pickle_matches = list(video_dir.glob('*_full.pickle'))
    if not full_pickle_matches:
        raise FileNotFoundError(f"Pose estimation did not produce expected *_full.pickle file in {video_dir}")

    # Derive tracklet and output paths from the _full.pickle filename
    full_pickle_path = full_pickle_matches[0]
    file_stem = str(full_pickle_path.name).replace('_full.pickle', '')
    args.tracklet_pickle = video_dir / f"{file_stem}_el.pickle"
    args.output_h5 = video_dir / f"{file_stem}_el.h5"
    args.pose_h5 = args.output_h5

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
    if args.parent_dir:
        print("\n[7/7] Running analysis...")
        cmd_analyze(args)
    else:
        print("\n[7/7] Skipping analysis (no parent directory specified)")

    print("\n" + "="*60)
    print(" Full pipeline complete!")
    print("="*60)


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

Examples:
  # Run pose estimation
  python main.py pose --video data/trial1.mp4 --dlc-config config.yaml

  # Run identity correction
  python main.py id-correction --tracklet-pickle data/trial1_el.pickle

  # Extract features from single video
  python main.py features --video data/trial1.mp4 --pose-h5 data/trial1.h5

  # Run full pipeline
  python main.py full --video data/trial1.mp4 --dlc-config config.yaml
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
    id_parser.add_argument('--tracklet-pickle', required=True, type=Path, help='Path to *_el.pickle file')
    id_parser.add_argument('--n-epochs', type=int, default=50, help='Training epochs')
    id_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    id_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    id_parser.add_argument('--patch-size', type=int, default=config.DEFAULT_PATCH_SIZE, help='Patch size')
    id_parser.add_argument('--padding', type=int, default=config.DEFAULT_PADDING, help='Padding around keypoints')
    id_parser.add_argument('--conf-threshold', type=float, default=config.DEFAULT_CONF_THRESHOLD, help='Confidence threshold')
    id_parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Device to use')
    id_parser.add_argument('--force-retrain', action='store_true', help='Force model retraining')
    id_parser.add_argument('--min-silhouette', type=float, default=config.DEFAULT_MIN_SILHOUETTE, help='Minimum silhouette score')
    id_parser.set_defaults(func=cmd_id_correction)

    # ========== TRACKLET STITCHING ==========
    stitch_parser = subparsers.add_parser('stitch', help='Stitch tracklets')
    stitch_parser.add_argument('--tracklet-pickle', required=True, type=Path, help='Path to *_el.pickle file')
    stitch_parser.add_argument('--output-h5', required=True, type=Path, help='Output H5 file path')
    stitch_parser.add_argument('--n-tracks', type=int, default=config.DEFAULT_N_FISH, help='Number of tracks')
    stitch_parser.add_argument('--min-length', type=int, default=config.DEFAULT_MIN_TRACKLET_LENGTH, help='Minimum tracklet length')
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
    features_parser.add_argument('--video', type=Path, help='Path to video file (single mode)')
    features_parser.add_argument('--pose-h5', type=Path, help='Path to pose H5 file (single mode)')
    features_parser.add_argument('--parent-dir', type=Path, help='Parent directory (batch mode)')
    features_parser.add_argument('--quivering-annotations', type=Path, help='Path to quivering annotations Excel file')
    features_parser.add_argument('--visualize', action='store_true', help='Generate feature visualizations')
    features_parser.add_argument('--n-minutes', type=int, help='Only analyze last N minutes')
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
    analyze_parser.add_argument('--parent-dir', required=True, type=Path, help='Parent directory containing Videos and Annotations')
    analyze_parser.add_argument('--plots', nargs='+', choices=['boxplots', 'correlation', 'heatmaps', 'all'], default='all', help='Types of plots to generate')
    analyze_parser.add_argument('--mouthing-dist-mm', type=float, default=config.DEFAULT_MOUTHING_DIST_MM, help='Mouthing distance threshold (mm)')
    analyze_parser.add_argument('--min-likelihood', type=float, default=config.DEFAULT_MIN_LIKELIHOOD, help='Minimum keypoint likelihood')
    analyze_parser.add_argument('--n-minutes', type=int, help='Time restriction in minutes')
    analyze_parser.add_argument('--bin-width', type=int, default=1800, help='Bin width in frames for heatmaps')
    analyze_parser.set_defaults(func=cmd_analyze)

    # ========== FULL PIPELINE ==========
    full_parser = subparsers.add_parser('full', help='Run complete pipeline')
    full_parser.add_argument('--video', required=True, type=Path, help='Path to video file')
    full_parser.add_argument('--dlc-config', required=True, type=Path, help='Path to DeepLabCut config.yaml')
    full_parser.add_argument('--parent-dir', type=Path, help='Parent directory for analysis stage')
    full_parser.add_argument('--quivering-annotations', type=Path, help='Path to quivering annotations')


    # Parameters (use defaults from config)
    full_parser.add_argument('--shuffle', type=int, default=config.DEFAULT_SHUFFLE)
    full_parser.add_argument('--n-fish', type=int, default=config.DEFAULT_N_FISH)
    full_parser.add_argument('--n-tracks', type=int, default=config.DEFAULT_N_FISH)
    full_parser.add_argument('--min-length', type=int, default=config.DEFAULT_MIN_TRACKLET_LENGTH)
    full_parser.add_argument('--animal-names', nargs='+')
    full_parser.add_argument('--min-likelihood', type=float, default=config.DEFAULT_MIN_LIKELIHOOD)
    full_parser.add_argument('--n-minutes', type=int)
    full_parser.add_argument('--mouthing-dist-mm', type=float, default=config.DEFAULT_MOUTHING_DIST_MM)
    full_parser.add_argument('--bin-width', type=int, default=1800)
    full_parser.add_argument('--force', action='store_true')
    full_parser.add_argument('--visualize', action='store_true')
    full_parser.add_argument('--plots', nargs='+', choices=['boxplots', 'correlation', 'heatmaps', 'all'], default=['all'])

    # ID correction parameters
    full_parser.add_argument('--n-epochs', type=int, default=50)
    full_parser.add_argument('--batch-size', type=int, default=32)
    full_parser.add_argument('--lr', type=float, default=0.001)
    full_parser.add_argument('--patch-size', type=int, default=config.DEFAULT_PATCH_SIZE)
    full_parser.add_argument('--padding', type=int, default=config.DEFAULT_PADDING)
    full_parser.add_argument('--conf-threshold', type=float, default=config.DEFAULT_CONF_THRESHOLD)
    full_parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    full_parser.add_argument('--force-retrain', action='store_true')
    full_parser.add_argument('--min-silhouette', type=float, default=config.DEFAULT_MIN_SILHOUETTE)

    full_parser.set_defaults(func=cmd_full)

    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Validate paths exist where required
    if hasattr(args, 'video') and args.video and not args.command == 'full':
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
