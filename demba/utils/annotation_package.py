"""Annotation package builder for tracklet evaluation."""

import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

from demba import config
from demba.utils.tracklet_sampler import sample_tracklets_stratified
from demba.utils.clip_generator import generate_tracklet_clip

logger = logging.getLogger(__name__)


INSTRUCTIONS_TEMPLATE = """FISH IDENTITY ANNOTATION INSTRUCTIONS
======================================

TASK
----
Watch video clips and label the identity of the fish shown in each clip.

SETUP
-----
1. Open the 'clips' folder and annotation_sheet.csv side by side
2. Use Excel, Google Sheets, or LibreOffice for the CSV
3. Use any video player (VLC recommended) for clips

PROCEDURE
---------
For each row in annotation_sheet.csv:

1. Note the clip_id (e.g., "clip_001")
2. Open clips/clip_001.mp4 in your video player
3. Verify the video name shown in the clip matches the video_name column
4. Watch the ENTIRE clip (some are 20+ seconds)
5. Determine the fish identity and fill in the 'label' column:

   Type "m" if the fish is MALE throughout the clip
   Type "f" if the fish is FEMALE throughout the clip
   Type "c" if the fish identity CHANGES during the clip (rare)

6. Add notes if helpful (optional)

CLIPS ARE GROUPED BY VIDEO
---------------------------
All clips from the same video appear together in the spreadsheet.
This helps you learn what the male and female look like in each specific pair.

When you start a new video, watch the first few clips carefully to identify
the distinguishing features of the two individuals.

IMPORTANT
---------
- Do NOT leave any labels blank
- Use lowercase letters only: m, f, or c
- The "c" label is for rare cases where the identity actually switches mid-clip
- Use the notes column for uncertain cases (e.g., "uncertain - poor visibility")

ESTIMATED TIME
--------------
This annotation task should take approximately {estimated_hours:.1f} hours.
"""


def check_required_files(trial_manager):
    """
    Check if all required files exist for a trial.

    Parameters
    ----------
    trial_manager : TrialManager
        Trial manager instance

    Returns
    -------
    tuple : (bool, list of str)
        (all_present, missing_files)
    """
    missing = []

    # Check video
    if not trial_manager.video_path().exists():
        missing.append("video file")

    # Check tracklet pickle
    if not trial_manager.el_pickle_path().exists():
        print(trial_manager.el_pickle_path())
        missing.append("tracklet pickle (*_el.pickle)")

    # Check embeddings
    embeddings_path = trial_manager.id_correction_dir() / 'embeddings.pkl'
    if not embeddings_path.exists():
        missing.append("embeddings.pkl")

    return (len(missing) == 0, missing)


def build_annotation_package(
    project_manager,
    output_dir='annotation_package',
    n_samples_per_video=None,
    min_tracklet_length=None,
    context_weights=None,
    random_seed=None
):
    """
    Build complete annotation package for external annotator.

    Parameters
    ----------
    project_manager : ProjectManager
        Manages all trials in project
    output_dir : str or Path
        Output directory for annotation package
    n_samples_per_video : int, optional
        Number of tracklets to sample per video.
        Defaults to config.DEFAULT_EVAL_N_SAMPLES
    min_tracklet_length : int, optional
        Minimum tracklet length filter. Defaults to config.DEFAULT_EVAL_MIN_TRACKLET_LENGTH
    context_weights : dict, optional
        Sampling weights for solo vs duo. Defaults to config values
    random_seed : int, optional
        Random seed for reproducibility. Defaults to config.DEFAULT_EVAL_RANDOM_SEED

    Returns
    -------
    dict : Package summary
        {
            'n_videos': int,
            'n_clips': int,
            'total_duration_sec': float,
            'estimated_annotation_time_hours': float,
            'output_dir': Path
        }

    Outputs Created
    ---------------
    1. clips/clip_001.mp4, clip_002.mp4, ...
    2. annotation_sheet.csv (for annotator)
    3. clip_metadata.json (for evaluation)
    4. instructions.txt (for annotator)
    5. sampling_summary.txt (technical report)
    """
    # Set defaults
    if n_samples_per_video is None:
        n_samples_per_video = config.DEFAULT_EVAL_N_SAMPLES
    if min_tracklet_length is None:
        min_tracklet_length = config.DEFAULT_EVAL_MIN_TRACKLET_LENGTH
    if random_seed is None:
        random_seed = config.DEFAULT_EVAL_RANDOM_SEED
    if context_weights is None:
        context_weights = {
            'duo': config.DEFAULT_EVAL_DUO_WEIGHT,
            'solo': config.DEFAULT_EVAL_SOLO_WEIGHT
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / 'clips'
    clips_dir.mkdir(exist_ok=True)

    # Get all trial directories
    trial_dirs = project_manager.list_trial_dirs()

    # Create TrialManagers for each trial
    trial_managers = []
    for trial_dir in trial_dirs:
        tm = project_manager.get_trial_manager(trial_dir.name)
        trial_managers.append(tm)

    # Check which videos have all required files
    print("Scanning project for videos with complete data...")
    valid_trials = []
    for tm in trial_managers:
        all_present, missing = check_required_files(tm)
        if all_present:
            valid_trials.append(tm)
            print(f"  ✓ {tm.trial_dir.name}: All files present")
        else:
            print(f"  ✗ {tm.trial_dir.name}: Missing {', '.join(missing)}")

    if len(valid_trials) == 0:
        raise ValueError(
            "No videos have all required files.\n"
            "Required: video file, *_el.pickle, id_correction/embeddings.pkl\n"
            "Run ID correction and tracklet stitching first."
        )

    print(f"\nFound {len(valid_trials)}/{len(trial_managers)} videos with complete data.")
    print(f"Proceeding with {len(valid_trials)} videos.\n")

    # Storage for all data
    all_clip_metadata = {}
    annotation_rows = []
    video_summaries = []
    clip_counter = 1
    total_duration = 0.0

    # Process each video
    for tm in tqdm(valid_trials, desc="Processing videos", position=0):
        video_name = tm.trial_dir.name

        try:
            # Sample tracklets
            samples = sample_tracklets_stratified(
                tm,
                n_samples=n_samples_per_video,
                min_length=min_tracklet_length,
                context_weights=context_weights,
                random_seed=random_seed
            )

            # Generate clips
            video_clips = []
            for sample in tqdm(samples, desc=f"  {video_name}", position=1, leave=False):
                clip_id = f"clip_{clip_counter:03d}"
                output_path = clips_dir / f"{clip_id}.mp4"

                # Generate clip
                clip_meta = generate_tracklet_clip(
                    tm,
                    tracklet_idx=sample['tracklet_idx'],
                    clip_id=clip_id,
                    output_path=output_path
                )

                # Merge with sample metadata
                full_metadata = {
                    **clip_meta,
                    'predicted_label': sample['predicted_id'],
                    'confidence': float(sample['mean_silhouette']),
                    'context': sample['context'],
                    'length_stratum': sample['length_stratum'],
                    'confidence_stratum': sample['confidence_stratum'],
                    'stratum': sample['stratum'],
                    'trial_dir': str(tm.trial_dir)
                }

                # Store metadata
                all_clip_metadata[clip_id] = full_metadata
                video_clips.append(full_metadata)

                # Add to annotation sheet
                annotation_rows.append({
                    'clip_id': clip_id,
                    'video_name': video_name,
                    'label': '',
                    'notes': ''
                })

                total_duration += clip_meta['duration_sec']
                clip_counter += 1

            # Compute summary statistics for this video
            video_summary = compute_video_summary(video_name, video_clips, samples)
            video_summaries.append(video_summary)

        except Exception as e:
            logger.error(f"Failed to process {video_name}: {e}")
            print(f"  ✗ Error processing {video_name}: {e}")

    # Create annotation sheet CSV (for annotator)
    annotation_df = pd.DataFrame(annotation_rows)
    annotation_csv_path = output_dir / 'annotation_sheet.csv'
    annotation_df.to_csv(annotation_csv_path, index=False)

    # Create clip metadata JSON (for evaluation)
    metadata_json_path = output_dir / 'clip_metadata.json'
    with open(metadata_json_path, 'w') as f:
        json.dump(all_clip_metadata, f, indent=2)

    # Create instructions
    total_clips = len(all_clip_metadata)
    estimated_hours = estimate_annotation_time(total_clips, total_duration)
    instructions_path = output_dir / 'instructions.txt'
    with open(instructions_path, 'w') as f:
        f.write(INSTRUCTIONS_TEMPLATE.format(estimated_hours=estimated_hours))

    # Create sampling summary
    summary_path = output_dir / 'sampling_summary.txt'
    write_sampling_summary(
        summary_path,
        project_manager.project_dir,
        video_summaries,
        n_samples_per_video,
        min_tracklet_length,
        context_weights,
        random_seed,
        total_clips,
        total_duration,
        estimated_hours
    )

    # Print final summary
    print("\n" + "="*60)
    print("ANNOTATION PACKAGE CREATED")
    print("="*60)
    print(f"Videos processed:     {len(valid_trials)}")
    print(f"Total clips:          {total_clips}")
    print(f"Total duration:       {total_duration/3600:.2f} hours")
    print(f"Estimated annotation: {estimated_hours:.1f} hours")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - clips/           ({total_clips} video files)")
    print(f"  - annotation_sheet.csv")
    print(f"  - clip_metadata.json")
    print(f"  - instructions.txt")
    print(f"  - sampling_summary.txt")
    print("="*60)

    return {
        'n_videos': len(valid_trials),
        'n_clips': total_clips,
        'total_duration_sec': total_duration,
        'estimated_annotation_time_hours': estimated_hours,
        'output_dir': output_dir
    }


def compute_video_summary(video_name, clips, samples):
    """Compute summary statistics for a single video."""
    durations = [c['duration_sec'] for c in clips]
    contexts = [c['context'] for c in clips]
    conf_strata = [c['confidence_stratum'] for c in clips]

    return {
        'video_name': video_name,
        'n_clips': len(clips),
        'n_solo': contexts.count('solo'),
        'n_duo': contexts.count('duo'),
        'n_low_conf': conf_strata.count('low'),
        'n_med_conf': conf_strata.count('medium'),
        'n_high_conf': conf_strata.count('high'),
        'min_duration': min(durations),
        'max_duration': max(durations),
        'mean_duration': np.mean(durations),
        'total_duration': sum(durations)
    }


def estimate_annotation_time(n_clips, total_duration_sec):
    """
    Estimate annotation time based on clip duration and overhead.

    Assumes: watch time + 5 sec overhead per clip
    """
    watch_time_hours = total_duration_sec / 3600
    overhead_hours = (n_clips * 5) / 3600  # 5 sec per clip
    return watch_time_hours + overhead_hours


def write_sampling_summary(
    output_path,
    project_dir,
    video_summaries,
    n_samples_per_video,
    min_tracklet_length,
    context_weights,
    random_seed,
    total_clips,
    total_duration,
    estimated_hours
):
    """Write detailed sampling summary report."""
    with open(output_path, 'w') as f:
        f.write("TRACKLET SAMPLING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Project: {project_dir}\n\n")

        f.write("PARAMETERS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Videos processed:              {len(video_summaries)}\n")
        f.write(f"Samples per video:             {n_samples_per_video}\n")
        f.write(f"Total clips generated:         {total_clips}\n")
        f.write(f"Minimum tracklet length:       {min_tracklet_length} frames ({min_tracklet_length/config.VIDEO_FPS:.1f} sec)\n")
        f.write(f"Context weights (duo/solo):    {context_weights['duo']*100:.0f}% / {context_weights['solo']*100:.0f}%\n")
        f.write(f"Random seed:                   {random_seed}\n\n")

        f.write("SAMPLING STATISTICS\n")
        f.write("-" * 60 + "\n")

        for vs in video_summaries:
            f.write(f"\nVideo: {vs['video_name']}\n")
            f.write(f"  Sampled:                     {vs['n_clips']} clips\n")
            f.write(f"\n")
            f.write(f"  Stratification:\n")
            f.write(f"    Low confidence:            {vs['n_low_conf']} clips\n")
            f.write(f"    Med confidence:            {vs['n_med_conf']} clips\n")
            f.write(f"    High confidence:           {vs['n_high_conf']} clips\n")
            f.write(f"\n")
            f.write(f"    Solo context:              {vs['n_solo']} clips\n")
            f.write(f"    Duo context:               {vs['n_duo']} clips\n")
            f.write(f"\n")
            f.write(f"  Duration:\n")
            f.write(f"    Min clip length:           {vs['min_duration']:.1f} sec\n")
            f.write(f"    Max clip length:           {vs['max_duration']:.1f} sec\n")
            f.write(f"    Mean clip length:          {vs['mean_duration']:.1f} sec\n")
            f.write(f"    Total duration:            {vs['total_duration']:.1f} sec\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("OVERALL TOTALS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total clips:                   {total_clips}\n")
        f.write(f"Total duration:                {total_duration/3600:.2f} hours\n")
        f.write(f"Estimated annotation time:     {estimated_hours:.1f} hours\n")