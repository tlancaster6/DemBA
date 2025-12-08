"""Tracklet annotation evaluation and metrics calculation."""

import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats

from demba.utils.dlc import load_tracklets
from demba import config

logger = logging.getLogger(__name__)


def load_and_validate_annotations(annotation_csv_path, metadata_json_path):
    """
    Load annotations and metadata, validate completeness.

    Parameters
    ----------
    annotation_csv_path : Path
        Path to completed annotation_sheet.csv
    metadata_json_path : Path
        Path to clip_metadata.json

    Returns
    -------
    pd.DataFrame
        Merged dataframe with columns:
        - clip_id
        - video_name
        - ground_truth_label (from annotator)
        - predicted_label (from DemBA)
        - confidence (silhouette score)
        - tracklet_id
        - length (n_frames)
        - context
        - confidence_stratum
        - length_stratum
        - stratum
        - notes
        - duration_sec
        - start_frame
        - end_frame

    Raises
    ------
    ValueError : If validation fails
    """
    # Load CSV
    annotations = pd.read_csv(annotation_csv_path)

    # Load JSON
    with open(metadata_json_path) as f:
        metadata = json.load(f)

    # Convert metadata to dataframe
    metadata_rows = []
    for clip_id, meta in metadata.items():
        metadata_rows.append({
            'clip_id': clip_id,
            'predicted_label': meta['predicted_label'],
            'confidence': meta['confidence'],
            'tracklet_id': meta['tracklet_id'],
            'length': meta['n_frames'],
            'context': meta['context'],
            'confidence_stratum': meta['confidence_stratum'],
            'length_stratum': meta['length_stratum'],
            'stratum': meta['stratum'],
            'duration_sec': meta['duration_sec'],
            'start_frame': meta['start_frame'],
            'end_frame': meta['end_frame'],
            'video_name_meta': meta['video_name']
        })
    metadata_df = pd.DataFrame(metadata_rows)

    # Validate
    issues = []

    # Check for missing labels
    missing = annotations[annotations['label'].isna() | (annotations['label'] == '')]
    if len(missing) > 0:
        issues.append(
            f"ERROR: {len(missing)} clips have missing labels:\n"
            f"  {missing['clip_id'].tolist()}"
        )

    # Check for invalid labels
    valid_labels = {'m', 'f', 'c'}
    # Filter out missing first, then check remaining
    labeled = annotations[~(annotations['label'].isna() | (annotations['label'] == ''))]
    invalid = labeled[~labeled['label'].isin(valid_labels)]
    if len(invalid) > 0:
        issues.append(
            f"ERROR: {len(invalid)} clips have invalid labels:\n"
            f"  Valid: m, f, c\n"
            f"  Found: {invalid[['clip_id', 'label']].values.tolist()}"
        )

    # Check for duplicate clip_ids
    duplicates = annotations[annotations.duplicated('clip_id', keep=False)]
    if len(duplicates) > 0:
        issues.append(
            f"ERROR: Duplicate clip_ids found:\n"
            f"  {duplicates['clip_id'].unique().tolist()}"
        )

    # Check that all metadata clips have annotations
    meta_clips = set(metadata_df['clip_id'])
    annot_clips = set(annotations['clip_id'])
    missing_annots = meta_clips - annot_clips
    if len(missing_annots) > 0:
        issues.append(
            f"WARNING: {len(missing_annots)} clips in metadata missing from annotations:\n"
            f"  {list(missing_annots)[:10]}"  # Show first 10
        )

    extra_annots = annot_clips - meta_clips
    if len(extra_annots) > 0:
        issues.append(
            f"WARNING: {len(extra_annots)} clips in annotations not in metadata:\n"
            f"  {list(extra_annots)[:10]}"
        )

    # Raise if there are errors (not warnings)
    errors = [issue for issue in issues if issue.startswith('ERROR')]
    if len(errors) > 0:
        raise ValueError("\n\n".join(errors))

    # Print warnings
    warnings = [issue for issue in issues if issue.startswith('WARNING')]
    for warning in warnings:
        logger.warning(warning)
        print(warning)

    # Merge on clip_id
    merged = annotations.merge(metadata_df, on='clip_id', how='inner')

    # Rename for clarity
    merged = merged.rename(columns={'label': 'ground_truth_label'})

    # Verify video names match
    name_mismatch = merged[merged['video_name'] != merged['video_name_meta']]
    if len(name_mismatch) > 0:
        logger.warning(
            f"{len(name_mismatch)} clips have mismatched video names between annotation and metadata"
        )

    # Drop the duplicate video name column
    merged = merged.drop(columns=['video_name_meta'])

    return merged


def calculate_accuracy_metrics(annotations_df):
    """
    Calculate tracklet-level accuracy metrics.

    Parameters
    ----------
    annotations_df : pd.DataFrame
        Merged annotations from load_and_validate_annotations()

    Returns
    -------
    dict : Metrics dictionary
        {
            'overall': {
                'n_tracklets': int,
                'n_conjoined': int,
                'n_evaluated': int (excluding conjoined),
                'accuracy': float,
                'frame_weighted_accuracy': float
            },
            'by_video': {
                'video1': {'accuracy': float, 'n': int, 'n_conjoined': int},
                ...
            },
            'by_confidence': {
                'low': {'accuracy': float, 'n': int},
                'medium': {'accuracy': float, 'n': int},
                'high': {'accuracy': float, 'n': int}
            },
            'by_context': {
                'solo': {'accuracy': float, 'n': int},
                'duo': {'accuracy': float, 'n': int}
            },
            'by_length': {
                'short': {'accuracy': float, 'n': int},
                'long': {'accuracy': float, 'n': int}
            },
            'confusion_matrix': {
                'true_m_pred_m': int,
                'true_m_pred_f': int,
                'true_f_pred_m': int,
                'true_f_pred_f': int
            }
        }
    """
    # Filter out conjoined
    eval_df = annotations_df[annotations_df['ground_truth_label'] != 'c'].copy()

    # Calculate correctness
    eval_df['correct'] = (eval_df['ground_truth_label'] == eval_df['predicted_label'])

    # Overall metrics
    n_total = len(annotations_df)
    n_conjoined = sum(annotations_df['ground_truth_label'] == 'c')
    n_evaluated = len(eval_df)

    overall = {
        'n_tracklets': n_total,
        'n_conjoined': n_conjoined,
        'n_evaluated': n_evaluated,
        'accuracy': eval_df['correct'].mean() if n_evaluated > 0 else 0.0,
        'frame_weighted_accuracy': (
            (eval_df['correct'] * eval_df['length']).sum() / eval_df['length'].sum()
            if n_evaluated > 0 else 0.0
        )
    }

    # By video
    by_video = {}
    for video_name, group in annotations_df.groupby('video_name'):
        group_eval = group[group['ground_truth_label'] != 'c']
        if len(group_eval) > 0:
            by_video[video_name] = {
                'accuracy': (group_eval['ground_truth_label'] == group_eval['predicted_label']).mean(),
                'n': len(group_eval),
                'n_conjoined': sum(group['ground_truth_label'] == 'c')
            }

    # By confidence stratum
    by_confidence = {}
    for conf_stratum in ['low', 'medium', 'high']:
        group = eval_df[eval_df['confidence_stratum'] == conf_stratum]
        if len(group) > 0:
            by_confidence[conf_stratum] = {
                'accuracy': group['correct'].mean(),
                'n': len(group)
            }

    # By context
    by_context = {}
    for context in ['solo', 'duo']:
        group = eval_df[eval_df['context'] == context]
        if len(group) > 0:
            by_context[context] = {
                'accuracy': group['correct'].mean(),
                'n': len(group)
            }

    # By length stratum
    by_length = {}
    for length_stratum in ['short', 'long']:
        group = eval_df[eval_df['length_stratum'] == length_stratum]
        if len(group) > 0:
            by_length[length_stratum] = {
                'accuracy': group['correct'].mean(),
                'n': len(group)
            }

    # Confusion matrix (only m/f, not c)
    confusion = {
        'true_m_pred_m': sum((eval_df['ground_truth_label'] == 'm') & (eval_df['predicted_label'] == 'm')),
        'true_m_pred_f': sum((eval_df['ground_truth_label'] == 'm') & (eval_df['predicted_label'] == 'f')),
        'true_f_pred_m': sum((eval_df['ground_truth_label'] == 'f') & (eval_df['predicted_label'] == 'm')),
        'true_f_pred_f': sum((eval_df['ground_truth_label'] == 'f') & (eval_df['predicted_label'] == 'f'))
    }

    return {
        'overall': overall,
        'by_video': by_video,
        'by_confidence': by_confidence,
        'by_context': by_context,
        'by_length': by_length,
        'confusion_matrix': confusion
    }


def calculate_confidence_intervals(annotations_df, metric='accuracy', n_bootstrap=10000):
    """
    Calculate bootstrap confidence intervals for accuracy metrics.

    Parameters
    ----------
    annotations_df : pd.DataFrame
        Annotations dataframe
    metric : str
        'accuracy' or 'frame_weighted_accuracy'
    n_bootstrap : int
        Number of bootstrap samples

    Returns
    -------
    tuple : (mean, ci_lower, ci_upper)
    """
    # Filter out conjoined
    eval_df = annotations_df[annotations_df['ground_truth_label'] != 'c'].copy()
    eval_df['correct'] = (eval_df['ground_truth_label'] == eval_df['predicted_label'])

    if len(eval_df) == 0:
        return (0.0, 0.0, 0.0)

    bootstrap_values = []
    n = len(eval_df)

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample_indices = np.random.choice(n, size=n, replace=True)
        sample = eval_df.iloc[sample_indices]

        if metric == 'accuracy':
            value = sample['correct'].mean()
        elif metric == 'frame_weighted_accuracy':
            value = (sample['correct'] * sample['length']).sum() / sample['length'].sum()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        bootstrap_values.append(value)

    bootstrap_values = np.array(bootstrap_values)
    mean = bootstrap_values.mean()
    ci_lower = np.percentile(bootstrap_values, 2.5)
    ci_upper = np.percentile(bootstrap_values, 97.5)

    return (mean, ci_lower, ci_upper)


def generate_evaluation_report(metrics, annotations_df, output_dir, create_plots=True):
    """
    Generate publication-ready evaluation report.

    Parameters
    ----------
    metrics : dict
        Output from calculate_accuracy_metrics()
    annotations_df : pd.DataFrame
        Annotations dataframe
    output_dir : Path
        Directory for output files
    create_plots : bool
        Whether to generate plots (requires matplotlib)

    Outputs
    -------
    1. evaluation_report.txt (text summary)
    2. metrics_table.csv (for publication)
    3. accuracy_by_confidence.png (plot, if create_plots=True)
    4. accuracy_by_video.png (plot, if create_plots=True)
    5. confusion_matrix.png (plot, if create_plots=True)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate confidence intervals
    print("Calculating bootstrap confidence intervals...")
    acc_mean, acc_ci_lower, acc_ci_upper = calculate_confidence_intervals(
        annotations_df, 'accuracy', n_bootstrap=10000
    )
    frame_mean, frame_ci_lower, frame_ci_upper = calculate_confidence_intervals(
        annotations_df, 'frame_weighted_accuracy', n_bootstrap=10000
    )

    # Calculate per-video statistics
    video_accuracies = [v['accuracy'] for v in metrics['by_video'].values()]
    video_acc_mean = np.mean(video_accuracies) if len(video_accuracies) > 0 else 0.0
    video_acc_std = np.std(video_accuracies) if len(video_accuracies) > 0 else 0.0

    # Write text report
    report_path = output_dir / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("TRACKLET ANNOTATION EVALUATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Annotation file: {annotations_df.attrs.get('annotation_file', 'N/A')}\n\n")

        f.write("OVERALL METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total tracklets:              {metrics['overall']['n_tracklets']}\n")
        f.write(f"Conjoined tracklets:          {metrics['overall']['n_conjoined']} ")
        f.write(f"({metrics['overall']['n_conjoined']/max(1,metrics['overall']['n_tracklets'])*100:.1f}%)\n")
        f.write(f"Evaluated tracklets:          {metrics['overall']['n_evaluated']} (excluding conjoined)\n\n")

        f.write(f"Tracklet Accuracy:            {metrics['overall']['accuracy']*100:.1f}%\n")
        f.write(f"  Per-video mean ± std:       {video_acc_mean*100:.1f}% ± {video_acc_std*100:.1f}%\n")
        f.write(f"Frame-Weighted Accuracy:      {metrics['overall']['frame_weighted_accuracy']*100:.1f}%\n\n")

        f.write("95% Confidence Intervals (bootstrap):\n")
        f.write(f"  Tracklet Accuracy:          [{acc_ci_lower*100:.1f}%, {acc_ci_upper*100:.1f}%]\n")
        f.write(f"  Frame-Weighted:             [{frame_ci_lower*100:.1f}%, {frame_ci_upper*100:.1f}%]\n\n")

        f.write("STRATIFIED RESULTS\n")
        f.write("-" * 70 + "\n\n")

        f.write("By Confidence Stratum:\n")
        for stratum in ['low', 'medium', 'high']:
            if stratum in metrics['by_confidence']:
                m = metrics['by_confidence'][stratum]
                f.write(f"  {stratum.capitalize():8s} (<p20/p20-p80/≥p80):  {m['accuracy']*100:5.1f}% ({m['n']} tracklets)\n")

        f.write("\nBy Context:\n")
        for context in ['solo', 'duo']:
            if context in metrics['by_context']:
                m = metrics['by_context'][context]
                f.write(f"  {context.capitalize():8s}:                   {m['accuracy']*100:5.1f}% ({m['n']} tracklets)\n")

        f.write("\nBy Tracklet Length:\n")
        for length in ['short', 'long']:
            if length in metrics['by_length']:
                m = metrics['by_length'][length]
                f.write(f"  {length.capitalize():8s} (<median/≥median):   {m['accuracy']*100:5.1f}% ({m['n']} tracklets)\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("PER-VIDEO BREAKDOWN\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Video':<40s} {'Accuracy':>10s} {'N':>6s} {'Conjoined':>10s}\n")
        f.write("-" * 70 + "\n")
        for video_name, m in sorted(metrics['by_video'].items()):
            f.write(f"{video_name:<40s} {m['accuracy']*100:9.1f}% {m['n']:6d} {m['n_conjoined']:10d}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 70 + "\n")
        cm = metrics['confusion_matrix']
        f.write(f"                  Predicted M    Predicted F\n")
        f.write(f"Actual M          {cm['true_m_pred_m']:11d}    {cm['true_m_pred_f']:11d}\n")
        f.write(f"Actual F          {cm['true_f_pred_m']:11d}    {cm['true_f_pred_f']:11d}\n\n")

        # Calculate precision/recall
        if (cm['true_m_pred_m'] + cm['true_f_pred_m']) > 0:
            precision_m = cm['true_m_pred_m'] / (cm['true_m_pred_m'] + cm['true_f_pred_m'])
            f.write(f"Precision (M):                {precision_m*100:.1f}%\n")
        if (cm['true_f_pred_f'] + cm['true_m_pred_f']) > 0:
            precision_f = cm['true_f_pred_f'] / (cm['true_f_pred_f'] + cm['true_m_pred_f'])
            f.write(f"Precision (F):                {precision_f*100:.1f}%\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("PUBLICATION-READY SUMMARY\n")
        f.write("-" * 70 + "\n")
        n_videos = len(metrics['by_video'])
        total_hours = annotations_df['duration_sec'].sum() / 3600
        f.write(f"We evaluated ID correction accuracy on {metrics['overall']['n_evaluated']} manually\n")
        f.write(f"annotated tracklets from {n_videos} videos (total duration: {total_hours:.1f} hours).\n")
        f.write(f"Tracklets were sampled using stratified sampling (length, silhouette\n")
        f.write(f"confidence, context) to ensure representative coverage.\n\n")
        f.write(f"Overall tracklet accuracy was {metrics['overall']['accuracy']*100:.1f}% ")
        f.write(f"(95% CI: [{acc_ci_lower*100:.1f}%, {acc_ci_upper*100:.1f}%]).\n")
        f.write(f"Frame-weighted accuracy was {metrics['overall']['frame_weighted_accuracy']*100:.1f}%,\n")
        f.write(f"indicating that {metrics['overall']['frame_weighted_accuracy']*100:.1f}% of analyzed frames\n")
        f.write(f"had correct identity assignments.\n")

    print(f"\n✓ Evaluation report written to: {report_path}")

    # Write metrics table CSV
    metrics_table = []
    metrics_table.append({'Metric': 'Overall Accuracy', 'Value': f"{metrics['overall']['accuracy']*100:.1f}%", 'N': metrics['overall']['n_evaluated']})
    metrics_table.append({'Metric': 'Frame-Weighted Accuracy', 'Value': f"{metrics['overall']['frame_weighted_accuracy']*100:.1f}%", 'N': metrics['overall']['n_evaluated']})

    for stratum, m in metrics['by_confidence'].items():
        metrics_table.append({'Metric': f'Accuracy ({stratum} confidence)', 'Value': f"{m['accuracy']*100:.1f}%", 'N': m['n']})

    for context, m in metrics['by_context'].items():
        metrics_table.append({'Metric': f'Accuracy ({context} context)', 'Value': f"{m['accuracy']*100:.1f}%", 'N': m['n']})

    metrics_df = pd.DataFrame(metrics_table)
    metrics_csv_path = output_dir / 'metrics_table.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"✓ Metrics table written to: {metrics_csv_path}")

    # Create plots if requested
    if create_plots:
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend

            # Plot 1: Accuracy by confidence
            fig, ax = plt.subplots(figsize=(8, 6))
            conf_labels = ['Low\n(<p20)', 'Medium\n(p20-p80)', 'High\n(≥p80)']
            conf_values = [metrics['by_confidence'].get(s, {'accuracy': 0})['accuracy']*100
                          for s in ['low', 'medium', 'high']]
            conf_ns = [metrics['by_confidence'].get(s, {'n': 0})['n']
                      for s in ['low', 'medium', 'high']]

            bars = ax.bar(conf_labels, conf_values, color=['#e74c3c', '#f39c12', '#2ecc71'])
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_xlabel('Silhouette Confidence Stratum', fontsize=12)
            ax.set_title('Tracklet Accuracy by Confidence Stratum', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 105)
            ax.axhline(metrics['overall']['accuracy']*100, color='black', linestyle='--',
                      label=f"Overall ({metrics['overall']['accuracy']*100:.1f}%)")
            ax.legend()

            # Add n labels on bars
            for bar, n in zip(bars, conf_ns):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'n={n}', ha='center', va='bottom', fontsize=10)

            plt.tight_layout()
            plot1_path = output_dir / 'accuracy_by_confidence.png'
            plt.savefig(plot1_path, dpi=300)
            plt.close()
            print(f"✓ Plot written to: {plot1_path}")

            # Plot 2: Accuracy by video
            fig, ax = plt.subplots(figsize=(12, 6))
            video_names = list(metrics['by_video'].keys())
            video_accs = [metrics['by_video'][v]['accuracy']*100 for v in video_names]

            # Sort by accuracy
            sorted_indices = np.argsort(video_accs)
            video_names = [video_names[i] for i in sorted_indices]
            video_accs = [video_accs[i] for i in sorted_indices]

            bars = ax.barh(range(len(video_names)), video_accs, color='#3498db')
            ax.set_yticks(range(len(video_names)))
            ax.set_yticklabels(video_names, fontsize=8)
            ax.set_xlabel('Accuracy (%)', fontsize=12)
            ax.set_title('Tracklet Accuracy by Video', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 105)
            ax.axvline(metrics['overall']['accuracy']*100, color='red', linestyle='--',
                      label=f"Overall ({metrics['overall']['accuracy']*100:.1f}%)")
            ax.legend()
            plt.tight_layout()
            plot2_path = output_dir / 'accuracy_by_video.png'
            plt.savefig(plot2_path, dpi=300)
            plt.close()
            print(f"✓ Plot written to: {plot2_path}")

            # Plot 3: Confusion matrix
            fig, ax = plt.subplots(figsize=(6, 6))
            cm = metrics['confusion_matrix']
            cm_array = np.array([[cm['true_m_pred_m'], cm['true_m_pred_f']],
                                [cm['true_f_pred_m'], cm['true_f_pred_f']]])

            im = ax.imshow(cm_array, cmap='Blues', aspect='auto')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Predicted M', 'Predicted F'], fontsize=12)
            ax.set_yticklabels(['Actual M', 'Actual F'], fontsize=12)
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

            # Add text annotations
            for i in range(2):
                for j in range(2):
                    text = ax.text(j, i, cm_array[i, j],
                                 ha="center", va="center", color="black", fontsize=20)

            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plot3_path = output_dir / 'confusion_matrix.png'
            plt.savefig(plot3_path, dpi=300)
            plt.close()
            print(f"✓ Plot written to: {plot3_path}")

        except ImportError:
            logger.warning("matplotlib not available, skipping plots")
            print("  (matplotlib not available, skipping plots)")


def detect_conjoined_tracklet(tracklet, min_run_length=None):
    """
    Detect if a tracklet is conjoined using the same logic as split_conjoined_tracklets().

    A "conjoined" tracklet is one that physically switches from tracking one fish
    to tracking another fish, resulting in sustained runs of different identity
    predictions (male vs female).

    Parameters
    ----------
    tracklet : Tracklet
        Tracklet object with identity predictions (4th column)
    min_run_length : int, optional
        Minimum consecutive frames of same ID to count as a "real" identity run.
        If None, uses config.DEFAULT_MIN_CONJOINED_RUN_LENGTH

    Returns
    -------
    bool
        True if tracklet is conjoined (has major runs of both male and female)
    dict
        Metadata about the detection:
        {
            'is_conjoined': bool,
            'has_identity_data': bool,
            'runs': list of dict,
            'major_runs': list of dict,
            'has_male_run': bool,
            'has_female_run': bool
        }
    """
    if min_run_length is None:
        min_run_length = config.DEFAULT_MIN_CONJOINED_RUN_LENGTH

    result = {
        'is_conjoined': False,
        'has_identity_data': False,
        'runs': [],
        'major_runs': [],
        'has_male_run': False,
        'has_female_run': False
    }

    # Check if tracklet has identity data
    if tracklet.data.shape[-1] < 4 or len(tracklet) < min_run_length:
        return False, result

    result['has_identity_data'] = True

    # Get frame-level identity (ID is same across all bodyparts per frame)
    # Just take the first bodypart's ID for each frame
    frame_ids = tracklet.data[:, 0, 3]  # shape: (nframes,)

    # Run-length encode to find consecutive runs
    runs = []
    if len(frame_ids) > 0:
        current_id = frame_ids[0]
        run_start = 0

        for i in range(1, len(frame_ids)):
            # Treat NaN as continuation of current run (ignore brief gaps)
            if frame_ids[i] == current_id or np.isnan(frame_ids[i]):
                continue
            else:
                # Run ended
                runs.append({
                    'id': current_id,
                    'start_idx': run_start,
                    'end_idx': i - 1,
                    'length': i - run_start
                })
                current_id = frame_ids[i]
                run_start = i

        # Add final run
        runs.append({
            'id': current_id,
            'start_idx': run_start,
            'end_idx': len(frame_ids) - 1,
            'length': len(frame_ids) - run_start
        })

    result['runs'] = runs

    # Find major runs (length >= min_run_length, ID in {0, 1})
    major_runs = [r for r in runs if r['length'] >= min_run_length and r['id'] in [0, 1]]
    result['major_runs'] = major_runs

    # Check if tracklet has major runs of BOTH male (0) and female (1)
    has_male = any(r['id'] == 0 for r in major_runs)
    has_female = any(r['id'] == 1 for r in major_runs)

    result['has_male_run'] = has_male
    result['has_female_run'] = has_female
    result['is_conjoined'] = has_male and has_female

    return result['is_conjoined'], result


def evaluate_conjoined_detection(annotations_df, metadata_json_path, min_run_length=None,
                                  exclude_videos=None):
    """
    Evaluate conjoined tracklet detection algorithm against manual annotations.

    This function simulates the conjoined detection algorithm on the evaluation set
    to determine how well it identifies tracklets that switch between tracking
    different individuals.

    Parameters
    ----------
    annotations_df : pd.DataFrame
        Merged annotations from load_and_validate_annotations()
    metadata_json_path : Path
        Path to clip_metadata.json (contains trial_dir for loading tracklets)
    min_run_length : int, optional
        Minimum consecutive frames of same ID to count as a "real" identity run.
        If None, uses config.DEFAULT_MIN_CONJOINED_RUN_LENGTH
    exclude_videos : list of str, optional
        List of video names to exclude from analysis. Video names should match
        the 'video_name' column in annotation_sheet.csv (e.g., ['trial1.mp4', 'trial2.mp4'])

    Returns
    -------
    dict
        Evaluation metrics:
        {
            'confusion_matrix': {
                'TP': int,  # Correctly identified conjoined
                'FP': int,  # Incorrectly flagged as conjoined
                'TN': int,  # Correctly identified as not conjoined
                'FN': int   # Missed conjoined tracklets
            },
            'precision': float,  # TP / (TP + FP)
            'recall': float,     # TP / (TP + FN)
            'f1': float,         # Harmonic mean of precision and recall
            'accuracy': float,   # (TP + TN) / total
            'false_positive_rate': float,  # FP / (FP + TN)
            'false_negative_rate': float,  # FN / (FN + TP)
            'n_total': int,
            'n_actual_conjoined': int,
            'n_predicted_conjoined': int,
            'n_excluded': int,  # Number of tracklets excluded
            'excluded_videos': list,  # List of excluded video names
            'examples': {
                'false_negatives': list,  # Missed conjoined tracklets
                'false_positives': list   # Incorrectly flagged tracklets
            },
            'detection_details': list  # Per-tracklet detection metadata
        }
    """
    if min_run_length is None:
        min_run_length = config.DEFAULT_MIN_CONJOINED_RUN_LENGTH

    if exclude_videos is None:
        exclude_videos = []

    # Load metadata to get trial directories
    with open(metadata_json_path) as f:
        metadata = json.load(f)

    # Initialize counters
    TP = 0  # True Positive: correctly identified conjoined
    FP = 0  # False Positive: incorrectly flagged as conjoined
    TN = 0  # True Negative: correctly identified as not conjoined
    FN = 0  # False Negative: missed conjoined tracklets
    n_excluded = 0

    false_negatives = []
    false_positives = []
    detection_details = []

    # Cache loaded tracklets by (trial_dir, pickle_path) to avoid reloading
    tracklet_cache = {}

    # Print exclusion info
    if exclude_videos:
        print(f"\nExcluding {len(exclude_videos)} video(s) from conjoined detection analysis:")
        for video_name in exclude_videos:
            print(f"  - {video_name}")

    print(f"\nEvaluating conjoined detection (min_run_length={min_run_length})...")

    for idx, row in annotations_df.iterrows():
        clip_id = row['clip_id']
        video_name = row['video_name']
        ground_truth = row['ground_truth_label']
        tracklet_idx = row['tracklet_id']

        # Check if this video should be excluded
        if video_name in exclude_videos:
            n_excluded += 1
            continue

        # Get metadata for this clip
        clip_meta = metadata.get(clip_id)
        if clip_meta is None:
            logger.warning(f"No metadata found for {clip_id}, skipping")
            continue

        trial_dir = Path(clip_meta['trial_dir'])

        # Construct path to tracklet pickle
        # Assuming the pickle is named *_el.pickle in the trial directory
        pickle_files = list(trial_dir.glob('*_el.pickle'))
        if len(pickle_files) == 0:
            logger.warning(f"No *_el.pickle found in {trial_dir}, skipping {clip_id}")
            continue

        pickle_path = pickle_files[0]

        # Load tracklets (use cache)
        cache_key = str(pickle_path)
        if cache_key not in tracklet_cache:
            try:
                tracklets, header = load_tracklets(pickle_path)
                tracklet_cache[cache_key] = tracklets
            except Exception as e:
                logger.error(f"Failed to load tracklets from {pickle_path}: {e}")
                continue

        tracklets = tracklet_cache[cache_key]

        # Get the specific tracklet
        if tracklet_idx >= len(tracklets):
            logger.warning(f"Tracklet index {tracklet_idx} out of range for {clip_id}, skipping")
            continue

        tracklet = tracklets[tracklet_idx]

        # Run conjoined detection
        is_predicted_conjoined, detection_meta = detect_conjoined_tracklet(tracklet, min_run_length)

        # Ground truth
        is_actual_conjoined = (ground_truth == 'c')

        # Update confusion matrix
        if is_actual_conjoined and is_predicted_conjoined:
            TP += 1
        elif is_actual_conjoined and not is_predicted_conjoined:
            FN += 1
            false_negatives.append({
                'clip_id': clip_id,
                'video_name': row['video_name'],
                'tracklet_idx': tracklet_idx,
                'length': row['length'],
                'confidence': row['confidence'],
                'context': row['context'],
                'detection_meta': detection_meta
            })
        elif not is_actual_conjoined and is_predicted_conjoined:
            FP += 1
            false_positives.append({
                'clip_id': clip_id,
                'video_name': row['video_name'],
                'tracklet_idx': tracklet_idx,
                'predicted_label': row['predicted_label'],
                'length': row['length'],
                'confidence': row['confidence'],
                'context': row['context'],
                'detection_meta': detection_meta
            })
        elif not is_actual_conjoined and not is_predicted_conjoined:
            TN += 1

        # Store detection details
        detection_details.append({
            'clip_id': clip_id,
            'ground_truth': ground_truth,
            'predicted_conjoined': is_predicted_conjoined,
            'detection_meta': detection_meta
        })

    # Calculate metrics
    n_total = TP + FP + TN + FN
    n_actual_conjoined = TP + FN
    n_predicted_conjoined = TP + FP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / n_total if n_total > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    return {
        'confusion_matrix': {
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN
        },
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'n_total': n_total,
        'n_actual_conjoined': n_actual_conjoined,
        'n_predicted_conjoined': n_predicted_conjoined,
        'n_excluded': n_excluded,
        'excluded_videos': exclude_videos,
        'examples': {
            'false_negatives': false_negatives,
            'false_positives': false_positives
        },
        'detection_details': detection_details
    }


def generate_conjoined_detection_report(conjoined_metrics, output_dir, min_run_length=None):
    """
    Generate report for conjoined detection evaluation.

    Parameters
    ----------
    conjoined_metrics : dict
        Output from evaluate_conjoined_detection()
    output_dir : Path
        Directory for output files
    min_run_length : int, optional
        The min_run_length parameter used for detection

    Outputs
    -------
    1. conjoined_detection_report.txt (text summary)
    2. conjoined_false_negatives.csv (missed conjoined tracklets)
    3. conjoined_false_positives.csv (incorrectly flagged tracklets)
    """
    if min_run_length is None:
        min_run_length = config.DEFAULT_MIN_CONJOINED_RUN_LENGTH

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = conjoined_metrics['confusion_matrix']

    # Write text report
    report_path = output_dir / 'conjoined_detection_report.txt'
    with open(report_path, 'w') as f:
        f.write("CONJOINED TRACKLET DETECTION EVALUATION\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Min Run Length Parameter: {min_run_length} frames\n\n")

        f.write("OVERVIEW\n")
        f.write("-" * 70 + "\n")
        f.write("This report evaluates how well the automatic conjoined tracklet\n")
        f.write("detection algorithm identifies tracklets that switch between tracking\n")
        f.write("different individuals (male/female).\n\n")

        # Exclusion info
        if conjoined_metrics['n_excluded'] > 0:
            f.write("EXCLUDED VIDEOS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Number of tracklets excluded: {conjoined_metrics['n_excluded']}\n")
            f.write(f"Videos excluded from analysis:\n")
            for video_name in conjoined_metrics['excluded_videos']:
                f.write(f"  - {video_name}\n")
            f.write("\n")

        f.write("CONFUSION MATRIX\n")
        f.write("-" * 70 + "\n")
        f.write(f"                      Predicted Not Conjoined    Predicted Conjoined\n")
        f.write(f"Actual Not Conjoined  {cm['TN']:22d}    {cm['FP']:19d}\n")
        f.write(f"Actual Conjoined      {cm['FN']:22d}    {cm['TP']:19d}\n\n")

        f.write("CLASSIFICATION METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total tracklets:              {conjoined_metrics['n_total']}\n")
        f.write(f"Actual conjoined:             {conjoined_metrics['n_actual_conjoined']} ")
        f.write(f"({conjoined_metrics['n_actual_conjoined']/max(1,conjoined_metrics['n_total'])*100:.1f}%)\n")
        f.write(f"Predicted conjoined:          {conjoined_metrics['n_predicted_conjoined']} ")
        f.write(f"({conjoined_metrics['n_predicted_conjoined']/max(1,conjoined_metrics['n_total'])*100:.1f}%)\n\n")

        f.write(f"Accuracy:                     {conjoined_metrics['accuracy']*100:.1f}%\n")
        f.write(f"Precision:                    {conjoined_metrics['precision']*100:.1f}%\n")
        f.write(f"Recall (Sensitivity):         {conjoined_metrics['recall']*100:.1f}%\n")
        f.write(f"F1 Score:                     {conjoined_metrics['f1']:.3f}\n\n")

        f.write(f"False Positive Rate:          {conjoined_metrics['false_positive_rate']*100:.1f}%\n")
        f.write(f"False Negative Rate:          {conjoined_metrics['false_negative_rate']*100:.1f}%\n\n")

        f.write("INTERPRETATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Precision ({conjoined_metrics['precision']*100:.1f}%):\n")
        f.write(f"  Of tracklets flagged as conjoined, {conjoined_metrics['precision']*100:.1f}% are\n")
        f.write(f"  actually conjoined. Low precision = many false alarms.\n\n")

        f.write(f"Recall ({conjoined_metrics['recall']*100:.1f}%):\n")
        f.write(f"  Of actually-conjoined tracklets, {conjoined_metrics['recall']*100:.1f}% are detected.\n")
        f.write(f"  Low recall = many conjoined tracklets are missed.\n\n")

        f.write("ERROR ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"False Negatives (FN={cm['FN']}): Conjoined tracklets that were MISSED\n")
        f.write(f"  These represent identity switches that the algorithm fails to detect.\n")
        f.write(f"  They will NOT be split and may cause downstream identity errors.\n\n")

        f.write(f"False Positives (FP={cm['FP']}): Non-conjoined tracklets INCORRECTLY flagged\n")
        f.write(f"  These are good tracklets that would be unnecessarily split.\n")
        f.write(f"  This could fragment valid tracks and reduce coverage.\n\n")

        if cm['FN'] > 0:
            f.write(f"See conjoined_false_negatives.csv for details on missed tracklets.\n")
        if cm['FP'] > 0:
            f.write(f"See conjoined_false_positives.csv for details on false alarms.\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 70 + "\n")

        if conjoined_metrics['recall'] < 0.7:
            f.write(f"⚠ Low recall ({conjoined_metrics['recall']*100:.1f}%): Many conjoined tracklets are missed.\n")
            f.write(f"  Consider DECREASING min_run_length (currently {min_run_length}) to detect\n")
            f.write(f"  shorter identity runs. WARNING: May increase false positives.\n\n")

        if conjoined_metrics['precision'] < 0.7:
            f.write(f"⚠ Low precision ({conjoined_metrics['precision']*100:.1f}%): Many false alarms.\n")
            f.write(f"  Consider INCREASING min_run_length (currently {min_run_length}) to require\n")
            f.write(f"  longer identity runs. WARNING: May decrease recall.\n\n")

        if conjoined_metrics['f1'] >= 0.8:
            f.write(f"✓ Good F1 score ({conjoined_metrics['f1']:.3f}): Detection algorithm performs well.\n")
            f.write(f"  Current min_run_length={min_run_length} appears appropriate.\n\n")

        f.write("\nNOTE: Parameter sweep analysis (TODO) can help optimize min_run_length\n")
        f.write("by testing values from 20-100 frames and plotting precision/recall curves.\n")

    print(f"\n✓ Conjoined detection report written to: {report_path}")

    # Write false negatives CSV
    if len(conjoined_metrics['examples']['false_negatives']) > 0:
        fn_df = pd.DataFrame(conjoined_metrics['examples']['false_negatives'])
        # Flatten detection_meta for CSV
        fn_df['has_male_run'] = fn_df['detection_meta'].apply(lambda x: x.get('has_male_run', False))
        fn_df['has_female_run'] = fn_df['detection_meta'].apply(lambda x: x.get('has_female_run', False))
        fn_df['n_major_runs'] = fn_df['detection_meta'].apply(lambda x: len(x.get('major_runs', [])))
        fn_df = fn_df.drop(columns=['detection_meta'])

        fn_path = output_dir / 'conjoined_false_negatives.csv'
        fn_df.to_csv(fn_path, index=False)
        print(f"✓ False negatives written to: {fn_path}")

    # Write false positives CSV
    if len(conjoined_metrics['examples']['false_positives']) > 0:
        fp_df = pd.DataFrame(conjoined_metrics['examples']['false_positives'])
        # Flatten detection_meta for CSV
        fp_df['has_male_run'] = fp_df['detection_meta'].apply(lambda x: x.get('has_male_run', False))
        fp_df['has_female_run'] = fp_df['detection_meta'].apply(lambda x: x.get('has_female_run', False))
        fp_df['n_major_runs'] = fp_df['detection_meta'].apply(lambda x: len(x.get('major_runs', [])))
        fp_df = fp_df.drop(columns=['detection_meta'])

        fp_path = output_dir / 'conjoined_false_positives.csv'
        fp_df.to_csv(fp_path, index=False)
        print(f"✓ False positives written to: {fp_path}")