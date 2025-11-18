"""Tracklet annotation evaluation and metrics calculation."""

import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats

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