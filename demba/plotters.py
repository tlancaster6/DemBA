import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import re

def plot_mouthing_event_summary(video_paths, plot_dir):
    event_counts = {}
    for vp in video_paths:
        df = pd.read_csv(str(vp).replace('.mp4', '_framefeatures.csv'), index_col=0)
        n_events = df.mouthing_event_id.max() + 1
        event_counts.update({vp.stem: n_events})
    df = pd.DataFrame(event_counts.items(), columns=['trial', 'n_mouthing_events'])
    fig, ax = plt.subplots()
    sns.barplot(df, x='trial', y='n_mouthing_events', ax=ax)
    ax.set(title='mouthing events counts by trial')
    ax.tick_params(axis='x', labelrotation=90)
    fig.tight_layout()
    fig.savefig(str(plot_dir / 'mouthing_counts.pdf'))
    plt.close(fig)

def plot_double_occupancy_event_summary(video_paths, plot_dir):
    event_counts = {}
    for vp in video_paths:
        df = pd.read_csv(str(vp).replace('.mp4', '_framefeatures.csv'), index_col=0)
        n_events = df.double_occupancy_event_id.max() + 1
        event_counts.update({vp.stem: n_events})
    df = pd.DataFrame(event_counts.items(), columns=['trial', 'n_double_occupancy_events'])
    fig, ax = plt.subplots()
    sns.barplot(df, x='trial', y='n_double_occupancy_events', ax=ax)
    ax.set(title='double occupancy events counts by trial')
    ax.tick_params(axis='x', labelrotation=90)
    fig.tight_layout()
    fig.savefig(str(plot_dir / 'double_occupancy_counts.pdf'))
    plt.close(fig)

def plot_quivering_event_summary(quivering_annotation_file, plot_dir):
    trial_ids = [f'{prefix}_group{suffix}' for prefix in ['BHVE', 'CTRL'] for suffix in [1, 2, 3, 5, 6, 7, 8, 9]]
    data = []
    for tid in trial_ids:
        quivering_data = pd.read_excel(quivering_annotation_file, sheet_name=tid, skiprows=1)
        quivering_data = quivering_data.loc[:, ['temporal_segment_start', 'temporal_segment_end']]
        quivering_lengths = quivering_data.temporal_segment_end - quivering_data.temporal_segment_start
        quivering_lengths.name = tid
        data.append(quivering_lengths)
    data = pd.concat(data, axis=1)
    counts = data.count()

    fig, ax = plt.subplots()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set(title='quivering event counts by trial')
    ax.tick_params(axis='x', labelrotation=90)
    fig.tight_layout()
    fig.savefig(str(plot_dir / 'quivering_counts.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.boxplot(data=data, ax=ax)
    ax.set(title='quivering event lengths by trial', ylabel='event length (secs)')
    ax.tick_params(axis='x', labelrotation=90)
    fig.tight_layout()
    fig.savefig(str(plot_dir / 'quivering_lengths.pdf'))
    plt.close(fig)

def plot_all_timeseries(vid_paths, plot_dir):
    data = {}
    names = []
    features = ['mouthing', 'quivering', 'double_occupancy']
    for vp in vid_paths:
        names.append(vp.stem)
        df = pd.read_csv(str(vp).replace('.mp4', '_framefeatures.csv'), index_col=0)
        df['mouthing_event_id'] = df['mouthing_event_id'] >= 0
        df.rename(columns={'mouthing_event_id':  'mouthing'}, inplace=True)
        df['double_occupancy_event_id'] = df['double_occupancy_event_id'] >= 0
        df.rename(columns={'double_occupancy_event_id':  'double_occupancy'}, inplace=True)
        data.update({names[-1]: df})
    for name in names:
        fig, axes = plt.subplots(len(features), 1, figsize=(13.3, 7.5), sharex=True)
        for i, feat in enumerate(features):
            try:
                sns.lineplot(x=data[name].index, y=data[name][feat], ax=axes[i], lw=0.5)
            except KeyError:
                pass
            axes[i].set_xlabel('frame')
            axes[i].set_ylabel(feat, rotation='horizontal', labelpad=40)
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(str(plot_dir / f'{name}_timeseries.pdf'))
        plt.close(fig)
    names_split = {'CTRL': [n for n in names if 'CTRL' in n], 'BHVE': [n for n in names if 'BHVE' in n]}
    for feat in features:
        for split, names in names_split.items():
            fig, axes = plt.subplots(len(names), 1, figsize=(13.3, 7.5), sharex=True, sharey=True)
            for i, name in enumerate(names):
                try:
                    sns.lineplot(x=data[name].index, y=data[name][feat], ax=axes[i], lw=0.5)
                except KeyError:
                    pass
                axes[i].set_xlabel('frame')
                axes[i].set_ylabel(name, rotation='horizontal', labelpad=40)
            fig.suptitle(feat)
            fig.tight_layout()
            fig.savefig(str(plot_dir / f'{split}_{feat}_timeseries.pdf'))
            plt.close(fig)



quivering_annotation_path = '/home/tlancaster/DLC/demasoni_singlenuc/quivering_annotations/Mbuna_behavior_annotations.xlsx'
parent_dir = Path('/home/tlancaster/DLC/demasoni_singlenuc/Analysis/Videos')
vid_paths = list(parent_dir.glob('**/*.mp4'))
pattern = '((CTRL)|(BHVE))_group\d.mp4'
vid_paths = sorted([p for p in vid_paths if re.fullmatch(pattern, p.name)])
plot_dir = Path('/home/tlancaster/DLC/demasoni_singlenuc/Analysis/Plots')
plot_all_timeseries(vid_paths, plot_dir)
plot_mouthing_event_summary(vid_paths, plot_dir)
plot_quivering_event_summary(quivering_annotation_path, plot_dir)
plot_double_occupancy_event_summary(vid_paths, plot_dir)
# plot_mouthing_event_summary(analysis_dir, plot_dir)
