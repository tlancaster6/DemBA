import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_mouthing_event_summary(analysis_dir, plot_dir):
    analysis_dir = Path(analysis_dir)
    event_counts = {}
    for video_dir in analysis_dir.glob('*'):
        trial_name = video_dir.name
        mouthing_event_summary = video_dir / 'output' / 'mouthing_events.csv'
        if mouthing_event_summary.exists():
            n_events = pd.read_csv(mouthing_event_summary, header=None, index_col=0).loc['n_mouthing_events', 1]
            event_counts.update({trial_name: n_events})
    df = pd.DataFrame(event_counts.items(), columns=['trial', 'n_mouthing_events'])
    fig, ax = plt.subplots()
    sns.barplot(df, x='trial', y='n_mouthing_events', ax=ax)
    ax.set(title='mouthing events counts by trial')
    ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()
    fig.savefig(str(plot_dir / 'mouthing_counts.pdf'))
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
    ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()
    fig.savefig(str(plot_dir / 'quivering_counts.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.boxplot(data=data, ax=ax)
    ax.set(title='quivering event lengths by trial', ylabel='event length (secs)')
    ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()
    fig.savefig(str(plot_dir / 'quivering_lengths.pdf'))
    plt.close(fig)





analysis_dir = Path(r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\analysis")
quivering_annotation_file = r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\quivering_annotations\Mbuna_behavior_annotations.xlsx"
plot_dir = Path(r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\plots")
# plot_mouthing_event_summary(analysis_dir, plot_dir)
