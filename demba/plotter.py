from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from math import ceil
import re

class Plotter:

    def __init__(self, parent_dir, mouthing_dist_mm=10, min_likelihood=0.5, n_minutes=None):
        """
        parent_dir should be a directory containing two folders: Videos and Annotations. Annotations should hold a single
        file: Annotations.xlsx. Videos should contain a directory for each video/trial analyzed. Each video/trial
        directory should contain the clipfeatures and framefeatures csvs. Plots will be output into a new directory
        in the parent_dir called "Summary"

        Parameters:
        - mouthing_dist_mm: Mouthing distance threshold in mm (default: 10)
        - min_likelihood: Minimum likelihood threshold (default: 0.5)
        - n_minutes: Time restriction in minutes (default: None)
        """
        self.parent_dir = Path(parent_dir)
        self.annotation_path = self.parent_dir / 'Annotations' / 'Annotations.xlsx'

        # Create parameter suffix for unique file naming (same as FeatureExtractor)
        self.param_suffix = f"_mdist{mouthing_dist_mm}mm_likelihood{min_likelihood}"
        if n_minutes is not None:
            self.param_suffix += f"_last{n_minutes}min"

        self.output_dir = self.parent_dir / 'Summary'
        self.output_dir.mkdir(exist_ok=True)
        self.data_dir = self.parent_dir / 'Videos'
        self.bhve_dirs, self.ctrl_dirs = self.get_bhve_ctrl_dir_paths()
        self.concat_clipfeature_csvs()
        self.clipfeature_df = pd.read_csv(str(self.output_dir / f'collated{self.param_suffix}_clipfeatures.csv'), index_col=0)

    def concat_clipfeature_csvs(self):
        rows = []
        for subdir in self.bhve_dirs:
            clipfeature_path = subdir / f'{subdir.name}{self.param_suffix}_clipfeatures.csv'
            if clipfeature_path.exists():
                row = pd.read_csv(str(clipfeature_path), index_col=0)
                row['split'] = 'bhve'
                rows.append(row)
            else:
                print(f"Warning: {clipfeature_path} not found, skipping {subdir.name}")
        for subdir in self.ctrl_dirs:
            clipfeature_path = subdir / f'{subdir.name}{self.param_suffix}_clipfeatures.csv'
            if clipfeature_path.exists():
                row = pd.read_csv(str(clipfeature_path), index_col=0)
                row['split'] = 'ctrl'
                rows.append(row)
            else:
                print(f"Warning: {clipfeature_path} not found, skipping {subdir.name}")

        if not rows:
            raise FileNotFoundError(f"No clipfeature CSV files found with suffix '{self.param_suffix}'. "
                                  "Make sure FeatureExtractor was run with the same parameters.")

        df = pd.concat(rows, axis=0)
        df = df.sort_index(key=trial_sort_key)
        df.to_csv(str(self.output_dir / f'collated{self.param_suffix}_clipfeatures.csv'))

    def get_bhve_ctrl_dir_paths(self):
        data_subdirs = list(self.data_dir.glob('*'))
        control_subdirs = [d for d in data_subdirs if ('CTRL' in d.name or 'DC' in d.name)]
        behave_subdirs = [d for d in data_subdirs if ('BHVE' in d.name or 'DB' in d.name)]
        if len(data_subdirs) != (len(control_subdirs) + len(behave_subdirs)):
            print('Warning: some directories not parsed into behave or control groups and will be omitted from analysis')
        return sorted(behave_subdirs), sorted(control_subdirs)

    def generate_clipfeature_boxplots(self):
        target_stats = [
             'n_mouthing_events',
             'n_double_occupancy_events',
             'n_spawning_events',
             'mouthing_event_fraction',
             'double_occupancy_event_fraction',
             'spawning_event_fraction',
             'raw_zero_occupancy_fraction',
             'raw_single_occupancy_fraction',
             'raw_double_occupancy_fraction',
        ]

        n_cols = 3
        n_rows = int(ceil(len(target_stats) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5, 10))
        flaxes = axes.flatten()
        for i, target in enumerate(target_stats):
            sns.boxplot(self.clipfeature_df, x=target, hue='split', ax=flaxes[i])
            flaxes[i].legend(loc='lower right')
        fig.tight_layout()
        fig.savefig(str(self.output_dir / f'clipfeature{self.param_suffix}_boxplots.pdf'))
        plt.close(fig)

    def generate_auto_manual_correlation_plots(self):
        n_circle_quivers = self.clipfeature_df.n_male_circle_quivers + self.clipfeature_df.n_female_circle_quivers
        circle_quiver_fraction = self.clipfeature_df.male_circle_quivering_fraction + self.clipfeature_df.female_circle_quivering_fraction

        fig, axes = plt.subplots(1, 2, figsize=(7.5, 5))
        sns.scatterplot(x=self.clipfeature_df.n_mouthing_events, y=n_circle_quivers, ax=axes[0])
        sns.scatterplot(x=self.clipfeature_df.mouthing_event_fraction, y=circle_quiver_fraction, ax=axes[1])
        fig.tight_layout()
        fig.savefig(str(self.output_dir / f'auto_manual_correlation{self.param_suffix}_plots.pdf'))
        plt.close(fig)

    def generate_event_timeseries_heatmaps(self, bin_width_frames=1800):
        """
        Generate heatmaps showing event occurrence over time.
        Creates 3 separate plots (one per event type) with trials as rows.

        Parameters:
        - bin_width_frames: Width of each time bin in frames (default: 60, which is 2 seconds at 30fps)
        """
        import numpy as np

        event_columns = ['mouthing_event_id', 'spawning_event_id', 'double_occupancy_event_id']
        event_names = ['Mouthing', 'Spawning', 'Double Occupancy']

        # Collect data from all trials
        trial_data = self._collect_trial_timeseries_data(event_columns, bin_width_frames)

        if not trial_data:
            print("Warning: No valid trial data found for heatmaps")
            return

        # Generate one plot per event type
        for event_idx, (event_col, event_name) in enumerate(zip(event_columns, event_names)):
            self._generate_single_metric_heatmap(trial_data, event_idx, event_name, bin_width_frames)

    def _collect_trial_timeseries_data(self, event_columns, bin_width_frames):
        """Collect timeseries data from all trials"""
        import numpy as np

        trial_data = []
        trial_names = []
        trial_groups = []
        max_n_bins = 0  # Track the maximum number of bins across all trials

        # Process BHVE trials first, then CTRL
        for trial_dir in self.bhve_dirs + self.ctrl_dirs:
            framefeatures_path = trial_dir / f'{trial_dir.name}{self.param_suffix}_framefeatures.csv'

            if not framefeatures_path.exists():
                print(f"Warning: {framefeatures_path} not found, skipping {trial_dir.name}")
                continue

            # Load framefeatures data
            framefeatures = pd.read_csv(framefeatures_path, index_col=0)

            # Check if required columns exist
            available_columns = [col for col in event_columns if col in framefeatures.columns]
            if not available_columns:
                print(f"Warning: No event columns found in {trial_dir.name}, skipping")
                continue

            # Create time bins with fixed width
            n_frames = len(framefeatures)
            n_bins = int(np.ceil(n_frames / bin_width_frames))
            max_n_bins = max(max_n_bins, n_bins)

            # Calculate event counts for this trial
            this_trial_data = np.zeros((len(event_columns), n_bins))

            for row_idx, event_col in enumerate(event_columns):
                if event_col in framefeatures.columns:
                    for bin_idx in range(n_bins):
                        start_frame = bin_idx * bin_width_frames
                        end_frame = min(start_frame + bin_width_frames, n_frames)

                        # Get events in this time bin
                        bin_events = framefeatures[event_col].iloc[start_frame:end_frame]

                        # Count unique event IDs (excluding -1)
                        unique_events = bin_events[bin_events >= 0].unique()
                        this_trial_data[row_idx, bin_idx] = len(unique_events)

            trial_data.append(this_trial_data)
            trial_names.append(trial_dir.name)
            trial_groups.append('BHVE' if trial_dir in self.bhve_dirs else 'CTRL')

        # Pad all trial data to have the same number of bins
        padded_trial_data = []
        for data in trial_data:
            if data.shape[1] < max_n_bins:
                # Pad with zeros
                padded_data = np.zeros((data.shape[0], max_n_bins))
                padded_data[:, :data.shape[1]] = data
                padded_trial_data.append(padded_data)
            else:
                padded_trial_data.append(data)

        return {
            'data': padded_trial_data,
            'names': trial_names,
            'groups': trial_groups,
            'n_bins': max_n_bins,
            'bin_width_frames': bin_width_frames
        }

    def _generate_single_metric_heatmap(self, trial_data, event_idx, event_name, bin_width_frames):
        """Generate a single heatmap for one event metric across all trials"""
        import numpy as np

        data = trial_data['data']
        names = trial_data['names']
        groups = trial_data['groups']
        n_bins = trial_data['n_bins']

        if not data:
            print(f"Warning: No data available for {event_name} heatmap")
            return

        # Extract data for this specific event type
        metric_data = np.array([trial_array[event_idx, :] for trial_array in data])

        # Create full-page figure (8.5 x 11 inches)
        fig, ax = plt.subplots(figsize=(8.5, 11))

        # Create heatmap
        im = ax.imshow(metric_data, aspect='auto', cmap='viridis', interpolation='nearest')

        # Set trial labels (y-axis)
        ax.set_yticks(range(len(names)))
        trial_labels = [f"{group}: {name}" for group, name in zip(groups, names)]
        ax.set_yticklabels(trial_labels, fontsize=8)

        # Add horizontal line to separate BHVE from CTRL
        n_bhve = sum(1 for g in groups if g == 'BHVE')
        if 0 < n_bhve < len(groups):
            ax.axhline(y=n_bhve - 0.5, color='red', linewidth=2, alpha=0.7)

        # Set time bin labels (x-axis) - show time in seconds
        n_ticks = min(10, n_bins)
        tick_indices = np.linspace(0, n_bins-1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        # Convert bin indices to time in seconds (assuming 30 fps)
        tick_times = [(idx * bin_width_frames) / 30 for idx in tick_indices]
        ax.set_xticklabels([f'{t:.0f}s' for t in tick_times])

        # Labels and title
        ax.set_xlabel(f'Time (bins = {bin_width_frames} frames = {bin_width_frames/30:.1f}s)', fontsize=12)
        ax.set_ylabel('Trial', fontsize=12)
        ax.set_title(f'{event_name} Events Over Time', fontsize=14, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.02)
        cbar.set_label('Number of Events per Bin', fontsize=10)

        # Adjust layout to fit full page
        plt.tight_layout()

        # Save plot as full-page PDF
        plot_filename = f'event_timeseries_{event_name.lower().replace(" ", "_")}{self.param_suffix}.pdf'
        fig.savefig(str(self.output_dir / plot_filename),
                   bbox_inches='tight',
                   orientation='portrait')
        plt.close(fig)

def trial_sort_key(index):
    def parse(name):
        name = str(name).upper()
        for pattern, is_control in [(r'DB(\d+)', False), (r'DC(\d+)', True),
                                   (r'BHVE.*?GROUP.*?(\d+)', False), (r'CTRL.*?GROUP.*?(\d+)', True)]:
            if m := re.search(pattern, name):
                return (int(m.group(1)), int(is_control))
        return (float('inf'), 2)
    return pd.Index([parse(name) for name in index])