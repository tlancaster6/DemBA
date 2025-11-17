"""Statistical analysis and plotting for behavioral features."""

from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from math import ceil
import re
import numpy as np
from demba.utils.dlc import parse_trial_name

class Plotter:

    def __init__(self, project_manager, mouthing_dist_mm=None, min_likelihood=None, n_minutes=None):
        """
        Initialize Plotter for statistical analysis and plotting across all trials in a project.

        Parameters:
        - project_manager (ProjectManager): ProjectManager instance for the project. Used to discover
            trials and resolve paths.
        - mouthing_dist_mm: Mouthing distance threshold in mm (default: from config)
        - min_likelihood: Minimum likelihood threshold (default: from config)
        - n_minutes: Time restriction in minutes (default: from config)
        """
        from demba import config

        # Store ProjectManager
        self.project_manager = project_manager

        # Set up paths
        self.project_dir = project_manager.project_dir

        # Load defaults from config
        if mouthing_dist_mm is None:
            mouthing_dist_mm = config.DEFAULT_MOUTHING_DIST_MM
        if min_likelihood is None:
            min_likelihood = config.DEFAULT_MIN_LIKELIHOOD
        if n_minutes is None:
            n_minutes = config.DEFAULT_N_MINUTES

        # Create parameter suffix for unique file naming (same as FeatureExtractor)
        self.param_suffix = f"_mdist{mouthing_dist_mm}mm_likelihood{min_likelihood}"
        if n_minutes is not None:
            self.param_suffix += f"_last{n_minutes}min"

        self.output_dir = self.project_dir / 'Summary'
        self.output_dir.mkdir(exist_ok=True)
        self.data_dir = project_manager.videos_dir
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
        # Use ProjectManager to discover all trial directories
        all_trial_dirs = self.project_manager.list_trial_dirs()
        control_subdirs = [d for d in all_trial_dirs if ('CTRL' in d.name or 'DC' in d.name)]
        behave_subdirs = [d for d in all_trial_dirs if ('BHVE' in d.name or 'DB' in d.name)]
        if len(all_trial_dirs) != (len(control_subdirs) + len(behave_subdirs)):
            print('Warning: some directories not parsed into behave or control groups and will be omitted from analysis')
        return sorted(behave_subdirs), sorted(control_subdirs)

    def generate_clipfeature_boxplots(self):
        """Generate boxplots for clip-level features including sex-specific metrics."""
        # Aggregate metrics (combined or non-sex-specific)
        target_stats = [
             'n_double_occupancy_events',
             'n_spawning_events',
             'double_occupancy_event_fraction',
             'spawning_event_fraction',
             'male_roi_occupancy_fraction',
             'female_roi_occupancy_fraction',
        ]

        # Add sex-specific mouthing metrics if available
        if 'n_male_mouthing_events' in self.clipfeature_df.columns:
            target_stats.extend([
                'n_male_mouthing_events',
                'n_female_mouthing_events',
                'male_mouthing_event_fraction',
                'female_mouthing_event_fraction',
            ])

        # Add quivering metrics if available
        if 'n_male_circle_quivers' in self.clipfeature_df.columns:
            target_stats.extend([
                'n_male_circle_quivers',
                'n_female_circle_quivers',
                'male_circle_quivering_fraction',
                'female_circle_quivering_fraction',
            ])

        # Filter out any stats that don't exist in the dataframe
        available_stats = [stat for stat in target_stats if stat in self.clipfeature_df.columns]

        if not available_stats:
            print("Warning: No valid clip features found for boxplots")
            return

        n_cols = 3
        n_rows = int(ceil(len(available_stats) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5, 2.5 * n_rows))
        flaxes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []

        for i, target in enumerate(available_stats):
            sns.boxplot(self.clipfeature_df, x=target, hue='split', ax=flaxes[i])
            flaxes[i].legend(loc='lower right')
            flaxes[i].set_title(target.replace('_', ' ').title(), fontsize=9)

        # Hide any unused subplots
        for i in range(len(available_stats), len(flaxes)):
            flaxes[i].set_visible(False)

        fig.tight_layout()
        fig.savefig(str(self.output_dir / f'clipfeature{self.param_suffix}_boxplots.pdf'))
        plt.close(fig)

    def generate_auto_manual_correlation_plots(self):
        """Generate correlation plots between automated mouthing detection and manual quivering annotations."""
        if "n_male_circle_quivers" not in self.clipfeature_df.columns:
            print('    no manual annotation data found')
            return

        # Check for sex-specific mouthing events
        has_sex_specific_mouthing = 'n_male_mouthing_events' in self.clipfeature_df.columns

        if not has_sex_specific_mouthing:
            print('    Warning: Sex-specific mouthing events not found in clipfeatures')
            return

        # Calculate combined metrics for aggregate comparison
        n_mouthing_events = self.clipfeature_df.n_male_mouthing_events + self.clipfeature_df.n_female_mouthing_events
        mouthing_event_fraction = self.clipfeature_df.male_mouthing_event_fraction + self.clipfeature_df.female_mouthing_event_fraction
        n_circle_quivers = self.clipfeature_df.n_male_circle_quivers + self.clipfeature_df.n_female_circle_quivers
        circle_quiver_fraction = self.clipfeature_df.male_circle_quivering_fraction + self.clipfeature_df.female_circle_quivering_fraction

        # Create aggregate correlation plots
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.regplot(x=n_mouthing_events, y=n_circle_quivers, ax=axes[0])
        axes[0].set_xlabel('Total Mouthing Events (Male + Female)')
        axes[0].set_ylabel('Total Circle Quivers (Manual)')
        axes[0].set_title('Event Counts: Auto vs Manual')

        sns.regplot(x=mouthing_event_fraction, y=circle_quiver_fraction, ax=axes[1])
        axes[1].set_xlabel('Total Mouthing Fraction (Male + Female)')
        axes[1].set_ylabel('Total Circle Quiver Fraction (Manual)')
        axes[1].set_title('Event Fractions: Auto vs Manual')

        fig.tight_layout()
        fig.savefig(str(self.output_dir / f'auto_manual_correlation{self.param_suffix}_plots.pdf'))
        plt.close(fig)

    def generate_cross_sex_correlation_plots(self):
        """
        Generate correlation plots exploring cross-sex behavioral relationships.
        Specifically examines male mouthing vs female quivering and vice versa,
        which are biologically relevant courtship interactions.
        """
        # Check for required columns
        required_cols = ['n_male_mouthing_events', 'n_female_mouthing_events',
                        'n_male_circle_quivers', 'n_female_circle_quivers']
        if not all(col in self.clipfeature_df.columns for col in required_cols):
            print('    Warning: Missing required columns for cross-sex correlation plots')
            return

        # Create 2x2 grid of correlation plots
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Male mouthing vs Female quivering (primary courtship pattern)
        sns.regplot(x=self.clipfeature_df.n_male_mouthing_events,
                       y=self.clipfeature_df.n_female_circle_quivers,
                       ax=axes[0, 0])
        axes[0, 0].set_xlabel('Male Mouthing Events')
        axes[0, 0].set_ylabel('Female Circle Quivers')
        axes[0, 0].set_title('Male Mouthing → Female Quivering')

        # Female mouthing vs Male quivering
        sns.regplot(x=self.clipfeature_df.n_female_mouthing_events,
                       y=self.clipfeature_df.n_male_circle_quivers,
                       ax=axes[0, 1])
        axes[0, 1].set_xlabel('Female Mouthing Events')
        axes[0, 1].set_ylabel('Male Circle Quivers')
        axes[0, 1].set_title('Female Mouthing → Male Quivering')

        # Male mouthing vs Female mouthing
        sns.regplot(x=self.clipfeature_df.n_male_mouthing_events,
                       y=self.clipfeature_df.n_female_mouthing_events,
                       ax=axes[1, 0])
        axes[1, 0].set_xlabel('Male Mouthing Events')
        axes[1, 0].set_ylabel('Female Mouthing Events')
        axes[1, 0].set_title('Male vs Female Mouthing')

        # Male quivering vs Female quivering
        sns.regplot(x=self.clipfeature_df.n_male_circle_quivers,
                       y=self.clipfeature_df.n_female_circle_quivers,
                       ax=axes[1, 1])
        axes[1, 1].set_xlabel('Male Circle Quivers')
        axes[1, 1].set_ylabel('Female Circle Quivers')
        axes[1, 1].set_title('Male vs Female Quivering')

        fig.tight_layout()
        fig.savefig(str(self.output_dir / f'cross_sex_correlation{self.param_suffix}_plots.pdf'))
        plt.close(fig)

    def generate_event_timeseries_heatmaps(self, bin_width_frames=1800, sex_specific=False):
        """
        Generate heatmaps showing event occurrence over time.
        Creates separate plots (one per event type) with trials as rows.

        Parameters:
        - bin_width_frames: Width of each time bin in frames (default: 1800, which is 60 seconds at 30fps)
        - sex_specific: If True, generate separate heatmaps for male and female mouthing events.
                       If False, combine male and female mouthing into a single combined heatmap.
        """

        if sex_specific:
            # Sex-specific event columns
            event_columns = ['male_mouthing_event_id', 'female_mouthing_event_id',
                           'spawning_event_id', 'double_occupancy_event_id']
            event_names = ['Male Mouthing', 'Female Mouthing', 'Spawning', 'Double Occupancy']
        else:
            # Combined mouthing events (backward compatible)
            event_columns = ['combined_mouthing_event_id', 'spawning_event_id', 'double_occupancy_event_id']
            event_names = ['Mouthing (Combined)', 'Spawning', 'Double Occupancy']

        # Collect data from all trials
        trial_data = self._collect_trial_timeseries_data(event_columns, bin_width_frames, sex_specific)

        if not trial_data:
            print("Warning: No valid trial data found for heatmaps")
            return

        # Generate one plot per event type
        for event_idx, (event_col, event_name) in enumerate(zip(event_columns, event_names)):
            self._generate_single_metric_heatmap(trial_data, event_idx, event_name, bin_width_frames, sex_specific)

    def _collect_trial_timeseries_data(self, event_columns, bin_width_frames, sex_specific=False):
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

            # If not sex_specific, create combined_mouthing_event_id from male and female
            if not sex_specific and 'combined_mouthing_event_id' not in framefeatures.columns:
                if 'male_mouthing_event_id' in framefeatures.columns and 'female_mouthing_event_id' in framefeatures.columns:
                    # Combine male and female mouthing events
                    # Use male events where they exist, female where male doesn't exist
                    framefeatures['combined_mouthing_event_id'] = framefeatures['male_mouthing_event_id'].copy()
                    # For frames where male has no event but female does, use female event
                    female_only_mask = (framefeatures['male_mouthing_event_id'] == -1) & (framefeatures['female_mouthing_event_id'] >= 0)
                    framefeatures.loc[female_only_mask, 'combined_mouthing_event_id'] = framefeatures.loc[female_only_mask, 'female_mouthing_event_id']

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

    def _generate_single_metric_heatmap(self, trial_data, event_idx, event_name, bin_width_frames, sex_specific=False):
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
        suffix = '_sex_specific' if sex_specific else ''
        plot_filename = f'event_timeseries_{event_name.lower().replace(" ", "_")}{self.param_suffix}{suffix}.pdf'
        fig.savefig(str(self.output_dir / plot_filename),
                   bbox_inches='tight',
                   orientation='portrait')
        plt.close(fig)

def trial_sort_key(index):
    """Sort trial names by group number and control/behavior status."""
    return pd.Index([parse_trial_name(name) for name in index])