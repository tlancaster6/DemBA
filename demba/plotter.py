from pathlib import Path
import pandas as pd
import seaborn as sns
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
from math import ceil

class Plotter:

    def __init__(self, parent_dir):
        """
        parent_dir should be a directory containing two folders: Videos and Annotations. Annotations should hold a single
        file: Annotations.xlsx. Videos should contain a directory for each video/trial analyzed. Each video/trial
        directory should contain the clipfeatures and framefeatures csvs. Plots will be output into a new directory
        in the parent_dir called "Summary"
        """
        self.parent_dir = Path(parent_dir)
        self.annotation_path = self.parent_dir / 'Annotations' / 'Annotations.xlsx'
        self.output_dir = self.parent_dir / 'Summary'
        self.output_dir.mkdir(exist_ok=True)
        self.data_dir = self.parent_dir / 'Videos'
        self.bhve_dirs, self.ctrl_dirs = self.get_bhve_ctrl_dir_paths()
        self.concat_clipfeature_csvs()

    def concat_clipfeature_csvs(self):
        rows = []
        for subdir in self.bhve_dirs:
            clipfeature_path = subdir / f'{subdir.name}_clipfeatures.csv'
            row = pd.read_csv(str(clipfeature_path), index_col=0)
            row['split'] = 'bhve'
            rows.append(row)
        for subdir in self.ctrl_dirs:
            clipfeature_path = subdir / f'{subdir.name}_clipfeatures.csv'
            row = pd.read_csv(str(clipfeature_path), index_col=0)
            row['split'] = 'ctrl'
            rows.append(row)
        df = pd.concat(rows, axis=0)
        df.to_csv(str(self.output_dir / 'collated_clipfeatures.csv'))

    def get_bhve_ctrl_dir_paths(self):
        data_subdirs = list(self.data_dir.glob('*'))
        control_subdirs = [d for d in data_subdirs if ('CTRL' in d.name or 'DC' in d.name)]
        behave_subdirs = [d for d in data_subdirs if ('BHVE' in d.name or 'DB' in d.name)]
        if len(data_subdirs) != (len(control_subdirs) + len(behave_subdirs)):
            print('Warning: some directories not parsed into behave or control groups and will be omitted from analysis')
        return sorted(behave_subdirs), sorted(control_subdirs)

    def generate_clipfeature_plots(self):
        collated_clipfeature_csv = pd.read_csv(str(self.output_dir / 'collated_clipfeatures.csv'), index_col=0)
        target_stats = ['male_lead_quivering_fraction',
                     'male_circle_quivering_fraction',
                     'female_circle_quivering_fraction',
                     'n_mouthing_events',
                     'n_double_occupancy_events',
                     'n_spawning_events',
                     'mouthing_event_fraction',
                     'double_occupancy_event_fraction',
                     'spawning_event_fraction',
                     'nfish_frame_max',
                     'nfish_pipe_max',
                     'raw_zero_occupancy_fraction',
                     'raw_single_occupancy_fraction',
                     'raw_double_occupancy_fraction',
                     'raw_triple_occupancy_fraction']

        n_cols = 3
        n_rows = int(ceil(len(target_stats) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5, 10))
        flaxes = axes.flatten()
        for i, target in enumerate(target_stats):
            sns.violinplot(collated_clipfeature_csv, x=target, y='split', ax=flaxes[i])
            flaxes[i].set(title=target)
        fig.tight_layout()
        fig.savefig(str(self.output_dir / 'clipfeature_plots.pdf'))
        plt.close(fig)


par_dir = "/home/tlancaster/DLC/demasoni_singlenuc-Victor-2025-06-06/Analysis"
plotter = Plotter(par_dir)
plotter.generate_clipfeature_plots()
