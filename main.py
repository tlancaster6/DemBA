from pathlib import Path
from demba.extract_features import FeatureExtractor
from demba.estimate_pose import estimate_pose


def run_analysis(analysis_targets, dlc_config, quivering_annotations, run_pose_estimation=True, featurize=True):
    for target in analysis_targets:
        video_path = analysis_dir_path / 'Videos' / target / f'{target}.mp4'
        if not video_path.exists():
            print(f'File Not Found: {video_path}. Skipping')
            continue
        if run_pose_estimation:
            print(f'running pose estimation on {video_path.name}')
            estimate_pose(dlc_config, video_path, visualize=True)
        if featurize:
            fe = FeatureExtractor(video_path, quivering_annotations)
            fe.extract_all_features()


dlc_config_path = Path('/home/tlancaster/DLC/demasoni_singlenuc/config.yaml')
analysis_dir_path = dlc_config_path.parent / 'Analysis'
quivering_annotation_path = analysis_dir_path / 'Annotations' / 'Mbuna_behavior_annotations.xlsx'
analysis_target_list = ['BHVE_group1']
run_analysis(analysis_target_list, dlc_config_path, quivering_annotation_path, False, True)
