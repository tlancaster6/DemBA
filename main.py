from pathlib import Path
from demba.extract_features import FeatureExtractor
from demba.estimate_pose import estimate_pose


def run_analysis(analysis_targets, dlc_config, quivering_annotations, run_pose_estimation=True, skip_tracking=False, featurize=True):
    for target in analysis_targets:
        video_path = analysis_dir_path / 'Videos' / target / f'{target}.mp4'
        if not video_path.exists():
            print(f'File Not Found: {video_path}. Skipping')
            continue
        if run_pose_estimation:
            print(f'running pose estimation on {video_path.name}')
            estimate_pose(dlc_config, video_path, skip_tracking=skip_tracking, visualize=True)
        if featurize:
            fe = FeatureExtractor(video_path, quivering_annotations)
            fe.extract_all_features()


dlc_config_path = Path('/home/tlancaster/DLC/demasoni_singlenuc/config.yaml')
analysis_dir_path = dlc_config_path.parent / 'Analysis'
quivering_annotation_path = analysis_dir_path / 'Annotations' / 'Annotations.xlsx'
analysis_target_list = [
    'knl_demasoni_BHVE_group1',
    'bgrb_t001_9.25.24_DC11',
    'bgrb_t011_10.21.24_DB13',
    'bgrb_t007_10.21.24_DC13',
    'bgrb_t001_10.21.24_DB12',
    'knl_demasoni_BHVE_group3',
    'knl_demasoni_CTRL_group5',
    'bgrb_t007_9.25.24_DB11',
    'knl_demasoni_BHVE_group8',
    'bgrb_t007_9.20.24_DC10',
    'bgrb_t011_10.29.24_DC14',
    'knl_demasoni_CTRL_group3',
    'knl_demasoni_CTRL_group8',
    'knl_demasoni_BHVE_group5',
    'bgrb_t001_9.20.24_DB10',
    'bgrb_t013_10.21.24_DC12',
    'knl_demasoni_CTRL_group1',
    'bgrb_t015_10.29.24_DB14']

run_analysis(analysis_target_list, dlc_config_path, quivering_annotation_path, True, True, False)
