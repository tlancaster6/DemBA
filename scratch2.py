from demba.extract_features import FeatureExtractor, process_video
from demba.estimate_pose import estimate_pose
from main import quivering_annotation_path

# config = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/config.yaml'
# video_path = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/testing/BHVE_group9_316800-318599.mp4'
# pose_h5_path = "/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/testing/BHVE_group9_316800-318599DLC_Resnet50_demasoni_singlenucSep11shuffle3_detector_best-130_snapshot_best-130.h5"
# config = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/config.yaml'
# estimate_pose(config, video_path, shuffle=3, n_fish=2, stop_before_stitching=True)
# process_video(video_path, None, pose_h5_path, True)

quivering_annotation_path = "/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/Annotations.xlsx"
video_path = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/videos/knl_demasoni_BHVE_group3cropped.mp4'
pose_h5_path = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/videos/knl_demasoni_BHVE_group3croppedDLC_Resnet50_demasoni_singlenucSep11shuffle3_detector_best-130_snapshot_best-130.h5'
process_video(video_path, quivering_annotation_path, pose_h5_path, False)