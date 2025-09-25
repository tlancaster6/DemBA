from demba.extract_features import FeatureExtractor, process_video, process_all
from demba.estimate_pose import estimate_pose
from demba.plotter import Plotter
from demba.plotters_old import plot_all_timeseries

# config = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/config.yaml'
# video_path = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/testing/BHVE_group9_316800-318599.mp4'
# pose_h5_path = "/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/testing/BHVE_group9_316800-318599DLC_Resnet50_demasoni_singlenucSep11shuffle3_detector_best-130_snapshot_best-130.h5"
# config = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/config.yaml'
# estimate_pose(config, video_path, shuffle=3, n_fish=2, stop_before_stitching=True)
# process_video(video_path, None, pose_h5_path, True)

# quivering_annotation_path = "/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/Annotations.xlsx"
# video_path = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/videos/knl_demasoni_BHVE_group3cropped.mp4'
# pose_h5_path = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/videos/knl_demasoni_BHVE_group3croppedDLC_Resnet50_demasoni_singlenucSep11shuffle3_detector_best-130_snapshot_best-130.h5'
# process_video(video_path, quivering_annotation_path, pose_h5_path, False)
# fe = FeatureExtractor(video_path, quivering_annotation_path, pose_h5_path)
# fe.extract_all_features()

parent_dir = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/Analysis/'
quivering_annotation_path = "/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/Analysis/Annotations/Annotations.xlsx"
process_all(parent_dir, quivering_annotation_path, visualize=True, n_minutes=None)
# plotter = Plotter(parent_dir, n_minutes=None)
# plotter.generate_clipfeature_boxplots()
# plotter.generate_event_timeseries_heatmaps()

# process_all(parent_dir, None, visualize=True)
# vid_path = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/testing/BHVE_group9_316800-318599.mp4'
# h5_path = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/testing/BHVE_group9_316800-318599DLC_Resnet50_demasoni_singlenucSep11shuffle3_detector_best-130_snapshot_best-130.h5'
# fe = FeatureExtractor(vid_path, None, h5_path)
# fe.extract_all_features()
# fe.visualize_features(full_vis=True)

# vid_path = 'projects/demasoni_singlenuc-tucker-2025-09-11/Analysis/Videos/bgrb_t007_10.21.24_DC13cropped/bgrb_t007_10.21.24_DC13cropped.mp4'
# h5_path = 'projects/demasoni_singlenuc-tucker-2025-09-11/Analysis/Videos/bgrb_t007_10.21.24_DC13cropped/bgrb_t007_10.21.24_DC13croppedDLC_Resnet50_demasoni_singlenucSep11shuffle3_detector_best-130_snapshot_best-130.h5'
# fe = FeatureExtractor(vid_path, None, h5_path)
# fe.extract_all_features()
# fe.visualize_features()