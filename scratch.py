from demba.estimate_pose import estimate_pose

config = '/home/tlancaster/DLC/demasoni_singlenuc-Victor-2025-06-06/config.yaml'
video_path = '/home/tlancaster/DLC/demasoni_singlenuc-Victor-2025-06-06/testing/iteration1/BHVE_group9_316800-318599.mp4'
estimate_pose(config_path=config, video_path=video_path, shuffle=2, n_fish=2, debug_visualize=True, track_method='box')
