from demba.estimate_pose import estimate_pose

config = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/config.yaml'
video_path = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/testing/BHVE_group9_316800-318599.mp4'
estimate_pose(config_path=config, video_path=video_path, shuffle=1, n_fish=2, debug_visualize=True, transreid=False)
