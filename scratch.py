from demba.estimate_pose import estimate_pose
from pathlib import Path
# config = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/config.yaml'
# video_path = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/testing/BHVE_group9_316800-318599.mp4'
# estimate_pose(config_path=config, video_path=video_path, shuffle=3, n_fish=2, debug_visualize=True, transreid=True)

config = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/config.yaml'
video_dir = Path('/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/videos')
video_paths = video_dir.glob('*cropped.mp4')
for vp in video_paths:
    estimate_pose(config_path=config, video_path=vp, shuffle=3, n_fish=2, debug_visualize=False, transreid=True)

