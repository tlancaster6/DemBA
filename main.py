import os
from demba.estimate_pose import analyze_video



project = 'demasoni_singlenuc'
prefix = 'C:\\Users\\tucke\\DLC_Projects'
project_path = os.path.join(prefix, project)
config = os.path.join(project_path, 'config.yaml')
shuffle = 1

video = os.path.join(project_path, 'full_videos\\BHVE_group3_2022-12-21T07_cropped.mp4')
# video = os.path.join(project_path, 'testclip\\testclip.mp4')
analyze_video(config, video)