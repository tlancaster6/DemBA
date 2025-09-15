import sys
import logging
from demba.estimate_pose import estimate_pose
from pathlib import Path

class TeeLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()



# config = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/config.yaml'
# video_path = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/testing/BHVE_group9_316800-318599.mp4'
# estimate_pose(config_path=config, video_path=video_path, shuffle=3, n_fish=2, debug_visualize=True, transreid=True)

sys.stdout = TeeLogger('scratch_output.log')
config = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/config.yaml'
video_dir = Path('/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/videos')
video_paths = video_dir.glob('*cropped.mp4')
for vp in video_paths:
    try:
        estimate_pose(config_path=config, video_path=vp, shuffle=3, n_fish=2, debug_visualize=False, transreid=True)
    except Exception as e:
        print('-'*30)
        print(f'EXCEPTION ENCOUNTERED: {e}')
        print('-'*30)

sys.stdout.close()
sys.stdout = sys.stdout.terminal
