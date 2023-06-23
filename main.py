import os
from demba.extract_features_v3 import FeatureExtractor



project = 'demasoni_singlenuc'
prefix = 'C:\\Users\\tucke\\DLC_Projects'
project_path = os.path.join(prefix, project)
config = os.path.join(project_path, 'config.yaml')
shuffle = 1

video = os.path.join(project_path, 'testclip\\testclip.mp4')
featurizer = FeatureExtractor(config, video)