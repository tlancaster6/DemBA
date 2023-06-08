from demba.utils.dlc_utils import h5_to_df
from demba.utils.roi_utils import estimate_roi

h5_path = r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\videos\BHVE_group2_sampleDLC_dlcrnetms5_demasoni_singlenucMay23shuffle1_50000_el_filtered.h5"
reference_img_path = r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\labeled-data\BHVE_group2_sample\img00533.png"

roi = estimate_roi(reference_img_path)

df = h5_to_df(h5_path)
individuals = list(df.columns.get_level_values(0).unique())
body_parts = list(df.columns.get_level_values(1).unique())


class FeatureExtractor:

    def __init__(self, config_path, video_path):
        self.config_path, self.video_path = config_path, video_path


