import pandas as pd


def h5_to_df(h5_path):
    df = pd.read_hdf(h5_path)
    scorer = df.columns.get_level_values(0)[0]
    return df[scorer]


