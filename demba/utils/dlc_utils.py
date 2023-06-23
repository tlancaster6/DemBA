import pandas as pd
idx = pd.IndexSlice


def h5_to_df(h5_path):
    df = pd.read_hdf(h5_path)
    scorer = df.columns.get_level_values(0)[0]
    return df[scorer]

def calc_rmse_per_bp(path_to_error_csv):
    """the input csv is produced after running 'evaluate_network' in the 'evaluation-results' directory. It will be
    called 'dist_x.csv', where x is the number of training iterations"""
    df = pd.read_csv(path_to_error_csv, index_col=0, header=[1, 2, 3])
    df = df.loc[..., idx[:, :, 'rmse']].droplevel(2, axis=1)
    df.columns = df.columns.get_level_values(1)
    df = df.melt()
    return df.groupby('body_parts').agg(['count', 'mean'])


