import pandas as pd
import numpy as np
import os

L = 60
STRIDE = 15
NON_FEATURE_COLS = ['Class', 'PathOrder', 'block_id', 'split']


def make_windows(group_df, feature_cols, L, stride):
    values = group_df[feature_cols].values
    n_rows = len(values)
    windows = []
    for start in range(0, n_rows - L + 1, stride):
        windows.append(values[start:start + L])
    return np.array(windows)


def process_dataset(csv_path, out_dir, L=60, stride=15):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    os.makedirs(out_dir, exist_ok=True)

    # Εδώ μας ενδιαφέρουν πραγματικά μόνο τα meta_train/test.
    # Κρατάμε και τα convtran_train/val μόνο για δομική συνέπεια με τα per-group datasets.
    for split_name in ['convtran_train', 'convtran_val', 'meta_train', 'test']:
        split_df = df[df['split'] == split_name]
        X_list, y_list = [], []
        for block_id, block_df in split_df.groupby('block_id'):
            windows = make_windows(block_df, feature_cols, L, stride)
            if len(windows) == 0:
                continue
            X_list.append(windows)
            y_list.append(np.full(len(windows), block_df['Class'].iloc[0]))

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        out_path = os.path.join(out_dir, f'{split_name}.npz')
        np.savez(out_path, X=X, y=y)
        print(f'{out_path}: X={X.shape}, y={y.shape}, class counts={dict(zip(*np.unique(y, return_counts=True)))}')


process_dataset('data/cleaned_full_split.csv', out_dir='data/windows/full', L=L, stride=STRIDE)