import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, SelectKBest


def OptimizedReadFeatures(absolute_path, key, value, feature_name):
    """
    Read one or multiple feature matrices from:
      <absolute_path>/extracted_dataset_<key.lower()>/<feature>.csv
    Returns:
      X: np.ndarray
      y: np.ndarray of shape (n_samples,), with first `value` as +1 and the rest -1
    """
    base = Path(absolute_path)
    path = base / f"extracted_dataset_{key.lower()}"

    if not path.exists():
        raise FileNotFoundError(f"Expected directory not found: {path}")

    def _read_one(name: str) -> pd.DataFrame:
        fpath = path / f"{name}.csv"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing feature file: {fpath}")
        return pd.read_csv(fpath, header=None)

    if isinstance(feature_name, list):
        # dict concat gives a MultiIndex over columns like (feature_name, col)
        frames = {name: _read_one(name) for name in feature_name}
        X_df = pd.concat(frames, axis=1)
        X = X_df.values
    else:
        X = _read_one(feature_name).values

    # Build labels: first `value` positives (+1), remaining negatives (-1)
    n = X.shape[0]
    if value > n:
        raise ValueError(
            f"`value` ({value}) cannot exceed number of rows in X ({n}). "
            f"Check your inputs for key={key}."
        )
    y = np.empty(n, dtype=np.int64)
    y[:value] = 1
    y[value:] = -1

    print(f"Features reading done for {key} from {path}")
    return X, y


def int_and_sel(K, X, y):
    """
    Feature selection with ANOVA F-test.
    K can be an int or 'all'.
    Returns:
      features: np.ndarray with selected columns
      cols: np.ndarray of selected column indices
    """
    sel = SelectKBest(score_func=f_classif, k=K)
    sel.fit(X, y)
    cols = sel.get_support(indices=True)
    features = X[:, cols]
    return features, cols
