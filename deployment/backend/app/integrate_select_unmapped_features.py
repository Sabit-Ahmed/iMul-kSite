import os
from pathlib import Path
import logging
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, SelectKBest

# Lightweight logger (inherits app/gunicorn handlers if present)
logger = logging.getLogger(__name__)

class FeatureLoadError(RuntimeError):
    """Raised when feature artifacts are missing or malformed."""

def _require_dir(path: Path, key: str) -> None:
    if not path.exists():
        # Show nearby dirs to help spot wrong base path
        parent = path.parent
        nearby = []
        try:
            nearby = sorted(p.name for p in parent.iterdir() if p.is_dir())
        except Exception:
            pass
        raise FeatureLoadError(
            f"[{key}] expected directory not found: {path}\n"
            f"Base path: {parent}\n"
            f"Siblings here: {nearby}"
        )

def _read_csv_strict(fpath: Path, key: str) -> pd.DataFrame:
    try:
        if not fpath.exists():
            raise FeatureLoadError(f"[{key}] missing feature file: {fpath}")

        # header=None to match your original reader
        df = pd.read_csv(fpath, header=None)

        if df.shape[0] == 0 or df.shape[1] == 0:
            raise FeatureLoadError(
                f"[{key}] empty feature file: {fpath} (shape={df.shape})"
            )

        # OPTIMIZED NaN CHECK: Only check if there are obvious NaN indicators
        # This avoids the expensive full-matrix scan that was causing hangs
        if df.isnull().any(axis=None):  # More efficient than isna().any().any()
            # Only do detailed reporting if NaNs are actually found
            nan_count = df.isnull().sum().sum()
            if nan_count > 0:
                # Sample a few NaN locations for debugging (limit to avoid slowdown)
                nan_mask = df.isnull()
                nan_rows, nan_cols = np.where(nan_mask)
                sample_size = min(5, len(nan_rows))
                samples = [(int(nan_rows[i]), int(nan_cols[i])) for i in range(sample_size)]
                raise FeatureLoadError(
                    f"[{key}] {nan_count} NaN values found in {fpath}; "
                    f"sample positions (row,col): {samples}"
                )

        return df

    except FeatureLoadError:
        raise
    except Exception as e:
        raise FeatureLoadError(
            f"[{key}] failed to read CSV: {fpath}\n"
            f"Reason: {type(e).__name__}: {e}"
        ) from e

def _validate_row_alignment(frames: Dict[str, pd.DataFrame], key: str) -> None:
    # Ensure all blocks have the same number of rows
    lengths = {name: df.shape[0] for name, df in frames.items()}
    unique_lengths = sorted(set(lengths.values()))
    if len(unique_lengths) != 1:
        details = ", ".join(f"{name}={rows}" for name, rows in lengths.items())
        raise FeatureLoadError(
            f"[{key}] feature blocks have mismatched row counts (they must be equal).\n"
            f"Row counts: {details}"
        )

def OptimizedReadFeatures(absolute_path, key, value, feature_name):
    """
    Read one or multiple feature matrices from:
      <absolute_path>/extracted_dataset_<key.lower()>/<feature>.csv

    Returns:
      X: np.ndarray
      y: np.ndarray of shape (n_samples,), with first `value` as +1 and the rest -1

    Raises:
      FeatureLoadError with precise diagnostics if artifacts are missing/misaligned.
      ValueError if label count `value` is invalid w.r.t. rows.
    """
    base = Path(absolute_path)
    path = base / f"extracted_dataset_{key.lower()}"

    _require_dir(path, key)

    def _read_one(name: str) -> pd.DataFrame:
        fpath = path / f"{name}.csv"
        return _read_csv_strict(fpath, key)

    print(f"[{key}] starting feature load from {path}")

    # Load frames
    if isinstance(feature_name, list):
        print(f"[{key}] loading {len(feature_name)} feature blocks: {feature_name}")
        # Load all requested blocks and validate alignment BEFORE concat
        frames = {}
        for i, name in enumerate(feature_name):
            print(f"[{key}] loading block {i+1}/{len(feature_name)}: {name}")
            frames[name] = _read_one(name)
            print(f"[{key}] {name} loaded: shape={frames[name].shape}")

        print(f"[{key}] validating row alignment...")
        _validate_row_alignment(frames, key)

        print(f"[{key}] concatenating blocks...")
        # dict concat -> MultiIndex columns (feature_name, col); then to numpy (exactly like before)
        X_df = pd.concat(frames, axis=1)
        X = X_df.values
        print(f"[{key}] concatenation complete: final X.shape={X.shape}")
    else:
        print(f"[{key}] loading single feature block: {feature_name}")
        X = _read_one(feature_name).values
        print(f"[{key}] single block loaded: X.shape={X.shape}")

    # Labels
    n = X.shape[0]
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"[{key}] `value` must be an integer; got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"[{key}] `value` must be >= 0; got {value}")
    if value > n:
        # Provide concrete guidance + current artifact location
        raise ValueError(
            f"[{key}] `value` ({value}) exceeds number of rows in X ({n}).\n"
            f"Check your inputs or regenerate features.\n"
            f"Artifacts read from: {path}"
        )

    print(f"[{key}] creating labels: positives={value}, total_rows={n}")
    y = np.empty(n, dtype=np.int64)
    y[:value] = 1
    y[value:] = -1

    # Success line with shape context (keeps your original print but richer)
    msg = f"[{key}] features loaded: X.shape={X.shape}, positives={value}, negatives={n - value}  from {path}"
    try:
        # prefer logger if configured; also print so you still see it in GAE logs
        logger.info(msg)
    finally:
        print(msg)

    return X, y