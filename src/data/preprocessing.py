"""Data cleaning and train/validation splitting."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def train_validation_split(
    X: np.ndarray, y: np.ndarray, val_fraction: float = 0.3, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random, reproducible split. Returns X_tr, y_tr, X_va, y_va, idx_tr, idx_va.

    The returned index arrays let callers split other aligned arrays (e.g. the
    noise-free values) identically. The two sets never overlap.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if not (0.05 <= val_fraction <= 0.5):
        raise ValueError("val_fraction must be between 0.05 and 0.5")
    n = X.shape[0]
    n_val = int(round(n * val_fraction))
    if n_val < 5 or n - n_val < 10:
        raise ValueError("Too few observations for this validation fraction")
    perm = np.random.default_rng(seed).permutation(n)
    idx_va, idx_tr = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    return X[idx_tr], y[idx_tr], X[idx_va], y[idx_va], idx_tr, idx_va


def drop_invalid_rows(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Remove rows containing NaN/inf. Returns cleaned arrays and the number dropped."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    return X[ok], y[ok], int((~ok).sum())


# --------------------------------------------------------------------------- user-supplied tables
MIN_ROWS = 30
MAX_ROWS = 50_000


def clean_column_names(columns) -> list:
    """Make header text usable as variable names: 'Length (m)' -> 'Length_m'. Names stay unique."""
    import re

    out, used = [], set()
    for i, c in enumerate(columns):
        name = re.sub(r"[^0-9A-Za-z_]+", "_", str(c).strip()).strip("_") or f"col{i + 1}"
        if name[0].isdigit():
            name = f"v_{name}"
        base, n = name, 2
        while name in used:
            name, n = f"{base}_{n}", n + 1
        used.add(name)
        out.append(name)
    return out


def prepare_numeric_dataframe(df, min_rows: int = MIN_ROWS, max_rows: int = MAX_ROWS):
    """Keep usable numeric columns, drop invalid rows, rename headers to valid names.

    Returns (clean_df, report). Raises ValueError with a clear message if the table is unusable.
    The report lists dropped rows, ignored text columns, constant columns and renamed headers.
    """
    import pandas as pd

    if df is None or df.empty:
        raise ValueError("The dataset is empty.")
    if len(df) > max_rows:
        raise ValueError(f"The dataset has {len(df)} rows; the demo limit is {max_rows}.")
    rows_in = len(df)
    original = [str(c) for c in df.columns]
    kept, ignored = {}, []
    for col, new in zip(df.columns, clean_column_names(original)):
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            ignored.append(str(col))
            continue
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce")
            if s.notna().mean() < 0.9:
                ignored.append(str(col))
                continue
        kept[new] = s.astype(float).replace([np.inf, -np.inf], np.nan)
    if len(kept) < 2:
        raise ValueError("At least two numeric columns are required. "
                         f"Found {len(kept)} (ignored non-numeric: {ignored or 'none'}).")
    clean = pd.DataFrame(kept).dropna().reset_index(drop=True)
    if len(clean) < min_rows:
        raise ValueError(f"Only {len(clean)} complete rows remain after removing missing or invalid "
                         f"values; at least {min_rows} are required.")
    constant = [c for c in clean.columns if clean[c].nunique() < 2]
    clean = clean.drop(columns=constant)
    if clean.shape[1] < 2:
        raise ValueError("After removing constant columns fewer than two columns remain.")
    renamed = {o: n for o, n in zip(original, clean_column_names(original)) if o != n and n in clean.columns}
    report = {"rows_in": rows_in, "rows_out": len(clean), "dropped_rows": rows_in - len(clean),
              "ignored_columns": ignored, "constant_columns": constant, "renamed_columns": renamed}
    return clean, report
