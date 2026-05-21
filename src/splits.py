"""Purged walk-forward cross-validation.

Standard k-fold leaks in time-series. Even a single shuffle would let the
model peek at the future. We need two things:

1. **Walk-forward**: train on the past, test on the future. Multiple folds
   walk the split forward in time.

2. **Purge gap**: triple-barrier labels at time t depend on prices up to
   t + horizon. If the test fold starts at t and the train fold ends at
   t - 1, the last `horizon` training labels were computed from prices
   that overlap the test window. We drop those training rows.

Reference: López de Prado, "Advances in Financial Machine Learning",
Chapter 7.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def purged_walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    purge: int = 10,
    min_train_size: int = 250,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) arrays for a purged walk-forward CV.

    The test windows are equal-sized, contiguous, and disjoint, advancing
    through the sample. The training window is everything before the test
    window, minus the last `purge` rows (to prevent label-overlap leakage).

    Parameters
    ----------
    n_samples
        Total rows in the labeled feature matrix.
    n_splits
        Number of (train, test) folds. The sample is divided into
        `n_splits + 1` chunks; the first chunk is initial training data,
        the remaining `n_splits` are successive test folds.
    purge
        Number of training rows to drop from the end of each training
        window. Set equal to the label horizon.
    min_train_size
        Skip folds whose training set ends up smaller than this. Useful
        when `n_splits` is large relative to `n_samples`.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if purge < 0:
        raise ValueError("purge must be >= 0")

    fold_size = n_samples // (n_splits + 1)
    if fold_size < 1:
        raise ValueError(
            f"n_samples={n_samples} too small for n_splits={n_splits}"
        )

    for k in range(n_splits):
        test_start = (k + 1) * fold_size
        test_end = test_start + fold_size if k < n_splits - 1 else n_samples
        train_end = max(0, test_start - purge)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) < min_train_size or len(test_idx) == 0:
            continue
        yield train_idx, test_idx
