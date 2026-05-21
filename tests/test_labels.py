"""Sanity tests for the triple-barrier labeler.

The labeler is the most error-prone piece (off-by-one, peek-ahead bugs,
same-bar tie handling). These tests pin down the contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.labels import BarrierConfig, triple_barrier_labels


def _series(prices):
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    s = pd.Series(prices, index=idx, dtype=float)
    return s, s, s  # close == high == low for simple cases


def test_lower_hit_labels_as_one():
    # Day 0 close = 100. Day 2 low = 94 < 95 (lower barrier).
    close, high, low = _series([100, 99, 94, 93, 92, 91, 90])
    cfg = BarrierConfig(lower_pct=0.05, upper_pct=0.05, horizon=5)
    out = triple_barrier_labels(close, high, low, cfg)
    assert out.iloc[0]["label"] == 1
    assert out.iloc[0]["outcome"] == "lower_hit"


def test_upper_hit_labels_as_zero():
    # Day 0 close = 100. Day 1 high = 106 > 105 (upper barrier).
    close, high, low = _series([100, 106, 108, 110, 112, 114, 116])
    cfg = BarrierConfig(lower_pct=0.05, upper_pct=0.05, horizon=5)
    out = triple_barrier_labels(close, high, low, cfg)
    assert out.iloc[0]["label"] == 0
    assert out.iloc[0]["outcome"] == "upper_hit"


def test_time_out_labels_as_zero():
    # Drifts gently inside both barriers.
    close, high, low = _series([100, 100.5, 101, 100, 99.5, 100, 100.5])
    cfg = BarrierConfig(lower_pct=0.05, upper_pct=0.05, horizon=5)
    out = triple_barrier_labels(close, high, low, cfg)
    assert out.iloc[0]["label"] == 0
    assert out.iloc[0]["outcome"] == "time_out"


def test_trailing_rows_dropped():
    close, high, low = _series([100] * 20)
    cfg = BarrierConfig(lower_pct=0.05, upper_pct=0.05, horizon=10)
    out = triple_barrier_labels(close, high, low, cfg)
    # Last `horizon` rows should be dropped (no future data to look at)
    assert len(out) == 20 - 10


def test_no_peek_at_entry_bar():
    # Day 0 itself has a low that would breach the lower barrier, but the
    # trade can't react until the next bar -- so entry-bar low must be
    # ignored. We construct a case where day 0's low = 90 (would trigger)
    # but day 1+ behave normally.
    close = pd.Series([100, 100, 100, 100, 100, 100],
                      index=pd.date_range("2020-01-01", periods=6, freq="B"),
                      dtype=float)
    high = close.copy()
    low = close.copy()
    low.iloc[0] = 80   # well below lower barrier, but on entry bar
    cfg = BarrierConfig(lower_pct=0.05, upper_pct=0.05, horizon=4)
    out = triple_barrier_labels(close, high, low, cfg)
    # Should NOT label as 1; entry-bar low is ignored.
    assert out.iloc[0]["outcome"] == "time_out"
    assert out.iloc[0]["label"] == 0


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        BarrierConfig(lower_pct=0, upper_pct=0.03).validate()
    with pytest.raises(ValueError):
        BarrierConfig(lower_pct=0.05, upper_pct=1.5).validate()
    with pytest.raises(ValueError):
        BarrierConfig(horizon=0).validate()


def test_simultaneous_touch_treated_as_stopout():
    # Day 1 high = 105 AND low = 95, both barriers touch on same bar.
    # Conservative rule: treat as upper_hit (stop fired first).
    close = pd.Series([100, 100, 100, 100, 100, 100],
                      index=pd.date_range("2020-01-01", periods=6, freq="B"),
                      dtype=float)
    high = close.copy()
    low = close.copy()
    high.iloc[1] = 106
    low.iloc[1] = 94
    cfg = BarrierConfig(lower_pct=0.05, upper_pct=0.05, horizon=4)
    out = triple_barrier_labels(close, high, low, cfg)
    assert out.iloc[0]["label"] == 0
    assert out.iloc[0]["outcome"] == "upper_hit"


def test_purged_walk_forward_no_overlap():
    """Independent sanity check on the CV splitter."""
    from src.splits import purged_walk_forward_splits
    splits = list(purged_walk_forward_splits(
        n_samples=1000, n_splits=4, purge=10, min_train_size=100
    ))
    assert len(splits) == 4
    for tr, te in splits:
        # No index appears in both
        assert len(set(tr) & set(te)) == 0
        # Purge gap is respected
        assert te.min() - tr.max() > 10
        # Walk-forward direction
        assert tr.max() < te.min()
