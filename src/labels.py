"""Triple-barrier labeling for short-signal classification.

Given a price series, for each day t we ask:
  Within the next `horizon` trading days, does the close fall to
  `(1 - lower_pct) * close[t]` BEFORE it rises to `(1 + upper_pct) * close[t]`?

If yes, label = 1 (a short opened at close[t] would have been profitable
before being stopped out or hitting the time limit).
If no -- either upper barrier hit first, or neither barrier hit within the
horizon -- label = 0.

This framing is asymmetric on purpose: shorts have unbounded loss potential
and brokers force covers on adverse moves, so a label that ignores that
mis-states the risk.

Reference: López de Prado, "Advances in Financial Machine Learning" (2018),
Chapter 3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BarrierConfig:
    lower_pct: float = 0.05   # short profit target (price falls by this much)
    upper_pct: float = 0.03   # stop-out (price rises by this much)
    horizon: int = 10         # max trading days to wait

    def validate(self) -> None:
        if not (0 < self.lower_pct < 1):
            raise ValueError("lower_pct must be in (0, 1)")
        if not (0 < self.upper_pct < 1):
            raise ValueError("upper_pct must be in (0, 1)")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")


def triple_barrier_labels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    cfg: BarrierConfig,
) -> pd.DataFrame:
    """Compute triple-barrier labels for a short trade.

    Uses intraday high/low to detect barrier touches (more realistic than
    close-only, since real stops fire intraday).

    Returns a DataFrame with columns:
        label       -- 1 if lower barrier hit first, else 0
        outcome     -- one of {"lower_hit", "upper_hit", "time_out"}
        days_to_exit -- trading days until the exit event (NaN if not enough
                        future data)

    The last `horizon` rows are dropped because their labels would require
    data beyond the end of the series.
    """
    cfg.validate()
    n = len(close)
    if not (len(high) == len(low) == n):
        raise ValueError("close, high, low must have equal length")

    close_arr = close.to_numpy(dtype=float)
    high_arr = high.to_numpy(dtype=float)
    low_arr = low.to_numpy(dtype=float)

    labels = np.zeros(n, dtype=np.int8)
    outcomes = np.empty(n, dtype=object)
    days_to_exit = np.full(n, np.nan)

    for i in range(n - cfg.horizon):
        entry = close_arr[i]
        lower_barrier = entry * (1 - cfg.lower_pct)
        upper_barrier = entry * (1 + cfg.upper_pct)

        # Look at days i+1 .. i+horizon (we enter at close of day i,
        # the trade can only react starting next day).
        window_high = high_arr[i + 1 : i + 1 + cfg.horizon]
        window_low = low_arr[i + 1 : i + 1 + cfg.horizon]

        lower_hits = np.where(window_low <= lower_barrier)[0]
        upper_hits = np.where(window_high >= upper_barrier)[0]

        first_lower = lower_hits[0] if len(lower_hits) else np.inf
        first_upper = upper_hits[0] if len(upper_hits) else np.inf

        if first_lower < first_upper:
            labels[i] = 1
            outcomes[i] = "lower_hit"
            days_to_exit[i] = first_lower + 1
        elif first_upper < first_lower:
            labels[i] = 0
            outcomes[i] = "upper_hit"
            days_to_exit[i] = first_upper + 1
        elif first_lower == first_upper and np.isfinite(first_lower):
            # Same-day touch of both barriers -- ambiguous, treat as stop-out
            # (conservative: in reality you'd likely have been stopped by the
            # rise before the fall on the same bar).
            labels[i] = 0
            outcomes[i] = "upper_hit"
            days_to_exit[i] = first_upper + 1
        else:
            labels[i] = 0
            outcomes[i] = "time_out"
            days_to_exit[i] = cfg.horizon

    # Mark the trailing rows as unusable
    for i in range(n - cfg.horizon, n):
        outcomes[i] = None

    out = pd.DataFrame(
        {
            "label": labels,
            "outcome": outcomes,
            "days_to_exit": days_to_exit,
        },
        index=close.index,
    )
    # Drop rows we can't label
    out = out.iloc[: n - cfg.horizon].copy()
    out["label"] = out["label"].astype(np.int8)
    return out
