"""Simple backtest of short signals.

Trading model (intentionally simple):
  - When proba >= threshold, "open a short" at next day's open.
  - Exit using the same barriers used for labeling, or at horizon.
  - Position size is fixed (1 unit of capital per trade).
  - Trades are non-overlapping per ticker: if a signal fires while a
    previous trade is still open, it is skipped.
  - No financing cost, no borrow fee, no slippage. **A real short would
    pay all of those** -- this is a research backtest, not a P&L claim.

Returned per-trade record:
  entry_date, entry_price, exit_date, exit_price, outcome, return_pct
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .labels import BarrierConfig


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    outcome: str
    return_pct: float  # positive = profitable short


def backtest_signals(
    prices: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    cfg: BarrierConfig,
) -> pd.DataFrame:
    """Simulate short trades for the given signal dates.

    `signal_dates` are dates on which the model emitted a positive signal
    (probability >= threshold). The trade enters at the NEXT day's open
    (avoiding any same-bar peek), and exits when either barrier is hit
    intraday or after `horizon` days.
    """
    px = prices.copy()
    px["next_open"] = px["Open"].shift(-1)
    px["next_date"] = px.index.to_series().shift(-1)

    trades: List[Trade] = []
    blocked_until: pd.Timestamp | None = None
    signal_set = set(signal_dates)

    for date in prices.index:
        if date not in signal_set:
            continue
        if blocked_until is not None and date <= blocked_until:
            continue

        entry_open = px.at[date, "next_open"]
        entry_date = px.at[date, "next_date"]
        if pd.isna(entry_open) or pd.isna(entry_date):
            continue  # no next day available

        lower_barrier = entry_open * (1 - cfg.lower_pct)
        upper_barrier = entry_open * (1 + cfg.upper_pct)

        # Look at the `horizon` bars starting from the entry day.
        entry_pos = prices.index.get_loc(entry_date)
        window = prices.iloc[entry_pos : entry_pos + cfg.horizon]

        exit_price = None
        exit_date = None
        outcome = "time_out"
        for ts, row in window.iterrows():
            hit_lower = row["Low"] <= lower_barrier
            hit_upper = row["High"] >= upper_barrier
            if hit_lower and hit_upper:
                # Ambiguous bar; assume stop fired first (conservative).
                exit_price = upper_barrier
                exit_date = ts
                outcome = "upper_hit"
                break
            if hit_lower:
                exit_price = lower_barrier
                exit_date = ts
                outcome = "lower_hit"
                break
            if hit_upper:
                exit_price = upper_barrier
                exit_date = ts
                outcome = "upper_hit"
                break

        if exit_price is None:
            exit_price = window["Close"].iloc[-1]
            exit_date = window.index[-1]
            outcome = "time_out"

        # Short P&L: profit when exit < entry
        ret_pct = (entry_open - exit_price) / entry_open
        trades.append(Trade(
            entry_date=entry_date,
            entry_price=float(entry_open),
            exit_date=exit_date,
            exit_price=float(exit_price),
            outcome=outcome,
            return_pct=float(ret_pct),
        ))
        blocked_until = exit_date

    if not trades:
        return pd.DataFrame(columns=[
            "entry_date", "entry_price", "exit_date", "exit_price",
            "outcome", "return_pct",
        ])
    return pd.DataFrame([t.__dict__ for t in trades])


def trade_stats(trades: pd.DataFrame) -> dict:
    """Summary statistics for a backtest."""
    if trades.empty:
        return {"n_trades": 0}
    rets = trades["return_pct"]
    return {
        "n_trades": int(len(trades)),
        "hit_rate": float((rets > 0).mean()),
        "mean_return": float(rets.mean()),
        "median_return": float(rets.median()),
        "worst_return": float(rets.min()),
        "best_return": float(rets.max()),
        "total_return": float(rets.sum()),  # equal-weight, non-compounding
        "lower_hit_pct": float(
            (trades["outcome"] == "lower_hit").mean()
        ),
        "upper_hit_pct": float(
            (trades["outcome"] == "upper_hit").mean()
        ),
        "time_out_pct": float(
            (trades["outcome"] == "time_out").mean()
        ),
    }


def equity_curve(trades: pd.DataFrame) -> pd.Series:
    """Cumulative return assuming equal-sized non-overlapping trades."""
    if trades.empty:
        return pd.Series(dtype=float)
    s = trades.set_index("exit_date")["return_pct"].sort_index()
    return s.cumsum()
