"""Data loading: OHLCV from yfinance and optional short-interest from CSV.

The model is designed to work with technical features alone. If you have
short-interest history (e.g. from FINRA's bi-monthly reports), pass a CSV
to `load_short_interest` and the features module will fold it in.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_prices(
    ticker: str,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance.

    Returns a DataFrame with columns Open, High, Low, Close, Volume and a
    DatetimeIndex. Uses auto-adjusted prices by default so splits/dividends
    don't create artificial signals.
    """
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker!r}.")

    # yfinance sometimes returns a column MultiIndex when one ticker is passed
    # in a list-shaped way; flatten it to keep downstream simple.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def load_short_interest(path: str | Path) -> pd.DataFrame:
    """Load a CSV of short-interest history.

    Expected columns (case-insensitive):
        date            -- settlement date
        short_interest  -- shares sold short
        avg_daily_volume (optional) -- used to derive days_to_cover

    Returns a DataFrame indexed by date, sorted ascending, with columns
    `short_interest` and (when computable) `days_to_cover`.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "short_interest" not in df.columns:
        raise ValueError(
            "short-interest CSV must have 'date' and 'short_interest' columns"
        )
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    out = pd.DataFrame(index=df.index)
    out["short_interest"] = df["short_interest"].astype(float)
    if "avg_daily_volume" in df.columns:
        adv = df["avg_daily_volume"].astype(float).replace(0, pd.NA)
        out["days_to_cover"] = out["short_interest"] / adv
    return out


def align_short_interest(prices: pd.DataFrame, si: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill short-interest values onto the daily price index.

    Short-interest is reported every two weeks. Forward-fill means each
    trading day uses the most recently *published* value, which is what a
    real trader would have seen on that day -- no peeking forward.
    """
    aligned = si.reindex(prices.index, method="ffill")
    return aligned
