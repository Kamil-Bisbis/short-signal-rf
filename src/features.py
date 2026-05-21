"""Feature engineering.

Two feature groups:

1. The six indicators from Khaidem, Saha & Dey (2016) -- kept for direct
   comparability with the original random-forests baseline:
       - RSI (14)
       - Stochastic Oscillator %K (14)
       - Williams %R (14)
       - MACD (12, 26, 9)
       - Price Rate of Change (10)
       - On-Balance Volume

2. Short-relevant additions: features that capture "is this stock stretched
   to the upside / due for a reversal":
       - distance from 50d and 200d moving averages (in pct)
       - Bollinger position (where in the 20d band the close sits)
       - ATR-normalized 5d return (how unusual the recent move is)
       - 20d log-volume z-score (unusual volume often precedes reversals)
       - 20d max drawup (frothy run-ups)

Optional short-interest features are added when an aligned SI frame is
passed in. All features use only past data -- nothing leaks from the future.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# --------- Original-paper indicators --------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 14) -> pd.Series:
    lowest = low.rolling(period).min()
    highest = high.rolling(period).max()
    return 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 14) -> pd.Series:
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": macd_line - signal_line,
    })


def price_rate_of_change(close: pd.Series, period: int = 10) -> pd.Series:
    return 100 * (close - close.shift(period)) / close.shift(period)


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


# --------- Short-relevant additions ---------------------------------------

def distance_from_sma(close: pd.Series, period: int) -> pd.Series:
    sma = close.rolling(period).mean()
    return 100 * (close - sma) / sma


def bollinger_position(close: pd.Series, period: int = 20,
                       k: float = 2.0) -> pd.Series:
    mean = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mean + k * std
    lower = mean - k * std
    band = (upper - lower).replace(0, np.nan)
    # 0 = at lower band, 1 = at upper band
    return (close - lower) / band


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def atr_normalized_return(close: pd.Series, high: pd.Series, low: pd.Series,
                          ret_period: int = 5, atr_period: int = 14) -> pd.Series:
    ret = close.diff(ret_period)
    a = atr(high, low, close, atr_period)
    return ret / a.replace(0, np.nan)


def volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    log_vol = np.log1p(volume)
    mean = log_vol.rolling(period).mean()
    std = log_vol.rolling(period).std().replace(0, np.nan)
    return (log_vol - mean) / std


def max_drawup(close: pd.Series, period: int = 20) -> pd.Series:
    """Largest peak-to-current pullback in pct over the past `period` days.

    High values = stock recently ran up and is currently at/near the high
    (potential short candidate by mean-reversion logic).
    """
    rolling_min = close.rolling(period).min()
    return 100 * (close - rolling_min) / rolling_min


# --------- Assembly --------------------------------------------------------

ORIGINAL_FEATURES = [
    "rsi_14", "stoch_k_14", "williams_r_14",
    "macd", "macd_signal", "macd_hist",
    "proc_10", "obv",
]

EXTRA_FEATURES = [
    "dist_sma_50", "dist_sma_200",
    "bollinger_pos_20", "atr_norm_ret_5",
    "vol_zscore_20", "max_drawup_20",
]


def build_features(
    prices: pd.DataFrame,
    short_interest: Optional[pd.DataFrame] = None,
    include_extras: bool = True,
) -> pd.DataFrame:
    """Compute the full feature matrix for one ticker.

    Parameters
    ----------
    prices
        DataFrame with Open, High, Low, Close, Volume and a DatetimeIndex.
    short_interest
        Optional DataFrame already aligned to `prices.index` (use
        `data.align_short_interest`). If passed, adds short_interest_log
        and (when present) days_to_cover as features.
    include_extras
        If False, return only the original-paper features for an apples-to-
        apples comparison with the 2016 baseline.
    """
    close, high, low, volume = (
        prices["Close"], prices["High"], prices["Low"], prices["Volume"]
    )

    feats = pd.DataFrame(index=prices.index)
    feats["rsi_14"] = rsi(close, 14)
    feats["stoch_k_14"] = stochastic_k(high, low, close, 14)
    feats["williams_r_14"] = williams_r(high, low, close, 14)
    m = macd(close)
    feats[["macd", "macd_signal", "macd_hist"]] = m
    feats["proc_10"] = price_rate_of_change(close, 10)
    feats["obv"] = on_balance_volume(close, volume)

    if include_extras:
        feats["dist_sma_50"] = distance_from_sma(close, 50)
        feats["dist_sma_200"] = distance_from_sma(close, 200)
        feats["bollinger_pos_20"] = bollinger_position(close, 20)
        feats["atr_norm_ret_5"] = atr_normalized_return(close, high, low, 5, 14)
        feats["vol_zscore_20"] = volume_zscore(volume, 20)
        feats["max_drawup_20"] = max_drawup(close, 20)

    if short_interest is not None and len(short_interest):
        if "short_interest" in short_interest.columns:
            feats["short_interest_log"] = np.log1p(
                short_interest["short_interest"].astype(float)
            )
        if "days_to_cover" in short_interest.columns:
            feats["days_to_cover"] = short_interest["days_to_cover"].astype(float)

    return feats.dropna()
