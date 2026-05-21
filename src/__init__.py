"""short-signal-rf: a tabular-ML baseline for short-trade classification."""

from . import backtest, data, features, labels, model, splits

__all__ = ["backtest", "data", "features", "labels", "model", "splits"]
__version__ = "0.1.0"
