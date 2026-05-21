"""Train a short-signal model on one ticker.

Usage:
    python -m scripts.train --ticker AAPL --model rf
    python -m scripts.train --ticker TSLA --model hgb --lower 0.07 --upper 0.04 \\
        --horizon 15 --start 2012-01-01 --out models/tsla_hgb.joblib
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/train.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import backtest_signals, trade_stats
from src.data import align_short_interest, load_prices, load_short_interest
from src.features import build_features
from src.labels import BarrierConfig, triple_barrier_labels
from src.model import cross_validate, fit_final, save_model, summarize_folds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker", required=True, help="Stock ticker, e.g. AAPL")
    p.add_argument("--start", default="2012-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--model", choices=["rf", "hgb"], default="rf")
    p.add_argument("--lower", type=float, default=0.05,
                   help="Short profit target (price drop fraction)")
    p.add_argument("--upper", type=float, default=0.03,
                   help="Stop-out (price rise fraction)")
    p.add_argument("--horizon", type=int, default=10,
                   help="Max trading days per trade")
    p.add_argument("--splits", type=int, default=5,
                   help="Number of walk-forward CV folds")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Probability threshold for emitting a signal")
    p.add_argument("--short-interest-csv", default=None,
                   help="Optional CSV with columns: date, short_interest, "
                        "[avg_daily_volume]")
    p.add_argument("--no-extras", action="store_true",
                   help="Use only the original 6 indicators (baseline mode)")
    p.add_argument("--out", default=None,
                   help="Where to save the final model (.joblib)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BarrierConfig(
        lower_pct=args.lower, upper_pct=args.upper, horizon=args.horizon
    )

    print(f"Loading {args.ticker} from {args.start} to {args.end or 'today'}")
    prices = load_prices(args.ticker, start=args.start, end=args.end)
    print(f"  {len(prices)} trading days")

    si_aligned = None
    if args.short_interest_csv:
        si = load_short_interest(args.short_interest_csv)
        si_aligned = align_short_interest(prices, si)
        print(f"  short-interest rows: {si_aligned.notna().any(axis=1).sum()}")

    print("Building features...")
    X = build_features(prices, short_interest=si_aligned,
                       include_extras=not args.no_extras)
    print(f"  feature matrix: {X.shape}")

    print(f"Labeling (lower={cfg.lower_pct}, upper={cfg.upper_pct}, "
          f"horizon={cfg.horizon})...")
    y_df = triple_barrier_labels(
        prices["Close"], prices["High"], prices["Low"], cfg,
    )

    aligned_idx = X.index.intersection(y_df.index)
    X = X.loc[aligned_idx]
    y = y_df.loc[aligned_idx, "label"]
    print(f"  labeled samples: {len(y)}  positive rate: {y.mean():.3f}")

    print(f"Cross-validating ({args.model}, {args.splits} folds)...")
    results = cross_validate(
        X, y,
        model_kind=args.model,
        n_splits=args.splits,
        purge=cfg.horizon,
        threshold=args.threshold,
    )
    summary = summarize_folds(results)
    print(summary.to_string(index=False))

    # Backtest on out-of-sample predictions concatenated across folds
    all_signal_dates = []
    for r in results:
        sig_mask = r.y_proba >= args.threshold
        all_signal_dates.extend(r.test_index[sig_mask])
    signal_idx = sorted(set(all_signal_dates))
    print(f"\nOut-of-sample signals fired: {len(signal_idx)}")
    if signal_idx:
        trades = backtest_signals(prices, signal_idx, cfg)
        stats = trade_stats(trades)
        print("Backtest stats:")
        print(json.dumps(stats, indent=2, default=str))

    if args.out:
        print(f"\nFitting final model on all data and saving to {args.out}...")
        final_model = fit_final(X, y, model_kind=args.model)
        save_model(final_model, list(X.columns), args.out)
        print("Saved.")


if __name__ == "__main__":
    main()
