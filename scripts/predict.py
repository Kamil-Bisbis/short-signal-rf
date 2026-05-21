"""Use a saved model to score the most recent day for one ticker.

Usage:
    python -m scripts.predict --ticker AAPL --model-path models/aapl_rf.joblib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import align_short_interest, load_prices, load_short_interest
from src.features import build_features
from src.model import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--start", default="2020-01-01",
                   help="How far back to download for feature warm-up")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--short-interest-csv", default=None)
    p.add_argument("--no-extras", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model(args.model_path)
    model = bundle["model"]
    expected = bundle["feature_names"]

    prices = load_prices(args.ticker, start=args.start)
    si_aligned = None
    if args.short_interest_csv:
        si = load_short_interest(args.short_interest_csv)
        si_aligned = align_short_interest(prices, si)

    X = build_features(prices, short_interest=si_aligned,
                       include_extras=not args.no_extras)

    missing = [c for c in expected if c not in X.columns]
    extra = [c for c in X.columns if c not in expected]
    if missing:
        raise RuntimeError(
            f"Feature mismatch: model expects {missing} but they are not "
            f"present. Did you change --no-extras or --short-interest-csv?"
        )
    X = X[expected]  # enforce order

    last_date = X.index[-1]
    proba = float(model.predict_proba(X.iloc[[-1]].to_numpy())[0, 1])
    signal = proba >= args.threshold

    print(f"Ticker:        {args.ticker}")
    print(f"As of:         {last_date.date()}  (close = "
          f"{prices.loc[last_date, 'Close']:.2f})")
    print(f"Short proba:   {proba:.3f}")
    print(f"Signal (>= {args.threshold}): {'SHORT' if signal else 'no signal'}")
    if extra:
        print(f"(ignored unexpected features: {extra})")


if __name__ == "__main__":
    main()
