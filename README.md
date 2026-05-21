# Machine Learning Short-Signal-RF Stock Predictor

<img width="1179" height="772" alt="image" src="https://github.com/user-attachments/assets/295897f9-7242-464c-9988-46fce0f6be4d" />

## Introduction

Welcome! I built this stock market machine learning app locally in Spring 2023 using Python, scikit-learn, and pandas to help my team's performance in the GCEE Stock Market Game. The core idea was a short-signal classifier: given a stock's recent technical data, predict whether opening a short position is likely to be profitable within a defined time window.

At the time, my team and I believed the aggressive growth of 2021 was still due for correction. The Federal Reserve was aggressively hiking rates, valuations in growth and tech names had stretched well beyond historical norms during 2020-2021, and the conditions that drove that expansion (near-zero rates, pandemic liquidity) were reversing. 

We were already planning on shorting going into that period regardless. This model helped with identifying *which* names had the technical profile of a stock that had run up too far and was likely to give back gains within a short window.

In 10 weeks, we grew our portfolio from $100,000 to ~$190,000 which led to a 1st place finish out of 200 teams in Fulton County (Atlanta) and a 4th place finish out of 3,000 teams in Georgia in the GCEE Stock Market Game.

I'm uploading this publicly now after cleaning it up, adding documentation, and improving the methodology. The core short-signal logic is the same as what I ran at the time; the main changes are better labeling (triple-barrier instead of a naive direction label), a proper walk-forward cross-validation setup to avoid data leakage, and a cleaner project structure.

---

> **Disclaimer:** This project does not guarantee any results whatsoever. The stock market is extremely hard to predict, and short selling in particular carries risks that no model can fully account for (short squeezes, unlimited downside, borrow costs, timing risk, etc.). Past performance in a simulated competition environment does not reflect real trading outcomes. Nothing here is financial advice. 

---

## What it does

The model uses:

- Six technical indicators from a 2016 academic baseline (RSI, Stochastic Oscillator, Williams %R, MACD, Price Rate of Change, On-Balance Volume)
- Six additional short-relevant signals I added: distance from moving averages, Bollinger Band position, ATR-normalized return, volume z-score, and 20-day max drawup
- Triple-barrier labels that define a "good short" as: price drops 5% before rising 3%, within 10 trading days
- Purged walk-forward cross-validation to prevent future data from leaking into training

Optional: if you have FINRA short-interest history for a ticker (CSV format), the loader will add short interest and days-to-cover as features automatically.


## Install

```bash
git clone https://github.com/Kamil-Bisbis/short-signal-rf
cd short-signal-rf
pip install -r requirements.txt
```

Python 3.9+ recommended.

## Quick start

Train and cross-validate on a ticker:

```bash
python -m scripts.train --ticker AAPL --out models/aapl_rf.joblib
```

Try the gradient-boosting variant with different barriers:

```bash
python -m scripts.train --ticker TSLA --model hgb \
    --lower 0.07 --upper 0.04 --horizon 15 \
    --out models/tsla_hgb.joblib
```

Score the latest bar using a saved model:

```bash
python -m scripts.predict --ticker AAPL --model-path models/aapl_rf.joblib
```

With short-interest data:

```bash
python -m scripts.train --ticker GME \
    --short-interest-csv data/gme_short_interest.csv \
    --out models/gme_rf.joblib
```

Run only the original six indicators to compare against the academic baseline:

```bash
python -m scripts.train --ticker AAPL --no-extras
```

## Notebook

`notebooks/demo.ipynb` walks through the full pipeline on AAPL with charts: feature inspection, CV results, backtest equity curve, and feature importances.

## CLI reference

`scripts/train.py`:

| flag | default | description |
| --- | --- | --- |
| `--ticker` | required | Stock ticker |
| `--start` | 2012-01-01 | Start date for price download |
| `--end` | today | End date |
| `--model` | rf | `rf` (Random Forest) or `hgb` (Gradient Boosting) |
| `--lower` | 0.05 | Short profit target as a fraction |
| `--upper` | 0.03 | Stop-out level as a fraction |
| `--horizon` | 10 | Max trading days per trade |
| `--splits` | 5 | Walk-forward CV folds |
| `--threshold` | 0.5 | Probability threshold for a signal |
| `--short-interest-csv` | None | Optional FINRA-style short-interest CSV |
| `--no-extras` | False | Use only the original 6 indicators |
| `--out` | None | Path to save the final model |

## Reading the results

The metric to focus on for shorting is **precision**: of the trades the model signaled, what fraction actually hit the profit target? The positive rate of the test fold is your floor. If model precision is at or below that, there is no edge at that threshold.

The backtest reports hit rate, mean return, worst return (a proxy for squeeze risk), and total return. None of this includes borrow fees, slippage, or financing costs, so treat the numbers as directional, not literal.

## Short-interest CSV format

FINRA publishes bi-monthly short-interest reports at https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data. The loader expects:

| column | required |
| --- | --- |
| date | yes |
| short_interest | yes |
| avg_daily_volume | no (used to compute days_to_cover) |

Values are forward-filled to daily frequency so each row only uses the most recently published figure.

## Project layout

```
short-signal-rf/
├── src/
│   ├── data.py          # price download + short-interest loader
│   ├── features.py      # all indicators
│   ├── labels.py        # triple-barrier labeling
│   ├── splits.py        # purged walk-forward CV
│   ├── model.py         # train / evaluate / save / load
│   └── backtest.py      # short-signal backtest
├── scripts/
│   ├── train.py         # CLI: train + report
│   └── predict.py       # CLI: score the latest bar
├── notebooks/
│   └── demo.ipynb       # walkthrough with charts
└── tests/
    └── test_labels.py   # 8 tests covering labeler and CV splitter
```

## Tests

```bash
pytest tests/ -v
```

## References

- Khaidem, L., Saha, S., and Dey, S. R. (2016). Predicting the direction of stock market prices using random forest. [arXiv:1605.00003](https://arxiv.org/abs/1605.00003)
- Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
- Reproduction of the above paper (which flagged its likely data leakage): https://github.com/jmartinezheras/reproduce-stock-market-direction-random-forests

## License

Apache 2.0
