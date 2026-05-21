"""Model training and evaluation.

Two model choices, both well-suited to tabular features at this scale:

  - RandomForestClassifier  -- baseline matching the 2016 paper
  - HistGradientBoostingClassifier  -- usually stronger, sklearn-native
                                        (no xgboost dependency)

We report ML metrics (precision/recall/F1/AUC) plus a simple
short-signal backtest (see backtest.py). For shorts, **precision is the
metric that matters most**: a false positive (you short a stock that then
rips upward) is worse than a missed opportunity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .splits import purged_walk_forward_splits


ModelKind = Literal["rf", "hgb"]


def make_model(kind: ModelKind = "rf", random_state: int = 42) -> Any:
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=400,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
        )
    raise ValueError(f"unknown model kind: {kind!r}")


@dataclass
class FoldResult:
    fold: int
    train_size: int
    test_size: int
    positive_rate_train: float
    positive_rate_test: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    avg_precision: float
    confusion: List[List[int]]
    test_index: pd.DatetimeIndex = field(repr=False)
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    y_proba: np.ndarray = field(repr=False)


def _score_fold(y_true: np.ndarray, y_pred: np.ndarray,
                y_proba: np.ndarray) -> Dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    # AUC undefined if only one class present in y_true
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
    else:
        auc = float("nan")
        ap = float("nan")
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "roc_auc": float(auc),
        "avg_precision": float(ap),
    }


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_kind: ModelKind = "rf",
    n_splits: int = 5,
    purge: int = 10,
    threshold: float = 0.5,
    random_state: int = 42,
) -> List[FoldResult]:
    """Run purged walk-forward CV and return per-fold results."""
    if len(X) != len(y):
        raise ValueError("X and y must align")
    X_arr = X.to_numpy()
    y_arr = y.to_numpy()

    results: List[FoldResult] = []
    for fold_idx, (tr, te) in enumerate(
        purged_walk_forward_splits(len(X), n_splits=n_splits, purge=purge)
    ):
        model = make_model(model_kind, random_state=random_state)
        model.fit(X_arr[tr], y_arr[tr])
        proba = model.predict_proba(X_arr[te])[:, 1]
        pred = (proba >= threshold).astype(int)

        scores = _score_fold(y_arr[te], pred, proba)
        cm = confusion_matrix(y_arr[te], pred, labels=[0, 1]).tolist()
        results.append(FoldResult(
            fold=fold_idx,
            train_size=len(tr),
            test_size=len(te),
            positive_rate_train=float(y_arr[tr].mean()),
            positive_rate_test=float(y_arr[te].mean()),
            precision=scores["precision"],
            recall=scores["recall"],
            f1=scores["f1"],
            roc_auc=scores["roc_auc"],
            avg_precision=scores["avg_precision"],
            confusion=cm,
            test_index=X.index[te],
            y_true=y_arr[te],
            y_pred=pred,
            y_proba=proba,
        ))
    return results


def summarize_folds(results: List[FoldResult]) -> pd.DataFrame:
    rows = [
        {
            "fold": r.fold,
            "train_size": r.train_size,
            "test_size": r.test_size,
            "pos_rate_test": round(r.positive_rate_test, 3),
            "precision": round(r.precision, 3),
            "recall": round(r.recall, 3),
            "f1": round(r.f1, 3),
            "roc_auc": round(r.roc_auc, 3),
            "avg_precision": round(r.avg_precision, 3),
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    if len(df):
        mean_row = {
            "fold": "mean",
            "train_size": int(df["train_size"].mean()),
            "test_size": int(df["test_size"].mean()),
            "pos_rate_test": round(df["pos_rate_test"].mean(), 3),
            "precision": round(df["precision"].mean(), 3),
            "recall": round(df["recall"].mean(), 3),
            "f1": round(df["f1"].mean(), 3),
            "roc_auc": round(df["roc_auc"].mean(), 3),
            "avg_precision": round(df["avg_precision"].mean(), 3),
        }
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df


def fit_final(
    X: pd.DataFrame,
    y: pd.Series,
    model_kind: ModelKind = "rf",
    random_state: int = 42,
) -> Any:
    """Fit on all available labeled data for production use."""
    model = make_model(model_kind, random_state=random_state)
    model.fit(X.to_numpy(), y.to_numpy())
    return model


def save_model(model: Any, feature_names: List[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names}, path)


def load_model(path: str | Path) -> Dict[str, Any]:
    return joblib.load(path)
