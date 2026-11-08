"""Logistic-regression head on frozen features, JSON persistence, seeded subsampling."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


@dataclass
class FittedHead:
    model: LogisticRegression
    C: float
    val_log_loss: float
    meta: dict = field(default_factory=dict)


def fit_head(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
             C_grid: Sequence[float], max_iter: int = 5000) -> FittedHead:
    """One fit per C; keep the lowest validation log loss (ties -> smaller C)."""
    best: FittedHead | None = None
    for C in sorted(C_grid):  # ascending, so a strict '<' keeps the smaller C on ties
        model = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=max_iter)
        model.fit(X_train, y_train)
        loss = float(log_loss(y_val, model.predict_proba(X_val)[:, 1], labels=[0, 1]))
        if best is None or loss < best.val_log_loss:
            best = FittedHead(model=model, C=float(C), val_log_loss=loss, meta={})
    if best is None:
        raise ValueError("C_grid is empty")
    return best


def predict_fake(head: FittedHead, X: np.ndarray) -> np.ndarray:
    """P(label 1) for each row; rejects a feature width that differs from training."""
    X = np.asarray(X)
    width = head.model.coef_.shape[1]
    if X.ndim != 2 or X.shape[1] != width:
        raise ValueError(f"expected features of width {width}, got shape {X.shape}")
    return head.model.predict_proba(X)[:, 1]


def save_head(head: FittedHead, path: Path) -> None:
    """Write the head as JSON; json emits floats via repr, so values round-trip exactly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = head.model
    payload = {
        "C": float(head.C),
        "coef": m.coef_.tolist(),
        "intercept": m.intercept_.tolist(),
        "classes": m.classes_.tolist(),
        "val_log_loss": float(head.val_log_loss),
        "meta": head.meta,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_head(path: Path) -> FittedHead:
    """Rebuild a head from JSON written by save_head."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    model = LogisticRegression(C=d["C"], class_weight="balanced", solver="lbfgs")
    model.coef_ = np.array(d["coef"], dtype=np.float64)
    model.intercept_ = np.array(d["intercept"], dtype=np.float64)
    model.classes_ = np.array(d["classes"])
    model.n_features_in_ = model.coef_.shape[1]
    return FittedHead(model=model, C=float(d["C"]), val_log_loss=float(d["val_log_loss"]), meta=d["meta"])


def stratified_subsample(rows: list[dict], fraction: float, seed: int) -> list[dict]:
    """Per (label, generator) stratum draw round(fraction*n) rows; return in input order."""
    strata: dict[tuple, list[dict]] = {}
    for r in rows:
        strata.setdefault((r["label"], r["generator"]), []).append(r)
    rng = random.Random(seed)
    keep: set[int] = set()
    for key in sorted(strata):  # fixed stratum order -> deterministic draws
        members = sorted(strata[key], key=lambda r: r["image_id"])
        for r in rng.sample(members, round(fraction * len(members))):
            keep.add(id(r))
    return [r for r in rows if id(r) in keep]
