"""Per-group detection metrics (AUROC, AP, balanced accuracy, oracle threshold)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _balanced_accuracy(y: np.ndarray, pred_fake: np.ndarray) -> tuple[float, float, float]:
    """Return (balanced accuracy, real specificity, fake recall) for boolean predictions."""
    spec = float(np.mean(~pred_fake[y == 0]))
    rec = float(np.mean(pred_fake[y == 1]))
    return (spec + rec) / 2.0, spec, rec


def group_metrics(y_true: np.ndarray, p_fake: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Metrics for one group. Fake if p_fake >= threshold.

    The oracle is the best balanced accuracy over every observed score used as
    a threshold (plus "everything real"); it is an upper bound, not a result.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_fake, dtype=float)
    n_real, n_fake = int((y == 0).sum()), int((y == 1).sum())
    if n_real == 0 or n_fake == 0:
        raise ValueError("group_metrics needs both classes present")

    ba, spec, rec = _balanced_accuracy(y, p >= threshold)

    # Candidate thresholds: each observed score, plus +inf (everything predicted real).
    best_ba, best_thr = -1.0, float("inf")
    for t in np.concatenate([np.unique(p), [np.inf]]):
        cand, _, _ = _balanced_accuracy(y, p >= t)
        if cand > best_ba:
            best_ba, best_thr = cand, float(t)

    return {
        "n_real": n_real,
        "n_fake": n_fake,
        "auroc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "balanced_accuracy": float(ba),
        "real_specificity": spec,
        "fake_recall": rec,
        "oracle_balanced_accuracy": float(best_ba),
        "oracle_threshold": best_thr,
    }
