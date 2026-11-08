"""Build the Part 2 report tables (Markdown + CSV) from results/part2/metrics/*.json.

Same-condition tables give every metric three ways (spec Goal 7): the deterministic full-data fit
(`<m>_full`) and the mean and sample sd (ddof=1) over the seeded 80%-subsample fits (`<m>_mean`,
`<m>_sd`), with the number of seeds behind them (`n_seeds`). Conditions that are incomplete, or
whose metric rows mix feature extractions or head settings, are printed as warnings on stderr.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import project_root_from_script

# Strict-protocol test groups, with Midjourney last because it is reported separately.
STRICT_GROUPS = ["ADM", "GLIDE", "SD15", "VQDM", "Wukong", "Midjourney"]
SEPARATE_GROUPS = {"Midjourney"}  # scored, but never part of the strict "overall"
CONDITION = ["backbone", "pipeline", "format"]
# The oracle picks its threshold on the test labels (spec Goal 6): the tables name it an upper bound.
ORACLE = "oracle_bacc_upper_bound"
OVERALL_METRICS = ["auroc", "ap", "balanced_accuracy", "fake_recall", "real_specificity", ORACLE]
PER_GENERATOR_METRICS = ["fake_recall", "auroc"]
LOGO_METRICS = ["auroc", "ap", "balanced_accuracy"]
CROSS_METRICS = ["auroc", "ap", "balanced_accuracy", "fake_recall", "real_specificity"]
CROSS_KEYS = ["backbone", "protocol", "pipeline", "format", "test_pipeline", "test_format"]
FOLD_MEAN = "mean_over_folds"  # fold label of each condition's LOGO summary row
# A complete condition has every seed of config/part2.yaml (4001-4003) and, under LOGO, all 7 folds.
EXPECTED_SEEDS = 3
EXPECTED_LOGO_FOLDS = 7
FINAL_TITLE = "Final classifier (all 7 generators): validation split, in-distribution only, not generalisation"
TITLES = {
    "strict_overall": "Strict protocol (BigGAN -> diffusion), overall over ADM/GLIDE/SD15/VQDM/Wukong: "
                      "full-data fit (_full) and mean/sd over seeds (_mean/_sd); oracle_bacc_upper_bound "
                      "picks its threshold on the test labels, so it is an upper bound, not a result",
    "strict_per_generator": "Strict protocol, per test generator: full-data fit (_full) and mean/sd over "
                            "seeds (_mean/_sd); Midjourney (in_overall = False) is reported separately",
    "logo_per_fold": "Leave-one-generator-out (main cross-generator evidence), per held-out generator: "
                     "full-data fit (_full) and mean/sd over seeds (_mean/_sd); mean_over_folds averages "
                     "each fit over its folds first, then takes mean/sd across seeds",
    "cross_condition": "Cross-condition transfer (full-data heads; LOGO averaged over folds): _same = the "
                       "head on its own training condition, _cross = on the test condition; "
                       "transfer drop = _same - _cross",
    "final_validation": FINAL_TITLE,
}


def _is_full(row: dict) -> bool:
    """The deterministic full-data fit; "full" is the only string seed."""
    return isinstance(row["seed"], str)


def summarise(rows: list[dict], by: list[str], value_keys: list[str]) -> pd.DataFrame:
    """Per `by` group: `n_seeds`, `<key>_full` (seed == "full"), `<key>_mean` and `<key>_sd` (ddof=1)
    over the integer seeds."""
    columns = list(by) + ["n_seeds"] + [f"{k}_{s}" for k in value_keys for s in ("full", "mean", "sd")]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    is_full = df["seed"].map(lambda s: isinstance(s, str))
    out = []
    for key, g in df.groupby(list(by), sort=False, dropna=False):
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        full, seeds = g[is_full.loc[g.index]], g[~is_full.loc[g.index]]
        rec["n_seeds"] = int(seeds["seed"].nunique())
        for k in value_keys:
            rec[f"{k}_full"] = full[k].iloc[0] if len(full) else np.nan
            rec[f"{k}_mean"] = seeds[k].mean() if len(seeds) else np.nan
            rec[f"{k}_sd"] = seeds[k].std(ddof=1) if len(seeds) > 1 else np.nan
        out.append(rec)
    return pd.DataFrame(out, columns=columns)


def _same_condition(row: dict) -> bool:
    """A row scored on the condition its head was trained on."""
    return (row.get("test_pipeline", row["pipeline"]) == row["pipeline"]
            and row.get("test_format", row["format"]) == row["format"])


def _with_oracle_label(rows: list[dict]) -> list[dict]:
    """Copies of the rows with the oracle metric renamed so every table calls it an upper bound."""
    return [{(ORACLE if k == "oracle_balanced_accuracy" else k): v for k, v in r.items()} for r in rows]


def _ordered(df: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    """Conditions in first-seen order; within each, `column` by its position in `order` (others last)."""
    if df.empty:
        return df
    conditions = [tuple(c) for c in df[CONDITION].itertuples(index=False)]
    cond_rank = {c: i for i, c in enumerate(dict.fromkeys(conditions))}
    rank = {v: i for i, v in enumerate(order)}
    keys = [(cond_rank[c], rank.get(v, len(order)), str(v)) for c, v in zip(conditions, df[column])]
    return df.iloc[sorted(range(len(df)), key=keys.__getitem__)].reset_index(drop=True)


def _strict_overall(rows: list[dict]) -> pd.DataFrame:
    sel = [r for r in rows if _same_condition(r) and r["protocol"] == "strict" and r["group"] == "overall"]
    return summarise(sel, CONDITION, OVERALL_METRICS)


def _strict_per_generator(rows: list[dict]) -> pd.DataFrame:
    """One row per condition x strict test group (Midjourney last, flagged in_overall = False)."""
    sel = [r for r in rows if _same_condition(r) and r["protocol"] == "strict" and r["group"] in STRICT_GROUPS]
    df = summarise(sel, CONDITION + ["group"], PER_GENERATOR_METRICS)
    df.insert(len(CONDITION) + 1, "in_overall", ~df["group"].isin(SEPARATE_GROUPS))
    return _ordered(df, "group", STRICT_GROUPS)


def _fold_means(rows: list[dict]) -> tuple[list[dict], dict[tuple, int]]:
    """Per condition and fit (full or seed), one row holding that fit's mean over the folds.

    A fit missing one of the condition's folds is left out, so every value of a summary row
    averages the same folds (the gap shows as a smaller n_seeds or a NaN `_full`). Also returns
    the number of folds per condition.
    """
    fits: dict[tuple, dict] = {}  # condition -> seed -> fold -> metric row
    for r in rows:
        cond = tuple(r[c] for c in CONDITION)
        fits.setdefault(cond, {}).setdefault(r["seed"], {})[r["fold"]] = r
    out, n_folds = [], {}
    for cond, by_seed in fits.items():
        folds = sorted(set().union(*by_seed.values()))
        n_folds[cond] = len(folds)
        for seed, by_fold in by_seed.items():
            if set(by_fold) != set(folds):
                continue
            out.append({**dict(zip(CONDITION, cond)), "fold": FOLD_MEAN, "seed": seed,
                        **{m: float(np.mean([by_fold[f][m] for f in folds])) for m in LOGO_METRICS}})
    return out, n_folds


def _logo_per_fold(rows: list[dict]) -> pd.DataFrame:
    """Per held-out generator, then one `mean_over_folds` row per condition.

    The summary row averages the folds within each fit first, so its `_sd` is the spread across
    seeds of the seven-fold mean (not an average of per-fold sds) and its `_full` is the full-data
    fold mean. `n_folds` is the number of folds a row averages.
    """
    columns = (CONDITION + ["fold", "n_seeds", "n_folds"]
               + [f"{m}_{s}" for m in LOGO_METRICS for s in ("full", "mean", "sd")])
    # the held-out generator's own group is the test group of each fold
    sel = [r for r in rows if _same_condition(r) and r["protocol"] == "logo" and r["group"] == r["fold"]]
    per_fold = summarise(sel, CONDITION + ["fold"], LOGO_METRICS).assign(n_folds=1).to_dict("records")
    mean_rows, n_folds = _fold_means(sel)
    means = summarise(mean_rows, CONDITION + ["fold"], LOGO_METRICS).to_dict("records")
    # A condition where no fit covers every fold still gets a (NaN) summary row, so the gap shows.
    covered = {tuple(r[c] for c in CONDITION) for r in means}
    means += [{**dict(zip(CONDITION, c)), "fold": FOLD_MEAN, "n_seeds": 0} for c in n_folds if c not in covered]
    for r in means:
        r["n_folds"] = n_folds[tuple(r[c] for c in CONDITION)]
    df = pd.DataFrame(per_fold + means, columns=columns)
    return _ordered(df, "fold", sorted({r["fold"] for r in per_fold}) + [FOLD_MEAN])


def _cross_condition(rows: list[dict]) -> pd.DataFrame:
    """Full-data heads scored off their training condition, next to the same head on its own condition.

    LOGO rows average the per-fold "overall" rows over folds, pairing each fold's cross value with
    the same fold's same-condition value; a fold without that value makes `<m>_same` NaN.
    """
    value_cols = [f"{m}_{kind}" for m in CROSS_METRICS for kind in ("same", "cross")]
    columns = CROSS_KEYS + ["n_folds"] + value_cols
    full = [r for r in rows if _is_full(r) and r["protocol"] in ("strict", "logo") and r["group"] == "overall"]
    same = {(r["backbone"], r["protocol"], r["pipeline"], r["format"], r["fold"]): r
            for r in full if _same_condition(r)}
    records = []
    for r in full:
        if _same_condition(r):
            continue
        s = same.get((r["backbone"], r["protocol"], r["pipeline"], r["format"], r["fold"]))
        rec = {k: r[k] for k in CROSS_KEYS}
        for m in CROSS_METRICS:
            rec[f"{m}_same"] = s[m] if s is not None else np.nan
            rec[f"{m}_cross"] = r[m]
        records.append(rec)
    if not records:
        return pd.DataFrame(columns=columns)
    g = pd.DataFrame(records).groupby(CROSS_KEYS, sort=False)
    out = g[value_cols].agg(lambda s: s.mean() if s.notna().all() else np.nan)
    out.insert(0, "n_folds", g.size())
    return out.reset_index()[columns]


def _final_validation(rows: list[dict]) -> pd.DataFrame:
    keep = CONDITION + ["auroc", "ap", "balanced_accuracy", "fake_recall", "real_specificity", "n_real", "n_fake"]
    sel = [r for r in rows if _is_full(r) and r["protocol"] == "final" and r["group"] == "validation"]
    return pd.DataFrame(sel, columns=keep) if sel else pd.DataFrame(columns=keep)


def build_tables(rows: list[dict]) -> dict[str, pd.DataFrame]:
    """The five report tables keyed by file stem."""
    rows = _with_oracle_label(rows)
    return {
        "strict_overall": _strict_overall(rows),
        "strict_per_generator": _strict_per_generator(rows),
        "logo_per_fold": _logo_per_fold(rows),
        "cross_condition": _cross_condition(rows),
        "final_validation": _final_validation(rows),
    }


def completeness_warnings(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Conditions with fewer than EXPECTED_SEEDS seeds or EXPECTED_LOGO_FOLDS folds, or no full-data fit."""
    out = []
    name = lambda rec: "/".join(str(rec[c]) for c in CONDITION)  # noqa: E731
    for rec in tables["strict_overall"].to_dict("records"):
        if rec["n_seeds"] < EXPECTED_SEEDS or pd.isna(rec["auroc_full"]):
            out.append(f"strict {name(rec)} is incomplete: {rec['n_seeds']} of {EXPECTED_SEEDS} seeds"
                       + ("" if pd.notna(rec["auroc_full"]) else ", no full-data fit"))
    for rec in tables["logo_per_fold"].to_dict("records"):
        if rec["fold"] == FOLD_MEAN and (rec["n_folds"] < EXPECTED_LOGO_FOLDS or rec["n_seeds"] < EXPECTED_SEEDS
                                         or pd.isna(rec["auroc_full"])):
            out.append(f"logo {name(rec)} is incomplete: {rec['n_folds']} of {EXPECTED_LOGO_FOLDS} folds, "
                       f"{rec['n_seeds']} of {EXPECTED_SEEDS} seeds covering every fold"
                       + ("" if pd.notna(rec["auroc_full"]) else ", full-data fit missing a fold"))
    for rec in tables["cross_condition"].to_dict("records"):
        few_folds = rec["protocol"] == "logo" and rec["n_folds"] < EXPECTED_LOGO_FOLDS
        if few_folds or pd.isna(rec["auroc_same"]):
            out.append(f"cross {rec['protocol']} {rec['backbone']}/{rec['pipeline']}/{rec['format']} -> "
                       f"{rec['test_pipeline']}/{rec['test_format']} is incomplete: {rec['n_folds']} folds"
                       + (", same-condition value missing" if pd.isna(rec["auroc_same"]) else ""))
    return out


def provenance_warnings(rows: list[dict]) -> list[str]:
    """Metric rows without a run fingerprint, or mixing feature extractions or head settings."""
    out = []
    unfingerprinted = 0
    versions: dict[str, set] = {}  # feature set -> {(spec_hash, created_at)} seen across rows
    settings: set[str] = set()
    for r in rows:
        fp = r.get("fingerprint")
        if not isinstance(fp, dict):
            unfingerprinted += 1
            continue
        # The fingerprint names its feature sets: a cross-condition run's validation row is
        # labelled with the train condition, so the row's test_* columns cannot be used here.
        for side in ("train", "test"):
            versions.setdefault(str(fp.get(f"{side}_features")), set()).add(
                (fp.get(f"{side}_spec_hash"), fp.get(f"{side}_created_at")))
        settings.add(json.dumps({k: fp.get(k) for k in ("C_grid", "max_iter", "subsample_fraction", "min_side")},
                                sort_keys=True))
    if unfingerprinted:
        out.append(f"{unfingerprinted} metric rows carry no run fingerprint (written before it existed); "
                   "re-run part2_run.py --grid, which re-runs stale runs")
    for fs_id, seen in sorted(versions.items()):
        if len(seen) > 1:
            out.append(f"metric rows on {fs_id} come from {len(seen)} different feature extractions; "
                       "re-run part2_run.py --grid, which re-runs stale runs")
    if len(settings) > 1:
        out.append(f"metric rows use {len(settings)} different head/exclusion settings; re-run part2_run.py --grid")
    return out


def to_markdown(df: pd.DataFrame, title: str) -> str:
    """Small dependency-free Markdown table (floats at 4 decimals)."""
    lines = [f"### {title}", ""]
    if df.empty:
        return "\n".join(lines + ["_No rows yet._", ""])
    fmt = lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    lines.append("| " + " | ".join(map(str, df.columns)) + " |")
    lines.append("|" + "|".join("---" for _ in df.columns) + "|")
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines + [""])


def load_rows(metrics_dir: Path) -> list[dict]:
    """All metric rows from every <run_id>.json (Python's json accepts Infinity)."""
    rows: list[dict] = []
    for path in sorted(Path(metrics_dir).glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def main(metrics_dir: Path | None = None, out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read the metric files, write each table as .md and .csv, and warn about gaps on stderr."""
    root = project_root_from_script(__file__)
    metrics_dir = Path(metrics_dir) if metrics_dir else root / "results" / "part2" / "metrics"
    out_dir = Path(out_dir) if out_dir else root / "results" / "part2" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(metrics_dir)
    tables = build_tables(rows)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
        (out_dir / f"{name}.md").write_text(to_markdown(df, TITLES[name]), encoding="utf-8")
    for message in provenance_warnings(rows) + completeness_warnings(tables):
        print(f"WARNING: {message}", file=sys.stderr)
    return tables


if __name__ == "__main__":
    main()
