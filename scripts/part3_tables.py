"""Part 3 tables and figures from the per-image predictions written by part3_run.py.

Outputs (committed, small) under results/part3/:
    tables/robustness_overall.{csv,md}    strict overall (ADM/GLIDE/SD15/VQDM/Wukong pooled) per head x perturbation,
                                          with the change from the clean condition
    tables/robustness_per_group.{csv,md}  the same per test generator, Midjourney included (in_overall = False)
    tables/clean_check.{csv,md}           clean results on our sample next to Part 2's full-test results
    figures/degradation_curves.png        AUROC and fake recall vs JPEG quality and downscale factor

Metrics come from Part 2's group_metrics, so they are defined exactly as in the Part 2 tables.
Accuracy-type metrics use the 0.5 threshold. AUROC and AP do not, so together they separate
"the scores moved" (threshold metrics drop, AUROC holds) from "the detector lost its signal" (AUROC drops).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config
from part2.metrics import group_metrics

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["head", "pipeline", "format", "perturbation"]
METRICS = ["auroc", "ap", "balanced_accuracy", "fake_recall", "real_specificity"]
# Colour = pipeline, line style = format (see the figure legend).
PIPELINE_COLOURS = {"A": "#2a78d6", "B": "#eb6834"}
FORMAT_STYLES = {"raw": "-", "matched_jpeg": "--"}


def metric_row(frame: pd.DataFrame) -> dict:
    """group_metrics on the scored rows, plus coverage (share of sampled images scored).

    Coverage below 1 means some perturbed images were too small for the pipeline; the metric
    then describes only the images that remained, so the tables flag it.
    """
    scored = frame[frame["scored"]]
    row = {"n_sampled": len(frame), "coverage": len(scored) / len(frame) if len(frame) else np.nan}
    y = scored["label"].to_numpy(int)
    if (y == 0).any() and (y == 1).any():
        row.update(group_metrics(y, scored["p_fake"].to_numpy(float)))
    else:  # a class vanished entirely: no defined AUROC/recall
        row.update({m: np.nan for m in METRICS})
    return row


def summarise(pred: pd.DataFrame, order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(overall, per_group) metric tables, each with deltas from the clean condition."""
    overall = pd.DataFrame([
        {**dict(zip(KEYS, key)), **metric_row(g)}
        for key, g in pred[pred["in_overall"]].groupby(KEYS, sort=False)
    ])
    per_group = pd.DataFrame([
        {**dict(zip(KEYS + ["evaluation_group", "in_overall"], key)), **metric_row(g)}
        for key, g in pred.groupby(KEYS + ["evaluation_group", "in_overall"], sort=False)
    ])
    for table, by in ((overall, ["head"]), (per_group, ["head", "evaluation_group"])):
        clean = table[table["perturbation"] == "clean"].set_index(by)[METRICS]
        joined = table.join(clean, on=by, rsuffix="_clean")
        for m in METRICS:
            # Negative = worse than clean (for every metric here, higher is better).
            table[f"{m}_change"] = joined[m] - joined[f"{m}_clean"]
        table["perturbation"] = pd.Categorical(table["perturbation"], order, ordered=True)
        table.sort_values(by + ["perturbation"], inplace=True, ignore_index=True)
    return overall, per_group


def clean_check(overall: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Clean overall metrics on our sample next to Part 2's full-test numbers.

    This is our reproduction gate: if they are not close, the scoring is wired up wrongly and no
    perturbation result can be trusted. They will not match exactly, because we test a sample.
    """
    part2 = pd.read_csv(root / "results/part2/tables/strict_overall.csv")
    clean = overall[overall["perturbation"] == "clean"]
    rows = []
    for _, r in clean.iterrows():
        backbone = r["head"].split("__")[0]
        ref = part2[(part2["backbone"] == backbone) & (part2["pipeline"] == r["pipeline"])
                    & (part2["format"] == r["format"])]
        rows.append({
            "head": r["head"],
            "auroc_part3_sample": r["auroc"],
            "auroc_part2_full_test": ref["auroc_full"].iloc[0] if len(ref) else np.nan,
            "balanced_accuracy_part3_sample": r["balanced_accuracy"],
            "balanced_accuracy_part2_full_test": ref["balanced_accuracy_full"].iloc[0] if len(ref) else np.nan,
        })
    out = pd.DataFrame(rows)
    out["auroc_difference"] = out["auroc_part3_sample"] - out["auroc_part2_full_test"]
    return out


def write_table(frame: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".csv"), index=False, float_format="%.4f")
    shown = frame.drop(columns=[c for c in ("oracle_threshold",) if c in frame])
    try:
        body = shown.to_markdown(index=False, floatfmt=".4f")  # needs the optional `tabulate` package
    except ImportError:
        body = "```\n" + shown.to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n```"
    path.with_suffix(".md").write_text(f"### {title}\n\n{body}\n", encoding="utf-8")


def plot_curves(overall: pd.DataFrame, perts: list[dict], path: Path) -> None:
    """2 x 2 small multiples: rows AUROC / fake recall, columns JPEG quality / downscale factor."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping the figure (pip install matplotlib)")
        return
    # Each sweep: the clean point, then single-setting perturbations ordered from mild to strong.
    jpeg = [("clean", "none")] + sorted(
        ((p["name"], f"Q{p['jpeg']}") for p in perts if p["jpeg"] is not None and p["scale"] is None),
        key=lambda t: -int(t[1][1:]))
    resize = [("clean", "1.0×")] + sorted(
        ((p["name"], f"{p['scale']:g}×") for p in perts if p["scale"] is not None and p["jpeg"] is None),
        key=lambda t: -float(t[1][:-1]))
    sweeps = [("JPEG quality", jpeg), ("Downscale factor", resize)]
    rows = [("auroc", "AUROC"), ("fake_recall", "Fake recall at 0.5")]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharey="row")
    for c, (xlabel, sweep) in enumerate(sweeps):
        names, ticks = [s[0] for s in sweep], [s[1] for s in sweep]
        for r, (metric, ylabel) in enumerate(rows):
            ax = axes[r][c]
            for head, g in overall.groupby("head", sort=True):
                g = g.set_index("perturbation", drop=False)
                ys = [g.loc[n, metric] if n in g.index else np.nan for n in names]
                cover = [g.loc[n, "coverage"] if n in g.index else np.nan for n in names]
                pipe, fmt = g["pipeline"].iloc[0], g["format"].iloc[0]
                colour = PIPELINE_COLOURS[pipe]
                ax.plot(range(len(names)), ys, FORMAT_STYLES[fmt], color=colour,
                        linewidth=2, marker="o", markersize=5, label=f"Pipeline {pipe}, {fmt}")
                # Hollow marker: some images were too small to score, so this point covers a subset.
                partial = [i for i, cv in enumerate(cover) if cv < 1 and not np.isnan(ys[i])]
                ax.plot(partial, [ys[i] for i in partial], "o", markersize=8,
                        markerfacecolor="white", markeredgecolor=colour, markeredgewidth=2)
            ax.set_xticks(range(len(names)), ticks)
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 1:
                ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel(ylabel)
    for ax in axes[1]:  # 0.5 reference: a coin flip would catch half the fakes
        ax.axhline(0.5, color="#8c8b87", linewidth=0.8, linestyle=":")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9,
               handlelength=3.5, bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.008, "Hollow marker: only part of the images could be scored (too small for pipeline B's "
             "112 px crop); missing point: none could. Coverage is in robustness_overall.csv.",
             ha="center", fontsize=8, color="#52514e")
    fig.suptitle("Strict overall (unseen diffusion generators), ViT-L/14 heads", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/part3.yaml")
    args = parser.parse_args(argv)

    cfg3 = load_config(ROOT, args.config)
    results = ROOT / cfg3["results_root"]
    pred_path = ROOT / cfg3["local_root"] / "predictions" / f"{cfg3['backbone']}__{cfg3['protocol']}.csv.gz"
    if not pred_path.exists():
        raise SystemExit(f"{pred_path} not found; run scripts/part3_run.py first")
    pred = pd.read_csv(pred_path)
    pred["scored"] = pred["scored"].astype(bool)
    pred["in_overall"] = pred["in_overall"].astype(bool)
    perts = json.loads((results / "run_meta.json").read_text(encoding="utf-8"))["perturbations"]
    order = [p["name"] for p in perts]
    if "clean" not in set(pred["perturbation"]):
        raise SystemExit("the predictions have no 'clean' condition, so no change can be measured")

    overall, per_group = summarise(pred, order)
    tables = results / "tables"
    write_table(overall, tables / "robustness_overall",
                "Strict overall (ADM/GLIDE/SD15/VQDM/Wukong pooled): metrics per perturbation and "
                "change from clean (negative = worse); coverage < 1 means some images were too small to score")
    write_table(per_group, tables / "robustness_per_group",
                "Per test generator (Midjourney: in_overall = False, reported separately)")
    check = clean_check(overall, ROOT)
    write_table(check, tables / "clean_check",
                "Clean condition on the Part 3 sample vs Part 2 full test set (reproduction check)")
    plot_curves(overall, perts, results / "figures" / "degradation_curves.png")

    print(check.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if np.nanmax(np.abs(check["auroc_difference"])) > 0.02:
        print("WARNING: clean AUROC differs from Part 2 by more than 0.02; check the setup before "
              "trusting the perturbation results")
    if (overall["coverage"] < 1).any():
        low = overall.loc[overall["coverage"] < 1, ["head", "perturbation", "coverage"]]
        print("NOTE: some conditions could not score every image (too small for the pipeline):")
        print(low.to_string(index=False))
    print(f"wrote {tables.relative_to(ROOT)}/ and {(results / 'figures').relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
