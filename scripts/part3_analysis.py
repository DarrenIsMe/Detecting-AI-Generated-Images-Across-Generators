"""Part 3 analysis: what the perturbations do to the detector, image by image.

Reads what part3_run.py saved (per-image predictions, cached embeddings, the sample) and, for the
failure-case pictures only, the original images. No GPU is needed. Run it after part3_run.py with
the same --config. Outputs under the config's results_root:

    tables/confusion_counts.csv        TN / FP / FN / TP at threshold 0.5, every condition x perturbation
    tables/flips.csv                   per test group: fakes caught on the clean image but missed after the
                                       perturbation (and reals newly flagged), plus the mean change in P(fake)
    tables/embedding_shift.csv         distance between the real and fake centroids of the embeddings
    tables/failure_cases.csv           the images shown in the failure-case figures
    figures/confusion_matrices.png     clean next to the analysis conditions, every detector condition
    figures/score_distributions.png    P(fake) histograms for reals and fakes, raw vs matched_jpeg
    figures/failure_cases_<cond>.png   fakes caught when clean but missed after <cond>
    figures/failure_cases_clean.png    the most confident mistakes on clean images
    figures/tsne_<pipeline>_<format>.png  t-SNE of clean and perturbed embeddings in one shared map

"Overall" rows are the ones the tables pool or average (strict: ADM/GLIDE/SD15/VQDM/Wukong;
LOGO: every fold), so the counts here line up with robustness_overall.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config
from part2.data import iter_image_bytes, load_manifest
from part2.formats import decode_rgb
from part3_tables import ROOT, load_predictions

THRESHOLD = 0.5  # same rule as Part 2's group_metrics: fake if p_fake >= 0.5
# Real / fake colours: categorical slots distinct from the pipeline colours in degradation_curves.png;
# shape (circle / triangle) and labels carry the class too, so colour is never the only cue.
REAL, FAKE = "#1baf7a", "#6250d6"
INK, MUTED = "#0b0b0b", "#52514e"
SEQUENTIAL = ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"]  # light -> dark, one hue
DEFAULTS = {"pipeline": "A", "format": "raw", "conditions": ["jpeg_q50"], "n_failures": 12,
            "tsne_per_class": 500, "tsne_seed": 4001}
ROW_KEYS = ["pipeline", "format", "evaluation_group", "image_id"]


def plt_or_none():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib not installed: skipping figures (pip install matplotlib)")
        return None


def save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)  # free the figure's memory


# ---------------------------------------------------------------- confusion matrices


def confusion_counts(pred: pd.DataFrame) -> pd.DataFrame:
    """TN/FP/FN/TP at 0.5 on the scored overall rows (real = negative, fake = positive)."""
    rows = []
    for (pipe, fmt, pert), g in pred[pred["in_overall"] & pred["scored"]].groupby(
            ["pipeline", "format", "perturbation"], sort=False):
        fake, flagged = g["label"] == 1, g["p_fake"] >= THRESHOLD
        rows.append({"pipeline": pipe, "format": fmt, "perturbation": pert,
                     "true_real_pred_real": int((~fake & ~flagged).sum()),
                     "true_real_pred_fake": int((~fake & flagged).sum()),
                     "true_fake_pred_real": int((fake & ~flagged).sum()),
                     "true_fake_pred_fake": int((fake & flagged).sum())})
    return pd.DataFrame(rows)


def plot_confusion(counts: pd.DataFrame, conditions: list[str], path: Path, plt) -> None:
    """Grid: one row per detector condition, one column per perturbation. Cells are coloured by the
    share of each true class (rows sum to 100%) and labelled with that share and the count."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", SEQUENTIAL)
    combos = counts[["pipeline", "format"]].drop_duplicates().sort_values(["pipeline", "format"]).values.tolist()
    cols = ["clean"] + conditions
    fig, axes = plt.subplots(len(combos), len(cols), figsize=(2.6 * len(cols) + 0.6, 2.3 * len(combos)),
                             squeeze=False)
    for r, (pipe, fmt) in enumerate(combos):
        for c, pert in enumerate(cols):
            ax = axes[r][c]
            sel = counts[(counts["pipeline"] == pipe) & (counts["format"] == fmt) & (counts["perturbation"] == pert)]
            if sel.empty:
                ax.axis("off")
                continue
            s = sel.iloc[0]
            m = np.array([[s["true_real_pred_real"], s["true_real_pred_fake"]],
                          [s["true_fake_pred_real"], s["true_fake_pred_fake"]]], float)
            share = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
            ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{share[i, j]:.0%}\n({int(m[i, j])})", ha="center", va="center",
                            fontsize=8, color="white" if share[i, j] > 0.6 else INK)
            ax.set_xticks([0, 1], ["real", "fake"], fontsize=8)
            ax.set_yticks([0, 1], ["real", "fake"], fontsize=8)
            if r == 0:
                ax.set_title(pert, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{pipe} / {fmt}\ntrue class", fontsize=8)
            if r == len(combos) - 1:
                ax.set_xlabel("predicted (threshold 0.5)", fontsize=8)
    fig.suptitle("Confusion matrices (rows sum to 100% per true class)", fontsize=10)
    fig.tight_layout()
    save(fig, path)


# ---------------------------------------------------------------- flips and score shifts


def paired(pred: pd.DataFrame, pert: str) -> pd.DataFrame:
    """Each (detector, group, image) row scored both clean and under ``pert``, side by side."""
    cols = ROW_KEYS + ["label", "generator", "in_overall", "p_fake"]
    clean = pred[(pred["perturbation"] == "clean") & pred["scored"]][cols]
    after = pred[(pred["perturbation"] == pert) & pred["scored"]][ROW_KEYS + ["p_fake"]]
    out = clean.merge(after, on=ROW_KEYS, suffixes=("_clean", "_after"))
    out["change"] = out["p_fake_after"] - out["p_fake_clean"]
    return out


def flip_table(pred: pd.DataFrame, perts: list[str]) -> pd.DataFrame:
    """Per detector condition, perturbation and test group (plus an 'overall' row): how many
    decisions the perturbation flipped, and the mean change in P(fake) per class."""
    rows = []
    for pert in perts:
        if pert == "clean":
            continue
        p = paired(pred, pert)
        overall = p[p["in_overall"]].assign(evaluation_group="overall")
        for (pipe, fmt, group), g in pd.concat([p, overall]).groupby(
                ["pipeline", "format", "evaluation_group"], sort=False):
            fake, real = g[g["label"] == 1], g[g["label"] == 0]
            was_fake = lambda d: d["p_fake_clean"] >= THRESHOLD  # noqa: E731
            now_fake = lambda d: d["p_fake_after"] >= THRESHOLD  # noqa: E731
            rows.append({
                "pipeline": pipe, "format": fmt, "perturbation": pert, "evaluation_group": group,
                "n_fake": len(fake),
                "fake_caught_clean": int(was_fake(fake).sum()),
                "fake_missed_after": int((was_fake(fake) & ~now_fake(fake)).sum()),   # evaded
                "fake_caught_after_only": int((~was_fake(fake) & now_fake(fake)).sum()),
                "mean_p_fake_change_fakes": float(fake["change"].mean()) if len(fake) else np.nan,
                "n_real": len(real),
                "real_flagged_after": int((~was_fake(real) & now_fake(real)).sum()),  # new false alarms
                "real_cleared_after": int((was_fake(real) & ~now_fake(real)).sum()),
                "mean_p_fake_change_reals": float(real["change"].mean()) if len(real) else np.nan,
            })
    return pd.DataFrame(rows)


def plot_score_distributions(pred: pd.DataFrame, pipeline: str, conditions: list[str], path: Path, plt) -> None:
    """Histograms of P(fake) for reals and fakes: rows = formats of the focus pipeline, columns =
    clean and the analysis conditions. Each class is normalised to its own total, so the two
    shapes compare directly; the dotted line is the 0.5 decision threshold."""
    sel = pred[pred["in_overall"] & pred["scored"] & (pred["pipeline"] == pipeline)]
    formats = sorted(sel["format"].unique(), key=lambda f: f != "raw")  # raw first
    cols = ["clean"] + conditions
    bins = np.linspace(0, 1, 26)
    fig, axes = plt.subplots(len(formats), len(cols), figsize=(3.1 * len(cols), 2.5 * len(formats)),
                             squeeze=False, sharex=True)
    for r, fmt in enumerate(formats):
        for c, pert in enumerate(cols):
            ax = axes[r][c]
            g = sel[(sel["format"] == fmt) & (sel["perturbation"] == pert)]
            for label, colour, name in ((0, REAL, "real"), (1, FAKE, "fake")):
                values = g.loc[g["label"] == label, "p_fake"].to_numpy()
                if len(values):
                    weights = np.full(len(values), 1 / len(values))
                    ax.hist(values, bins=bins, weights=weights, color=colour, alpha=0.25)
                    ax.hist(values, bins=bins, weights=weights, histtype="step", color=colour,
                            linewidth=2, label=name)
            ax.axvline(THRESHOLD, color=MUTED, linewidth=0.8, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"{fmt}, {pert}", fontsize=9)
            if c == 0:
                ax.set_ylabel("share of class", fontsize=8)
            if r == len(formats) - 1:
                ax.set_xlabel("P(fake)", fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Pipeline {pipeline}: where the detector's scores fall, before and after perturbation",
                 fontsize=10)
    fig.tight_layout()
    save(fig, path)


# ---------------------------------------------------------------- failure cases


def pick_failures(pred: pd.DataFrame, pipeline: str, fmt: str, conditions: list[str],
                  n: int) -> dict[str, pd.DataFrame]:
    """Images for each failure figure, keyed by figure name.

    clean:  the most confident clean mistakes (half fakes scored lowest, half reals scored highest).
    <cond>: fakes the detector caught when clean but missed after <cond>, biggest score drop first.
            When none flipped, the fakes with the biggest drops are shown and the title says so.
    """
    focus = pred[(pred["pipeline"] == pipeline) & (pred["format"] == fmt) & pred["in_overall"]]
    picks: dict[str, pd.DataFrame] = {}
    clean = focus[(focus["perturbation"] == "clean") & focus["scored"]]
    wrong_fake = clean[(clean["label"] == 1) & (clean["p_fake"] < THRESHOLD)].nsmallest(n - n // 2, "p_fake")
    wrong_real = clean[(clean["label"] == 0) & (clean["p_fake"] >= THRESHOLD)].nlargest(n // 2, "p_fake")
    picks["clean"] = (pd.concat([wrong_fake, wrong_real]).drop_duplicates("image_id")
                      .assign(p_fake_clean=lambda d: d["p_fake"], p_fake_after=np.nan, flipped=False))
    for pert in conditions:
        p = paired(focus, pert)
        fakes = p[p["label"] == 1].assign(drop=lambda d: d["p_fake_clean"] - d["p_fake_after"])
        flipped = fakes[(fakes["p_fake_clean"] >= THRESHOLD) & (fakes["p_fake_after"] < THRESHOLD)]
        chosen = (flipped if len(flipped) else fakes).nlargest(n, "drop").drop_duplicates("image_id")
        picks[pert] = chosen.assign(flipped=len(flipped) > 0)
    return picks


def load_thumbnails(rows: pd.DataFrame, root: Path, size: int = 224) -> dict[str, object]:
    """Original images (as stored in the dataset) for the given rows, keyed by image_id."""
    by_id = {r["image_id"]: r for r in load_manifest(root)}
    wanted = [by_id[i] for i in rows["image_id"].unique() if i in by_id]
    thumbs = {}
    for row, data in iter_image_bytes(wanted, root):
        image = decode_rgb(data)
        image.thumbnail((size, size))
        thumbs[row["image_id"]] = image
    return thumbs


def plot_failures(chosen: pd.DataFrame, thumbs: dict, title: str, after: str | None, path: Path, plt) -> None:
    cols = 4
    n_rows = max(1, int(np.ceil(len(chosen) / cols)))
    fig, axes = plt.subplots(n_rows, cols, figsize=(2.6 * cols, 2.9 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, chosen.iterrows()):
        if r["image_id"] in thumbs:
            ax.imshow(thumbs[r["image_id"]])
        truth = "fake" if r["label"] == 1 else "real"
        score = (f"P(fake) {r['p_fake_clean']:.2f}" if after is None
                 else f"P(fake) {r['p_fake_clean']:.2f} → {r['p_fake_after']:.2f}")
        ax.set_title(f"{truth}: {r['generator']}\n{score}", fontsize=8, color=INK)
    if chosen.empty:
        axes[0][0].text(0, 0.5, "No images in this category.", fontsize=10, color=MUTED)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save(fig, path)


# ---------------------------------------------------------------- embeddings: t-SNE and centroid gap


def load_embeddings(cfg3: dict, pert: str, combo: tuple[str, str]) -> tuple[np.ndarray, np.ndarray] | None:
    path = ROOT / cfg3["local_root"] / "embeddings" / f"{cfg3['backbone']}__{pert}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        key = f"emb__{combo[0]}__{combo[1]}"
        return (archive["ids"], archive[key]) if key in archive else None


def overall_images(pred: pd.DataFrame) -> pd.DataFrame:
    """Each overall image once, with its label (LOGO reals appear in several folds)."""
    rows = pred[pred["in_overall"] & (pred["perturbation"] == "clean")]
    return rows[["image_id", "label"]].drop_duplicates("image_id").set_index("image_id")


def embedding_shift(cfg3: dict, pred: pd.DataFrame, perts: list[str]) -> pd.DataFrame:
    """Per (pipeline, format) and perturbation: distance between the mean real and mean fake
    embedding. A shrinking gap means the perturbation pushes the two classes together in CLIP's
    feature space, before any classifier is involved."""
    labels = overall_images(pred)["label"]
    combos = pred[["pipeline", "format"]].drop_duplicates().sort_values(["pipeline", "format"]).values
    rows = []
    for pipe, fmt in combos:
        gap_clean = np.nan
        for pert in perts:
            loaded = load_embeddings(cfg3, pert, (pipe, fmt))
            if loaded is None:
                continue
            ids, emb = loaded
            y = labels.reindex(ids.astype(str)).to_numpy()
            ok = ~np.isnan(emb).any(axis=1) & ~np.isnan(y)
            if not ((y[ok] == 0).any() and (y[ok] == 1).any()):
                continue
            gap = float(np.linalg.norm(emb[ok & (y == 0)].mean(axis=0) - emb[ok & (y == 1)].mean(axis=0)))
            if pert == "clean":
                gap_clean = gap
            rows.append({"pipeline": pipe, "format": fmt, "perturbation": pert, "n_images": int(ok.sum()),
                         "centroid_gap": gap, "centroid_gap_change": gap - gap_clean})
    return pd.DataFrame(rows)


def plot_tsne(cfg3: dict, pred: pd.DataFrame, combo: tuple[str, str], conditions: list[str],
              per_class: int, seed: int, path: Path, plt) -> None:
    """One t-SNE fitted on clean and perturbed embeddings together (so all panels share one map),
    then drawn per condition. Uses only images that could be embedded under every condition."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    cols = ["clean"] + conditions
    loaded = {pert: load_embeddings(cfg3, pert, combo) for pert in cols}
    if any(v is None for v in loaded.values()):
        print(f"t-SNE {combo}: embeddings missing for some of {cols}; skipped")
        return
    labels = overall_images(pred)["label"]
    ids = loaded["clean"][0].astype(str)
    ok = labels.reindex(ids).notna().to_numpy(copy=True)  # writable: narrowed per condition below
    for pert in cols:
        if not np.array_equal(loaded[pert][0].astype(str), ids):
            print(f"t-SNE {combo}: image lists differ between runs; skipped")
            return
        ok &= ~np.isnan(loaded[pert][1]).any(axis=1)
    y_all = labels.reindex(ids).to_numpy()
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([
        rng.choice(idx, min(per_class, len(idx)), replace=False)
        for idx in (np.flatnonzero(ok & (y_all == 0)), np.flatnonzero(ok & (y_all == 1))) if len(idx)])
    if len(chosen) < 10:
        print(f"t-SNE {combo}: too few images; skipped")
        return
    X = np.concatenate([loaded[pert][1][chosen] for pert in cols])
    X = PCA(n_components=min(50, X.shape[1], len(X) - 1), random_state=seed).fit_transform(X)
    xy = TSNE(n_components=2, init="pca", perplexity=min(30, (len(X) - 1) // 3),
              random_state=seed).fit_transform(X)
    y = y_all[chosen]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.8), sharex=True, sharey=True,
                             squeeze=False)
    for c, pert in enumerate(cols):
        ax, part = axes[0][c], xy[c * len(chosen):(c + 1) * len(chosen)]
        ax.scatter(*part[y == 0].T, s=9, marker="o", color=REAL, alpha=0.6, linewidths=0, label="real")
        ax.scatter(*part[y == 1].T, s=11, marker="^", color=FAKE, alpha=0.6, linewidths=0, label="fake")
        ax.set_title(pert, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0][0].legend(frameon=False, fontsize=8, markerscale=1.5)
    fig.suptitle(f"t-SNE of ViT-L/14 embeddings, pipeline {combo[0]} / {combo[1]} "
                 f"({len(chosen)} images per panel, one shared map)", fontsize=10)
    fig.tight_layout()
    save(fig, path)


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/part3.yaml")
    parser.add_argument("--no-images", action="store_true",
                        help="skip the failure-case pictures (they need the dataset's Parquet shards)")
    args = parser.parse_args(argv)

    cfg3 = load_config(ROOT, args.config)
    a = {**DEFAULTS, **(cfg3.get("analysis") or {})}
    pred, meta = load_predictions(cfg3)
    perts = [p["name"] for p in meta["perturbations"] if p["name"] in set(pred["perturbation"])]
    missing = [c for c in a["conditions"] if c not in perts]
    if missing:
        raise SystemExit(f"analysis.conditions {missing} have no predictions; choose from {perts}")
    if pred[(pred["pipeline"] == a["pipeline"]) & (pred["format"] == a["format"])].empty:
        raise SystemExit(f"no predictions for analysis.pipeline/format = {a['pipeline']}/{a['format']}")
    results = ROOT / cfg3["results_root"]
    tables, figures = results / "tables", results / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    plt = plt_or_none()

    counts = confusion_counts(pred)
    counts.to_csv(tables / "confusion_counts.csv", index=False)
    flips = flip_table(pred, perts)
    flips.to_csv(tables / "flips.csv", index=False, float_format="%.4f")
    shift = embedding_shift(cfg3, pred, perts)
    shift.to_csv(tables / "embedding_shift.csv", index=False, float_format="%.4f")

    picks = pick_failures(pred, a["pipeline"], a["format"], a["conditions"], int(a["n_failures"]))
    failure_rows = pd.concat([d.assign(figure=name) for name, d in picks.items()], ignore_index=True)
    keep = ["figure", "image_id", "evaluation_group", "generator", "label", "p_fake_clean", "p_fake_after", "flipped"]
    failure_rows[keep].to_csv(tables / "failure_cases.csv", index=False, float_format="%.4f")

    if plt is not None:
        plot_confusion(counts, a["conditions"], figures / "confusion_matrices.png", plt)
        plot_score_distributions(pred, a["pipeline"], a["conditions"], figures / "score_distributions.png", plt)
        if not args.no_images:
            try:
                thumbs = load_thumbnails(failure_rows, ROOT)
            except (FileNotFoundError, OSError) as error:
                print(f"failure-case pictures skipped (dataset not readable: {error}); "
                      "tables/failure_cases.csv still lists the images")
                thumbs = None
            if thumbs is not None:
                focus = f"pipeline {a['pipeline']} / {a['format']}"
                plot_failures(picks["clean"], thumbs,
                              f"Most confident mistakes on clean images ({focus})", None,
                              figures / "failure_cases_clean.png", plt)
                for pert in a["conditions"]:
                    chosen = picks[pert]
                    flipped = bool(len(chosen)) and bool(chosen["flipped"].iloc[0])
                    title = (f"Fakes caught when clean but missed after {pert} ({focus})" if flipped else
                             f"No fake flipped to real under {pert}; largest score drops shown ({focus})")
                    plot_failures(chosen, thumbs, title, pert, figures / f"failure_cases_{pert}.png", plt)
        plot_tsne(cfg3, pred, (a["pipeline"], a["format"]), a["conditions"], int(a["tsne_per_class"]),
                  int(a["tsne_seed"]), figures / f"tsne_{a['pipeline']}_{a['format']}.png", plt)

    overall = flips[flips["evaluation_group"] == "overall"]
    print(overall[["pipeline", "format", "perturbation", "n_fake", "fake_caught_clean", "fake_missed_after",
                   "real_flagged_after", "mean_p_fake_change_fakes"]].to_string(index=False))
    print(f"wrote {tables.relative_to(ROOT)}/ and {figures.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
