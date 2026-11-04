from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from common import load_config, project_root_from_script, read_csv, stable_id, write_csv, write_json


ROLE_PRIORITY = {"train": 0, "validation": 1, "test": 2, "separate_test": 2}


def grouped(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["duplicate_group_id"]].append(row)
    return list(groups.values())


def select_groups(groups: list[list[dict[str, str]]], target: int, seed: int, salt: str) -> tuple[list[dict[str, str]], list[list[dict[str, str]]]]:
    rng = random.Random(int(stable_id(seed, salt, length=15), 16))
    groups = list(groups)
    rng.shuffle(groups)
    selected: list[dict[str, str]] = []
    remainder: list[list[dict[str, str]]] = []
    for group in groups:
        if len(selected) < target:
            selected.extend(group)
        else:
            remainder.append(group)
    return selected, remainder


def assign_real_partitions(real_rows: list[dict[str, str]], targets: dict[str, int], seed: int, salt: str) -> dict[str, list[dict[str, str]]]:
    remaining = grouped(real_rows)
    assignments: dict[str, list[dict[str, str]]] = {}
    for role in ("train", "validation", "test"):
        assignments[role], remaining = select_groups(remaining, int(targets[role]), seed, f"{salt}:{role}")
    assignments["unused"] = [row for group in remaining for row in group]
    return assignments


def with_protocol(rows: list[dict[str, str]], protocol: str, fold: str, split: str, seed: int, evaluation_group: str = "") -> list[dict[str, str]]:
    result = []
    for row in rows:
        result.append({
            **row,
            "protocol": protocol,
            "fold": fold,
            "split": split,
            "evaluation_group": evaluation_group,
            "seed": seed,
        })
    return result


def assert_safe(rows: list[dict[str, str]], held_out: str | None = None) -> None:
    ids_by_split: dict[str, set[str]] = collections.defaultdict(set)
    groups_by_split: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        ids_by_split[row["split"]].add(row["image_id"])
        groups_by_split[row["split"]].add(row["duplicate_group_id"])
    roles = list(ids_by_split)
    for index, left in enumerate(roles):
        for right in roles[index + 1:]:
            if ids_by_split[left] & ids_by_split[right]:
                raise ValueError(f"Image overlap between {left} and {right}")
            if groups_by_split[left] & groups_by_split[right]:
                raise ValueError(f"Duplicate-group overlap between {left} and {right}")
    if held_out:
        leaked = [row for row in rows if row["generator"] == held_out and row["split"] in {"train", "validation"}]
        if leaked:
            raise ValueError(f"Held-out generator leaked into development: {held_out}")


def write_fold(root: Path, name: str, rows: list[dict[str, str]], held_out: str | None = None) -> dict:
    assert_safe(rows, held_out)
    path = root / "data" / "splits" / f"{name}.csv"
    fields = list(rows[0].keys()) if rows else []
    write_csv(path, rows, fields)
    counts = collections.Counter((row["split"], row["label_name"], row["generator"]) for row in rows)
    return {
        "name": name,
        "held_out_generator": held_out,
        "rows": len(rows),
        "counts": [
            {"split": key[0], "label": key[1], "generator": key[2], "count": value}
            for key, value in sorted(counts.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create generator-aware, duplicate-group-safe split manifests.")
    parser.add_argument("--project-root", type=Path, default=project_root_from_script(__file__))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = load_config(root, args.config)
    seed = args.seed if args.seed is not None else int(config["reproducibility"]["seed"])
    manifest_path = args.manifest or root / "data" / "manifests" / "master_manifest.csv"
    rows = [row for row in read_csv(manifest_path) if row["processing_status"] == "ok"]
    real_rows = [row for row in rows if row["label_name"] == "real"]
    fake_rows = [row for row in rows if row["label_name"] == "fake"]
    generators = sorted({row["generator"] for row in fake_rows})
    summaries = []

    real_logo = assign_real_partitions(real_rows, config["splits"]["logo_real_targets"], seed, "logo-real")
    for held_out in generators:
        fold_rows: list[dict[str, str]] = []
        for generator in generators:
            candidates = [row for row in fake_rows if row["generator"] == generator]
            if generator == held_out:
                fold_rows += with_protocol(candidates, "leave_one_generator_out", held_out, "test", seed, held_out)
            else:
                train = [row for row in candidates if row["source_split"] == "train"]
                validation = [row for row in candidates if row["source_split"] == "validation"]
                fold_rows += with_protocol(train, "leave_one_generator_out", held_out, "train", seed)
                fold_rows += with_protocol(validation, "leave_one_generator_out", held_out, "validation", seed)
        fold_rows += with_protocol(real_logo["train"], "leave_one_generator_out", held_out, "train", seed)
        fold_rows += with_protocol(real_logo["validation"], "leave_one_generator_out", held_out, "validation", seed)
        fold_rows += with_protocol(real_logo["test"], "leave_one_generator_out", held_out, "test", seed, held_out)

        # If an exact/perceptual group crosses roles because it spans generators, exclude the entire group.
        group_roles: dict[str, set[str]] = collections.defaultdict(set)
        for row in fold_rows:
            group_roles[row["duplicate_group_id"]].add(row["split"])
        unsafe_groups = {group for group, roles in group_roles.items() if len(roles) > 1}
        fold_rows = [row for row in fold_rows if row["duplicate_group_id"] not in unsafe_groups]
        summaries.append(write_fold(root, f"logo_{held_out.lower()}", fold_rows, held_out))

    strict = config["splits"]["strict_biggan_to_diffusion"]
    strict_generators = set(strict["test_generators"])
    separate_generators = set(strict["separate_test_generators"])
    strict_real_targets = {"train": 2000, "validation": 500, "test": 12500}
    strict_real = assign_real_partitions(real_rows, strict_real_targets, seed, "strict-real")
    separate_real = strict_real["unused"][:2500]
    protocol_rows: list[dict[str, str]] = []
    biggan = [row for row in fake_rows if row["generator"] == "BigGAN"]
    protocol_rows += with_protocol([row for row in biggan if row["source_split"] == "train"], "biggan_to_diffusion_strict", "strict", "train", seed)
    protocol_rows += with_protocol([row for row in biggan if row["source_split"] == "validation"], "biggan_to_diffusion_strict", "strict", "validation", seed)
    protocol_rows += with_protocol(strict_real["train"], "biggan_to_diffusion_strict", "strict", "train", seed)
    protocol_rows += with_protocol(strict_real["validation"], "biggan_to_diffusion_strict", "strict", "validation", seed)

    strict_test_real = list(strict_real["test"])
    cursor = 0
    for generator in sorted(strict_generators):
        generated = [row for row in fake_rows if row["generator"] == generator]
        protocol_rows += with_protocol(generated, "biggan_to_diffusion_strict", "strict", "test", seed, generator)
        block = strict_test_real[cursor: cursor + len(generated)]
        cursor += len(generated)
        protocol_rows += with_protocol(block, "biggan_to_diffusion_strict", "strict", "test", seed, generator)
    summaries.append(write_fold(root, "biggan_to_diffusion_strict", protocol_rows))

    for generator in sorted(separate_generators):
        generated = [row for row in fake_rows if row["generator"] == generator]
        separate_rows = with_protocol(generated, "biggan_to_midjourney_separate", generator, "test", seed, generator)
        separate_rows += with_protocol(separate_real[: len(generated)], "biggan_to_midjourney_separate", generator, "test", seed, generator)
        # This file is evaluation-only; train/validation settings are referenced from the strict protocol.
        summaries.append(write_fold(root, f"biggan_to_{generator.lower()}_separate_test", separate_rows))

    summary = {"seed": seed, "folds": summaries}
    write_json(root / "results" / "dataset_statistics" / "split_statistics.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
