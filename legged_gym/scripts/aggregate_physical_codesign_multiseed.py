#!/usr/bin/env python3
"""Aggregate per-seed physical co-design tables without pooling trials."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


RAW_FIELDS = (
    "H_clear90_m",
    "theta90_deg",
    "theta90_midpoint_deg",
    "M90_kg",
    "V90_mps",
    "A_jump_m2",
)
NORMALIZED_FIELDS = (
    "H_clear90_m_normalized",
    "theta90_deg_normalized",
    "M90_kg_normalized",
    "V90_mps_normalized",
    "A_jump_m2_normalized",
)
CENSOR_FIELDS = (
    "H_clear90_m_censor",
    "theta90_deg_censor",
    "M90_kg_censor",
    "V90_mps_censor",
    "A_jump_m2_censor",
)
SENSITIVITY_FIELDS = (
    "score_jump_10pct",
    "score_jump_20pct",
    "score_jump_30pct",
)
TASKS = ("clearance", "slope", "payload", "speed", "jump")
ALL_SENSITIVITY_FIELDS = tuple(
    f"score_{task}_{weight}pct"
    for task in TASKS
    for weight in (10, 20, 30)
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="SEED=AGGREGATE_DIR",
    )
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_inputs(values):
    parsed = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"input must be SEED=AGGREGATE_DIR: {value}")
        seed_raw, path_raw = value.split("=", 1)
        seed = int(seed_raw)
        if seed in seen:
            raise ValueError(f"duplicate seed: {seed}")
        seen.add(seed)
        path = Path(path_raw)
        parsed.append((seed, path))
    return sorted(parsed)


def finite_float(row, field, path):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path}: non-finite {field}")
    return value


def mean_std(values):
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_candidates(path):
    rows = read_csv(path)
    labels = [row["label"] for row in rows]
    if len(labels) != len(set(labels)):
        raise ValueError("candidate labels must be unique")
    return rows


def load_seed_tables(inputs, candidates):
    expected = {row["label"]: row for row in candidates}
    metric_tables = {}
    sensitivity_tables = {}
    all_sensitivity_tables = {}
    for seed, aggregate_dir in inputs:
        metrics_path = aggregate_dir / "five_metrics.csv"
        sensitivity_path = aggregate_dir / "weight_sensitivity.csv"
        all_sensitivity_path = (
            aggregate_dir / "all_metric_weight_sensitivity.csv"
        )
        metric_rows = {
            row["label"]: row for row in read_csv(metrics_path)
        }
        sensitivity_rows = {
            row["label"]: row for row in read_csv(sensitivity_path)
        }
        all_sensitivity_rows = {
            row["label"]: row for row in read_csv(all_sensitivity_path)
        }
        if set(metric_rows) != set(expected):
            raise ValueError(
                f"{metrics_path}: labels do not match candidate shortlist"
            )
        if set(sensitivity_rows) != set(expected):
            raise ValueError(
                f"{sensitivity_path}: labels do not match candidate shortlist"
            )
        if set(all_sensitivity_rows) != set(expected):
            raise ValueError(
                f"{all_sensitivity_path}: labels do not match shortlist"
            )
        for label, candidate in expected.items():
            row = metric_rows[label]
            if row.get("status") != "complete":
                raise ValueError(f"{metrics_path}: {label} is incomplete")
            for field in ("thigh_scale", "shank_scale", "motor_scale"):
                if abs(float(row[field]) - float(candidate[field])) > 5.0e-4:
                    raise ValueError(
                        f"{metrics_path}: {label} design mismatch in {field}"
                    )
        metric_tables[seed] = metric_rows
        sensitivity_tables[seed] = sensitivity_rows
        all_sensitivity_tables[seed] = all_sensitivity_rows
    return metric_tables, sensitivity_tables, all_sensitivity_tables


def per_seed_ranks(tables, field):
    result = {}
    for seed, rows in tables.items():
        ordered = sorted(
            rows,
            key=lambda label: finite_float(
                rows[label], field, f"seed {seed}"
            ),
            reverse=True,
        )
        result[seed] = {
            label: rank for rank, label in enumerate(ordered, start=1)
        }
    return result


def aggregate_metrics(inputs, candidates, tables):
    seeds = [seed for seed, _ in inputs]
    score_ranks = per_seed_ranks(tables, "equal_weight_score")
    rows = []
    for candidate in candidates:
        label = candidate["label"]
        row = {
            "label": label,
            "thigh_scale": candidate["thigh_scale"],
            "shank_scale": candidate["shank_scale"],
            "motor_scale": candidate["motor_scale"],
            "role": candidate.get("role", ""),
            "seed_count": len(seeds),
            "seeds": ";".join(str(seed) for seed in seeds),
            "rank_by_seed": ";".join(
                f"{seed}:{score_ranks[seed][label]}" for seed in seeds
            ),
        }
        for field in RAW_FIELDS + NORMALIZED_FIELDS + (
            "equal_weight_score",
        ):
            values = [
                finite_float(tables[seed][label], field, f"seed {seed}")
                for seed in seeds
            ]
            mean, std = mean_std(values)
            row[f"{field}_mean"] = mean
            row[f"{field}_std"] = std
        for field in CENSOR_FIELDS:
            statuses = [tables[seed][label][field] for seed in seeds]
            row[f"{field}_by_seed"] = ";".join(
                f"{seed}:{status}"
                for seed, status in zip(seeds, statuses)
            )
            row[f"{field}_left_count"] = statuses.count("left")
            row[f"{field}_right_count"] = statuses.count("right")
        rows.append(row)

    rows.sort(key=lambda row: row["equal_weight_score_mean"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank_mean_equal_weight"] = rank
    return rows


def aggregate_sensitivity(inputs, candidates, tables):
    seeds = [seed for seed, _ in inputs]
    rows = []
    for candidate in candidates:
        label = candidate["label"]
        row = {
            "label": label,
            "thigh_scale": candidate["thigh_scale"],
            "shank_scale": candidate["shank_scale"],
            "motor_scale": candidate["motor_scale"],
        }
        for field in SENSITIVITY_FIELDS:
            values = [
                finite_float(tables[seed][label], field, f"seed {seed}")
                for seed in seeds
            ]
            mean, std = mean_std(values)
            row[f"{field}_mean"] = mean
            row[f"{field}_std"] = std
        rows.append(row)
    for field in SENSITIVITY_FIELDS:
        ordered = sorted(
            rows, key=lambda row: row[f"{field}_mean"], reverse=True
        )
        rank_field = field.replace("score_", "rank_mean_")
        for rank, row in enumerate(ordered, start=1):
            row[rank_field] = rank
    return rows


def aggregate_all_sensitivity(inputs, candidates, tables):
    seeds = [seed for seed, _ in inputs]
    rows = []
    for candidate in candidates:
        label = candidate["label"]
        row = {
            "label": label,
            "thigh_scale": candidate["thigh_scale"],
            "shank_scale": candidate["shank_scale"],
            "motor_scale": candidate["motor_scale"],
        }
        for field in ALL_SENSITIVITY_FIELDS:
            values = [
                finite_float(tables[seed][label], field, f"seed {seed}")
                for seed in seeds
            ]
            mean, std = mean_std(values)
            row[f"{field}_mean"] = mean
            row[f"{field}_std"] = std
        rows.append(row)
    for field in ALL_SENSITIVITY_FIELDS:
        ordered = sorted(
            rows, key=lambda row: row[f"{field}_mean"], reverse=True
        )
        rank_field = field.replace("score_", "rank_mean_")
        for rank, row in enumerate(ordered, start=1):
            row[rank_field] = rank
    return rows


def main():
    args = parse_args()
    inputs = parse_inputs(args.input)
    candidates = load_candidates(args.candidates)
    (
        metric_tables,
        sensitivity_tables,
        all_sensitivity_tables,
    ) = load_seed_tables(inputs, candidates)
    metric_rows = aggregate_metrics(inputs, candidates, metric_tables)
    sensitivity_rows = aggregate_sensitivity(
        inputs, candidates, sensitivity_tables
    )
    all_sensitivity_rows = aggregate_all_sensitivity(
        inputs, candidates, all_sensitivity_tables
    )
    args.out.mkdir(parents=True, exist_ok=True)

    identity = [
        "rank_mean_equal_weight",
        "label",
        "thigh_scale",
        "shank_scale",
        "motor_scale",
        "role",
        "seed_count",
        "seeds",
        "rank_by_seed",
    ]
    value_columns = []
    for field in RAW_FIELDS + NORMALIZED_FIELDS + (
        "equal_weight_score",
    ):
        value_columns.extend((f"{field}_mean", f"{field}_std"))
    censor_columns = []
    for field in CENSOR_FIELDS:
        censor_columns.extend(
            (
                f"{field}_by_seed",
                f"{field}_left_count",
                f"{field}_right_count",
            )
        )
    write_csv(
        args.out / "multiseed_five_metrics.csv",
        identity + value_columns + censor_columns,
        metric_rows,
    )

    sensitivity_columns = [
        "label",
        "thigh_scale",
        "shank_scale",
        "motor_scale",
    ]
    for field in SENSITIVITY_FIELDS:
        sensitivity_columns.extend(
            (
                f"{field}_mean",
                f"{field}_std",
                field.replace("score_", "rank_mean_"),
            )
        )
    write_csv(
        args.out / "multiseed_weight_sensitivity.csv",
        sensitivity_columns,
        sensitivity_rows,
    )

    all_sensitivity_columns = [
        "label",
        "thigh_scale",
        "shank_scale",
        "motor_scale",
    ]
    for field in ALL_SENSITIVITY_FIELDS:
        all_sensitivity_columns.extend(
            (
                f"{field}_mean",
                f"{field}_std",
                field.replace("score_", "rank_mean_"),
            )
        )
    write_csv(
        args.out / "multiseed_all_metric_weight_sensitivity.csv",
        all_sensitivity_columns,
        all_sensitivity_rows,
    )

    summary = {
        "inputs": [
            {"seed": seed, "aggregate_dir": str(path.resolve())}
            for seed, path in inputs
        ],
        "candidate_count": len(candidates),
        "seed_count": len(inputs),
        "ranking": metric_rows,
        "weight_sensitivity": sensitivity_rows,
        "all_metric_weight_sensitivity": all_sensitivity_rows,
    }
    with (args.out / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    print(
        f"[aggregate-multiseed] candidates={len(candidates)} "
        f"seeds={len(inputs)} out={args.out}"
    )


if __name__ == "__main__":
    main()
