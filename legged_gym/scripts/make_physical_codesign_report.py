#!/usr/bin/env python3
"""Create paper-ready tables and figures from a multi-seed aggregate."""

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = (
    ("H_clear90_m", "Clearance", "m", 3),
    ("theta90_midpoint_deg", "Slope", "deg", 2),
    ("M90_kg", "Payload", "kg", 1),
    ("V90_mps", "Speed", "m/s", 2),
    ("A_jump_m2", "Jump area", "m^2", 5),
)
NORMALIZED_FIELDS = (
    ("H_clear90_m_normalized", "Clearance"),
    ("theta90_deg_normalized", "Slope"),
    ("M90_kg_normalized", "Payload"),
    ("V90_mps_normalized", "Speed"),
    ("A_jump_m2_normalized", "Jump area"),
)
CENSOR_BY_NORMALIZED_FIELD = {
    "H_clear90_m_normalized": "H_clear90_m",
    "theta90_deg_normalized": "theta90_deg",
    "M90_kg_normalized": "M90_kg",
    "V90_mps_normalized": "V90_mps",
    "A_jump_m2_normalized": "A_jump_m2",
}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--title",
        default="Five-metric physical co-design results",
    )
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def as_float(row, field):
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} for {row.get('label')}")
    return value


def metric_cell(row, field, digits):
    mean = as_float(row, f"{field}_mean")
    std = as_float(row, f"{field}_std")
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def censor_cell(row, field):
    left = int(row.get(f"{field}_censor_left_count", 0))
    right = int(row.get(f"{field}_censor_right_count", 0))
    flags = []
    if left:
        flags.append(f"L{left}")
    if right:
        flags.append(f"R{right}")
    return "/".join(flags) if flags else "-"


def load_per_seed_scores(summary, labels, aggregate_dir):
    scores = {label: [] for label in labels}
    seeds = []
    for item in summary.get("inputs", []):
        seed = int(item["seed"])
        seed_dir = Path(item["aggregate_dir"])
        if not seed_dir.is_dir():
            relocated = aggregate_dir.parent / f"seed_{seed}"
            if not relocated.is_dir():
                raise FileNotFoundError(
                    f"missing per-seed aggregate for {seed}: "
                    f"{seed_dir} or {relocated}"
                )
            seed_dir = relocated
        table = read_csv(seed_dir / "five_metrics.csv")
        by_label = {row["label"]: row for row in table}
        seeds.append(seed)
        for label in labels:
            scores[label].append(
                (seed, as_float(by_label[label], "equal_weight_score"))
            )
    return seeds, scores


def write_markdown(
    path,
    title,
    rows,
    sensitivity,
    summary,
    aggregate_dir,
):
    labels = [row["label"] for row in rows]
    seeds, per_seed_scores = load_per_seed_scores(
        summary, labels, aggregate_dir
    )
    nominal = next(row for row in rows if row["label"] == "nominal")

    lines = [
        f"# {title}",
        "",
        (
            f"Independent repetitions: {len(seeds)} "
            f"({', '.join(str(seed) for seed in seeds)}). "
            "Values are mean +/- sample standard deviation."
        ),
        "",
        "## Physical metrics",
        "",
        (
            "| Rank | Design `(thigh, shank, motor)` | "
            "`H_clear,90` (m) | `theta90` midpoint (deg) | "
            "`M90` (kg) | `V90` (m/s) | `A_jump` (m^2) | "
            "Equal-weight score |"
        ),
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        design = (
            f"({float(row['thigh_scale']):g}, "
            f"{float(row['shank_scale']):g}, "
            f"{float(row['motor_scale']):g})"
        )
        cells = [
            metric_cell(row, field, digits)
            for field, _, _, digits in METRICS
        ]
        score = metric_cell(row, "equal_weight_score", 3)
        lines.append(
            f"| {row['rank_mean_equal_weight']} | "
            f"`{row['label']}` {design} | "
            + " | ".join(cells)
            + f" | {score} |"
        )

    lines.extend(
        [
            "",
            "## Censoring",
            "",
            "L means capability is below the tested lower boundary (for "
            "jump, no grid cell is reliable); R means the full tested "
            "continuous range or grid passed. Counts are across seeds.",
            "",
            "| Design | Clearance | Slope | Payload | Speed | Jump |",
            "|:---|---:|---:|---:|---:|---:|",
        ]
    )
    censor_fields = (
        "H_clear90_m",
        "theta90_deg",
        "M90_kg",
        "V90_mps",
        "A_jump_m2",
    )
    for row in rows:
        cells = [censor_cell(row, field) for field in censor_fields]
        lines.append(f"| `{row['label']}` | " + " | ".join(cells) + " |")

    sensitivity_by_label = {row["label"]: row for row in sensitivity}
    rank_fields = [
        field
        for field in sensitivity[0]
        if field.startswith("rank_mean_")
    ]
    lines.extend(
        [
            "",
            "## Robustness",
            "",
            "| Design | Seeds above nominal | Paired score difference "
            "range | Weight-sensitivity rank range | First-place settings |",
            "|:---|---:|:---|:---|---:|",
        ]
    )
    nominal_scores = dict(per_seed_scores["nominal"])
    for row in rows:
        label = row["label"]
        diffs = [
            score - nominal_scores[seed]
            for seed, score in per_seed_scores[label]
        ]
        wins = sum(diff > 1.0e-12 for diff in diffs)
        sensitivity_row = sensitivity_by_label[label]
        ranks = [int(sensitivity_row[field]) for field in rank_fields]
        lines.append(
            f"| `{label}` | {wins}/{len(diffs)} | "
            f"[{min(diffs):+.3f}, {max(diffs):+.3f}] | "
            f"{min(ranks)}--{max(ranks)} | {sum(rank == 1 for rank in ranks)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"The top mean equal-weight score is "
                f"`{rows[0]['label']}` ({as_float(rows[0], 'equal_weight_score_mean'):.3f}). "
                f"The nominal reference is "
                f"{as_float(nominal, 'equal_weight_score_mean'):.3f}."
            ),
            "",
            "The slope component uses `1 + (theta - theta_nominal) / 12 deg` "
            "and jump uses `1 + (A - A_nominal) / 0.0224 m^2`. Clearance, "
            "payload, and speed use within-seed nominal ratios. The fixed "
            "spans avoid ill-conditioned division when a nominal threshold "
            "is at or near zero.",
            "",
            "The jump metric is the reliable achieved H-D area inside the "
            "fixed in-training command envelope, not an unconstrained "
            "mechanical jump limit.",
            "",
            "The conclusion is limited to the tested discrete designs and "
            "the frozen adaptation budget. It is not a continuous-space "
            "global optimum claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def make_plots(out_dir, rows, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [row["label"] for row in rows]
    x = np.arange(len(NORMALIZED_FIELDS), dtype=float)
    width = 0.82 / len(rows)
    fig, ax = plt.subplots(figsize=(10.2, 4.8), constrained_layout=True)
    for index, row in enumerate(rows):
        values = [
            as_float(row, f"{field}_mean")
            for field, _ in NORMALIZED_FIELDS
        ]
        errors = [
            as_float(row, f"{field}_std")
            for field, _ in NORMALIZED_FIELDS
        ]
        error_bars = np.asarray(
            (
                [min(value, error) for value, error in zip(values, errors)],
                errors,
            )
        )
        offset = (index - 0.5 * (len(rows) - 1)) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            yerr=error_bars,
            capsize=2.5,
            color=COLORS[index % len(COLORS)],
            label=row["label"],
            linewidth=0,
        )
        for metric_index, (field, _) in enumerate(NORMALIZED_FIELDS):
            censor_field = CENSOR_BY_NORMALIZED_FIELD[field]
            left = int(row.get(f"{censor_field}_censor_left_count", 0))
            right = int(row.get(f"{censor_field}_censor_right_count", 0))
            flags = []
            if left:
                flags.append(f"L{left}")
            if right:
                flags.append(f"R{right}")
            if flags:
                ax.text(
                    x[metric_index] + offset,
                    values[metric_index] + errors[metric_index] + 0.03,
                    "/".join(flags),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_xticks(x, [label for _, label in NORMALIZED_FIELDS])
    ax.set_ylabel("Within-seed normalized capability")
    ax.set_title(title)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncol=min(3, len(rows)), frameon=False, fontsize=8)
    fig.savefig(out_dir / "normalized_capabilities.png", dpi=300)
    plt.close(fig)

    ordered = list(reversed(rows))
    fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    y = np.arange(len(ordered))
    means = [as_float(row, "equal_weight_score_mean") for row in ordered]
    errors = [as_float(row, "equal_weight_score_std") for row in ordered]
    colors = [
        COLORS[labels.index(row["label"]) % len(COLORS)]
        for row in ordered
    ]
    ax.barh(y, means, xerr=errors, capsize=3, color=colors)
    ax.axvline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_yticks(y, [row["label"] for row in ordered])
    ax.set_xlabel("Five-metric equal-weight score")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.savefig(out_dir / "composite_scores.png", dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.aggregate_dir / "multiseed_five_metrics.csv"
    sensitivity_path = (
        args.aggregate_dir / "multiseed_all_metric_weight_sensitivity.csv"
    )
    summary_path = args.aggregate_dir / "summary.json"
    rows = read_csv(metrics_path)
    sensitivity = read_csv(sensitivity_path)
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    write_markdown(
        args.out / "PAPER_RESULTS.md",
        args.title,
        rows,
        sensitivity,
        summary,
        args.aggregate_dir,
    )
    make_plots(args.out, rows, args.title)
    print(f"[report] rows={len(rows)} out={args.out}")


if __name__ == "__main__":
    main()
