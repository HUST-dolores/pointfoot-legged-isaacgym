"""Build the final five-metric co-design table with frozen-policy jump results.

The candidate CSV must contain:
  label, thigh, shank, motor, obstacle, slope, load, accel, jump_summary

Each jump_summary points to the JSON sidecar written by measure_jumpfwd.py. All
five capability dimensions are divided by the nominal candidate before equal-
weight aggregation. Raw scores and jump diagnostics are retained.
"""

import argparse
import csv
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np


SCENARIO_METRICS = ("obstacle", "slope", "load", "accel")
ALL_METRICS = SCENARIO_METRICS + ("jump_return",)


def read_candidates(path):
    base_dir = os.path.dirname(os.path.abspath(path))
    rows = []
    with open(path, newline="", encoding="utf-8") as csv_file:
        for source in csv.DictReader(csv_file):
            summary_path = source["jump_summary"]
            if not os.path.isabs(summary_path):
                summary_path = os.path.join(base_dir, summary_path)
            with open(summary_path, encoding="utf-8") as summary_file:
                jump = json.load(summary_file)
            row = {
                "label": source["label"],
                "thigh": float(source["thigh"]),
                "shank": float(source["shank"]),
                "motor": float(source["motor"]),
                "jump_summary": os.path.abspath(summary_path),
                "jump_return": float(jump["jump_return_mean_all_terminal"]),
            }
            row.update({name: float(source[name]) for name in SCENARIO_METRICS})
            row.update({
                "trigger_count": int(jump["trigger_count"]),
                "terminal_rate": float(jump["terminal_rate"]),
                "completion_rate": float(jump["completion_rate"]),
                "single_jump_success_rate": float(jump["single_jump_success_rate_per_trigger"]),
                "height_mean_m": float(jump["height_mean_single_m"]),
                "distance_mean_m": float(jump["distance_mean_single_m"]),
                "height_mae_m": float(jump["height_mae_single_m"]),
                "distance_mae_m": float(jump["distance_mae_single_m"]),
                "takeoff_quality": float(jump["takeoff_quality_mean_single"]),
                "lateral_abs_p95_m": float(jump["lateral_abs_p95_single_m"]),
                "max_roll_p95_deg": float(jump["max_roll_p95_single_deg"]),
                "torque_sat_fraction": float(jump["torque_sat_fraction_mean"]),
                "peak_torque_ratio_p95": float(jump["peak_torque_ratio_p95"]),
                "mechanical_energy_mean_j": float(jump["mechanical_energy_mean_j"]),
            })
            rows.append(row)
    return rows


def normalize(rows, nominal_label):
    nominal_matches = [row for row in rows if row["label"] == nominal_label]
    if len(nominal_matches) != 1:
        raise ValueError(f"expected one nominal row labelled {nominal_label!r}")
    nominal = nominal_matches[0]
    for metric in ALL_METRICS:
        value = nominal[metric]
        if not math.isfinite(value) or abs(value) < 1.0e-8:
            raise ValueError(f"nominal {metric}={value} cannot be used for ratio normalization")

    for row in rows:
        for metric in ALL_METRICS:
            row[f"{metric}_norm"] = row[metric] / nominal[metric]
        row["five_metric_index"] = float(np.mean([row[f"{name}_norm"] for name in ALL_METRICS]))
        row["five_metric_change_percent"] = 100.0 * (row["five_metric_index"] - 1.0)
    ranked = sorted(rows, key=lambda row: row["five_metric_index"], reverse=True)
    for rank, row in enumerate(ranked, 1):
        row["five_metric_rank"] = rank
    return nominal


def add_weight_sensitivity(rows, jump_weights=(0.1, 0.2, 0.3)):
    for jump_weight in jump_weights:
        key = f"score_jump_weight_{int(round(100 * jump_weight))}"
        scenario_weight = (1.0 - jump_weight) / len(SCENARIO_METRICS)
        for row in rows:
            row[key] = (
                scenario_weight * sum(row[f"{name}_norm"] for name in SCENARIO_METRICS)
                + jump_weight * row["jump_return_norm"]
            )
        for rank, row in enumerate(sorted(rows, key=lambda item: item[key], reverse=True), 1):
            row[key.replace("score_", "rank_")] = rank


def write_csv(path, rows):
    fieldnames = [
        "five_metric_rank", "label", "thigh", "shank", "motor",
        *SCENARIO_METRICS, "jump_return",
        *[f"{name}_norm" for name in ALL_METRICS],
        "five_metric_index", "five_metric_change_percent",
        "trigger_count", "terminal_rate", "completion_rate", "single_jump_success_rate",
        "height_mean_m", "distance_mean_m", "height_mae_m", "distance_mae_m",
        "takeoff_quality", "lateral_abs_p95_m", "max_roll_p95_deg",
        "torque_sat_fraction", "peak_torque_ratio_p95", "mechanical_energy_mean_j",
        "score_jump_weight_10", "rank_jump_weight_10",
        "score_jump_weight_20", "rank_jump_weight_20",
        "score_jump_weight_30", "rank_jump_weight_30",
        "jump_summary",
    ]
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["five_metric_rank"]))


def write_sensitivity_csv(path, rows):
    fieldnames = [
        "label", "thigh", "shank", "motor",
        "score_jump_weight_10", "rank_jump_weight_10",
        "score_jump_weight_20", "rank_jump_weight_20",
        "score_jump_weight_30", "rank_jump_weight_30",
    ]
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["rank_jump_weight_20"]))


def plot_normalized(path, rows):
    ordered = sorted(rows, key=lambda row: row["five_metric_rank"])
    values = np.asarray([[row[f"{name}_norm"] for name in ALL_METRICS] for row in ordered])
    x = np.arange(len(ordered))
    width = 0.15
    colors = ("#276FBF", "#41A66A", "#E2A72E", "#A45A9B", "#D1495B")
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    for index, (metric, color) in enumerate(zip(ALL_METRICS, colors)):
        offset = (index - (len(ALL_METRICS) - 1) / 2.0) * width
        ax.bar(x + offset, values[:, index], width, label=metric, color=color)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in ordered], rotation=18, ha="right")
    ax.set_ylabel("Nominal-normalized capability")
    ax.set_title("Final five-metric co-design pilot")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_weight_sensitivity(path, rows):
    weights = np.asarray([10, 20, 30])
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for row in sorted(rows, key=lambda item: item["five_metric_rank"]):
        scores = [row[f"score_jump_weight_{weight}"] for weight in weights]
        ax.plot(weights, scores, marker="o", linewidth=1.6, label=row["label"])
    ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xticks(weights)
    ax.set_xlabel("Jump metric weight (%)")
    ax.set_ylabel("Nominal-normalized weighted score")
    ax.set_title("Five-metric weight sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_methodology(path, rows, nominal):
    ranked = sorted(rows, key=lambda row: row["five_metric_rank"])
    lines = [
        "# Five-Metric Co-Design Pilot",
        "",
        "## Protocol",
        "",
        "- Final dimensions: obstacle, slope, load, acceleration, and jump.",
        "- The first four dimensions use mean episode reward from the existing fixed-difficulty evaluation after the same +400 per-design adaptation budget.",
        "- Jump dimension: mean trigger-to-terminal reward over first-touchdown completions, no-takeoff timeouts, and in-cycle episode resets.",
        "- Controller: one frozen jump policy for every design; identical reward, command grid, seed, rollout length, and nominal disturbance-free environment.",
        "- Normalization: each raw dimension is divided by the nominal design value. The preliminary aggregate is the equal-weight mean of the five ratios.",
        "- Weight sensitivity: jump receives 10%, 20%, or 30%; the remaining weight is divided equally among the other four dimensions.",
        "- Interpretation: this is zero-shot controller-design compatibility screening, not the design-specific performance ceiling.",
        "- Budget caveat: the first four dimensions are adapted per design, while jump is zero-shot; the pilot is not yet a fully budget-aligned five-task comparison.",
        "",
        "## Ranking",
        "",
        "| Rank | Design | (thigh, shank, motor) | Obstacle | Slope | Load | Accel | Jump return | Five-metric index |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['five_metric_rank']} | {row['label']} | "
            f"({row['thigh']:.4f}, {row['shank']:.4f}, {row['motor']:.4f}) | "
            f"{row['obstacle']:.2f} | {row['slope']:.2f} | {row['load']:.2f} | "
            f"{row['accel']:.2f} | {row['jump_return']:.3f} | {row['five_metric_index']:.3f} |"
        )
    lines.extend([
        "",
        "## Nominal Reference",
        "",
        f"Nominal raw vector: obstacle={nominal['obstacle']:.3f}, slope={nominal['slope']:.3f}, "
        f"load={nominal['load']:.3f}, accel={nominal['accel']:.3f}, "
        f"jump_return={nominal['jump_return']:.3f}.",
        "",
        "## Weight Sensitivity",
        "",
        "| Design | Jump 10% score/rank | Jump 20% score/rank | Jump 30% score/rank |",
        "|---|---:|---:|---:|",
    ])
    for row in ranked:
        lines.append(
            f"| {row['label']} | {row['score_jump_weight_10']:.3f}/{row['rank_jump_weight_10']} | "
            f"{row['score_jump_weight_20']:.3f}/{row['rank_jump_weight_20']} | "
            f"{row['score_jump_weight_30']:.3f}/{row['rank_jump_weight_30']} |"
        )
    lines.extend([
        "",
        "## Paper Boundary",
        "",
        "Use this pilot as workflow evidence and an initial optimization indicator. For a final mechanical-capability claim, retrain or fine-tune every shortlisted design with the same seed set and update budget, then repeat the same held-out command-grid evaluation.",
    ])
    with open(path, "w", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nominal-label", default="nominal")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = read_candidates(args.candidates)
    nominal = normalize(rows, args.nominal_label)
    add_weight_sensitivity(rows)
    write_csv(os.path.join(args.output_dir, "five_metric_final_table.csv"), rows)
    write_sensitivity_csv(os.path.join(args.output_dir, "weight_sensitivity.csv"), rows)
    plot_normalized(os.path.join(args.output_dir, "five_metric_normalized.png"), rows)
    plot_weight_sensitivity(os.path.join(args.output_dir, "weight_sensitivity.png"), rows)
    write_methodology(os.path.join(args.output_dir, "README.md"), rows, nominal)
    best = max(rows, key=lambda row: row["five_metric_index"])
    print(
        f"[jump-codesign] candidates={len(rows)} best={best['label']} "
        f"index={best['five_metric_index']:.4f} "
        f"delta={best['five_metric_change_percent']:+.1f}%"
    )


if __name__ == "__main__":
    main()
