#!/usr/bin/env python3
"""Analyze whole-body equivalent inverted-pendulum angles from play MAT files."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat


COLORS = {
    "Model-guided": "#E69F00",
    "RL-only": "#D55E00",
}

METRICS = (
    ("system_ip_angle_rms_deg", "Whole-system IP angle RMS (deg)"),
    ("system_ip_angle_mae_deg", "Whole-system IP angle MAE (deg)"),
    ("system_ip_angle_p95_abs_deg", "Whole-system IP angle p95 abs. (deg)"),
    ("robot_ip_angle_rms_deg", "Robot-only IP angle RMS (deg)"),
    ("base_pitch_rms_deg", "Base pitch RMS (deg)"),
    ("system_com_offset_rms_cm", "Whole-system CoM offset RMS (cm)"),
)


def parse_run_spec(value):
    parts = value.split("=", 3)
    if len(parts) != 4 or not all(parts):
        raise argparse.ArgumentTypeError(
            "run must be LABEL=METHOD=PAIR_ID=/path/to/play_data.mat"
        )
    label, method, pair_id, path = parts
    return {"label": label, "method": method, "pair_id": pair_id, "path": Path(path)}


def as_matrix(mat, key):
    if key not in mat:
        raise KeyError(f"required MAT field is missing: {key}")
    value = np.asarray(mat[key], dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"{key} must be a 2-D array, got {value.shape}")
    return value


def masked_rms(value, mask):
    return float(np.sqrt(np.mean(np.square(value[mask]))))


def load_run(spec, settle_s):
    mat = loadmat(spec["path"])
    system_angle = as_matrix(mat, "system_ip_pitch_all")
    robot_angle = as_matrix(mat, "robot_ip_pitch_all")
    base_pitch = as_matrix(mat, "base_pitch_all")
    system_offset = as_matrix(mat, "system_com_forward_from_support_all")
    system_height = as_matrix(mat, "system_com_height_from_support_all")
    support_force = as_matrix(mat, "support_force_z_all")
    load_on_body = as_matrix(mat, "load_on_body_all") > 0.5
    command = as_matrix(mat, "command_x_all")
    payload_mass = as_matrix(mat, "payload_mass_ref_all")

    arrays = {
        "robot_ip_pitch_all": robot_angle,
        "base_pitch_all": base_pitch,
        "system_com_forward_from_support_all": system_offset,
        "system_com_height_from_support_all": system_height,
        "support_force_z_all": support_force,
        "load_on_body_all": load_on_body,
        "command_x_all": command,
        "payload_mass_ref_all": payload_mass,
    }
    for name, value in arrays.items():
        if value.shape != system_angle.shape:
            raise ValueError(
                f"{name} shape {value.shape} does not match {system_angle.shape}"
            )

    dt = float(np.asarray(mat["dt"]).squeeze())
    time_s = np.arange(system_angle.shape[0], dtype=np.float64) * dt
    nominal_command = float(np.median(command[load_on_body]))
    command_is_nominal = np.isclose(command, nominal_command, atol=1e-6)
    loaded_nominal = load_on_body & command_is_nominal
    settled = np.broadcast_to((time_s >= settle_s)[:, None], system_angle.shape)
    metric_mask = loaded_nominal & settled
    if not np.any(metric_mask):
        raise ValueError(f"no valid loaded samples in {spec['path']}")

    rad_to_deg = 180.0 / np.pi
    system_abs_deg = np.abs(system_angle[metric_mask]) * rad_to_deg
    row = {
        "label": spec["label"],
        "method": spec["method"],
        "pair_id": spec["pair_id"],
        "source_mat": str(spec["path"].resolve()),
        "dt_s": dt,
        "duration_s": system_angle.shape[0] * dt,
        "num_envs": system_angle.shape[1],
        "settle_s": settle_s,
        "nominal_roll_command_mps": nominal_command,
        "payload_mass_mean_kg": float(np.mean(payload_mass[metric_mask])),
        "valid_sample_count": int(metric_mask.sum()),
        "loaded_nominal_coverage_percent": 100.0 * float(np.mean(loaded_nominal)),
        "contact_support_percent": 100.0 * float(np.mean(support_force[metric_mask] > 1.0)),
        "system_ip_angle_rms_deg": rad_to_deg * masked_rms(system_angle, metric_mask),
        "system_ip_angle_mae_deg": float(np.mean(system_abs_deg)),
        "system_ip_angle_bias_deg": rad_to_deg * float(np.mean(system_angle[metric_mask])),
        "system_ip_angle_p95_abs_deg": float(np.percentile(system_abs_deg, 95.0)),
        "robot_ip_angle_rms_deg": rad_to_deg * masked_rms(robot_angle, metric_mask),
        "base_pitch_rms_deg": rad_to_deg * masked_rms(base_pitch, metric_mask),
        "system_com_offset_rms_cm": 100.0 * masked_rms(system_offset, metric_mask),
        "system_com_height_mean_m": float(np.mean(system_height[metric_mask])),
    }
    return {
        "spec": spec,
        "row": row,
        "dt": dt,
        "time_s": time_s,
        "system_angle": system_angle,
        "robot_angle": robot_angle,
        "base_pitch": base_pitch,
        "valid": loaded_nominal,
    }


def rms_curve(values, valid, min_valid_fraction):
    valid_count = valid.sum(axis=1)
    valid_fraction = valid_count / valid.shape[1]
    enough = (valid_count > 0) & (valid_fraction >= min_valid_fraction)
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    squared_sum = np.sum(np.square(values) * valid, axis=1)
    result[enough] = np.degrees(np.sqrt(squared_sum[enough] / valid_count[enough]))
    return result, valid_count, valid_fraction


def pool_method(runs, min_valid_fraction):
    if not runs:
        raise ValueError("each method must have at least one run")
    dt = runs[0]["dt"]
    sample_count = runs[0]["system_angle"].shape[0]
    for run in runs[1:]:
        if not np.isclose(run["dt"], dt) or run["system_angle"].shape[0] != sample_count:
            raise ValueError("all runs must have matching time axes")

    system = np.concatenate([run["system_angle"] for run in runs], axis=1)
    robot = np.concatenate([run["robot_angle"] for run in runs], axis=1)
    base = np.concatenate([run["base_pitch"] for run in runs], axis=1)
    valid = np.concatenate([run["valid"] for run in runs], axis=1)
    system_curve, count, fraction = rms_curve(system, valid, min_valid_fraction)
    robot_curve, _, _ = rms_curve(robot, valid, min_valid_fraction)
    base_curve, _, _ = rms_curve(base, valid, min_valid_fraction)
    return {
        "time_s": np.arange(sample_count, dtype=np.float64) * dt,
        "system_ip_angle_rms_deg": system_curve,
        "robot_ip_angle_rms_deg": robot_curve,
        "base_pitch_rms_deg": base_curve,
        "valid_env_count": count,
        "valid_env_fraction": fraction,
        "num_envs": system.shape[1],
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def method_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    result = []
    for method, method_rows in grouped.items():
        for key, display in METRICS:
            values = np.asarray([row[key] for row in method_rows], dtype=np.float64)
            result.append(
                {
                    "method": method,
                    "metric": key,
                    "display_name": display,
                    "run_count": values.size,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                }
            )
    return result


def compare_methods(rows, baseline_method, candidate_method):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    if not grouped[baseline_method] or not grouped[candidate_method]:
        raise ValueError("baseline and candidate methods are both required")

    by_pair = {(row["method"], row["pair_id"]): row for row in rows}
    common_pairs = sorted(
        {row["pair_id"] for row in grouped[baseline_method]}
        & {row["pair_id"] for row in grouped[candidate_method]}
    )
    result = []
    for key, display in METRICS:
        baseline = float(np.mean([row[key] for row in grouped[baseline_method]]))
        candidate = float(np.mean([row[key] for row in grouped[candidate_method]]))
        paired_changes = []
        for pair_id in common_pairs:
            left = by_pair[(baseline_method, pair_id)][key]
            right = by_pair[(candidate_method, pair_id)][key]
            paired_changes.append(100.0 * (left - right) / abs(left))
        result.append(
            {
                "metric": key,
                "display_name": display,
                "baseline_method": baseline_method,
                "candidate_method": candidate_method,
                "baseline_mean": baseline,
                "candidate_mean": candidate,
                "candidate_improvement_percent": 100.0 * (baseline - candidate) / abs(baseline),
                "paired_count": len(paired_changes),
                "candidate_better_pair_count": int(sum(value > 0.0 for value in paired_changes)),
            }
        )
    return result


def write_timeseries_csv(path, rlonly, model_guided):
    fields = [
        "time_s",
        "system_ip_angle_rms_model_guided_deg",
        "system_ip_angle_rms_rlonly_deg",
        "robot_ip_angle_rms_model_guided_deg",
        "robot_ip_angle_rms_rlonly_deg",
        "base_pitch_rms_model_guided_deg",
        "base_pitch_rms_rlonly_deg",
        "valid_env_count_model_guided",
        "valid_env_count_rlonly",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, time_s in enumerate(rlonly["time_s"]):
            writer.writerow(
                {
                    "time_s": time_s,
                    "system_ip_angle_rms_model_guided_deg": model_guided["system_ip_angle_rms_deg"][index],
                    "system_ip_angle_rms_rlonly_deg": rlonly["system_ip_angle_rms_deg"][index],
                    "robot_ip_angle_rms_model_guided_deg": model_guided["robot_ip_angle_rms_deg"][index],
                    "robot_ip_angle_rms_rlonly_deg": rlonly["robot_ip_angle_rms_deg"][index],
                    "base_pitch_rms_model_guided_deg": model_guided["base_pitch_rms_deg"][index],
                    "base_pitch_rms_rlonly_deg": rlonly["base_pitch_rms_deg"][index],
                    "valid_env_count_model_guided": model_guided["valid_env_count"][index],
                    "valid_env_count_rlonly": rlonly["valid_env_count"][index],
                }
            )


def plot_timeseries(path, rlonly, model_guided, time_limit_s):
    figure, axis = plt.subplots(figsize=(7.6, 4.3))
    for method, data in (("Model-guided", model_guided), ("RL-only", rlonly)):
        axis.plot(
            data["time_s"],
            data["system_ip_angle_rms_deg"],
            color=COLORS[method],
            linewidth=1.7,
            label=method,
        )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Equivalent IP angle RMS (deg)")
    axis.set_xlim(0.0, time_limit_s)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, color="#D9D9D9", linewidth=0.6)
    axis.legend(loc="upper right", frameon=True)
    axis.tick_params(direction="out")
    figure.tight_layout()
    figure.savefig(path, dpi=240, facecolor="white")
    plt.close(figure)


def write_report(path, rows, comparisons, settle_s):
    comparison = {row["metric"]: row for row in comparisons}
    lines = [
        "# Equivalent Inverted-Pendulum Angle",
        "",
        "Definition:",
        "",
        "`theta_IP = atan2((p_COM - p_support) dot e_forward, z_COM - z_support)`",
        "",
        "The primary angle uses the mass-weighted CoM of all robot links plus the payload while it remains on the robot.",
        "The support point is the vertical-contact-force-weighted wheel-center position; the wheel midpoint is used when both sampled contact forces are zero.",
        f"Scalar metrics use t >= {settle_s:g} s, payload-on-body samples, and the nominal logged command.",
        "",
        "## Per-run results",
        "",
        "| Run | Method | System IP RMS (deg) | Robot IP RMS (deg) | Base pitch RMS (deg) | CoM offset RMS (cm) | Loaded coverage (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['method']} | {row['system_ip_angle_rms_deg']:.3f} | "
            f"{row['robot_ip_angle_rms_deg']:.3f} | {row['base_pitch_rms_deg']:.3f} | "
            f"{row['system_com_offset_rms_cm']:.3f} | {row['loaded_nominal_coverage_percent']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Method means",
            "",
            "Positive improvement means Model-guided has the lower error.",
            "",
            "| Metric | RL-only | Model-guided | Improvement (%) | Better pairs |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, display in METRICS:
        row = comparison[key]
        lines.append(
            f"| {display} | {row['baseline_mean']:.4f} | {row['candidate_mean']:.4f} | "
            f"{row['candidate_improvement_percent']:+.2f} | "
            f"{row['candidate_better_pair_count']}/{row['paired_count']} |"
        )
    lines.extend(
        [
            "",
            "This is an associative policy comparison, not a same-policy causal ablation of estimation quality.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=parse_run_spec)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-method", default="RL-only")
    parser.add_argument("--candidate-method", default="Model-guided")
    parser.add_argument("--settle-s", type=float, default=2.0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.25)
    parser.add_argument("--time-limit-s", type=float, default=40.0)
    args = parser.parse_args()
    if not 0.0 < args.min_valid_fraction <= 1.0:
        parser.error("--min-valid-fraction must be in (0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_run(spec, args.settle_s) for spec in args.run]
    rows = [run["row"] for run in runs]
    summary = method_summary(rows)
    comparisons = compare_methods(rows, args.baseline_method, args.candidate_method)
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["spec"]["method"]].append(run)
    rlonly = pool_method(grouped[args.baseline_method], args.min_valid_fraction)
    model_guided = pool_method(grouped[args.candidate_method], args.min_valid_fraction)

    write_csv(args.output_dir / "run_metrics.csv", rows)
    write_csv(args.output_dir / "method_summary.csv", summary)
    write_csv(args.output_dir / "method_comparison.csv", comparisons)
    write_timeseries_csv(args.output_dir / "ip_angle_rms_timeseries.csv", rlonly, model_guided)
    savemat(
        args.output_dir / "ip_angle_rms_timeseries.mat",
        {
            "time_s": rlonly["time_s"],
            "system_ip_angle_rms_model_guided_deg": model_guided["system_ip_angle_rms_deg"],
            "system_ip_angle_rms_rlonly_deg": rlonly["system_ip_angle_rms_deg"],
            "robot_ip_angle_rms_model_guided_deg": model_guided["robot_ip_angle_rms_deg"],
            "robot_ip_angle_rms_rlonly_deg": rlonly["robot_ip_angle_rms_deg"],
            "base_pitch_rms_model_guided_deg": model_guided["base_pitch_rms_deg"],
            "base_pitch_rms_rlonly_deg": rlonly["base_pitch_rms_deg"],
            "valid_env_count_model_guided": model_guided["valid_env_count"],
            "valid_env_count_rlonly": rlonly["valid_env_count"],
            "definition": "atan2(forward COM offset from wheel support, COM height above wheel support)",
            "valid_definition": "payload on body and logged roll command is nominal",
        },
        do_compression=True,
    )
    plot_timeseries(
        args.output_dir / "ip_angle_rms_timeseries.png",
        rlonly,
        model_guided,
        args.time_limit_s,
    )
    write_report(args.output_dir / "ip_angle_report.md", rows, comparisons, args.settle_s)
    payload = {
        "settings": {
            "settle_s": args.settle_s,
            "min_valid_fraction": args.min_valid_fraction,
            "time_limit_s": args.time_limit_s,
            "baseline_method": args.baseline_method,
            "candidate_method": args.candidate_method,
        },
        "runs": rows,
        "method_summary": summary,
        "method_comparison": comparisons,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "ip_angle_report.md")
    print(args.output_dir / "ip_angle_rms_timeseries.png")


if __name__ == "__main__":
    main()
