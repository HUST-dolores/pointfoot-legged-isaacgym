#!/usr/bin/env python3
"""Compute payload-estimation and motion metrics from play.py MAT exports."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat


METRICS = (
    ("mass_rmse_kg", "Mass RMSE (kg)", True),
    ("speed_tracking_rmse_mps", "Speed tracking RMSE (m/s)", True),
    ("speed_tracking_mae_mps", "Speed tracking MAE (m/s)", True),
    ("pitch_rms_deg", "Pitch RMS (deg)", True),
    ("pitch_mae_deg", "Pitch MAE (deg)", True),
    ("roll_rms_deg", "Roll RMS (deg)", True),
    ("upright_percent", "Upright samples (%)", False),
    ("load_coverage_percent", "Loaded samples, all envs (%)", False),
    ("load_retention_percent", "Load retention, load-seen envs (%)", False),
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


def masked_mae(value, mask):
    return float(np.mean(np.abs(value[mask])))


def summarize_per_env(value, mask, transform=None, min_samples=1):
    result = []
    for env_id in range(value.shape[1]):
        env_mask = mask[:, env_id]
        if int(env_mask.sum()) < min_samples:
            continue
        metric = masked_rms(value[:, env_id], env_mask)
        result.append(transform(metric) if transform else metric)
    if not result:
        return float("nan"), float("nan"), 0
    values = np.asarray(result)
    return float(values.mean()), float(values.std(ddof=0)), int(values.size)


def analyze_run(spec, settle_s, upright_deg, min_loaded_s, cmd_step_mps):
    mat = loadmat(spec["path"])
    pitch = as_matrix(mat, "base_pitch_all")
    roll = as_matrix(mat, "base_roll_all")
    velocity = as_matrix(mat, "base_lin_vel_x_all")
    command = as_matrix(mat, "command_x_all")
    load_on_body = as_matrix(mat, "load_on_body_all") > 0.5
    mass_estimate = as_matrix(mat, "encoder_mass_all")
    mass_reference = as_matrix(mat, "payload_mass_ref_all")

    shape = pitch.shape
    for name, value in (
        ("base_roll_all", roll),
        ("base_lin_vel_x_all", velocity),
        ("command_x_all", command),
        ("load_on_body_all", load_on_body),
        ("encoder_mass_all", mass_estimate),
        ("payload_mass_ref_all", mass_reference),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} does not match {shape}")

    dt = float(np.asarray(mat["dt"]).squeeze())
    time_s = np.arange(shape[0], dtype=np.float64) * dt
    steady = np.broadcast_to((time_s >= settle_s)[:, None], shape)
    loaded = steady & load_on_body
    if not np.any(loaded):
        raise ValueError(f"no loaded samples after {settle_s:g} s in {spec['path']}")

    load_seen_envs = np.any(loaded, axis=0)
    steady_seen = steady[:, load_seen_envs]
    loaded_seen = loaded[:, load_seen_envs]
    nominal_roll_command = float(np.median(command[loaded]))
    command_is_nominal = np.isclose(command, nominal_roll_command, atol=1e-6)
    motion_mask = loaded & command_is_nominal
    if not np.any(motion_mask):
        raise ValueError(f"no nominal-command motion samples in {spec['path']}")

    upright_limit = np.deg2rad(upright_deg)
    upright = (np.abs(pitch) < upright_limit) & (np.abs(roll) < upright_limit)
    speed_target = command + cmd_step_mps
    speed_error = velocity - speed_target
    mass_error = mass_estimate - mass_reference
    rad_to_deg = 180.0 / np.pi
    min_loaded_samples = max(1, int(round(min_loaded_s / dt)))

    per_env_speed_mean, per_env_speed_std, per_env_count = summarize_per_env(
        speed_error, motion_mask, min_samples=min_loaded_samples
    )
    per_env_pitch_mean, per_env_pitch_std, _ = summarize_per_env(
        pitch,
        motion_mask,
        transform=lambda value: value * rad_to_deg,
        min_samples=min_loaded_samples,
    )
    per_env_mass_mean, per_env_mass_std, _ = summarize_per_env(
        mass_error, loaded, min_samples=min_loaded_samples
    )

    row = {
        "label": spec["label"],
        "method": spec["method"],
        "pair_id": spec["pair_id"],
        "source_mat": str(spec["path"].resolve()),
        "dt_s": dt,
        "duration_s": shape[0] * dt,
        "settle_s": settle_s,
        "nominal_roll_command_mps": nominal_roll_command,
        "cmd_step_mps": cmd_step_mps,
        "total_base_speed_target_mps": nominal_roll_command + cmd_step_mps,
        "num_envs": shape[1],
        "load_seen_envs": int(load_seen_envs.sum()),
        "per_env_metric_count": per_env_count,
        "loaded_sample_count": int(loaded.sum()),
        "motion_sample_count": int(motion_mask.sum()),
        "load_coverage_percent": 100.0 * float(loaded.sum() / steady.sum()),
        "load_retention_percent": 100.0 * float(loaded_seen.sum() / steady_seen.sum()),
        "command_override_percent_loaded": 100.0 * float(np.mean(~command_is_nominal[loaded])),
        "mass_rmse_kg": masked_rms(mass_error, loaded),
        "speed_tracking_rmse_mps": masked_rms(speed_error, motion_mask),
        "speed_tracking_mae_mps": masked_mae(speed_error, motion_mask),
        "pitch_rms_deg": rad_to_deg * masked_rms(pitch, motion_mask),
        "pitch_mae_deg": rad_to_deg * masked_mae(pitch, motion_mask),
        "roll_rms_deg": rad_to_deg * masked_rms(roll, motion_mask),
        "upright_percent": 100.0 * float(np.mean(upright[motion_mask])),
        "per_env_speed_rmse_mean_mps": per_env_speed_mean,
        "per_env_speed_rmse_std_mps": per_env_speed_std,
        "per_env_pitch_rms_mean_deg": per_env_pitch_mean,
        "per_env_pitch_rms_std_deg": per_env_pitch_std,
        "per_env_mass_rmse_mean_kg": per_env_mass_mean,
        "per_env_mass_rmse_std_kg": per_env_mass_std,
    }

    per_env_rows = []
    for env_id in range(shape[1]):
        env_mask = motion_mask[:, env_id]
        if int(env_mask.sum()) < min_loaded_samples:
            continue
        per_env_rows.append(
            {
                "label": spec["label"],
                "method": spec["method"],
                "pair_id": spec["pair_id"],
                "env_id": env_id,
                "loaded_samples": int(env_mask.sum()),
                "load_retention_percent": 100.0 * float(env_mask.sum() / steady[:, env_id].sum()),
                "mass_rmse_kg": masked_rms(mass_error[:, env_id], env_mask),
                "speed_tracking_rmse_mps": masked_rms(speed_error[:, env_id], env_mask),
                "pitch_rms_deg": rad_to_deg * masked_rms(pitch[:, env_id], env_mask),
                "roll_rms_deg": rad_to_deg * masked_rms(roll[:, env_id], env_mask),
                "upright_percent": 100.0 * float(np.mean(upright[:, env_id][env_mask])),
            }
        )
    return row, per_env_rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_method_summary(run_rows):
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[row["method"]].append(row)
    summary = []
    for method, rows in grouped.items():
        for key, display, lower_is_better in METRICS:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            summary.append(
                {
                    "method": method,
                    "metric": key,
                    "display_name": display,
                    "lower_is_better": lower_is_better,
                    "run_count": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                }
            )
    return summary


def make_comparisons(run_rows, baseline_method, candidate_method):
    by_method_pair = {(row["method"], row["pair_id"]): row for row in run_rows}
    baseline_rows = [row for row in run_rows if row["method"] == baseline_method]
    candidate_rows = [row for row in run_rows if row["method"] == candidate_method]
    if not baseline_rows or not candidate_rows:
        raise ValueError("baseline and candidate methods must both have at least one run")

    comparisons = []
    pair_rows = []
    common_pairs = sorted(
        {row["pair_id"] for row in baseline_rows}
        & {row["pair_id"] for row in candidate_rows}
    )
    for key, display, lower_is_better in METRICS:
        baseline_mean = float(np.mean([row[key] for row in baseline_rows]))
        candidate_mean = float(np.mean([row[key] for row in candidate_rows]))
        sign = 1.0 if lower_is_better else -1.0
        improvement = sign * (baseline_mean - candidate_mean) / abs(baseline_mean) * 100.0
        pair_improvements = []
        for pair_id in common_pairs:
            baseline = by_method_pair[(baseline_method, pair_id)][key]
            candidate = by_method_pair[(candidate_method, pair_id)][key]
            pair_improvement = sign * (baseline - candidate) / abs(baseline) * 100.0
            pair_improvements.append(pair_improvement)
            pair_rows.append(
                {
                    "metric": key,
                    "display_name": display,
                    "pair_id": pair_id,
                    "baseline": baseline,
                    "candidate": candidate,
                    "candidate_improvement_percent": pair_improvement,
                }
            )
        comparisons.append(
            {
                "metric": key,
                "display_name": display,
                "lower_is_better": lower_is_better,
                "baseline_method": baseline_method,
                "candidate_method": candidate_method,
                "baseline_mean": baseline_mean,
                "candidate_mean": candidate_mean,
                "candidate_improvement_percent": improvement,
                "paired_count": len(pair_improvements),
                "candidate_better_pair_count": int(sum(value > 0.0 for value in pair_improvements)),
                "paired_direction_consistent": bool(pair_improvements)
                and all(value > 0.0 for value in pair_improvements),
            }
        )
    return comparisons, pair_rows


def write_report(path, run_rows, comparisons, settle_s, upright_deg, cmd_step_mps):
    comparison_by_key = {row["metric"]: row for row in comparisons}
    lines = [
        "# Payload Motion Metrics",
        "",
        f"Evaluation window: samples at t >= {settle_s:g} s while the payload is on the body.",
        f"Base-speed target: logged roll command + fixed step command ({cmd_step_mps:g} m/s).",
        "Internal command-resampling frames with a temporary randomized roll command are excluded from motion metrics.",
        f"Upright criterion: abs(roll) and abs(pitch) < {upright_deg:g} deg.",
        "",
        "## Per-run results",
        "",
        "| Run | Method | Mass RMSE (kg) | Speed RMSE (m/s) | Pitch RMS (deg) | Roll RMS (deg) | Load coverage (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in run_rows:
        lines.append(
            f"| {row['label']} | {row['method']} | {row['mass_rmse_kg']:.3f} | "
            f"{row['speed_tracking_rmse_mps']:.3f} | {row['pitch_rms_deg']:.3f} | "
            f"{row['roll_rms_deg']:.3f} | {row['load_coverage_percent']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Method comparison",
            "",
            "Positive improvement means the candidate method is better.",
            "",
            "| Metric | Baseline mean | Candidate mean | Improvement (%) | Better pairs | Consistent |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for key, _, _ in METRICS:
        row = comparison_by_key[key]
        lines.append(
            f"| {row['display_name']} | {row['baseline_mean']:.4f} | "
            f"{row['candidate_mean']:.4f} | {row['candidate_improvement_percent']:+.2f} | "
            f"{row['candidate_better_pair_count']}/{row['paired_count']} | "
            f"{'yes' if row['paired_direction_consistent'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The comparison is associative, not causal: the policies differ in architecture and training.",
            "A same-policy estimate-corruption or perfect-estimate ablation is needed for causal attribution.",
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
    parser.add_argument("--upright-deg", type=float, default=25.0)
    parser.add_argument("--min-loaded-s", type=float, default=1.0)
    parser.add_argument("--cmd-step-mps", type=float, default=0.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = []
    per_env_rows = []
    for spec in args.run:
        row, env_rows = analyze_run(
            spec,
            args.settle_s,
            args.upright_deg,
            args.min_loaded_s,
            args.cmd_step_mps,
        )
        run_rows.append(row)
        per_env_rows.extend(env_rows)

    method_summary = make_method_summary(run_rows)
    comparisons, pair_rows = make_comparisons(
        run_rows, args.baseline_method, args.candidate_method
    )
    write_csv(args.output_dir / "run_metrics.csv", run_rows)
    write_csv(args.output_dir / "per_env_metrics.csv", per_env_rows)
    write_csv(args.output_dir / "method_summary.csv", method_summary)
    write_csv(args.output_dir / "method_comparison.csv", comparisons)
    write_csv(args.output_dir / "paired_comparison.csv", pair_rows)
    write_report(
        args.output_dir / "metrics_report.md",
        run_rows,
        comparisons,
        args.settle_s,
        args.upright_deg,
        args.cmd_step_mps,
    )
    settings = dict(vars(args))
    settings["output_dir"] = str(settings["output_dir"])
    settings["run"] = [
        {**spec, "path": str(spec["path"])} for spec in settings["run"]
    ]
    payload = {
        "settings": settings,
        "runs": run_rows,
        "method_summary": method_summary,
        "method_comparison": comparisons,
        "paired_comparison": pair_rows,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "metrics_report.md")


if __name__ == "__main__":
    main()
