#!/usr/bin/env python3
"""Measure body-pitch fluctuation without assuming a zero-pitch target."""

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
    ("pitch_temporal_sd_full_deg", "Pitch temporal SD, loaded interval (deg)"),
    ("pitch_temporal_sd_steady_deg", "Pitch temporal SD, t >= settle (deg)"),
    ("pitch_abs_deviation_p95_steady_deg", "Pitch abs. deviation p95 (deg)"),
    ("pitch_rmssd_steady_deg", "Pitch RMSSD (deg/sample)"),
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


def center_per_env(values, valid, min_samples):
    count = valid.sum(axis=0)
    usable = count >= min_samples
    mean = np.full(values.shape[1], np.nan, dtype=np.float64)
    value_sum = np.sum(np.where(valid, values, 0.0), axis=0)
    mean[usable] = value_sum[usable] / count[usable]
    centered = values - mean[None, :]
    centered_valid = valid & usable[None, :]
    return centered, centered_valid, mean, count


def pooled_rms(values, valid):
    return float(np.sqrt(np.mean(np.square(values[valid]))))


def rmssd(values, valid):
    consecutive = valid[1:] & valid[:-1]
    delta = values[1:] - values[:-1]
    return float(np.sqrt(np.mean(np.square(delta[consecutive]))))


def load_run(spec, settle_s, min_valid_s):
    mat = loadmat(spec["path"])
    pitch = as_matrix(mat, "base_pitch_all")
    load_on_body = as_matrix(mat, "load_on_body_all") > 0.5
    command = as_matrix(mat, "command_x_all")
    if load_on_body.shape != pitch.shape or command.shape != pitch.shape:
        raise ValueError(f"array shape mismatch in {spec['path']}")

    dt = float(np.asarray(mat["dt"]).squeeze())
    time_s = np.arange(pitch.shape[0], dtype=np.float64) * dt
    nominal_command = float(np.median(command[load_on_body]))
    valid_full = load_on_body & np.isclose(command, nominal_command, atol=1.0e-6)
    steady = np.broadcast_to((time_s >= settle_s)[:, None], pitch.shape)
    valid_steady = valid_full & steady
    min_samples = max(2, int(round(min_valid_s / dt)))

    full_centered, full_valid, full_mean, _ = center_per_env(
        pitch, valid_full, min_samples
    )
    steady_centered, steady_valid, steady_mean, steady_count = center_per_env(
        pitch, valid_steady, min_samples
    )
    if not np.any(full_valid) or not np.any(steady_valid):
        raise ValueError(f"no valid loaded samples in {spec['path']}")

    rad_to_deg = 180.0 / np.pi
    per_env_sd = []
    for env_id in np.flatnonzero(steady_count >= min_samples):
        mask = steady_valid[:, env_id]
        per_env_sd.append(
            rad_to_deg * np.sqrt(np.mean(np.square(steady_centered[:, env_id][mask])))
        )
    per_env_sd = np.asarray(per_env_sd)
    row = {
        "label": spec["label"],
        "method": spec["method"],
        "pair_id": spec["pair_id"],
        "source_mat": str(spec["path"].resolve()),
        "dt_s": dt,
        "duration_s": pitch.shape[0] * dt,
        "num_envs": pitch.shape[1],
        "usable_envs": int(per_env_sd.size),
        "settle_s": settle_s,
        "nominal_roll_command_mps": nominal_command,
        "mean_operating_pitch_full_deg": rad_to_deg * float(np.mean(full_mean[np.isfinite(full_mean)])),
        "mean_operating_pitch_steady_deg": rad_to_deg * float(np.mean(steady_mean[np.isfinite(steady_mean)])),
        "pitch_temporal_sd_full_deg": rad_to_deg * pooled_rms(full_centered, full_valid),
        "pitch_temporal_sd_steady_deg": rad_to_deg * pooled_rms(steady_centered, steady_valid),
        "pitch_abs_deviation_p95_steady_deg": rad_to_deg * float(
            np.percentile(np.abs(steady_centered[steady_valid]), 95.0)
        ),
        "pitch_rmssd_steady_deg": rad_to_deg * rmssd(pitch, steady_valid),
        "per_env_pitch_sd_mean_deg": float(per_env_sd.mean()),
        "per_env_pitch_sd_std_deg": float(per_env_sd.std(ddof=0)),
        "loaded_nominal_coverage_percent": 100.0 * float(np.mean(valid_full)),
    }
    return {
        "spec": spec,
        "row": row,
        "dt": dt,
        "time_s": time_s,
        "pitch": pitch,
        "valid": valid_full,
    }


def rolling_pooled_sd(values, valid, window_samples, min_window_fraction, min_env_fraction):
    sample_count, env_count = values.shape
    curve = np.full(sample_count, np.nan, dtype=np.float64)
    valid_env_count = np.zeros(sample_count, dtype=np.int64)
    min_window_samples = max(2, int(np.ceil(window_samples * min_window_fraction)))
    min_env_count = max(1, int(np.ceil(env_count * min_env_fraction)))

    for index in range(sample_count):
        start = max(0, index - window_samples + 1)
        window_value = values[start : index + 1]
        window_valid = valid[start : index + 1]
        count = window_valid.sum(axis=0)
        usable = count >= min_window_samples
        valid_env_count[index] = int(usable.sum())
        if valid_env_count[index] < min_env_count:
            continue
        mean = np.sum(np.where(window_valid, window_value, 0.0), axis=0)
        mean[usable] /= count[usable]
        centered = window_value[:, usable] - mean[usable][None, :]
        usable_valid = window_valid[:, usable]
        curve[index] = np.degrees(
            np.sqrt(np.mean(np.square(centered[usable_valid])))
        )
    return curve, valid_env_count


def pool_method(runs, window_s, min_window_fraction, min_env_fraction):
    dt = runs[0]["dt"]
    sample_count = runs[0]["pitch"].shape[0]
    for run in runs[1:]:
        if not np.isclose(run["dt"], dt) or run["pitch"].shape[0] != sample_count:
            raise ValueError("all runs must have matching time axes")
    pitch = np.concatenate([run["pitch"] for run in runs], axis=1)
    valid = np.concatenate([run["valid"] for run in runs], axis=1)
    window_samples = max(2, int(round(window_s / dt)))
    curve, valid_env_count = rolling_pooled_sd(
        pitch,
        valid,
        window_samples,
        min_window_fraction,
        min_env_fraction,
    )
    return {
        "time_s": np.arange(sample_count, dtype=np.float64) * dt,
        "pitch_fluctuation_sd_deg": curve,
        "valid_env_count": valid_env_count,
        "window_s": window_samples * dt,
        "num_envs": pitch.shape[1],
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    result = []
    for method, method_rows in grouped.items():
        for key, display in METRICS:
            values = np.asarray([row[key] for row in method_rows])
            result.append(
                {
                    "method": method,
                    "metric": key,
                    "display_name": display,
                    "run_count": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                }
            )
    return result


def compare(rows, baseline_method, candidate_method):
    grouped = defaultdict(list)
    by_pair = {}
    for row in rows:
        grouped[row["method"]].append(row)
        by_pair[(row["method"], row["pair_id"])] = row
    common_pairs = sorted(
        {row["pair_id"] for row in grouped[baseline_method]}
        & {row["pair_id"] for row in grouped[candidate_method]}
    )
    result = []
    for key, display in METRICS:
        baseline = float(np.mean([row[key] for row in grouped[baseline_method]]))
        candidate = float(np.mean([row[key] for row in grouped[candidate_method]]))
        paired = []
        for pair_id in common_pairs:
            left = by_pair[(baseline_method, pair_id)][key]
            right = by_pair[(candidate_method, pair_id)][key]
            paired.append(100.0 * (left - right) / abs(left))
        result.append(
            {
                "metric": key,
                "display_name": display,
                "baseline_mean": baseline,
                "candidate_mean": candidate,
                "candidate_improvement_percent": 100.0 * (baseline - candidate) / abs(baseline),
                "paired_count": len(paired),
                "candidate_better_pair_count": int(sum(value > 0.0 for value in paired)),
            }
        )
    return result


def write_timeseries_csv(path, rlonly, model_guided):
    fields = [
        "time_s",
        "pitch_fluctuation_sd_model_guided_deg",
        "pitch_fluctuation_sd_rlonly_deg",
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
                    "pitch_fluctuation_sd_model_guided_deg": model_guided["pitch_fluctuation_sd_deg"][index],
                    "pitch_fluctuation_sd_rlonly_deg": rlonly["pitch_fluctuation_sd_deg"][index],
                    "valid_env_count_model_guided": model_guided["valid_env_count"][index],
                    "valid_env_count_rlonly": rlonly["valid_env_count"][index],
                }
            )


def plot_timeseries(path, rlonly, model_guided, time_limit_s):
    figure, axis = plt.subplots(figsize=(7.6, 4.3))
    for method, data in (("Model-guided", model_guided), ("RL-only", rlonly)):
        axis.plot(
            data["time_s"],
            data["pitch_fluctuation_sd_deg"],
            color=COLORS[method],
            linewidth=1.7,
            label=method,
        )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Pitch fluctuation, 1 s SD (deg)")
    axis.set_xlim(0.0, time_limit_s)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, color="#D9D9D9", linewidth=0.6)
    axis.legend(loc="upper right", frameon=True)
    axis.tick_params(direction="out")
    figure.tight_layout()
    figure.savefig(path, dpi=240, facecolor="white")
    plt.close(figure)


def write_report(path, rows, comparisons, settle_s, window_s):
    comparison = {row["metric"]: row for row in comparisons}
    lines = [
        "# Body Pitch Fluctuation",
        "",
        "The metric does not assume that the optimal body pitch is zero.",
        "For each environment, temporal SD is the RMS pitch deviation from that environment's own valid-time mean.",
        f"Steady metrics use t >= {settle_s:g} s while the payload remains on the body and the logged command is nominal.",
        f"The time-series curve is a trailing {window_s:g} s pooled temporal SD, with each environment centered on its own window mean.",
        "No temporal smoothing, fitting, or zero-angle reference is used.",
        "",
        "## Per-run results",
        "",
        "| Run | Method | Mean pitch (deg) | Full SD (deg) | Steady SD (deg) | p95 deviation (deg) | RMSSD (deg/sample) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['method']} | {row['mean_operating_pitch_full_deg']:.3f} | "
            f"{row['pitch_temporal_sd_full_deg']:.3f} | {row['pitch_temporal_sd_steady_deg']:.3f} | "
            f"{row['pitch_abs_deviation_p95_steady_deg']:.3f} | {row['pitch_rmssd_steady_deg']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Method comparison",
            "",
            "Positive improvement means lower fluctuation for Model-guided.",
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
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=parse_run_spec)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-method", default="RL-only")
    parser.add_argument("--candidate-method", default="Model-guided")
    parser.add_argument("--settle-s", type=float, default=2.0)
    parser.add_argument("--min-valid-s", type=float, default=1.0)
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--min-window-fraction", type=float, default=0.5)
    parser.add_argument("--min-env-fraction", type=float, default=0.25)
    parser.add_argument("--time-limit-s", type=float, default=40.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_run(spec, args.settle_s, args.min_valid_s) for spec in args.run]
    rows = [run["row"] for run in runs]
    summary = summarize(rows)
    comparisons = compare(rows, args.baseline_method, args.candidate_method)
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["spec"]["method"]].append(run)
    rlonly = pool_method(
        grouped[args.baseline_method],
        args.window_s,
        args.min_window_fraction,
        args.min_env_fraction,
    )
    model_guided = pool_method(
        grouped[args.candidate_method],
        args.window_s,
        args.min_window_fraction,
        args.min_env_fraction,
    )

    write_csv(args.output_dir / "pitch_fluctuation_run_metrics.csv", rows)
    write_csv(args.output_dir / "pitch_fluctuation_method_summary.csv", summary)
    write_csv(args.output_dir / "pitch_fluctuation_method_comparison.csv", comparisons)
    write_timeseries_csv(
        args.output_dir / "pitch_fluctuation_timeseries.csv", rlonly, model_guided
    )
    savemat(
        args.output_dir / "pitch_fluctuation_timeseries.mat",
        {
            "time_s": rlonly["time_s"],
            "pitch_fluctuation_sd_model_guided_deg": model_guided["pitch_fluctuation_sd_deg"],
            "pitch_fluctuation_sd_rlonly_deg": rlonly["pitch_fluctuation_sd_deg"],
            "valid_env_count_model_guided": model_guided["valid_env_count"],
            "valid_env_count_rlonly": rlonly["valid_env_count"],
            "window_s": rlonly["window_s"],
            "definition": "trailing-window temporal SD after subtracting each environment's window mean",
            "valid_definition": "payload on body and logged roll command is nominal",
        },
        do_compression=True,
    )
    plot_timeseries(
        args.output_dir / "pitch_fluctuation_timeseries.png",
        rlonly,
        model_guided,
        args.time_limit_s,
    )
    write_report(
        args.output_dir / "pitch_fluctuation_report.md",
        rows,
        comparisons,
        args.settle_s,
        args.window_s,
    )
    settings = dict(vars(args))
    settings["output_dir"] = str(settings["output_dir"])
    settings["run"] = [
        {**spec, "path": str(spec["path"])} for spec in settings["run"]
    ]
    payload = {
        "settings": settings,
        "runs": rows,
        "method_summary": summary,
        "method_comparison": comparisons,
    }
    (args.output_dir / "pitch_fluctuation_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "pitch_fluctuation_report.md")
    print(args.output_dir / "pitch_fluctuation_timeseries.png")


if __name__ == "__main__":
    main()
