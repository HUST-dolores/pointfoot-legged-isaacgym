#!/usr/bin/env python3
"""Analyze inference-only GRU history-length ablation play exports."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat


FILENAME_RE = re.compile(r"habl_s2_(rlonly|mg)_h(10|20|50|100)\.mat$")
METHOD_NAMES = {"rlonly": "RL-only", "mg": "Model-guided"}


def as_matrix(mat, key):
    value = np.asarray(mat[key], dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"{key} must be a 2-D array, got {value.shape}")
    return value


def analyze(path, method_key, history_length, settle_s, min_valid_s):
    mat = loadmat(path)
    pitch = as_matrix(mat, "base_pitch_all")
    load_on_body = as_matrix(mat, "load_on_body_all") > 0.5
    command = as_matrix(mat, "command_x_all")
    mass_estimate = as_matrix(mat, "encoder_mass_all")
    mass_reference = as_matrix(mat, "payload_mass_ref_all")
    arrays = (load_on_body, command, mass_estimate, mass_reference)
    if any(value.shape != pitch.shape for value in arrays):
        raise ValueError(f"array shape mismatch in {path}")

    dt = float(np.asarray(mat["dt"]).squeeze())
    time_s = np.arange(pitch.shape[0], dtype=np.float64) * dt
    nominal_command = float(np.median(command[load_on_body]))
    steady = np.broadcast_to((time_s >= settle_s)[:, None], pitch.shape)
    valid = (
        steady
        & load_on_body
        & np.isclose(command, nominal_command, atol=1.0e-6)
    )
    min_samples = max(2, int(round(min_valid_s / dt)))
    valid_count = valid.sum(axis=0)
    usable = valid_count >= min_samples
    valid &= usable[None, :]
    if not np.any(valid):
        raise ValueError(f"no valid samples in {path}")

    pitch_sum = np.sum(np.where(valid, pitch, 0.0), axis=0)
    pitch_mean = np.zeros(pitch.shape[1], dtype=np.float64)
    pitch_mean[usable] = pitch_sum[usable] / valid_count[usable]
    pitch_deviation_deg = np.degrees(
        np.abs(pitch - pitch_mean[None, :])[valid]
    )
    mass_error = mass_estimate - mass_reference
    per_step_dim = 54 if method_key == "rlonly" else 94
    return {
        "method": METHOD_NAMES[method_key],
        "checkpoint": (
            "Jul05_15-21-33_rlonly_loadest_s2/model_16000.pt"
            if method_key == "rlonly"
            else "Jul05_23-33-51_mg_e1_s2/model_16000.pt"
        ),
        "history_length_steps": history_length,
        "history_context_s": history_length * dt,
        "per_step_encoder_dim": per_step_dim,
        "flat_encoder_input_dim": per_step_dim * history_length,
        "actor_latent_dim": 7,
        "gru_hidden_dim": 128,
        "num_envs": pitch.shape[1],
        "usable_envs": int(usable.sum()),
        "valid_sample_count": int(valid.sum()),
        "payload_mass_mean_kg": float(np.mean(mass_reference[valid])),
        "mass_rmse_kg": float(np.sqrt(np.mean(np.square(mass_error[valid])))),
        "mass_mae_kg": float(np.mean(np.abs(mass_error[valid]))),
        "mass_bias_kg": float(np.mean(mass_error[valid])),
        "pitch_demeaned_abs_p95_deg": float(np.percentile(pitch_deviation_deg, 95.0)),
        "pitch_demeaned_sd_deg": float(np.sqrt(np.mean(np.square(pitch_deviation_deg)))),
        "load_coverage_percent": 100.0 * float(np.mean(load_on_body)),
        "source_mat": str(path.resolve()),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows, settle_s):
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    lines = [
        "# GRU History-Length Sensitivity",
        "",
        "This is an inference-only context-length sensitivity test. The trained H=10 checkpoints and all learned weights are held fixed; the same GRU is unrolled over H=10/20/50/100 observations without retraining.",
        "It isolates deployment-time context length but is not a from-scratch architecture-training ablation.",
        f"Metrics use t >= {settle_s:g} s, payload-on-body samples, and the nominal logged command.",
        "",
        "## Architecture",
        "",
        "| Method | Trained H | Context | Per-step input | Actor z_t | GRU hidden |",
        "|---|---:|---:|---:|---:|---:|",
        "| RL-only | 10 | 0.20 s | 54 | 7 | 128 |",
        "| Model-guided | 10 | 0.20 s | 94 | 7 | 128 |",
        "",
        "The 7-D actor-visible z_t contains 3-D base linear velocity and 4-D payload state (mass plus 3-D CoM offset). Model-guided forms the payload components as the quasi-static baseline plus a learned residual; RL-only predicts them directly.",
        "",
        "## Results",
        "",
        "| Method | H (steps) | Context (s) | Mass RMSE (kg) | Pitch demeaned p95 (deg) | Usable envs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['history_length_steps']} | "
            f"{row['history_context_s']:.2f} | {row['mass_rmse_kg']:.3f} | "
            f"{row['pitch_demeaned_abs_p95_deg']:.3f} | {row['usable_envs']} |"
        )
    lines.extend(["", "## Change from H=10", ""])
    for method, method_rows in by_method.items():
        method_rows.sort(key=lambda row: row["history_length_steps"])
        baseline = method_rows[0]
        lines.append(f"**{method}**")
        lines.append("")
        lines.append("| H | Mass RMSE change (%) | Pitch p95 change (%) |")
        lines.append("|---:|---:|---:|")
        for row in method_rows:
            mass_change = 100.0 * (
                row["mass_rmse_kg"] - baseline["mass_rmse_kg"]
            ) / baseline["mass_rmse_kg"]
            pitch_change = 100.0 * (
                row["pitch_demeaned_abs_p95_deg"]
                - baseline["pitch_demeaned_abs_p95_deg"]
            ) / baseline["pitch_demeaned_abs_p95_deg"]
            lines.append(
                f"| {row['history_length_steps']} | {mass_change:+.2f} | {pitch_change:+.2f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--settle-s", type=float, default=2.0)
    parser.add_argument("--min-valid-s", type=float, default=1.0)
    args = parser.parse_args()

    candidates = []
    for path in args.input_dir.glob("play_data_*habl_s2_*_h*.mat"):
        match = FILENAME_RE.search(path.name)
        if match:
            candidates.append((path, match.group(1), int(match.group(2))))
    if len(candidates) != 8:
        raise ValueError(f"expected 8 ablation MAT files, found {len(candidates)}")
    rows = [
        analyze(path, method, history, args.settle_s, args.min_valid_s)
        for path, method, history in candidates
    ]
    rows.sort(key=lambda row: (row["method"], row["history_length_steps"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "history_length_ablation.csv", rows)
    write_report(args.output_dir / "history_length_ablation.md", rows, args.settle_s)
    payload = {
        "protocol": {
            "mode": "inference-only fixed-weight GRU unroll",
            "trained_history_length": 10,
            "tested_history_lengths": [10, 20, 50, 100],
            "settle_s": args.settle_s,
            "min_valid_s": args.min_valid_s,
        },
        "rows": rows,
    }
    (args.output_dir / "history_length_ablation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "history_length_ablation.md")


if __name__ == "__main__":
    main()
