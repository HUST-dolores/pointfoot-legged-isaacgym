#!/usr/bin/env python3
"""Export and plot time-resolved pitch RMS for payload play runs."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat


COLORS = {
    "Model-guided": "#E69F00",
    "RL-only": "#D55E00",
}


def load_group(paths, min_valid_fraction):
    pitches = []
    valid_masks = []
    all_env_curves = []
    loaded_curves = []
    dt = None
    sample_count = None

    for path in paths:
        mat = loadmat(path)
        pitch = np.asarray(mat["base_pitch_all"], dtype=np.float64)
        load_on_body = np.asarray(mat["load_on_body_all"], dtype=np.float64) > 0.5
        command = np.asarray(mat["command_x_all"], dtype=np.float64)
        run_dt = float(np.asarray(mat["dt"]).squeeze())

        if pitch.shape != load_on_body.shape or pitch.shape != command.shape:
            raise ValueError(f"array shape mismatch in {path}")
        if dt is None:
            dt = run_dt
            sample_count = pitch.shape[0]
        elif not np.isclose(dt, run_dt) or sample_count != pitch.shape[0]:
            raise ValueError("all runs must use the same dt and sample count")

        nominal_command = float(np.median(command[load_on_body]))
        valid = load_on_body & np.isclose(command, nominal_command, atol=1e-6)
        pitches.append(pitch)
        valid_masks.append(valid)
        all_env_curves.append(np.degrees(np.sqrt(np.mean(np.square(pitch), axis=1))))
        loaded_curves.append(pitch_rms_curve(pitch, valid, min_valid_fraction))

    pitch = np.concatenate(pitches, axis=1)
    valid = np.concatenate(valid_masks, axis=1)
    pooled_curve, valid_count, valid_fraction = pitch_rms_curve(
        pitch, valid, min_valid_fraction, return_counts=True
    )
    return {
        "dt": dt,
        "time_s": np.arange(sample_count, dtype=np.float64) * dt,
        "pitch_rms_deg": pooled_curve,
        "valid_env_count": valid_count,
        "valid_env_fraction": valid_fraction,
        "pitch_rms_all_env_deg": np.degrees(
            np.sqrt(np.mean(np.square(pitch), axis=1))
        ),
        "seed_pitch_rms_deg": np.asarray(loaded_curves),
        "seed_pitch_rms_all_env_deg": np.asarray(all_env_curves),
        "num_runs": len(paths),
        "num_envs": pitch.shape[1],
    }


def pitch_rms_curve(pitch, valid, min_valid_fraction, return_counts=False):
    valid_count = valid.sum(axis=1)
    valid_fraction = valid_count / pitch.shape[1]
    enough = valid_fraction >= min_valid_fraction
    curve = np.full(pitch.shape[0], np.nan, dtype=np.float64)
    pitch_sq_sum = np.sum(np.square(pitch) * valid, axis=1)
    curve[enough] = np.degrees(
        np.sqrt(pitch_sq_sum[enough] / valid_count[enough])
    )
    if return_counts:
        return curve, valid_count, valid_fraction
    return curve


def write_csv(path, rlonly, model_guided):
    fieldnames = [
        "time_s",
        "pitch_rms_model_guided_deg",
        "pitch_rms_rlonly_deg",
        "valid_env_count_model_guided",
        "valid_env_count_rlonly",
        "valid_env_fraction_model_guided",
        "valid_env_fraction_rlonly",
        "pitch_rms_all_env_model_guided_deg",
        "pitch_rms_all_env_rlonly_deg",
    ]
    seed_count = max(
        model_guided["seed_pitch_rms_deg"].shape[0],
        rlonly["seed_pitch_rms_deg"].shape[0],
    )
    for seed_id in range(seed_count):
        fieldnames.extend(
            [
                f"pitch_rms_model_guided_s{seed_id + 1}_deg",
                f"pitch_rms_rlonly_s{seed_id + 1}_deg",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, time_s in enumerate(rlonly["time_s"]):
            row = {
                "time_s": time_s,
                "pitch_rms_model_guided_deg": model_guided["pitch_rms_deg"][index],
                "pitch_rms_rlonly_deg": rlonly["pitch_rms_deg"][index],
                "valid_env_count_model_guided": model_guided["valid_env_count"][index],
                "valid_env_count_rlonly": rlonly["valid_env_count"][index],
                "valid_env_fraction_model_guided": model_guided["valid_env_fraction"][index],
                "valid_env_fraction_rlonly": rlonly["valid_env_fraction"][index],
                "pitch_rms_all_env_model_guided_deg": model_guided[
                    "pitch_rms_all_env_deg"
                ][index],
                "pitch_rms_all_env_rlonly_deg": rlonly["pitch_rms_all_env_deg"][index],
            }
            for seed_id in range(seed_count):
                if seed_id < model_guided["seed_pitch_rms_deg"].shape[0]:
                    row[f"pitch_rms_model_guided_s{seed_id + 1}_deg"] = model_guided[
                        "seed_pitch_rms_deg"
                    ][seed_id, index]
                if seed_id < rlonly["seed_pitch_rms_deg"].shape[0]:
                    row[f"pitch_rms_rlonly_s{seed_id + 1}_deg"] = rlonly[
                        "seed_pitch_rms_deg"
                    ][seed_id, index]
            writer.writerow(row)


def write_mat(path, rlonly, model_guided, min_valid_fraction):
    savemat(
        path,
        {
            "time_s": rlonly["time_s"],
            "pitch_rms_model_guided_deg": model_guided["pitch_rms_deg"],
            "pitch_rms_rlonly_deg": rlonly["pitch_rms_deg"],
            "valid_env_count_model_guided": model_guided["valid_env_count"],
            "valid_env_count_rlonly": rlonly["valid_env_count"],
            "valid_env_fraction_model_guided": model_guided["valid_env_fraction"],
            "valid_env_fraction_rlonly": rlonly["valid_env_fraction"],
            "pitch_rms_all_env_model_guided_deg": model_guided[
                "pitch_rms_all_env_deg"
            ],
            "pitch_rms_all_env_rlonly_deg": rlonly["pitch_rms_all_env_deg"],
            "pitch_rms_model_guided_per_seed_deg": model_guided[
                "seed_pitch_rms_deg"
            ],
            "pitch_rms_rlonly_per_seed_deg": rlonly["seed_pitch_rms_deg"],
            "min_valid_fraction": min_valid_fraction,
            "definition": "sqrt(mean(pitch_rad^2 over valid envs)) converted to deg",
            "valid_definition": "payload on body and logged roll command is nominal",
        },
        do_compression=True,
    )


def plot(path, rlonly, model_guided, time_limit_s):
    figure, axis = plt.subplots(figsize=(7.6, 4.3))
    axis.plot(
        model_guided["time_s"],
        model_guided["pitch_rms_deg"],
        color=COLORS["Model-guided"],
        linewidth=1.7,
        label="Model-guided",
    )
    axis.plot(
        rlonly["time_s"],
        rlonly["pitch_rms_deg"],
        color=COLORS["RL-only"],
        linewidth=1.7,
        label="RL-only",
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Pitch RMS (deg)")
    axis.set_xlim(0.0, time_limit_s)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, color="#D9D9D9", linewidth=0.6)
    axis.legend(loc="upper right", frameon=True)
    axis.tick_params(direction="out")
    figure.tight_layout()
    figure.savefig(path, dpi=240, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlonly", action="append", required=True, type=Path)
    parser.add_argument("--model-guided", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-valid-fraction", type=float, default=0.25)
    parser.add_argument("--time-limit-s", type=float, default=40.0)
    args = parser.parse_args()

    if not 0.0 < args.min_valid_fraction <= 1.0:
        parser.error("--min-valid-fraction must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rlonly = load_group(args.rlonly, args.min_valid_fraction)
    model_guided = load_group(args.model_guided, args.min_valid_fraction)
    if not np.array_equal(rlonly["time_s"], model_guided["time_s"]):
        raise ValueError("RL-only and Model-guided time axes do not match")

    csv_path = args.output_dir / "pitch_rms_timeseries.csv"
    mat_path = args.output_dir / "pitch_rms_timeseries.mat"
    png_path = args.output_dir / "pitch_rms_timeseries.png"
    write_csv(csv_path, rlonly, model_guided)
    write_mat(mat_path, rlonly, model_guided, args.min_valid_fraction)
    plot(
        png_path,
        rlonly,
        model_guided,
        args.time_limit_s,
    )
    print(csv_path)
    print(mat_path)
    print(png_path)


if __name__ == "__main__":
    main()
