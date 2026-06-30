#!/usr/bin/env python3
"""Analyze the WF_TRON1A payload load-estimation experiment.

The CSV is expected to live beside this script. The script produces the same
three PNG outputs described by the original MATLAB README.
"""

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_DIR / "wf_deploy_log_real_load_fall_latest.csv"

LOAD_MASSES = np.array([4.425, 3.48, 4.50])
ADD_TIMES = np.array([135.4, 149.0, 158.1])
REMOVE_TIMES = np.array([166.0, 169.8, 177.6])
RIGHT_ADD_TIMES = np.array([182.5, 192.1])
FALL_TIME = 198.75
MASS_PLOT_XLIM = (130.0, 180.0)
MASS_EST_OFFSET = 1.0
BODY_MASS = 9.58
BODY_COM0 = np.array([0.04576, 0.00014, -0.16398])


def require_columns(table: pd.DataFrame, columns: List[str]) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")


def add_vertical_marker(ax: plt.Axes, x: float, label: str, color: str, style: str,
                        y: float = 0.98, va: str = "top") -> None:
    ax.axvline(x, color=color, linestyle=style, linewidth=1.0)
    offset_y = -4 if va == "top" else 4
    ax.annotate(
        label,
        xy=(x, y),
        xycoords=("data", "axes fraction"),
        xytext=(3, offset_y),
        textcoords="offset points",
        ha="left",
        va=va,
        fontsize=8,
        color=color,
    )


def marker_is_visible(x: float, xlim: tuple) -> bool:
    return xlim[0] <= x <= xlim[1]


def add_load_markers(ax: plt.Axes, include_right_adds: bool = False,
                     xlim: tuple = None) -> None:
    for index, add_time in enumerate(ADD_TIMES, start=1):
        if xlim is not None and not marker_is_visible(add_time, xlim):
            continue
        add_vertical_marker(ax, add_time, f"add {index}", "black", "--")
    for index, (remove_index, remove_time) in enumerate(zip([3, 2, 1], REMOVE_TIMES)):
        if xlim is not None and not marker_is_visible(remove_time, xlim):
            continue
        add_vertical_marker(
            ax,
            remove_time,
            f"remove {remove_index}",
            "m",
            "--",
            y=0.98 - 0.07 * (index % 2),
        )
    if include_right_adds:
        for index, add_time in enumerate(RIGHT_ADD_TIMES, start=1):
            if xlim is not None and not marker_is_visible(add_time, xlim):
                continue
            add_vertical_marker(
                ax,
                add_time,
                f"right add {index}",
                "black",
                ":",
                y=0.08 + 0.08 * (index - 1),
                va="bottom",
            )
    if xlim is None or marker_is_visible(FALL_TIME, xlim):
        add_vertical_marker(ax, FALL_TIME, "fall", "r", "--")


def save_figure(fig: plt.Figure, filename: str) -> Path:
    output_path = SCRIPT_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def time_weighted_mean(t: np.ndarray, values: np.ndarray, start: float, end: float) -> float:
    idx = (t > start) & (t < end)
    t_window = np.concatenate(([start], t[idx], [end]))
    if t_window.size < 2:
        return float("nan")
    values_window = np.interp(t_window, t, values)
    return float(np.trapz(values_window, t_window) / (end - start))


def actual_loaded_mean() -> float:
    phase_starts = np.array([ADD_TIMES[0], ADD_TIMES[1], ADD_TIMES[2],
                             REMOVE_TIMES[0], REMOVE_TIMES[1]])
    phase_ends = np.array([ADD_TIMES[1], ADD_TIMES[2], REMOVE_TIMES[0],
                           REMOVE_TIMES[1], REMOVE_TIMES[2]])
    phase_masses = np.array([
        LOAD_MASSES[0],
        LOAD_MASSES[0] + LOAD_MASSES[1],
        LOAD_MASSES.sum(),
        LOAD_MASSES[0] + LOAD_MASSES[1],
        LOAD_MASSES[0],
    ])
    return float(np.sum((phase_ends - phase_starts) * phase_masses)
                 / (REMOVE_TIMES[-1] - ADD_TIMES[0]))


def infer_payload_position_from_robot_com_delta(
    com_delta: np.ndarray,
    payload_mass: np.ndarray,
) -> np.ndarray:
    payload_pos = np.full_like(com_delta, np.nan, dtype=float)
    valid = payload_mass > 1e-6
    scale = (BODY_MASS + payload_mass[valid]) / payload_mass[valid]
    payload_pos[valid] = BODY_COM0 + scale[:, None] * com_delta[valid]
    return payload_pos


def main() -> None:
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

    table = pd.read_csv(CSV_FILE)
    require_columns(table, ["t", "mass_est", "comx", "comy", "comz"])

    t = table["t"].to_numpy(dtype=float)
    mass_est_raw = table["mass_est"].to_numpy(dtype=float)
    mass_est = mass_est_raw + MASS_EST_OFFSET
    comx = table["comx"].to_numpy(dtype=float)
    comy = table["comy"].to_numpy(dtype=float)
    comz = table["comz"].to_numpy(dtype=float)
    robot_com_delta = np.column_stack((comx, comy, comz))

    mass_smooth = (
        pd.Series(mass_est)
        .rolling(window=51, center=True, min_periods=1)
        .median()
        .to_numpy()
    )

    actual_times = np.array([t.min(), *ADD_TIMES, *REMOVE_TIMES, t.max()])
    actual_mass = np.array([
        0.0,
        LOAD_MASSES[0],
        LOAD_MASSES[0] + LOAD_MASSES[1],
        LOAD_MASSES.sum(),
        LOAD_MASSES[0] + LOAD_MASSES[1],
        LOAD_MASSES[0],
        0.0,
        0.0,
    ])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, mass_est, color=(0.75, 0.75, 0.75),
            label=f"estimated raw +{MASS_EST_OFFSET:.1f} kg")
    ax.plot(t, mass_smooth, color="b", linewidth=1.5,
            label=f"estimated movmedian +{MASS_EST_OFFSET:.1f} kg")
    ax.step(actual_times, actual_mass, where="post", color="r", linewidth=2.0,
            label="actual payload")
    add_load_markers(ax, xlim=MASS_PLOT_XLIM)
    ax.grid(True)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mass [kg]")
    ax.set_title("Estimated payload mass vs actual payload mass")
    ax.set_xlim(MASS_PLOT_XLIM)
    ax.legend(loc="lower center")
    mass_png = save_figure(fig, "payload_mass_comparison.png")

    print("\nApproximate middle-load mass comparison")
    print(
        "Assumed load masses: "
        f"{LOAD_MASSES[0]:.3f}, {LOAD_MASSES[1]:.3f}, {LOAD_MASSES[2]:.3f} kg"
    )
    print(
        "Add times: "
        f"{ADD_TIMES[0]:.1f}, {ADD_TIMES[1]:.1f}, {ADD_TIMES[2]:.1f} s; "
        "remove times, last-in first-out: "
        f"{REMOVE_TIMES[0]:.1f}, {REMOVE_TIMES[1]:.1f}, {REMOVE_TIMES[2]:.1f} s\n"
    )
    print(f"Applied estimated mass offset: +{MASS_EST_OFFSET:.3f} kg\n")
    print(f"{'window':>11s} {'actual_kg':>14s} {'est_med_kg':>14s} {'error_kg':>14s}")

    plateau_starts = ADD_TIMES
    plateau_ends = np.array([ADD_TIMES[1], ADD_TIMES[2], REMOVE_TIMES[0]])
    actual_plateaus = np.cumsum(LOAD_MASSES)

    for start, end, actual in zip(plateau_starts, plateau_ends, actual_plateaus):
        t0 = start + 2.0
        t1 = min(end - 1.0, t0 + 8.0)
        idx = (t >= t0) & (t <= t1)
        est_med = np.nanmedian(mass_est[idx])
        print(f"{t0:5.1f}-{t1:<5.1f} {actual:14.3f} {est_med:14.3f} {est_med - actual:14.3f}")

    loaded_start = ADD_TIMES[0]
    loaded_end = REMOVE_TIMES[-1]
    actual_mean = actual_loaded_mean()
    raw_mean = time_weighted_mean(t, mass_est, loaded_start, loaded_end)
    smooth_mean = time_weighted_mean(t, mass_smooth, loaded_start, loaded_end)
    print("\nWhole loaded-period mean bias")
    print(f"Loaded period: {loaded_start:.1f}-{loaded_end:.1f} s")
    print(f"Actual red mean:          {actual_mean:8.3f} kg")
    print(f"Estimated raw mean:       {raw_mean:8.3f} kg, residual {actual_mean - raw_mean:8.3f} kg")
    print(
        f"Estimated movmedian mean: {smooth_mean:8.3f} kg, "
        f"residual {actual_mean - smooth_mean:8.3f} kg"
    )

    payload_pos = infer_payload_position_from_robot_com_delta(robot_com_delta, mass_est)
    loaded_mask = (t >= ADD_TIMES[0]) & (t < REMOVE_TIMES[-1])
    payload_pos_plot = payload_pos.copy()
    payload_pos_plot[~loaded_mask] = np.nan

    print("\nPayload position inferred from robot COM delta")
    print(
        "Formula: payload_pos = body_com0 + "
        "(body_mass + payload_mass) / payload_mass * robot_com_delta"
    )
    print(f"body_mass = {BODY_MASS:.3f} kg, body_com0 = {BODY_COM0.tolist()}")
    print(f"{'window':>11s} {'load_x_m':>12s} {'load_y_m':>12s} {'load_z_m':>12s}")
    for start, end in zip(plateau_starts, plateau_ends):
        t0 = start + 2.0
        t1 = min(end - 1.0, t0 + 8.0)
        idx = (t >= t0) & (t <= t1)
        med = np.nanmedian(payload_pos[idx], axis=0)
        print(f"{t0:5.1f}-{t1:<5.1f} {med[0]:12.3f} {med[1]:12.3f} {med[2]:12.3f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, comx, linewidth=1.2, label="comx")
    ax.plot(t, comy, linewidth=1.2, label="comy")
    ax.plot(t, comz, linewidth=1.2, label="comz")
    add_load_markers(ax, include_right_adds=True)
    ax.grid(True)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("estimated robot COM offset [m]")
    ax.set_title("Estimated robot COM offset coordinates")
    ax.legend(loc="best")
    com_png = save_figure(fig, "payload_com_coordinates.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(comx, comy, c=t, s=8, cmap="viridis")
    ax.plot(comx, comy, color=(0.7, 0.7, 0.7, 0.35))
    ax.plot(comx[0], comy[0], "go", markersize=8, linewidth=1.5, label="start")
    fall_idx = int(np.argmin(np.abs(t - FALL_TIME)))
    ax.plot(comx[fall_idx], comy[fall_idx], "rx", markersize=10, linewidth=2.0,
            label="near fall")
    ax.grid(True)
    ax.axis("equal")
    fig.colorbar(scatter, ax=ax, label="time [s]")
    ax.set_xlabel("robot COM delta x [m]")
    ax.set_ylabel("robot COM delta y [m]")
    ax.set_title("Estimated robot COM offset x-y trajectory, color = time [s]")
    ax.legend(loc="best")
    xy_png = save_figure(fig, "payload_com_xy_trajectory.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, payload_pos_plot[:, 0], linewidth=1.2, label="payload x inferred")
    ax.plot(t, payload_pos_plot[:, 1], linewidth=1.2, label="payload y inferred")
    ax.plot(t, payload_pos_plot[:, 2], linewidth=1.2, label="payload z inferred")
    add_load_markers(ax, xlim=MASS_PLOT_XLIM)
    ax.grid(True)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("estimated payload position [m]")
    ax.set_title("Payload position inferred from robot COM offset")
    ax.set_xlim(MASS_PLOT_XLIM)
    ax.legend(loc="best")
    payload_pos_png = save_figure(fig, "payload_position_from_robot_com.png")

    print("\nSaved figures:")
    for output_path in [mass_png, com_png, xy_png, payload_pos_png]:
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
