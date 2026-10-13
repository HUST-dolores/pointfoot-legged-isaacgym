#!/usr/bin/env python3
"""Plot raw payload estimate and jump phase from a jump-load MAT file."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


PHASES = {
    0: ("IDLE", "#F2F2F2"),
    1: ("CROUCH", "#DCEAF7"),
    2: ("FLIGHT", "#FCE8C3"),
    3: ("LANDED", "#DDEEDB"),
}


def add_phase_spans(axis, t, state):
    start = 0
    for i in range(1, len(state) + 1):
        if i == len(state) or state[i] != state[start]:
            phase = int(state[start])
            label, color = PHASES.get(phase, (str(phase), "#FFFFFF"))
            axis.axvspan(
                t[start],
                t[min(i, len(t) - 1)],
                color=color,
                alpha=0.55,
                linewidth=0,
            )
            start = i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mat_path")
    parser.add_argument("--output")
    parser.add_argument("--time_start", type=float)
    parser.add_argument("--time_end", type=float)
    args = parser.parse_args()

    data = loadmat(args.mat_path, squeeze_me=True)
    t = np.asarray(data["t"], dtype=float)
    true_mass = np.asarray(data["true_mass"], dtype=float)
    estimate = np.asarray(data["estimate"], dtype=float)
    jump_state = np.asarray(data["jump_state"], dtype=int)
    base_height = np.asarray(data["base_height"], dtype=float)
    base_vz = np.asarray(data["base_vz"], dtype=float)

    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.mat_path)),
        "jump_load_estimation.png",
    )
    figure, axes = plt.subplots(
        2, 1, figsize=(9.0, 6.0), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    for axis in axes:
        add_phase_spans(axis, t, jump_state)
        axis.grid(True, color="#D9D9D9", linewidth=0.6)

    axes[0].plot(t, true_mass, color="black", linewidth=2.2, label="true")
    axes[0].plot(t, estimate, color="#D55E00", linewidth=1.4, label="RL-only")
    axes[0].set_ylabel("Payload mass [kg]")
    axes[0].legend(loc="best")

    axes[1].plot(t, base_height, color="#0072B2", linewidth=1.5, label="Base height")
    axes[1].plot(t, base_vz, color="#009E73", linewidth=1.0, label=r"Base $v_z$")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Height [m] / velocity [m/s]")
    axes[1].legend(loc="best")
    axes[1].set_xlim(
        float(t[0]) if args.time_start is None else args.time_start,
        float(t[-1]) if args.time_end is None else args.time_end,
    )

    figure.tight_layout()
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
