"""Plot aligned measured traces from paired single-environment retention demos."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from .geometry import write_json, sha256


def main(a):
    scenes = [json.loads((p / "scenarios.json").read_text()) for p in (a.baseline, a.trained)]
    assert scenes[0] == scenes[1], "demonstrations must be exactly paired"
    a.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 1, figsize=(8.2, 9.0), sharex=True)
    table = {}
    for directory, label, color, style in ((a.baseline, "Frozen baseline", "#666b70", "--"),
                                           (a.trained, "Retention policy", "#087e8b", "-")):
        with np.load(directory / "trace.npz", allow_pickle=False) as f:
            t = f["time_s"][:, 0]
            root = f["root"][:, 0]
            local = f["local"][:, 0]
            margin = f["margins"][:, 0].min(-1)
            loaded = f["has_load"][:, 0].astype(bool)
            visible_local = np.where(loaded[:, None], local, np.nan)
            visible_margin = np.where(loaded, margin, np.nan)
            force = f["force"][:, 0]
            angles = Rotation.from_quat(root[:, 3:7]).as_euler("xyz", degrees=True)
            axes[0].plot(t, visible_local[:, 0] * 100, style, color=color, label=label)
            axes[1].plot(t, visible_local[:, 1] * 100, style, color=color)
            axes[2].plot(t, visible_margin * 100, style, color=color)
            axes[3].plot(t, angles[:, 1], style, color=color, label=label + " pitch")
            axes[3].plot(t, angles[:, 0], style, color=color, alpha=.5, label=label + " roll")
            axes[4].plot(t, root[:, 0] - root[0, 0], style, color=color)
            dropped = np.flatnonzero(f["drop"][:, 0])
            if len(dropped):
                for ax in axes:
                    ax.axvline(t[dropped[0]], color=color, linestyle=":", alpha=.6)
            table[label] = {"trace_sha256": sha256(directory / "trace.npz"),
                "first_drop_s": float(t[dropped[0]]) if len(dropped) else None,
                "peak_external_force_N": float(np.linalg.norm(force, axis=-1).max()),
                "final_relative_xy_m": local[-1, :2].tolist(),
                "final_robot_displacement_xy_m": (root[-1, :2] - root[0, :2]).tolist()}
    for ax, label in zip(axes, ("Payload x / plate (cm)", "Payload y / plate (cm)",
                                "Min footprint margin (cm)", "Body attitude (deg)", "Robot world-x shift (m)")):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.axvspan(2., 2.04, color="#c55136", alpha=.22)
    axes[2].axhline(0, color="black", lw=.8)
    axes[0].legend(frameon=False)
    axes[3].legend(frameon=False, fontsize=8, ncol=2)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(a.output / ("paired_recovery_timeseries." + ext), dpi=220)
    plt.close(fig)
    write_json(a.output / "timeseries_manifest.json", {"paired_scene": scenes[0], "measurements": table,
        "curves": "recorded simulation states, no synthetic response curves",
        "missing_before_spawn": "payload position and margin undefined before spawn; all post-drop states retained",
        "partial_overhang_is_not_drop": True, "camera_follow_motion_not_used_as_position_measurement": True})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--trained", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
