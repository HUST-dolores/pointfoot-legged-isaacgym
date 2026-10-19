"""Build tables and measured-data figures from completed retention evaluations."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .geometry import write_json, sha256


def label(result):
    checkpoint = result["checkpoint"]
    if checkpoint == "original_frozen_balancing_policy":
        return "Frozen baseline"
    p = Path(checkpoint)
    seed = p.parent.name.rsplit("_s", 1)[-1]
    return "Retention S" + seed if seed.isdigit() else p.parent.name + ":" + p.stem.replace("model_", "")


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(a):
    a.output.mkdir(parents=True, exist_ok=True)
    batches, rows = [], []
    seen = set()
    for directory in a.batches:
        result = json.loads((directory / "result.json").read_text())
        assert abs(result["settings"]["duration"] - 10.) < 1e-6, "Keep long-duration checks separate from the 10 s main protocol"
        assert result["cleanup_completed"] and result["no_pose_writes_after_spawn"] and result["all_trials_single_spawn"]
        cases = json.loads((directory / "cases.json").read_text())
        identity = (label(result), result["seed"])
        if identity in seen:
            raise ValueError(f"Duplicate policy/evaluation-seed batch: {identity}")
        seen.add(identity)
        scenes = json.loads((directory / "scenarios.json").read_text())
        batches.append({"directory": str(directory), "label": label(result), "result": result,
                        "scenario_digest": sha256(directory / "scenarios.json")})
        rows.extend({"policy": label(result), "batch": directory.name, **case} for case in cases)
    # Every comparison within an evaluation seed must use the exact initial placement/pulse list.
    for seed in {b["result"]["seed"] for b in batches}:
        digests = {b["scenario_digest"] for b in batches if b["result"]["seed"] == seed}
        if len(digests) != 1:
            raise ValueError(f"Unpaired scenarios for seed {seed}")
    csv_write(a.output / "all_trials.csv", rows)
    summaries = []
    policy_names = list(dict.fromkeys(row["policy"] for row in rows))
    for policy in policy_names:
        hashes = {b["result"]["checkpoint_sha256"] for b in batches if b["label"] == policy}
        assert len(hashes) == 1, "Do not pool different checkpoints under one training-seed label"
    def stats(group):
        seeds = sorted({r["seed"] for r in group})
        rates = [float(np.mean([r["success"] for r in group if r["seed"] == seed])) for seed in seeds]
        return {"trials": len(group), "successes": sum(r["success"] for r in group),
                "success_rate": float(np.mean([r["success"] for r in group])),
                "success_rate_eval_seed_sd": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.,
                "drop_rate": float(np.mean([r["drop"] for r in group])),
                "fall_rate": float(np.mean([r["fall"] for r in group])),
                "mean_robot_max_displacement_before_failure_m": float(np.mean([r["max_robot_displacement_before_failure_m"] for r in group])),
                "mean_max_supported_relative_motion_m": float(np.mean([r["max_supported_slip_m"] for r in group]))}
    domain_rows, overall, direction_rows = [], {}, []
    for policy in policy_names:
        policy_rows = [r for r in rows if r["policy"] == policy]
        for domain, predicate in (("control", lambda d: d == 0), ("in_domain", lambda d: 0 < d <= .75),
                                  ("stress_extrapolation", lambda d: d > .75)):
            group = [r for r in policy_rows if predicate(r["nominal_dv"])]
            if group:
                domain_rows.append({"policy": policy, "domain": domain, **stats(group)})
        overall[policy] = stats([r for r in policy_rows if r["nominal_dv"] > 0])
        for dv, direction in sorted({(r["nominal_dv"], r["direction"]) for r in policy_rows}):
            group = [r for r in policy_rows if r["nominal_dv"] == dv and r["direction"] == direction]
            direction_rows.append({"policy": policy, "nominal_dv_mps": dv, "direction": direction, **stats(group)})
    csv_write(a.output / "performance_by_domain.csv", domain_rows)
    csv_write(a.output / "performance_by_direction.csv", direction_rows)
    write_json(a.output / "summary.json", {"disjoint_domain_groups": domain_rows, "all_disturbed": overall,
        "unique_trials": len(rows), "evaluation_seeds": sorted({r["seed"] for r in rows}),
        "failure_trials_not_removed": True, "drift_is_not_position_hold_error": True,
        "relative_motion_diagnostic_is_not_pure_contact_slip": True})
    for policy in policy_names:
        for dv in (0., .25, .45, .65, .85):
            group = [r for r in rows if r["policy"] == policy and r["nominal_dv"] == dv]
            if not group:
                continue
            summaries.append({"policy": policy, "nominal_impulse_dv_mps": dv, "trials": len(group),
                "success_rate": float(np.mean([r["success"] for r in group])),
                "drop_rate": float(np.mean([r["drop"] for r in group])),
                "fall_rate": float(np.mean([r["fall"] for r in group])),
                "supported_sliding_rate": float(np.mean([r["sliding_observed"] for r in group])),
                "mean_max_supported_slip_m": float(np.mean([r["max_supported_slip_m"] for r in group])),
                "mean_robot_max_displacement_m": float(np.mean([r["max_robot_displacement_before_failure_m"] for r in group]))})
    csv_write(a.output / "performance_by_strength.csv", summaries)
    directions = ["forward", "backward", "left", "right"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#60666b", "#087e8b", "#c55136", "#659947", "#a66caa", "#d99b2b"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.0), sharex=True, sharey=True)
    for ax, direction in zip(axes.flat, directions):
        for series, (policy, color) in enumerate(zip(policy_names, colors)):
            means, errors = [], []
            for dv in (.25, .45, .65, .85):
                group = [r for r in rows if r["policy"] == policy and r["direction"] == direction and r["nominal_dv"] == dv]
                if not group:
                    means.append(np.nan)
                    errors.append(np.nan)
                    continue
                seed_means = [np.mean([r["success"] for r in group if r["seed"] == s]) for s in sorted({r["seed"] for r in group})]
                means.append(np.mean(seed_means))
                errors.append(np.std(seed_means, ddof=1) if len(seed_means) > 1 else 0.)
            x = np.array([.25, .45, .65, .85])
            y, e = np.array(means) * 100., np.array(errors) * 100.
            style, marker = (("--", "s"), ("-", "o"), (":", "^"))[series % 3]
            ax.plot(x, y, linestyle=style, marker=marker, color=color, label=policy, lw=1.5, ms=4.5,
                    markerfacecolor="white" if series > 1 else color)
            ax.fill_between(x, np.clip(y - e, 0., 100.), np.clip(y + e, 0., 100.), color=color, alpha=.13)
        ax.set_title(direction.capitalize())
        ax.set_ylim(-3, 103)
        ax.set_xticks([.25, .45, .65, .85])
        ax.grid(alpha=.2)
        ax.axvline(.75, color="0.7", ls=":", lw=1.)
    for ax in axes[-1]:
        ax.set_xlabel("Nominal impulse / payload mass (m/s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("10 s retention success (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, 1))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(a.output / ("retention_by_direction." + ext), dpi=220)
    plt.close(fig)
    write_json(a.output / "report_manifest.json", {"batches": [{k: v for k, v in b.items() if k != "result"} for b in batches],
        "error_band": "sample SD of evaluation-seed success rates, not training-seed uncertainty",
        "vertical_line": "training maximum nominal impulse-dv .75 m/s; .85 is a stress extrapolation",
        "all_assigned_trials_kept": True, "initial_scenarios_exactly_paired": True,
        "no_disturbance_controls": "dv=0 rows in performance_by_strength.csv", "runtime_seconds_is_not_simulated_seconds": True})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
