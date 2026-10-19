#!/usr/bin/env python3
"""Validate real paired rollouts, retain failures, and export paper figures."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from legged_gym.scripts.payload_step_protocol import (
    DT, MASS_KG, POLICIES, SEEDS, sha256_file, trajectory_events, write_json,
)


MOTION = ("speed_rmse_mps", "pitch_rms_deg", "pitch_sd_deg", "pitch_mean_deg",
          "speed_mean_mps", "speed_sd_mps", "speed_bias_mps")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    usable = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(usable)) if usable else None


def rms(value):
    return float(np.sqrt(np.mean(np.square(value))))


def pitch_deg(trace, env_id):
    q = trace["root_state"][:, env_id, 3:7].astype(np.float64)
    return np.rad2deg(np.arcsin(np.clip(2 * (q[:, 3] * q[:, 1] - q[:, 2] * q[:, 0]), -1, 1)))


def motion_metrics(trace, env_id, mask):
    if not np.any(mask):
        return {key: None for key in MOTION}
    pitch = pitch_deg(trace, env_id)[mask]
    speed = trace["body_velocity"][mask, env_id, 0].astype(np.float64)
    return {"speed_rmse_mps": rms(speed - 0.9), "pitch_rms_deg": rms(pitch),
            "pitch_sd_deg": float(np.std(pitch)), "pitch_mean_deg": float(np.mean(pitch)),
            "speed_mean_mps": float(np.mean(speed)), "speed_sd_mps": float(np.std(speed)),
            "speed_bias_mps": float(np.mean(speed) - 0.9)}


def estimation_metrics(trace, env_id, mask, suffix):
    fields = ["mass_rmse_kg"] + [f"com_{axis}_rmse_m" for axis in "xyz"]
    fields += ["com_xyz_rmse_m"]
    if not np.any(mask):
        return {f"{key}_{suffix}": None for key in fields}
    mass_err = trace["estimate"][mask, env_id, 3] - trace["mass_label"][mask, env_id]
    com_err = trace["estimate"][mask, env_id, 4:7] - trace["com_label"][mask, env_id]
    values = [rms(mass_err)] + [rms(com_err[:, i]) for i in range(3)]
    values += [float(np.sqrt(np.mean(np.sum(com_err ** 2, axis=-1))))]
    return {f"{key}_{suffix}": value for key, value in zip(fields, values)}


def aggregate(rows, groups, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in groups)].append(row)
    result = []
    for keys, members in sorted(grouped.items()):
        row = dict(zip(groups, keys))
        row["unit_count"] = len(members)
        for metric in metrics:
            row[metric] = mean(m.get(metric) for m in members)
            row[f"n_{metric}"] = sum(m.get(metric) is not None for m in members)
        result.append(row)
    return result


def load_and_validate(root, preflight=False):
    records = {}
    folder = root / ("preflight" if preflight else "batches")
    for path in sorted(folder.glob("*/result.json")):
        meta = json.loads(path.read_text())
        if meta["minimal_recorder"]:
            continue
        with np.load(path.parent / "trace.npz", allow_pickle=False) as saved:
            trace = {key: value.copy() for key, value in saved.items()}
        if sha256_file(path.parent / "trace.npz") != meta["trace_sha256"]:
            raise ValueError(f"trace checksum mismatch: {path}")
        np.testing.assert_allclose(meta["measured_payload_mass_kg"], MASS_KG, atol=2e-6)
        records[(meta["policy_id"], meta["evaluation_seed"], meta["condition"])] = (meta, trace, path.parent)
    seeds = (12345,) if preflight else SEEDS
    expected = {(p, s, c) for p in POLICIES for s in seeds for c in ("empty", "free_box")}
    if set(records) != expected:
        raise ValueError(f"missing/extra batches: missing={expected-set(records)}, extra={set(records)-expected}")
    audits = []
    for (policy, seed, condition), (meta, z, directory) in records.items():
        steps, count = z["root_state"].shape[:2]
        np.testing.assert_allclose(z["time_s"], np.arange(1, steps + 1) * DT, atol=5e-6)
        np.testing.assert_allclose(z["task_commands"], np.broadcast_to([.5, 0, 0, .4], (steps, count, 4)), atol=1e-7)
        np.testing.assert_allclose(z["actor_commands"], np.broadcast_to([1., 0, 0], (steps, count, 3)), atol=1e-7)
        np.testing.assert_allclose(z["observations"][:, :, -3], .4, atol=1e-7)
        np.testing.assert_allclose(z["mass_label"], z["on_body"] * MASS_KG, atol=3e-6)
        factor = z["mass_label"] / (z["base_mass0"][None] + z["mass_label"])
        expected_com = factor[..., None] * (z["box_relative_position"] - z["base_com0"][None])
        np.testing.assert_allclose(z["com_label"], expected_com, atol=1e-6)
        if condition == "empty":
            if z["has_load"].any() or z["on_body"].any():
                raise ValueError(f"unexpected box in empty baseline: {policy}/{seed}")
        else:
            first_spawn = np.flatnonzero(z["has_load"][:, 0])[0]
            np.testing.assert_allclose(z["time_s"][first_spawn], .5, atol=2e-6)
        if (np.diff(z["first_failure"].astype(int), axis=0) < 0).any():
            raise ValueError("failure masks rearmed: a reset may have contaminated the first trial")
        if not np.isfinite(z["actions"]).all() or not np.isfinite(z["estimate"]).all():
            raise ValueError("nonfinite action/estimate")
        scale = np.array([.05, 5, 5, 5], dtype=np.float32)
        if policy.startswith("mg"):
            expected_est = z["network_raw_mass_com"] + z["observations"][:, :, 36:40] / scale
        else:
            expected_est = z["network_raw_mass_com"]
        np.testing.assert_allclose(z["estimate"][:, :, 3:7], expected_est, atol=3e-5, rtol=2e-6)
        props = json.loads((directory / "actual_body_properties.json").read_text())
        np.testing.assert_allclose(z["base_mass0"], [row["mass_kg"][0] for row in props], atol=2e-6)
        audits.append({"batch_id": meta["batch_id"], "commands_constant": True,
                       "mass_and_com_supervision_verified": True, "estimate_fusion_verified": True,
                       "no_reset_contamination": True, "samples": steps * count,
                       "robot_failures": int(z["first_failure"][-1].sum())})
    for seed in seeds:
        reference = records[("rl_s1", seed, "empty")]
        for policy in POLICIES:
            for condition in ("empty", "free_box"):
                meta, z, path = records[(policy, seed, condition)]
                ref_meta, ref, ref_path = reference
                if meta["scene_array_sha256"] != ref_meta["scene_array_sha256"]:
                    raise ValueError("unpaired scene digest")
                for key in ("base_mass0", "base_com0", "imu_offset_quaternion", "noise_injected", "noise_effective"):
                    np.testing.assert_array_equal(z[key], ref[key], err_msg=f"unpaired {key}")
                np.testing.assert_allclose(z["root_state"][0], ref["root_state"][0], atol=1e-6,
                                           err_msg="initial zero-step state differs")
                if json.loads((path / "actual_body_properties.json").read_text()) != json.loads((ref_path / "actual_body_properties.json").read_text()):
                    raise ValueError("physical body properties differ across paired scenes")
        for policy in POLICIES:
            empty = records[(policy, seed, "empty")][1]
            loaded = records[(policy, seed, "free_box")][1]
            before_load = empty["time_s"] < .5 - 1e-6
            np.testing.assert_allclose(empty["root_state"][before_load], loaded["root_state"][before_load],
                                       atol=1e-6, err_msg="paired trajectories differ before loading")
    return records, {"status": "passed", "batch_count": len(records), "batches": audits,
                     "paired_scene_physics_noise_and_initial_state_verified": True,
                     "scope": "preflight" if preflight else "formal"}


def analyze(records, output):
    per_env, all_events, paired = [], [], []
    event_lookup = {}
    for (policy, seed, condition), (meta, z, _) in sorted(records.items()):
        for env_id in range(meta["num_envs"]):
            event = trajectory_events(z, env_id, condition, meta["duration_s"])
            identity = {key: meta[key] for key in ("policy_id", "method", "training_repeat", "training_seed",
                                                  "evaluation_seed", "condition")}
            identity.update(env_id=env_id, pair_id=f"{seed}:{env_id}")
            all_events.append({**identity, **event})
            event_lookup[(policy, seed, condition, env_id)] = event
            t = z["time_s"]
            before_failure = ~z["first_failure"][:, env_id]
            own = (t >= 2 - 1e-6) & (t < event["motion_end_exclusive_s"] - 1e-7) & before_failure
            post = (t >= 2 - 1e-6) & before_failure
            on = post & z["on_body"][:, env_id]
            transient = (t >= .5 - 1e-6) & (t < 2 - 1e-6) & before_failure
            row = {**identity, "robot_completed": int(event["robot_completed"]),
                   "loaded_completed": None if condition == "empty" else int(event["loaded_completed"]),
                   "catch_failed": None if condition == "empty" else int(event["catch_failed"]),
                   "drop_confirmed": None if condition == "empty" else int(event["drop_confirmed"]),
                   "motion_samples": int(own.sum()), "motion_valid_s": float(own.sum() * DT),
                   "supported_samples": int(on.sum()), "on_body_fraction": event["on_body_fraction"],
                   **motion_metrics(z, env_id, own),
                   **estimation_metrics(z, env_id, post, "all_supervision"),
                   **estimation_metrics(z, env_id, on, "supported"),
                   **estimation_metrics(z, env_id, transient, "transient")}
            per_env.append(row)
    for policy in POLICIES:
        for seed in SEEDS:
            empty = records[(policy, seed, "empty")][1]
            meta, loaded, _ = records[(policy, seed, "free_box")]
            for env_id in range(meta["num_envs"]):
                le = event_lookup[(policy, seed, "free_box", env_id)]
                ee = event_lookup[(policy, seed, "empty", env_id)]
                end = min(le["motion_end_exclusive_s"], ee["motion_end_exclusive_s"])
                mask = (loaded["time_s"] >= 2 - 1e-6) & (loaded["time_s"] < end - 1e-7)
                loaded_metrics = motion_metrics(loaded, env_id, mask)
                empty_metrics = motion_metrics(empty, env_id, mask)
                row = {key: meta[key] for key in ("policy_id", "method", "training_repeat", "training_seed", "evaluation_seed")}
                row.update(env_id=env_id, pair_id=f"{seed}:{env_id}", common_samples=int(mask.sum()),
                           common_valid_s=float(mask.sum() * DT),
                           both_full_horizon=int(le["loaded_completed"] and ee["robot_completed"]))
                for metric in MOTION:
                    lval, eval_ = loaded_metrics[metric], empty_metrics[metric]
                    row[f"loaded_{metric}"] = lval
                    row[f"empty_{metric}"] = eval_
                    row[f"delta_{metric}"] = None if lval is None else lval - eval_
                paired.append(row)
    excluded = {"env_id", "training_repeat", "training_seed", "evaluation_seed"}
    numeric = [k for k, v in per_env[0].items() if k not in excluded and isinstance(v, (int, float))]
    numeric += [k for k in per_env[0] if k not in excluded and k not in numeric and
                any(isinstance(row.get(k), (int, float)) for row in per_env)]
    eval_rows = aggregate(per_env, ("policy_id", "method", "condition", "evaluation_seed"), numeric)
    runs = aggregate(eval_rows, ("policy_id", "method", "condition"), numeric)
    methods = aggregate(runs, ("method", "condition"), numeric)
    pair_numeric = [k for k, v in paired[0].items() if k not in excluded and isinstance(v, (int, float))]
    pair_numeric = list(dict.fromkeys(pair_numeric + [k for k in paired[0] if k.startswith(("loaded_", "empty_", "delta_"))]))
    pair_eval = aggregate(paired, ("policy_id", "method", "evaluation_seed"), pair_numeric)
    pair_runs = aggregate(pair_eval, ("policy_id", "method"), pair_numeric)
    pair_methods = aggregate(pair_runs, ("method",), pair_numeric)
    for name, rows in (("trajectory_metrics", per_env), ("trajectory_events", all_events),
                       ("paired_trajectory_metrics", paired), ("evaluation_seed_summary", eval_rows),
                       ("training_run_summary", runs), ("method_summary", methods),
                       ("paired_evaluation_seed_summary", pair_eval), ("paired_training_run_summary", pair_runs),
                       ("paired_method_summary", pair_methods)):
        write_csv(output / f"{name}.csv", rows)
    example = []
    for policy in ("rl_s1", "mg_s1"):
        meta, z, _ = records[(policy, 12345, "free_box")]
        for index, t in enumerate(z["time_s"]):
            row = {"policy_id": policy, "evaluation_seed": 12345, "env_id": 0,
                   "time_s": float(t), "valid_before_robot_failure": int(not z["first_failure"][index, 0]),
                   "mass_label_kg": float(z["mass_label"][index, 0]),
                   "mass_estimate_kg": float(z["estimate"][index, 0, 3]),
                   "on_body": int(z["on_body"][index, 0]),
                   "left_wheel_Fz_N": float(z["wheel_forces"][index, 0, 0, 2]),
                   "right_wheel_Fz_N": float(z["wheel_forces"][index, 0, 1, 2])}
            for dim, axis in enumerate("xyz"):
                row[f"base_com_delta_{axis}_label_m"] = float(z["com_label"][index, 0, dim])
                row[f"base_com_delta_{axis}_estimate_m"] = float(z["estimate"][index, 0, dim + 4])
                row[f"box_relative_{axis}_m"] = float(z["box_relative_position"][index, 0, dim])
            example.append(row)
    write_csv(output / "preregistered_example_timeseries.csv", example)
    from scipy.io import savemat
    matlab = {"load_mass_kg": MASS_KG,
              "supervision": "effective-base CoM shift, not box position",
              "coordinates": "base frame; positions in metres; quaternion xyzw"}
    for policy in ("rl_s1", "mg_s1"):
        z = records[(policy, 12345, "free_box")][1]
        matlab["time_s"] = z["time_s"]
        matlab[policy] = {
            "mass_label_kg": z["mass_label"][:, 0], "estimate_mass_kg": z["estimate"][:, 0, 3],
            "com_delta_truth_m": z["com_label"][:, 0], "estimate_com_delta_m": z["estimate"][:, 0, 4:7],
            "box_relative_position_m": z["box_relative_position"][:, 0],
            "wheel_contact_forces_N": z["wheel_forces"][:, 0], "first_failure": z["first_failure"][:, 0],
            "body_velocity_mps": z["body_velocity"][:, 0], "root_state": z["root_state"][:, 0],
            "task_commands": z["task_commands"][:, 0], "on_body": z["on_body"][:, 0]}
    savemat(output / "preregistered_example_timeseries.mat", matlab,
            do_compression=True, long_field_names=True)
    return per_env, paired, runs, methods, pair_runs, pair_methods


def figures(records, pair_runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from legged_gym.scripts.figs_chx.fig0_sixd_design_variables import configure_fonts
    from matplotlib import font_manager
    import shutil
    first_batch_directory = next(iter(records.values()))[2]
    archived_fonts = first_batch_directory.parents[1] / "analysis" / "fonts"
    if (archived_fonts / "CodesignNotoSansSC-Regular.ttf").exists():
        (output / "fonts").mkdir(parents=True, exist_ok=True)
        for source in archived_fonts.glob("*.ttf"):
            target = output / "fonts" / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            font_manager.fontManager.addfont(str(target))
        plt.rcParams.update({"font.family": "sans-serif",
                             "font.sans-serif": ["Codesign Noto Sans SC", "DejaVu Sans"],
                             "pdf.fonttype": 42, "ps.fonttype": 42,
                             "svg.fonttype": "path", "axes.unicode_minus": False})
    else:
        configure_fonts(output, extra_text=Path(__file__).read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "normal", "axes.labelweight": "normal"})
    colors = {"RL-only": "#347BB4", "Model-guided": "#1C927C"}
    for language in ("zh", "en"):
        chinese = language == "zh"
        for window, xmax in (("full", 40), ("loading", 8)):
            fig, axes = plt.subplots(4, 2, figsize=(13.8, 10.5), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1, 1, .50]})
            for col, policy in enumerate(("rl_s1", "mg_s1")):
                meta, z, _ = records[(policy, 12345, "free_box")]
                time = z["time_s"]
                valid = ~z["first_failure"][:, 0]
                shown = (time <= xmax + 1e-6) & valid
                for row, (target, estimate) in enumerate(((z["mass_label"][:, 0], z["estimate"][:, 0, 3]),
                                                         (z["com_label"][:, 0, 0] * 100, z["estimate"][:, 0, 4] * 100),
                                                         (z["com_label"][:, 0, 1] * 100, z["estimate"][:, 0, 5] * 100))):
                    ax = axes[row, col]
                    ax.plot(time[shown], target[shown], color="#29343B", lw=1.6,
                            label="监督真值" if chinese else "Supervision truth")
                    ax.plot(time[shown], estimate[shown], color="#1C927C", lw=1.25,
                            label="最终估计" if chinese else "Final estimate")
                    if policy == "mg_s1":
                        prior = z["observations"][:, 0, 36 + row] / (.05 if row == 0 else 5)
                        ax.plot(time[shown], prior[shown] * (1 if row == 0 else 100),
                                color="#C28C36", lw=1, alpha=.85, label="解析先验" if chinese else "Analytic prior")
                    ax.grid(axis="y", color="#DEE3E7", lw=.6)
                event = trajectory_events(z, 0, "free_box")
                for row in range(4):
                    axes[row, col].axvline(.5, color="#8B5B9A", ls="--", lw=.9)
                    if event["support_established_s"] is not None and event["support_established_s"] <= xmax:
                        axes[row, col].axvline(event["support_established_s"], color="#7E8A90", ls="-.", lw=.8)
                    if event["first_robot_failure_s"] is not None and event["first_robot_failure_s"] <= xmax:
                        axes[row, col].axvline(event["first_robot_failure_s"], color="#BE5349", lw=1)
                    if event["first_drop_start_s"] is not None and event["first_drop_start_s"] <= xmax:
                        axes[row, col].axvline(event["first_drop_start_s"], color="#BE5349", lw=1, ls=":")
                for wheel in range(2):
                    contact = z["wheel_forces"][:, 0, wheel, 2] > 1.0
                    y0 = 1 - wheel
                    axes[3, col].fill_between(time, y0, y0 + .65, where=contact & valid,
                                              step="mid", color=("#7386B3", "#BA9B52")[wheel])
                axes[3, col].set_yticks([.325, 1.325])
                axes[3, col].set_yticklabels(["右轮", "左轮"] if chinese else ["R", "L"])
                axes[3, col].set_ylim(-.1, 1.85)
                axes[3, col].set_xlim(0, xmax)
                axes[3, col].set_xlabel("时间 (s)" if chinese else "Time (s)")
            for row in range(3):
                bounds = [ax.get_ylim() for ax in axes[row]]
                limits = min(b[0] for b in bounds), max(b[1] for b in bounds)
                for ax in axes[row]:
                    ax.set_ylim(*limits)
            if window == "full":
                from matplotlib.ticker import FuncFormatter
                for ax in axes[0]:
                    ax.set_yscale("symlog", linthresh=10, linscale=1.3)
                    ax.set_yticks([0, 5.93, 20, 100])
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            mass_label = ("监督质量 (kg, symlog)" if chinese else "Supervised mass (kg, symlog)")
            if window != "full":
                mass_label = "监督质量 (kg)" if chinese else "Supervised mass (kg)"
            axes[0, 0].set_ylabel(mass_label)
            axes[1, 0].set_ylabel("质心增量 x (cm)" if chinese else "Base CoM shift x (cm)")
            axes[2, 0].set_ylabel("质心增量 y (cm)" if chinese else "Base CoM shift y (cm)")
            axes[3, 0].set_ylabel("轮接触" if chinese else "Wheel contact")
            handles, labels = axes[0, 1].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .995))
            fig.subplots_adjust(left=.10, right=.98, bottom=.07, top=.895, hspace=.20, wspace=.16)
            for col, method in enumerate(("RL-only", "Model-guided")):
                box = axes[0, col].get_position()
                fig.text((box.x0 + box.x1) / 2, .920, method + " / s1",
                         ha="center", va="bottom", fontsize=14)
            save_figure(fig, output / f"payload_sensing_contact_{window}_{language}")
            plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.4))
        keys = ("speed_rmse_mps", "pitch_rms_deg", "pitch_sd_deg")
        names = (["速度跟踪 RMSE (m/s)", "俯仰 RMS (deg)", "俯仰标准差 (deg)"] if chinese else
                 ["Speed RMSE (m/s)", "Pitch RMS (deg)", "Pitch SD (deg)"])
        for col, (key, label) in enumerate(zip(keys, names)):
            for method_index, method in enumerate(("RL-only", "Model-guided")):
                selected = [row for row in pair_runs if row["method"] == method]
                for condition, shift, filled in (("empty", -.12, False), ("loaded", .12, True)):
                    values = [row[f"{condition}_{key}"] for row in selected]
                    x = method_index + shift
                    for index, value in enumerate(values):
                        if value is not None:
                            axes[0, col].scatter(x + (index - .5) * .045, value, s=45,
                                marker=("o", "s")[index], facecolor=colors[method] if filled else "white",
                                edgecolor=colors[method], zorder=3)
                    value_mean = mean(values)
                    if value_mean is not None:
                        axes[0, col].plot([x-.065, x+.065], [value_mean]*2, color=colors[method], lw=2)
                values = [row[f"delta_{key}"] for row in selected]
                for index, value in enumerate(values):
                    if value is not None:
                        axes[1, col].scatter(method_index + (index-.5)*.07, value, s=50,
                            marker=("o", "s")[index], color=colors[method], zorder=3)
                value_mean = mean(values)
                if value_mean is not None:
                    axes[1, col].plot([method_index-.12, method_index+.12], [value_mean]*2,
                                      color=colors[method], lw=2)
            axes[0, col].set_ylabel(label)
            axes[1, col].set_ylabel(("带载 - 空载\n" if chinese else "Loaded - empty\n") + label)
            axes[1, col].axhline(0, color="#737D83", ls="--", lw=1)
            for ax in axes[:, col]:
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["RL-only", "Model-guided"])
                ax.set_xlim(-.4, 1.4)
                ax.grid(axis="y", color="#DEE3E7", lw=.6)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#4F5F68",
                          label="空载" if chinese else "Empty"),
                   Line2D([], [], marker="o", ls="", color="#4F5F68", label="带载" if chinese else "Loaded"),
                   Line2D([], [], marker="o", ls="", color="#73848D", label="s1"),
                   Line2D([], [], marker="s", ls="", color="#73848D", label="s2")]
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
        fig.subplots_adjust(left=.085, right=.985, bottom=.09, top=.91, wspace=.43, hspace=.30)
        save_figure(fig, output / f"payload_motion_paired_increments_{language}")
        plt.close(fig)


def save_figure(fig, stem):
    fig.canvas.draw()
    from matplotlib.text import Text
    renderer = fig.canvas.get_renderer()
    outside = []
    ignored = set()
    for ax in fig.axes:
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not min(limits) - 1e-9 <= tick.get_loc() <= max(limits) + 1e-9:
                    ignored.update((tick.label1, tick.label2))
    for item in fig.findobj(Text):
        if item in ignored:
            continue
        if item.get_visible() and item.get_text().strip():
            bbox = item.get_window_extent(renderer)
            if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.width + 1 or bbox.y1 > fig.bbox.height + 1:
                outside.append(item.get_text())
    if outside:
        raise ValueError(f"text outside figure canvas: {outside}")
    for legend in fig.legends:
        legend_box = legend.get_window_extent(renderer)
        for item in fig.texts:
            if legend_box.overlaps(item.get_window_extent(renderer)):
                raise ValueError(f"figure legend overlaps column heading: {stem}: "
                                 f"{legend_box.extents} vs {item.get_window_extent(renderer).extents}")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(str(stem) + "." + extension, dpi=300, facecolor="white")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, validation = load_and_validate(args.input, args.preflight)
    write_json(args.output / ("preflight_validation.json" if args.preflight else "validation.json"), validation)
    if args.preflight:
        print(json.dumps(validation, indent=2))
        return
    data = analyze(records, args.output)
    if not args.no_figures:
        figures(records, data[4], args.output)
    write_json(args.output / "summary.json", {"methods": data[3], "paired_methods": data[5],
        "training_runs": data[2], "paired_training_runs": data[4], "trajectory_count": len(data[0]),
        "pair_count": len(data[1]), "independent_training_repeats_per_method": 2,
        "new_training_performed": False, "all_completed_batches": len(records)})
    print(json.dumps({"validation": "passed", "batches": len(records), "trajectories": len(data[0]),
                      "paired_trajectories": len(data[1]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
