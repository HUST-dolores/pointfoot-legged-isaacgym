#!/usr/bin/env python3
"""Four empirical supplements with independent provenance and no synthetic traces."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, SymmetricalLogLocator
import numpy as np
from scipy.io import loadmat

import fig0_sixd_design_variables as style


HERE = Path(__file__).resolve().parent
FONT_SIZE = 14
MG_COLOR = "#238B7B"
RL_COLOR = "#B34959"
BLUE = "#487FB7"
GOLD = "#B38A31"
PURPLE = "#85719C"
STEMS = {14: "controlled_condition_scans", 15: "per_environment_error_distribution",
         16: "synchronized_payload_response", 17: "recorded_failure_cases"}
KEYS = {
    14: ["mass1", "mass3", "mass6", "slope26", "slope27", "slope28", "slope26_result", "slope27_result",
         "slope28_result", "slope_manifest", "slope_evaluator_source", "slope_terrain_source", "grid1141", "grid2217", "grid1141_log", "grid2217_log",
         "grid1141_wrapper", "grid2217_wrapper"],
    15: ["motion_rl1", "motion_rl2", "motion_mg1", "motion_mg2", "motion_reference_metrics", "motion_protocol"],
    16: ["a2_mg", "a2_config"],
    17: ["diagnostic", "diagnostic_protocol", "legacy_jump", "legacy_jump_log", "nominal_urdf"],
}
TEXT = {
    "zh": {
        "14_titles": ["(a)  固定策略：负载质量扫描", "(b)  固定硬件：真实坡度扫描",
                      "(c)  固定距离命令：高度扫描", "(d)  固定高度命令：距离扫描"],
        "mass": "附加载荷 (kg)", "mass_rmse": "质量估计 RMSE (kg)", "slope": "坡角 (°)",
        "stand_rate": "站立判据通过率", "height_cmd": "高度命令 (m)", "distance_cmd": "距离命令 (m)",
        "height_actual": "实际轮端峰值净空 (m)", "distance_actual": "实际飞行跳距 (m)",
        "phases": ["待命", "准备", "腾空", "着陆"], "slope_rate": "静态站立，摩擦系数 0.8",
        "output": "逐环境均值与 P10–P90", "identity": "命令＝结果",
        "14_footer": "各子图来自独立固定策略批次；区间是试次离散性，不是置信区间，不合并为最终策略结论",
        "15_titles": ["(a)  逐环境质量 RMSE", "(b)  RMSE 累积分布", "(c)  在体样本覆盖累积分布"],
        "env_rmse": "逐环境质量 RMSE (kg)", "cdf": "环境累积比例", "coverage": "带载样本覆盖率",
        "low_coverage": "在体覆盖低于 90%", "15_footer": "约 5.93 kg 自由负载；各训练种子单独展示；低覆盖与无有效样本环境另行记录，不删长尾",
        "16_titles": ["(a)  命令与实际运动", "(b)  同步负载估计", "(c)  双轮接触力", "(d)  机体姿态"],
        "command": "记录的前向命令", "actual_speed": "实际机体系前向速度", "speed": "速度 (m/s)",
        "truth": "负载真值", "prior": "解析先验", "fused": "融合估计", "mass_axis": "质量 (kg)",
        "left": "左轮", "right": "右轮", "force": "竖直接触力 (N)", "pitch": "俯仰", "roll": "横滚",
        "angle": "角度 (°)", "time": "时间 (s)", "load": "加载", "unload": "卸载",
        "16_footer": "同一 MAT、环境 0、共同采样时钟；原始记录不平滑，事件取负载状态变化，不宣称全部运动量已恢复",
        "17_titles": ["(a)  解析反演失效：质量尖峰", "(b)  同事件：有效力臂接近零",
                      "(c)  旧跳跃策略：侧摆角失配", "(d)  同批逐跳横向偏移"],
        "raw_qs": "原始解析估计", "filtered_qs": "滤波解析估计", "arm": "有效力臂 (mm)",
        "mass_symlog": "质量 (kg，symlog)",
        "left_arm": "左腿", "right_arm": "右腿", "left_abad": "左侧摆角", "mirrored_right": "右侧镜像角",
        "lower_limit": "左关节下限", "record_id": "逐跳记录序号（非时间）", "abad_deg": "侧摆角 (°)",
        "lat_ratio": "最大横向偏移比", "halfwidth": "轮距半宽参考",
        "17_footer": "历史诊断与旧策略记录；不代表最终策略失效率；横向偏移按机体原点定义，关联不单独证明因果",
    },
    "en": {
        "14_titles": ["(a)  Fixed policy: payload sweep", "(b)  Fixed hardware: real slopes",
                      "(c)  Fixed distance command: height", "(d)  Fixed height command: distance"],
        "mass": "Added payload (kg)", "mass_rmse": "Mass-estimation RMSE (kg)", "slope": "Slope (deg)",
        "stand_rate": "Standing-criterion pass rate", "height_cmd": "Height command (m)", "distance_cmd": "Distance command (m)",
        "height_actual": "Actual peak wheel clearance (m)", "distance_actual": "Actual airborne distance (m)",
        "phases": ["Idle", "Crouch", "Flight", "Landed"], "slope_rate": "Static stance, friction 0.8",
        "output": "Per-env means, P10–P90", "identity": "Command = outcome",
        "14_footer": "Separate fixed-policy campaigns; intervals show dispersion, not confidence or a combined final-policy result",
        "15_titles": ["(a)  Per-environment mass RMSE", "(b)  RMSE cumulative distribution", "(c)  Loaded coverage distribution"],
        "env_rmse": "Per-environment mass RMSE (kg)", "cdf": "Cumulative environment fraction", "coverage": "Loaded-sample coverage",
        "low_coverage": "Loaded coverage below 90%", "15_footer": "Free payload about 5.93 kg; training seeds shown separately; low coverage and missing-error cases are retained in the audit",
        "16_titles": ["(a)  Command and actual motion", "(b)  Same-run payload estimates", "(c)  Wheel contact forces", "(d)  Body attitude"],
        "command": "Recorded forward command", "actual_speed": "Actual body-frame forward speed", "speed": "Speed (m/s)",
        "truth": "True payload mass", "prior": "Analytical prior", "fused": "Fused estimate", "mass_axis": "Mass (kg)",
        "left": "Left wheel", "right": "Right wheel", "force": "Vertical contact force (N)", "pitch": "Pitch", "roll": "Roll",
        "angle": "Angle (deg)", "time": "Time (s)", "load": "Load", "unload": "Unload",
        "16_footer": "One MAT, environment 0, common clock; unsmoothed records and load-state events, not a claim of complete recovery",
        "17_titles": ["(a)  Analytical inversion: mass spike", "(b)  Same event: vanishing moment arm",
                      "(c)  Legacy jump: abduction mismatch", "(d)  Same campaign: lateral offsets"],
        "raw_qs": "Raw analytical estimate", "filtered_qs": "Filtered analytical estimate", "arm": "Effective moment arm (mm)",
        "mass_symlog": "Mass (kg, symlog)",
        "left_arm": "Left leg", "right_arm": "Right leg", "left_abad": "Left abduction", "mirrored_right": "Mirrored right angle",
        "lower_limit": "Left lower joint limit", "record_id": "Jump-record index (not time)", "abad_deg": "Abduction (deg)",
        "lat_ratio": "Peak lateral-offset ratio", "halfwidth": "Half-wheel-spacing reference",
        "17_footer": "Historical diagnosis / legacy policy, not final-policy failure rates; body-origin lateral proxy, not causal attribution",
    },
}


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refuse to export an empty evidence table")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def encode_json(value):
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(value) for key, value in item.items()}
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    return json.dumps(clean(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def mat(path, keys):
    data = loadmat(path, variable_names=[*keys, "dt", "meta", "summary"], simplify_cells=True)
    for key in keys:
        if key not in data:
            raise KeyError(f"{path}: missing {key}")
    return data


def axis_style(ax, title, x_label, y_label):
    ax.set_title(title, loc="left", fontsize=FONT_SIZE, color=style.INK, pad=14, fontweight="normal")
    ax.set_xlabel(x_label, fontsize=FONT_SIZE, labelpad=8)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE, labelpad=8)
    ax.tick_params(labelsize=FONT_SIZE, colors=style.MUTED, width=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for edge in ("left", "bottom"):
        ax.spines[edge].set_color("#97A4AC")
    ax.grid(axis="y", color="#DFE6EA", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))


def layout_check(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside_ticks = set()
    for ax in fig.axes:
        for axis, limits in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if not limits[0] <= tick.get_loc() <= limits[1]:
                    outside_ticks.update((id(tick.label1), id(tick.label2)))
    labels = [obj for obj in fig.findobj(Text) if obj.get_visible() and obj.get_text() and id(obj) not in outside_ticks]
    for obj in labels:
        box = obj.get_window_extent(renderer)
        if not np.isclose(obj.get_fontsize(), FONT_SIZE):
            raise ValueError("All labels must use 14 pt: " + obj.get_text())
        if not all(fig.bbox.contains(*corner) for corner in box.get_points()):
            raise ValueError("Text outside figure: " + obj.get_text())
    return {"visible_text_count": len(labels), "all_text_within_export": True, "font_size_pt": FONT_SIZE}


def condition_data(files, output):
    rows, mass_values = [], []
    fixed = None
    for payload in (1, 3, 6):
        data = mat(files[f"mass{payload}"], ["true_mass_all", "estimate_all", "jump_state_all"])
        summary = data["summary"]
        signature = [summary[key] for key in ("task", "load_run", "checkpoint", "payload_mode", "jump_height_m", "jump_dist_m", "dt_s")]
        if fixed is not None and signature != fixed:
            raise ValueError("Payload sweep changed a supposedly fixed setting")
        fixed = signature
        true, estimate, state = [np.asarray(data[key]) for key in ("true_mass_all", "estimate_all", "jump_state_all")]
        if true.shape != (600, 32) or not (true.shape == estimate.shape == state.shape):
            raise ValueError("Payload sweep must use 32 environments and 600 steps in every condition")
        np.testing.assert_allclose(true, payload)
        scores = []
        for phase in range(4):
            mask = state == phase
            error = estimate.astype(float) - true
            rmse = float(np.sqrt(np.mean(error[mask] ** 2)))
            if not np.isfinite(rmse):
                raise ValueError("Missing/invalid phase in payload sweep")
            scores.append(rmse)
            rows.append({"scan": "payload", "condition": payload, "series": phase,
                         "value": rmse, "lower": rmse, "upper": rmse,
                         "n_units": int(np.any(mask, axis=0).sum()), "n_samples": int(mask.sum()),
                         "metric": "pooled_phase_mass_rmse_kg"})
        mass_values.append(scores)
    slope_values = []
    slope_signature = None
    for seed in (26, 27, 28):
        raw = np.load(files[f"slope{seed}"])
        result = json.loads(files[f"slope{seed}_result"].read_text())
        signature = {key: result[key] for key in ("run", "checkpoint", "design", "friction", "settle_s", "evaluation_s", "replicas_per_level", "command_decomposition")}
        if slope_signature is not None and signature != slope_signature:
            raise ValueError("Slope scan changed fixed policy/hardware/protocol")
        slope_signature = signature
        if result["friction"] != 0.8 or set(result["design"].values()) != {1.0}:
            raise ValueError("Slope figure must fix nominal hardware and friction 0.8")
        reconstructed_success = (~raw["failed"]) & (raw["roll_p95_rad"] <= np.deg2rad(15)) & (raw["pitch_p95_rad"] <= np.deg2rad(20))
        np.testing.assert_array_equal(reconstructed_success, raw["slope_stand_success"])
        rates = []
        for level in result["levels"]:
            mask = raw["level"] == level
            if mask.sum() != 64:
                raise ValueError("Each slope needs exactly 64 replicas per evaluation seed")
            rate = float(raw["slope_stand_success"][mask].mean())
            expected = next(item["stand_success_rate"] for item in result["per_level"] if item["level"] == level)
            np.testing.assert_allclose(rate, expected)
            rates.append(rate)
        slope_values.append(rates)
    slopes = np.asarray(result["levels"])
    slope_values = np.asarray(slope_values)
    for i, level in enumerate(slopes):
        rows.append({"scan": "slope", "condition": level, "series": 0,
                     "value": float(slope_values[:, i].mean()), "lower": float(slope_values[:, i].min()),
                     "upper": float(slope_values[:, i].max()), "n_units": 192, "n_samples": 192,
                     "metric": "static_stand_pass_rate_mean_and_seed_range"})

    grid_rows = []
    log_policy = []
    for seed in (1141, 2217):
        log = files[f"grid{seed}_log"].read_text()
        log_policy.append(next(line for line in log.splitlines() if line.startswith("Loading model from:")))
        if "nominal=1" not in files[f"grid{seed}_wrapper"].read_text():
            raise ValueError("Grid must explicitly fix nominal hardware")
        for row in read_csv(files[f"grid{seed}"]):
            row = {key: float(value) for key, value in row.items()}
            row["eval_seed"] = seed
            grid_rows.append(row)
    if len(set(log_policy)) != 1:
        raise ValueError("Command scans must use the same policy checkpoint")
    curves = []
    units = []
    for scan, varying, held, fixed_value, outcome in (("height", "hstar", "dstar", 0.095, "height"),
                                                    ("distance", "dstar", "hstar", 0.13, "disp")):
        subset = [row for row in grid_rows if np.isclose(row[held], fixed_value, atol=1e-6)]
        levels = sorted(set(round(row[varying], 6) for row in subset))
        scan_curve = []
        for value in levels:
            point = [row for row in subset if np.isclose(row[varying], value, atol=1e-6)]
            env_keys = sorted(set((int(row["eval_seed"]), int(row["env_id"])) for row in point))
            if len(env_keys) != 32:
                raise ValueError("Expected 16 replicas for each of two seeds per grid point")
            means = []
            for seed, env in env_keys:
                jumps = [row for row in point if row["eval_seed"] == seed and row["env_id"] == env]
                mean = float(np.mean([row[outcome] for row in jumps]))
                means.append(mean)
                units.append({"scan": scan, "command": value, "fixed_other_command": fixed_value,
                              "eval_seed": seed, "env_id": env, "recorded_completed_jumps": len(jumps), "mean_outcome_m": mean})
            summary = {"scan": scan, "condition": value, "series": 0, "value": float(np.mean(means)),
                       "lower": float(np.quantile(means, .10)), "upper": float(np.quantile(means, .90)),
                       "n_units": len(means), "n_samples": len(point), "metric": "completed_jump_per_env_mean_m"}
            rows.append(summary)
            scan_curve.append(summary)
        curves.append(scan_curve)
    write_csv(output / "condition_scan_summary.csv", rows)
    write_csv(output / "command_scan_environment_units.csv", units)
    audit = {"payload_fixed_settings": fixed, "payload_conditions_kg": [1, 3, 6],
             "payload_each_shape": [600, 32], "slope_fixed_settings": slope_signature,
             "slope_is_true_trimesh_with_vertical_world_gravity": True,
             "slope_protocol": json.loads(files["slope_manifest"].read_text()),
             "command_policy": log_policy[0], "command_observed_completed_jumps_only": True,
             "grid_timeouts_not_in_event_csv": True,
             "independent_campaigns_not_one_common_policy": True,
             "uncertainty": "Slope band is min/max across 3 evaluation seeds; command bars P10/P90 across per-environment means, not CI",
             "rejected_alternative": "Legacy June05 gravity-tilt runs changed sampled payloads and are not used as a strict slope-only scan"}
    (output / "controlled_variables_audit.json").write_text(encode_json(audit))
    return np.asarray(mass_values), slopes, slope_values, curves


def figure14(data, words):
    mass_values, slopes, rates, curves = data
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.0), dpi=140)
    fig.subplots_adjust(left=.09, right=.97, top=.90, bottom=.13, wspace=.27, hspace=.56)
    colors = (BLUE, GOLD, PURPLE, MG_COLOR)
    for phase in range(4):
        axes[0, 0].plot([1, 3, 6], mass_values[:, phase], "-o", color=colors[phase], label=words["phases"][phase], linewidth=1.6, markersize=5)
    axis_style(axes[0, 0], words["14_titles"][0], words["mass"], words["mass_rmse"])
    axes[0, 0].set_xticks([1, 3, 6])
    axes[0, 0].legend(frameon=False, fontsize=FONT_SIZE, ncol=2, loc="upper right")
    ax = axes[0, 1]
    ax.fill_between(slopes, rates.min(axis=0), rates.max(axis=0), color=BLUE, alpha=.16)
    ax.plot(slopes, rates.mean(axis=0), "-o", color=BLUE, linewidth=1.7, markersize=4, label=words["slope_rate"])
    axis_style(ax, words["14_titles"][1], words["slope"], words["stand_rate"])
    ax.set_ylim(-.03, 1.05)
    ax.legend(frameon=False, fontsize=FONT_SIZE, loc="lower left")
    for ax, curve, scan, title in zip(axes[1], curves, ("height", "distance"), words["14_titles"][2:]):
        x, y, lo, hi = [np.array([row[key] for row in curve]) for key in ("condition", "value", "lower", "upper")]
        ax.errorbar(x, y, yerr=[y-lo, hi-y], fmt="o-", color=MG_COLOR, capsize=4, linewidth=1.6, label=words["output"])
        ax.plot([x.min(), x.max()], [x.min(), x.max()], color="#8D9AA3", linestyle=(0, (4, 3)), label=words["identity"])
        axis_style(ax, title, words[scan + "_cmd"], words[scan + "_actual"])
        ax.set_xticks(x, [f"{value:g}" for value in x], fontsize=FONT_SIZE)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=FONT_SIZE, loc="lower right")
    fig.text(.5, .035, words["14_footer"], ha="center", fontsize=FONT_SIZE, color=style.MUTED)
    return fig


def distribution_data(files, output):
    rows, audit = [], []
    for key, label, method, seed in (("motion_rl1", "rlonly_s1", "RL-only", "s1"), ("motion_rl2", "rlonly_s2", "RL-only", "s2"),
                                     ("motion_mg1", "mg_s1", "Model-guided", "s1"), ("motion_mg2", "mg_s2", "Model-guided", "s2")):
        data = mat(files[key], ["payload_mass_ref_all", "encoder_mass_all", "load_on_body_all", "command_x_all"])
        true, estimate, loaded, command = [np.asarray(data[name]) for name in ("payload_mass_ref_all", "encoder_mass_all", "load_on_body_all", "command_x_all")]
        dt = float(data["dt"])
        if true.shape != (2000, 256) or not all(value.shape == true.shape for value in (estimate, loaded, command)):
            raise ValueError("Fixed-payload experiment shape differs from documented protocol")
        np.testing.assert_allclose(true[true > .3], 5.9268727, atol=1e-5)
        if data["meta"]["verify"]["play_cfg"]["seed"] != 12345 or data["meta"]["corrupt_estimate"] != "none":
            raise ValueError("Evaluation seed/corruption mismatch")
        steady = np.arange(true.shape[0]) * dt >= 2
        nominal = np.isclose(command, .5, atol=1e-6)
        eligible = steady[:, None] & nominal
        valid = eligible & (loaded > .5)
        for env in range(256):
            mask = valid[:, env]
            rmse = float(np.sqrt(np.mean((estimate[mask, env].astype(float) - true[mask, env]) ** 2))) if mask.any() else None
            if rmse is not None and not np.isfinite(rmse):
                raise ValueError("Non-finite RMSE must not silently disappear from a distribution")
            coverage = float(mask.sum() / eligible[:, env].sum())
            rows.append({"label": label, "method": method, "training_seed_label": seed, "env_id": env,
                         "valid_loaded_samples": int(mask.sum()), "eligible_nominal_samples": int(eligible[:, env].sum()),
                         "loaded_coverage": coverage, "mass_rmse_kg": rmse, "no_error_samples": not bool(mask.any()),
                         "coverage_below_90pct_diagnostic": coverage < .90})
        audit.append({"label": label, "source_meta": data["meta"], "total_environments": 256,
                      "mass_kg": float(np.max(true)), "no_valid_error_environments": int(np.sum(valid.sum(axis=0) == 0))})
    old = {(row["label"], int(row["env_id"])): float(row["mass_rmse_kg"]) for row in read_csv(files["motion_reference_metrics"])}
    differences = [abs(row["mass_rmse_kg"] - old[(row["label"], row["env_id"])]) for row in rows
                   if (row["label"], row["env_id"]) in old and row["mass_rmse_kg"] is not None]
    if not differences or max(differences) > 1e-6:
        raise ValueError("Recomputed per-environment RMSE disagrees with archived metrics")
    write_csv(output / "all_environment_error_and_coverage.csv", rows)
    stats = []
    for label in ("rlonly_s1", "rlonly_s2", "mg_s1", "mg_s2"):
        subset = [row for row in rows if row["label"] == label]
        values = np.array([row["mass_rmse_kg"] for row in subset if row["mass_rmse_kg"] is not None])
        stats.append({"label": label, "environments_total": 256, "environments_with_error": len(values),
                      "median_rmse_kg": float(np.median(values)), "p95_rmse_kg": float(np.quantile(values, .95)),
                      "max_rmse_kg": float(values.max()), "low_coverage_environments": sum(row["loaded_coverage"] < .90 for row in subset),
                      "no_error_samples": sum(row["no_error_samples"] for row in subset)})
    write_csv(output / "distribution_summary.csv", stats)
    (output / "distribution_audit.json").write_text(encode_json({"runs": audit, "old_metric_overlap_count": len(differences),
        "old_metric_maximum_abs_difference_kg": max(differences), "outliers_removed": False,
        "minimum_error_samples": 1, "coverage_is_not_a_direct_drop_or_fall_rate": True,
        "statistical_unit": "Environment within each saved training run; not independent training seeds",
        "empty_error_units_handling": "Keep in CSV and coverage CDF, omit from RMSE CDF with counts, never replace by zero"}))
    return rows


def ecdf(ax, values, **kwargs):
    x = np.sort(np.asarray(values, dtype=float))
    ax.step(np.r_[x[0], x], np.r_[0, np.arange(1, len(x)+1) / len(x)], where="post", **kwargs)


def figure15(rows, words):
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.8), dpi=140)
    fig.subplots_adjust(left=.065, right=.985, top=.78, bottom=.23, wspace=.40)
    labels = ("rlonly_s1", "rlonly_s2", "mg_s1", "mg_s2")
    for i, label in enumerate(labels):
        subset = [row for row in rows if row["label"] == label]
        good = [row for row in subset if row["mass_rmse_kg"] is not None]
        values = np.array([row["mass_rmse_kg"] for row in good])
        low = np.array([row["loaded_coverage"] < .9 for row in good])
        color = RL_COLOR if i < 2 else MG_COLOR
        boxes = axes[0].boxplot([values], positions=[i+1], widths=.50, patch_artist=True, showfliers=False,
                                medianprops={"color": style.INK, "linewidth": 1.4},
                                whiskerprops={"color": color}, capprops={"color": color})
        boxes["boxes"][0].set(facecolor=color, alpha=.20, edgecolor=color)
        jitter = ((np.arange(len(values)) * 0.61803398875) % 1 - .5) * .29
        axes[0].scatter(i+1+jitter, values, s=9, color=color, alpha=.35, linewidths=0)
        axes[0].scatter(i+1+jitter[low], values[low], s=17, facecolors="none", edgecolors=style.INK, linewidths=.55)
        linestyle = "-" if i % 2 == 0 else (0, (4, 3))
        ecdf(axes[1], values, color=color, linestyle=linestyle, linewidth=1.8)
        ecdf(axes[2], [row["loaded_coverage"] for row in subset], color=color, linestyle=linestyle, linewidth=1.8)
    for ax, title, x, y in zip(axes, words["15_titles"], ("", words["env_rmse"], words["coverage"]),
                              (words["mass_rmse"], words["cdf"], words["cdf"])):
        axis_style(ax, title, x, y)
    counts = [sum(row["label"] == label and row["mass_rmse_kg"] is not None for row in rows) for label in labels]
    axes[0].set_xticks([1, 2, 3, 4], [f"{label}\n{count}/256" for label, count in zip(("RL s1", "RL s2", "MG s1", "MG s2"), counts)], fontsize=FONT_SIZE)
    for ax in axes[1:]:
        ax.set_ylim(-.02, 1.03)
    axes[1].set_xlim(left=0)
    axes[2].set_xlim(0, 1.01)
    handles = [Line2D([], [], color=RL_COLOR, label="RL-only"), Line2D([], [], color=MG_COLOR, label="Model-guided"),
               Line2D([], [], color=style.MUTED, label="s1", linestyle="-"), Line2D([], [], color=style.MUTED, label="s2", linestyle=(0,(4,3))),
               Line2D([], [], linestyle="none", marker="o", markerfacecolor="none", markeredgecolor=style.INK, label=words["low_coverage"])]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.96), ncol=5, frameon=False, fontsize=FONT_SIZE)
    fig.text(.5,.065,words["15_footer"],ha="center",fontsize=FONT_SIZE,color=style.MUTED)
    return fig


def synchronized_data(files, output):
    fields = ["payload_mass_ref_all", "encoder_mass_all", "payload_mass_all", "base_lin_vel_x_all", "base_lin_vel_y_all", "base_lin_vel_z_all",
              "base_pitch_all", "base_roll_all", "command_x_all", "contact_forces_z", "base_vel_x", "base_vel_y", "base_vel_z",
              "payload_mass_ref", "mass_est"]
    data = mat(files["a2_mg"], fields)
    cfg = json.loads(files["a2_config"].read_text())
    for single, matrix in (("base_vel_x", "base_lin_vel_x_all"), ("base_vel_y", "base_lin_vel_y_all"), ("base_vel_z", "base_lin_vel_z_all")):
        np.testing.assert_array_equal(data[single], data[matrix][:, 0])
    true, raw_prior, fused = [np.asarray(data[key])[:,0].astype(float) for key in ("payload_mass_ref_all", "payload_mass_all", "encoder_mass_all")]
    threshold = cfg["asset"]["load_estimation_position_zero_mass_threshold"]
    prior = np.where(raw_prior >= threshold, raw_prior, 0)
    dt = float(data["dt"])
    t = np.arange(len(true))*dt
    contact = np.asarray(data["contact_forces_z"])
    if contact.shape != (len(true),2):
        raise ValueError("Need the two-wheel contact force trace")
    signals = {"time_s":t, "recorded_forward_command_mps":np.asarray(data["command_x_all"])[:,0],
               "actual_body_forward_speed_mps":np.asarray(data["base_lin_vel_x_all"])[:,0], "true_payload_mass_kg":true,
               "analytical_prior_kg":prior, "fused_payload_mass_kg":fused,
               "left_contact_force_z_N":contact[:,0], "right_contact_force_z_N":contact[:,1],
               "body_pitch_deg":np.rad2deg(data["base_pitch_all"][:,0]), "body_roll_deg":np.rad2deg(data["base_roll_all"][:,0])}
    if not all(np.isfinite(value).all() for value in signals.values()):
        raise ValueError("Non-finite synchronous channels require explicit handling")
    changes = np.flatnonzero((true[1:]>.3)!=(true[:-1]>.3))+1
    events = [{"sample":int(i),"time_s":float(t[i]),"type":"load" if true[i]>.3 else "unload",
               "mass_before_kg":float(true[i-1]),"mass_after_kg":float(true[i])} for i in changes]
    write_csv(output/"synchronized_full_trace.csv",[dict(zip(signals,(value[i] for value in signals.values()))) for i in range(len(t))])
    write_csv(output/"load_state_events.csv",events)
    (output/"synchronization_audit.json").write_text(encode_json({"environment":0,"dt_s":dt,"meta":data["meta"],
        "display_window_s":[9.5,17.0],"contact_logger_identity_check":"Three uniquely varying single-env velocity traces exactly match all-env column 0",
        "events":"Actual all-environment truth transitions, not prescribed timestamps",
        "legacy_single_mass_label_max_difference_kg":float(np.max(np.abs(data["payload_mass_ref"]-true))),
        "legacy_single_estimate_max_difference_kg":float(np.max(np.abs(data["mass_est"]-fused))),
        "mass_channel_choice":"Use authoritative *_all[:,0]; legacy scalar mass labels differ and are not mixed with them",
        "command_definition":"Recorded forward command channel only, not an assumed total roll+step target",
        "smoothing_or_interpolation":False,"reset_flags_available":False,"settling_time_or_recovery_success_claimed":False}))
    return signals,events


def figure16(record,words):
    signals,events=record
    fig,axes=plt.subplots(4,1,figsize=(12.8,10.5),sharex=True,dpi=140)
    fig.subplots_adjust(left=.105,right=.97,top=.91,bottom=.115,hspace=.75)
    t=signals["time_s"]
    mask=(t>=9.5)&(t<=17.0)
    specs=[[("recorded_forward_command_mps","command",style.INK),("actual_body_forward_speed_mps","actual_speed",BLUE)],
           [("true_payload_mass_kg","truth",style.INK),("analytical_prior_kg","prior",GOLD),("fused_payload_mass_kg","fused",MG_COLOR)],
           [("left_contact_force_z_N","left",BLUE),("right_contact_force_z_N","right",PURPLE)],
           [("body_pitch_deg","pitch",BLUE),("body_roll_deg","roll",RL_COLOR)]]
    for ax,series,title,ylabel in zip(axes,specs,words["16_titles"],("speed","mass_axis","force","angle")):
        for key,label,color in series:
            ax.plot(t[mask],signals[key][mask],color=color,linewidth=1.4,label=words[label])
        axis_style(ax,title,"",words[ylabel])
        ax.set_title(title,loc="left",fontsize=FONT_SIZE,color=style.INK,y=1.23,pad=6,fontweight="normal")
        ax.legend(frameon=False,fontsize=FONT_SIZE,loc="lower right",bbox_to_anchor=(1,1.02),ncol=len(series),borderaxespad=0)
        ax.set_xlim(9.5,17.0)
        for event in events:
            if 9.5<=event["time_s"]<=17:
                ax.axvline(event["time_s"],color="#8D9AA3",linestyle=(0,(3,3)),linewidth=.8)
    for event in events:
        if 9.5<=event["time_s"]<=17:
            label_y = .65 if event["type"] == "unload" else .97
            axes[0].text(event["time_s"],label_y,words[event["type"]],transform=axes[0].get_xaxis_transform(),ha="center",va="top",fontsize=FONT_SIZE,color=style.MUTED)
    axes[-1].set_xlabel(words["time"],fontsize=FONT_SIZE)
    fig.text(.5,.028,words["16_footer"],fontsize=FONT_SIZE,ha="center",color=style.MUTED)
    return fig


def failure_data(files,output):
    keys=["payload_mass_ref_all","payload_mass_all","qs_payload_mass_raw_all","qs_mass_moment_arm_left_all","qs_mass_moment_arm_right_all"]
    data=mat(files["diagnostic"],keys)
    env,event=7,1398
    t=np.arange(len(data[keys[0]]))*float(data["dt"])
    signals={"time_s":t,"truth_kg":data[keys[0]][:,env],"filtered_qs_kg":data[keys[1]][:,env],"raw_qs_kg":data[keys[2]][:,env],
             "left_arm_mm":data[keys[3]][:,env]*1000,"right_arm_mm":data[keys[4]][:,env]*1000}
    if abs(signals["right_arm_mm"][event])>.10 or signals["raw_qs_kg"][event]<1000:
        raise ValueError("The predeclared recorded singularity does not match its source")
    window=(t>=27.5)&(t<=28.7)
    write_csv(output/"recorded_singularity_window.csv",[dict(zip(signals,(value[i] for value in signals.values()))) for i in np.flatnonzero(window)])
    jumps=[{key:float(value) for key,value in row.items()} for row in read_csv(files["legacy_jump"])]
    urdf=ET.parse(files["nominal_urdf"]).getroot()
    lower=float(urdf.find("joint[@name='abad_L_Joint']/limit").get("lower"))
    lat=np.array([row["max_lat_ratio"] for row in jumps])
    left=np.array([row["takeoff_abad_left"] for row in jumps])
    right=np.array([row["takeoff_abad_right"] for row in jumps])
    if not np.any(np.isclose(left,lower,atol=1e-4)):
        raise ValueError("Cannot substantiate an abduction-limit example from these records")
    rows=[{"record_index":i,"left_abad_deg":float(np.rad2deg(left[i])),"mirrored_right_abad_deg":float(np.rad2deg(-right[i])),
           "max_lateral_body_origin_ratio":float(lat[i]),"left_at_recorded_nominal_lower_limit":bool(abs(left[i]-lower)<1e-4)} for i in range(len(jumps))]
    write_csv(output/"legacy_jump_boundary_records.csv",rows)
    audit={"singularity_source_meta":data["meta"],"singularity_env_id":env,"singularity_step":event,"singularity_time_s":float(t[event]),
           "singularity_raw_estimate_kg":float(signals["raw_qs_kg"][event]),"singularity_right_moment_arm_mm":float(signals["right_arm_mm"][event]),
           "legacy_jump_count":len(jumps),"left_lower_limit_rad":lower,"at_limit_fraction":float(np.mean(np.isclose(left,lower,atol=1e-4))),
           "lateral_body_origin_ratio_above_one_fraction":float(np.mean(lat>1)),
           "legacy_csv_takeoff_counts":{str(value):int(count) for value,count in zip(*np.unique([row['takeoffs'] for row in jumps],return_counts=True))},
           "repeat_jump_failure_example_not_available_in_selected_csv":True,"torque_saturation_not_inferred_from_position_limit":True,
           "failure_causes":"Analytical near-zero denominator is a formula failure; legacy abduction and lateral offsets are associated diagnostics, not isolated causal evidence",
           "final_policy_failure_rate_claim":False,"timestamp_for_jump_records_available":False}
    (output/"failure_case_audit.json").write_text(encode_json(audit))
    return signals,rows,lower,float(t[event])


def figure17(data,words):
    signals,rows,lower,event=data
    fig,axes=plt.subplots(2,2,figsize=(14.4,8.8),dpi=140)
    fig.subplots_adjust(left=.09,right=.97,top=.91,bottom=.15,wspace=.28,hspace=.60)
    mask=(signals['time_s']>=27.5)&(signals['time_s']<=28.7)
    for key,label,color in (("truth_kg","truth",style.INK),("raw_qs_kg","raw_qs",RL_COLOR),("filtered_qs_kg","filtered_qs",BLUE)):
        axes[0,0].plot(signals['time_s'][mask],signals[key][mask],label=words[label],color=color,linewidth=1.5)
    axes[0,0].set_yscale('symlog',linthresh=10)
    for key,label,color in (("left_arm_mm","left_arm",BLUE),("right_arm_mm","right_arm",RL_COLOR)):
        axes[0,1].plot(signals['time_s'][mask],signals[key][mask],label=words[label],color=color,linewidth=1.5)
    axes[0,1].axhline(0,color=style.MUTED,linestyle=(0,(4,3)),linewidth=.9)
    for ax,title,ylabel in zip(axes[0],words['17_titles'][:2],('mass_symlog','arm')):
        axis_style(ax,title,words['time'],words[ylabel]);ax.axvline(event,color=GOLD,linestyle=(0,(3,3)),linewidth=.9)
        ax.legend(frameon=False,fontsize=FONT_SIZE,loc='upper right')
    axes[0,0].legend(frameon=False,fontsize=FONT_SIZE,loc='lower right')
    axes[0,0].yaxis.set_major_locator(SymmetricalLogLocator(base=10, linthresh=10))
    x=np.array([row['record_index'] for row in rows])
    axes[1,0].scatter(x,[row['left_abad_deg'] for row in rows],color=BLUE,s=8,alpha=.6,linewidths=0,label=words['left_abad'])
    axes[1,0].scatter(x,[row['mirrored_right_abad_deg'] for row in rows],color=PURPLE,s=8,alpha=.6,linewidths=0,label=words['mirrored_right'])
    axes[1,0].axhline(np.rad2deg(lower),color=RL_COLOR,linestyle=(0,(4,3)),linewidth=1.1,label=words['lower_limit'])
    axes[1,1].scatter(x,[row['max_lateral_body_origin_ratio'] for row in rows],s=9,color=PURPLE,alpha=.55,linewidths=0)
    axes[1,1].axhline(1,color=RL_COLOR,linestyle=(0,(4,3)),linewidth=1.1,label=words['halfwidth'])
    for ax,title,ylabel in zip(axes[1],words['17_titles'][2:],('abad_deg','lat_ratio')):
        axis_style(ax,title,words['record_id'],words[ylabel])
    axes[1,0].legend(frameon=False,fontsize=FONT_SIZE,loc='center left',markerscale=2.5)
    axes[1,1].legend(frameon=False,fontsize=FONT_SIZE,loc='lower right')
    fig.text(.5,.035,words['17_footer'],ha='center',fontsize=FONT_SIZE,color=style.MUTED)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure',type=int,choices=tuple(STEMS),required=True)
    parser.add_argument('--inputs',type=Path,default=HERE/'empirical_supplement_inputs.json')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--package-dir',type=Path)
    parser.add_argument('--language',choices=('zh','en','both'),default='both')
    parser.add_argument('--dpi',type=int,default=300)
    args=parser.parse_args()
    config=json.loads(args.inputs.read_text())
    files={key:(args.inputs.parent/Path(config['files'][key])).resolve() for key in KEYS[args.figure]}
    output=args.package_dir or args.output_dir or HERE/'out'/STEMS[args.figure]
    output.mkdir(parents=True,exist_ok=True)
    provenance={key:{'source_path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for key,path in files.items()}
    if args.package_dir:
        raw=output/'raw_sources';raw.mkdir(exist_ok=True)
        packaged={}
        for key,path in files.items():
            target=raw/(key+path.suffix)
            if path.resolve()!=target.resolve():shutil.copy2(path,target)
            packaged[key]=str(target.relative_to(output))
        shutil.copy2(__file__,output/Path(__file__).name)
        shutil.copy2(style.__file__,output/Path(style.__file__).name)
        (output/'empirical_supplement_inputs.json').write_text(encode_json({'files':packaged,'original_sources':provenance}))
        files={key:output/path for key,path in packaged.items()}
    corpus=''.join(str(words) for words in TEXT.values()) + "−"
    style.configure_fonts(output,extra_text=corpus)
    plt.rcParams.update({'font.size':FONT_SIZE,'font.weight':'normal','axes.titleweight':'normal'})
    loaders={14:condition_data,15:distribution_data,16:synchronized_data,17:failure_data}
    plotters={14:figure14,15:figure15,16:figure16,17:figure17}
    data=loaders[args.figure](files,output)
    reports={}
    for language in ('zh','en') if args.language=='both' else (args.language,):
        fig=plotters[args.figure](data,TEXT[language]);reports[language]=layout_check(fig)
        for extension in ('png','pdf','svg'):
            target=output/f'{STEMS[args.figure]}_{language}.{extension}'
            fig.savefig(target,dpi=args.dpi,facecolor='white');print(target)
        plt.close(fig)
    manifest={'figure':args.figure,'type':'Empirical reanalysis of archived recordings; no new training/simulation',
              'original_sources':config.get('original_sources',provenance),'current_inputs':provenance,
              'no_synthetic_physical_curves':True,'previous_results_overwritten':False,'layout_checks':reports,
              'is_one_final_policy_comparison':False,'font_size_pt':FONT_SIZE,
              'raster_size_px':[round(value*args.dpi) for value in fig.get_size_inches()]}
    (output/'figure_manifest.json').write_text(encode_json(manifest))
    print('Empirical source audits passed; no training or simulator launched')


if __name__=='__main__':main()
