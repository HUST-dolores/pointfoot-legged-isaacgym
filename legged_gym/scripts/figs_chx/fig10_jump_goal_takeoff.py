#!/usr/bin/env python3
"""Separate ideal centroidal reference planning from the archived RL policy path."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
from matplotlib.text import Text
import numpy as np

import fig0_sixd_design_variables as style
import fig03_four_capability_definitions as geometry


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SNAPSHOT = HERE / "source_snapshot"
EXPERIMENT = Path("/home/xu/Desktop/codesign_worklog/nested_four_objective_sixd_2026-07-30/results/final_nested_bo")
DEFAULT_URDF = (HERE / "nominal_source.urdf" if (HERE / "nominal_source.urdf").is_file()
                else REPO / "resources/robots/WF_TRON1A/urdf/robot.urdf")
DEFAULT_CONFIG = (SNAPSHOT / "jump_winner_env_cfg.json" if SNAPSHOT.is_dir() else
                  EXPERIMENT / "winner_policies/jump/seed_20260811/env_cfg.json")
DEFAULT_ENV = (SNAPSHOT / "wheelfoot_flat.py" if SNAPSHOT.is_dir() else
               EXPERIMENT / "code_snapshot/legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py")
DEFAULT_OUTPUT = HERE / "out/jump_goal_takeoff"
FONT_SIZE = 14
FIGURE_SIZE = (14.4, 8.0)
DISTANCE_M = 0.60
COM_RISE_M = 0.18
AIR_PITCH_DEG = 18.0
COM_COLOR = "#B34959"
VERTICAL_COLOR = "#238B7B"
HORIZONTAL_COLOR = "#487FB7"
ANGLE_COLOR = "#85719C"
FORCE_COLOR = "#B38A31"
STEMS = {"model": "jump_model_reference", "policy": "jump_formal_policy"}
SUBDIRS = {"model": "model_reference", "policy": "formal_policy"}
PUSH_POSE = {"hip_L_Joint": 0.45, "knee_L_Joint": -0.20,
             "hip_R_Joint": -0.45, "knee_R_Joint": 0.20}
AIR_POSE = {"hip_L_Joint": 0.425, "knee_L_Joint": 0.85,
            "hip_R_Joint": -0.425, "knee_R_Joint": -0.85}
TEXT = {
    "zh": {
        "model_ballistic": "(a)  理想质心弹道与离地状态",
        "policy_ballistic": "(a)  理想质心弹道（仅理论背景）",
        "push_title": "(b)  虚拟伸展与蹬地",
        "quantities_title": "(c)  高度与俯仰量",
        "model_flow": "(d)  模型参考规划",
        "policy_flow": "(d)  正式策略执行与训练引导",
        "takeoff": "离地",
        "landing": "着陆",
        "com": "质心 C",
        "push_impulse": "蹬地净冲量",
        "model_1": "任务目标",
        "model_2": "离地状态",
        "model_3": "伸展与偏角轨迹",
        "model_4": "关节参考",
        "policy_1": "任务命令",
        "policy_2": "策略输出",
        "policy_3": "相位动作修正／侧摆约束",
        "policy_4": "关节目标与 PD／轮速",
        "training": "奖励引导（仅训练）",
        "policy_update": "策略更新",
        "observation_feedback": "观测反馈",
        "feedback": "接触／腾空反馈",
        "model_caution": "理想模型与参考规划示意，非实测轨迹或正式策略执行链",
        "policy_caution": "正式策略未启用显式轨迹；质心抛物线仅作理论背景，非策略跟踪参考",
    },
    "en": {
        "model_ballistic": "(a)  Ideal CoM flight and takeoff state",
        "policy_ballistic": "(a)  Ideal CoM flight (background only)",
        "push_title": "(b)  Virtual extension and push-off",
        "quantities_title": "(c)  Heights and body pitch",
        "model_flow": "(d)  Model-reference planning",
        "policy_flow": "(d)  Archived policy and training guidance",
        "takeoff": "Takeoff",
        "landing": "Landing",
        "com": "CoM C",
        "push_impulse": "Net push-off impulse",
        "model_1": "Task goals",
        "model_2": "Takeoff state",
        "model_3": "Extension and offset trajectories",
        "model_4": "Joint references",
        "policy_1": "Task commands",
        "policy_2": "Policy output",
        "policy_3": "Phase / abduction corrections",
        "policy_4": "Joint targets: PD + wheel speed",
        "training": "Reward guidance (training only)",
        "policy_update": "Policy update",
        "observation_feedback": "Observation feedback",
        "feedback": "Contact / flight feedback",
        "model_caution": "Ideal model/reference schematic, not measured flight or the archived policy execution path",
        "policy_caution": "Explicit trajectories disabled; CoM parabola is background, not a tracked policy reference",
    },
}


def load_policy(config_path):
    cfg = json.loads(config_path.read_text())
    control = cfg["control"]
    if not cfg["env"]["use_jump"] or not cfg["env"]["use_jump_distance"]:
        raise ValueError("The archived figure requires the forward-jump command interface")
    if control["jump_use_explicit_trajectory"]:
        raise ValueError("The archived-policy figure is specifically for the trajectory-disabled final policy")
    if control["jump_residual_profile"] != "step":
        raise ValueError("The illustrated phase-dependent command corrections require the archived step profile")
    gravity = cfg["sim"]["gravity"]
    if gravity[:2] != [0.0, 0.0] or gravity[2] >= 0:
        raise ValueError("The illustrative flight assumes uniform vertical gravity")
    return cfg


def load_inertials(urdf):
    inertials = []
    for link in ET.parse(urdf).getroot().findall("link"):
        node = link.find("inertial")
        if node is not None:
            mass = float(node.find("mass").get("value"))
            if mass > 0:
                origin = node.find("origin")
                xyz = style.values(origin, "xyz") if origin is not None else np.zeros(3)
                inertials.append((link.get("name"), mass, np.append(xyz, 1.0)))
    if not inertials or not all(np.isfinite(mass) for _, mass, _ in inertials):
        raise ValueError("A positive finite nominal mass model is needed to locate the actual rigid-body CoM")
    return inertials


def center_of_mass(scene, inertials):
    mass = sum(value for _, value, _ in inertials)
    return sum(value * (scene.frames[name] @ local)[:3] for name, value, local in inertials) / mass


def rotate_pitch(scene, angle_deg):
    rotation = style.rotation_rpy(np.deg2rad([0, angle_deg, 0]))
    for frame in scene.frames.values():
        frame[:3, :3] = rotation @ frame[:3, :3]
        frame[:3, 3] = rotation @ frame[:3, 3]
    for point in scene.centers.values():
        point[:] = rotation @ point
    for collision in scene.collisions:
        collision.transform[:3, :3] = rotation @ collision.transform[:3, :3]
        collision.transform[:3, 3] = rotation @ collision.transform[:3, 3]


def wheel_bottom(scene, side):
    wheel = next(c for c in scene.collisions if c.link == f"wheel_{side}_Link")
    points = np.concatenate(style.collision_faces(wheel))
    lowest = points[:, 2].min()
    return points[np.isclose(points[:, 2], lowest, atol=1e-10, rtol=0)].mean(axis=0)


def make_scenes(robot, inertials):
    push = geometry.make_scene(robot, PUSH_POSE)
    z = min(wheel_bottom(push, side)[2] for side in "LR")
    com = center_of_mass(push, inertials)
    geometry.translate_scene(push, np.array([-com[0], 0.0, -z]))
    c0 = center_of_mass(push, inertials)
    air = geometry.make_scene(robot, AIR_POSE)
    rotate_pitch(air, AIR_PITCH_DEG)
    ca = center_of_mass(air, inertials)
    geometry.translate_scene(air, np.array([-ca[0], 0.0, c0[2] + COM_RISE_M - ca[2]]))
    if max(abs(wheel_bottom(push, side)[2]) for side in "LR") > 1e-8:
        raise ValueError("The push-off illustration requires both wheels on the ground")
    if min(wheel_bottom(air, side)[2] for side in "LR") <= 0:
        raise ValueError("The posture-definition illustration must be airborne")
    return push, air


def ballistic_reference(gravity):
    vz = np.sqrt(2 * gravity * COM_RISE_M)
    duration = 2 * vz / gravity
    vx = DISTANCE_M / duration
    time = np.linspace(0, duration, 601)
    return time, vx * time, vz * time - 0.5 * gravity * time ** 2, vx, vz


def arrow(ax, start, finish, color, double=False, linewidth=1.4, dashed=False):
    ax.add_patch(FancyArrowPatch(start, finish, arrowstyle="<->" if double else "-|>",
                                 linewidth=linewidth, color=color, mutation_scale=11,
                                 linestyle=(0, (4, 3)) if dashed else "-",
                                 shrinkA=0, shrinkB=0, zorder=6))


def draw_ballistic(ax, gravity, variant, words):
    time, x, z, vx, vz = ballistic_reference(gravity)
    ax.plot(x, z, color=COM_COLOR, linewidth=1.9,
            linestyle="-" if variant == "model" else (0, (4, 3)))
    ax.plot([-0.09, DISTANCE_M + 0.08], [0, 0], color="#9CAAB3", linewidth=0.9,
            linestyle=(0, (4, 3)))
    ax.scatter([0, DISTANCE_M / 2, DISTANCE_M], [0, COM_RISE_M, 0],
                c=COM_COLOR, s=[31, 26, 31], zorder=7)
    velocity_scale = 0.07
    dx, dz = velocity_scale * vx, velocity_scale * vz
    arrow(ax, (0, 0), (dx, 0), HORIZONTAL_COLOR)
    arrow(ax, (0, 0), (0, dz), VERTICAL_COLOR)
    arrow(ax, (0, 0), (dx, dz), COM_COLOR)
    ax.plot([dx, dx], [0, dz], color="#BCC8CE", linestyle=(0, (3, 3)), linewidth=0.75)
    ax.plot([0, dx], [dz, dz], color="#BCC8CE", linestyle=(0, (3, 3)), linewidth=0.75)
    vx_label = r"$v_{C,x,0}^\star$" if variant == "model" else r"$v_{C,x,0}$"
    vz_label = r"$v_{C,z,0}^\star$" if variant == "model" else r"$v_{C,z,0}$"
    v_label = r"$\mathbf{v}_{C,0}^\star$" if variant == "model" else r"$\mathbf{v}_{C,0}$"
    ax.text(dx * 0.65, -0.026, vx_label, fontsize=FONT_SIZE, color=HORIZONTAL_COLOR, ha="center")
    ax.text(-0.025, dz / 2, vz_label, fontsize=FONT_SIZE, color=VERTICAL_COLOR,
            ha="right", va="center")
    ax.text(dx + 0.012, dz + 0.015, v_label, fontsize=FONT_SIZE, color=COM_COLOR)
    arrow(ax, (DISTANCE_M / 2 + 0.025, 0), (DISTANCE_M / 2 + 0.025, COM_RISE_M),
          VERTICAL_COLOR, double=True)
    rise = r"$H_C^\star$" if variant == "model" else r"$\Delta z_C$"
    ax.text(DISTANCE_M / 2 + 0.047, COM_RISE_M / 2, rise, fontsize=FONT_SIZE,
            color=VERTICAL_COLOR, va="center")
    arrow(ax, (0, -0.080), (DISTANCE_M, -0.080), HORIZONTAL_COLOR, double=True)
    for endpoint in (0, DISTANCE_M):
        ax.plot([endpoint, endpoint], [-0.075, 0], color="#AEBBC4", linewidth=0.7,
                linestyle=(0, (3, 3)))
    distance = r"$D_C^\star$" if variant == "model" else r"$D_C$"
    ax.text(DISTANCE_M / 2, -0.105, distance, fontsize=FONT_SIZE, color=HORIZONTAL_COLOR,
            ha="center", va="center")
    ax.text(-0.015, -0.049, words["takeoff"] + r" $C_0$", ha="right", fontsize=FONT_SIZE, color=style.INK)
    ax.text(DISTANCE_M + 0.010, -0.049, words["landing"] + r" $C_T$", ha="left", fontsize=FONT_SIZE, color=style.INK)
    ax.set(xlim=(-0.20, DISTANCE_M + 0.22), ylim=(-0.14, COM_RISE_M + 0.065), aspect="equal")
    ax.axis("off")


def draw_push(ax, push, inertials, words):
    projection = style.Projection([0, -1, 0])
    ax.fill_between([-0.60, 0.56], -0.025, 0, color="#E6ECEF", zorder=0)
    ax.plot([-0.60, 0.56], [0, 0], color="#93A3AE", linewidth=0.9)
    geometry.draw_robot(ax, push, projection, side="L")
    com = center_of_mass(push, inertials)
    wheel = np.mean([push.centers[f"wheel_{side}_Joint"] for side in "LR"], axis=0)
    c, w = projection.xy([com, wheel])
    ax.scatter(*c, s=31, color=COM_COLOR, zorder=8)
    ax.text(c[0] + 0.045, c[1], r"$C$", fontsize=FONT_SIZE, color=COM_COLOR, va="center")
    ax.scatter(*w, s=18, color=style.MUTED, zorder=8)
    ax.text(w[0] - 0.15, w[1] - 0.035, r"$W$", fontsize=FONT_SIZE, color=style.MUTED,
            ha="right", va="center")
    ax.plot([w[0], c[0]], [w[1], c[1]], color="#8296A1", linewidth=0.85, linestyle=(0, (3, 3)))
    vector = c - w
    normal = np.array([-vector[1], vector[0]]) / np.linalg.norm(vector)
    shift = 0.10 * normal
    arrow(ax, w + shift, c + shift, HORIZONTAL_COLOR, double=True)
    midpoint = (w + c) / 2 + 1.7 * shift
    ax.text(*midpoint, r"$\ell_C$", color=HORIZONTAL_COLOR, fontsize=FONT_SIZE, ha="right", va="center")
    alpha = float(np.arctan2(vector[0], vector[1]))
    ax.plot([w[0], w[0]], [w[1], w[1] + 0.24], color="#AAB7BF", linewidth=0.8, linestyle=(0, (3, 3)))
    ax.add_patch(Arc(w, 0.36, 0.36, theta1=90 - np.rad2deg(alpha), theta2=90,
                     color=ANGLE_COLOR, linewidth=1.2))
    angle_target = w + 0.18 * np.array([np.sin(alpha / 2), np.cos(alpha / 2)])
    geometry.leader(ax, angle_target, (-0.50, 0.26), r"$\alpha_C$", ANGLE_COLOR, align="right")
    contact = projection.xy(wheel_bottom(push, "L"))
    force_x, force_z = 0.22, 0.32
    arrow(ax, contact, contact + [force_x, 0], HORIZONTAL_COLOR)
    arrow(ax, contact + [force_x, 0], contact + [force_x, force_z], VERTICAL_COLOR)
    arrow(ax, contact, contact + [force_x, force_z], FORCE_COLOR)
    ax.text(contact[0] + force_x + 0.09, 0.025, r"$F_x$", color=HORIZONTAL_COLOR, fontsize=FONT_SIZE, ha="center")
    ax.text(contact[0] + force_x + 0.045, force_z / 2, r"$F_z$", color=VERTICAL_COLOR, fontsize=FONT_SIZE)
    ax.text(0, -0.17, words["push_impulse"] + r" $J_x,\ J_z^{\mathrm{net}}$",
            fontsize=FONT_SIZE, color=style.INK, ha="center")
    ax.set(xlim=(-0.64, 0.58), ylim=(-0.22, 1.08), aspect="equal")
    ax.axis("off")
    return {"virtual_length_m": float(np.linalg.norm(vector)), "virtual_angle_rad": alpha}


def draw_quantities(ax, air, inertials, words):
    projection = style.Projection([0, -1, 0])
    ax.fill_between([-0.58, 0.60], -0.025, 0, color="#E6ECEF", zorder=0)
    ax.plot([-0.58, 0.60], [0, 0], color="#93A3AE", linewidth=0.9)
    geometry.draw_robot(ax, air, projection, side="L")
    com = center_of_mass(air, inertials)
    c = projection.xy(com)
    ax.scatter(*c, color=COM_COLOR, s=31, zorder=8)
    geometry.leader(ax, c, (-0.31, c[1] + 0.075), words["com"], COM_COLOR, align="right")
    base_origin = projection.xy(air.frames["base_Link"][:3, 3])
    ax.scatter(*base_origin, color=HORIZONTAL_COLOR, s=20, edgecolors="white", linewidths=0.5, zorder=8)
    ax.text(base_origin[0] - 0.08, base_origin[1] + 0.015, r"$B$", fontsize=FONT_SIZE,
            color=HORIZONTAL_COLOR, ha="right")
    x_height = -0.39
    ax.plot([x_height, c[0]], [c[1], c[1]], color=COM_COLOR, linewidth=0.8, linestyle=(0, (3, 3)))
    arrow(ax, (x_height, 0), (x_height, com[2]), COM_COLOR, double=True)
    ax.text(x_height - 0.035, com[2] / 2, r"$z_C$", color=COM_COLOR, fontsize=FONT_SIZE,
            ha="right", va="center")
    bottom = min((wheel_bottom(air, side) for side in "LR"), key=lambda point: point[2])
    floor = bottom.copy()
    floor[2] = 0
    p, ground = projection.xy([bottom, floor])
    shift = np.array([0.20, 0])
    for endpoint in (p, ground):
        ax.plot([endpoint[0], endpoint[0] + shift[0]], [endpoint[1], endpoint[1]],
                color=VERTICAL_COLOR, linewidth=0.8)
    ax.scatter(*p, s=20, color=style.INK, zorder=8)
    arrow(ax, ground + shift, p + shift, VERTICAL_COLOR, double=True)
    ax.text(p[0] + 0.24, p[1] / 2, r"$h_w$", color=VERTICAL_COLOR, fontsize=FONT_SIZE, va="center")
    base_collision = next(c for c in air.collisions if c.link == "base_Link")
    center = projection.xy(base_collision.transform[:3, 3])
    theta = np.deg2rad(AIR_PITCH_DEG)
    up = np.array([np.sin(theta), np.cos(theta)])
    ax.plot([center[0], center[0]], [center[1], center[1] + 0.24], color="#AAB7BF",
            linewidth=0.85, linestyle=(0, (3, 3)))
    ax.plot([center[0], center[0] + 0.24 * up[0]], [center[1], center[1] + 0.24 * up[1]],
            color=ANGLE_COLOR, linewidth=1.2)
    ax.add_patch(Arc(center, 0.32, 0.32, theta1=90 - AIR_PITCH_DEG, theta2=90,
                     color=ANGLE_COLOR, linewidth=1.1))
    ax.text(center[0] + 0.24, center[1] + 0.15, r"$\theta_b$", fontsize=FONT_SIZE, color=ANGLE_COLOR)
    ax.set(xlim=(-0.60, 0.63), ylim=(-0.08, 1.20), aspect="equal")
    ax.axis("off")
    return {"com_height_m": float(com[2]), "wheel_clearance_m": float(bottom[2]),
            "body_pitch_deg": AIR_PITCH_DEG}


def flow_node(ax, y, label, formula, color):
    ax.text(0.45, y + 0.035, label, fontsize=FONT_SIZE, color=style.INK, ha="center", va="center")
    ax.text(0.45, y - 0.035, formula, fontsize=FONT_SIZE, color=color, ha="center", va="center")


def draw_flow(ax, variant, words):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    if variant == "model":
        nodes = ((0.88, "model_1", r"$D_C^\star,\ H_C^\star$"),
                 (0.66, "model_2", r"$v_{C,x,0}^\star,\ v_{C,z,0}^\star$"),
                 (0.44, "model_3", r"$\ell_C(t),\ \alpha_C(t)$"),
                 (0.22, "model_4", r"$q^{\mathrm{ref}}(t),\ \dot q^{\mathrm{ref}}(t)$"))
        for y, key, formula in nodes:
            flow_node(ax, y, words[key], formula, HORIZONTAL_COLOR)
        for y in (0.88, 0.66, 0.44):
            arrow(ax, (0.45, y - 0.10), (0.45, y - 0.14), style.MUTED)
    else:
        nodes = ((0.90, "policy_1", r"$h_{\mathrm{cmd}},\ D_{\mathrm{cmd}},\ j$"),
                 (0.70, "policy_2", r"$a_t=\pi(o_t,c_t)$"),
                 (0.50, "policy_3", r"$\delta q_k(h),\ \delta q_h(D)$"),
                 (0.30, "policy_4", r"$q^{\mathrm{des}},\ \omega_w^{\mathrm{des}}$"))
        for y, key, formula in nodes:
            flow_node(ax, y, words[key], formula, HORIZONTAL_COLOR)
        for y in (0.90, 0.70, 0.50):
            arrow(ax, (0.45, y - 0.095), (0.45, y - 0.125), style.MUTED)
        flow_node(ax, 0.075, words["training"], r"$v_{B,z}^\star,\ \bar v_{w,\parallel}^\star,\ h_w,\ D_w$", COM_COLOR)
        arrow(ax, (0.45, 0.24), (0.45, 0.225), style.MUTED)
        ax.text(0.45, 0.185, words["feedback"], fontsize=FONT_SIZE, color=style.MUTED, ha="center", va="center")
        arrow(ax, (0.45, 0.15), (0.45, 0.135), style.MUTED)
        ax.plot([0.45, 0.04, 0.04], [0.155, 0.155, 0.70], color=style.MUTED, linewidth=0.9)
        arrow(ax, (0.04, 0.70), (0.20, 0.70), style.MUTED)
        ax.text(-0.02, 0.43, words["observation_feedback"], rotation=90, fontsize=FONT_SIZE,
                color=style.MUTED, ha="center", va="center")
        ax.plot([0.73, 0.89], [0.08, 0.08], color=COM_COLOR, linewidth=1, linestyle=(0, (4, 3)))
        ax.plot([0.89, 0.89], [0.08, 0.70], color=COM_COLOR, linewidth=1,
                linestyle=(0, (4, 3)))
        arrow(ax, (0.89, 0.70), (0.72, 0.70), COM_COLOR, dashed=True)
        ax.text(0.94, 0.40, words["policy_update"], rotation=90, fontsize=FONT_SIZE,
                color=COM_COLOR, ha="center", va="center")


def create_figure(robot, inertials, scenes, cfg, variant, language):
    words = TEXT[language]
    push, air = scenes
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=140)
    for position, key in (((0.04, 0.95), variant + "_ballistic"),
                          ((0.04, 0.515), "push_title"), ((0.345, 0.515), "quantities_title"),
                          ((0.70, 0.95), variant + "_flow")):
        fig.text(*position, words[key], fontsize=FONT_SIZE, color=style.INK)
    draw_ballistic(fig.add_axes([0.045, 0.585, 0.59, 0.31]), abs(cfg["sim"]["gravity"][2]), variant, words)
    draw_push(fig.add_axes([0.035, 0.13, 0.28, 0.34]), push, inertials, words)
    draw_quantities(fig.add_axes([0.34, 0.13, 0.29, 0.34]), air, inertials, words)
    draw_flow(fig.add_axes([0.695, 0.13, 0.28, 0.76]), variant, words)
    fig.text(0.52, 0.045, words[variant + "_caution"], fontsize=FONT_SIZE, ha="center", color=style.INK)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            if not np.isclose(text.get_fontsize(), FONT_SIZE):
                raise ValueError("All figure labels must use 14 pt")
            if not all(fig.bbox.contains(*corner) for corner in text.get_window_extent(renderer).get_points()):
                raise ValueError("Text lies outside the export: " + text.get_text())
    return fig


def policy_targets(cfg, height, distance):
    height = np.asarray(height, dtype=float)
    distance = np.asarray(distance, dtype=float)
    flight_time = 2 * np.sqrt(2 * np.maximum(height, 0.05) / 9.81)
    root_vz = cfg["rewards"]["jump_takeoff_vz_command_gain"] * np.sqrt(2 * 9.81 * np.maximum(height, 0.02))
    wheel_vx = np.minimum(distance / np.maximum(flight_time, 0.15), 3.0)
    return root_vz, wheel_vx, flight_time


def write_data(output, cfg, inertials, scenes, sources, variant, dpi):
    gravity = abs(cfg["sim"]["gravity"][2])
    t, x, z, vx, vz = ballistic_reference(gravity)
    with (output / "ideal_com_ballistic_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model_time_s", "com_horizontal_displacement_m", "com_rise_from_takeoff_m", "status"])
        for row in zip(t, x, z):
            writer.writerow([*row, "illustrative_ideal_model_not_rollout_or_achieved_capability"])
    lo_h, hi_h = cfg["commands"]["jump_height_cmd_range"]
    lo_d, hi_d = cfg["commands"]["jump_dist_cmd_range"]
    with (output / "archived_policy_reward_targets.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["height_command_m", "distance_command_m", "root_vertical_reward_target_mps",
                         "wheel_midpoint_forward_reward_target_mps", "nominal_proxy_flight_time_s", "status"])
        for h in np.linspace(lo_h, hi_h, 3):
            for d in np.linspace(lo_d, hi_d, 3):
                bvz, wvx, duration = policy_targets(cfg, h, d)
                writer.writerow([h, d, float(bvz), float(wvx), float(duration), "training_target_not_achieved_velocity"])
    push, air = scenes
    c0, ca = (center_of_mass(scene, inertials) for scene in scenes)
    w0 = np.mean([push.centers[f"wheel_{side}_Joint"] for side in "LR"], axis=0)
    vector = (c0 - w0)[[0, 2]]
    manifest = {
        "variant": variant,
        "figure_type": "Concept/reference planning" if variant == "model" else "Archived policy architecture with ideal physics background",
        "sources": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for name, path in sources.items()},
        "rendered_curves_are_measurements": False,
        "ideal_model": {"equal_takeoff_landing_com_height": True, "gravity_mps2": gravity,
                        "illustrative_com_rise_m": COM_RISE_M, "illustrative_com_distance_m": DISTANCE_M,
                        "takeoff_com_vx_mps": float(vx), "takeoff_com_vz_mps": float(vz),
                        "flight_time_s": float(t[-1]), "reachability_or_training_domain_claim": False},
        "model_goal_warning": "H_C and D_C are centroidal model goals, NOT aliases for archived wheel-clearance/distance commands",
        "centroid_definition": "Mass-weighted CoM of all positive-mass URDF links, including fixed links; not base origin or visual-box center",
        "nominal_mass_kg": sum(mass for _, mass, _ in inertials),
        "push_com_m": c0.tolist(), "air_com_m": ca.tolist(),
        "air_body_reference_B_m": air.frames["base_Link"][:3, 3].tolist(),
        "push_pose_rad": PUSH_POSE, "air_pose_rad": AIR_POSE,
        "virtual_leg_definition": "Sagittal vector from mean wheel center W to full rigid-body CoM C; NOT hip-to-wheel length",
        "virtual_length_m": float(np.linalg.norm(vector)),
        "virtual_offset_rad": float(np.arctan2(vector[0], vector[1])),
        "air_instantaneous_wheel_clearance_m": float(min(wheel_bottom(air, side)[2] for side in "LR")),
        "body_pitch_definition": "Rotation about +Y, measured clockwise from world vertical to body +Z in the side view",
        "illustrative_air_body_pitch_deg": AIR_PITCH_DEG,
        "model_reference_warning": "Four-step model-reference diagram is not a claim that an implemented full CoM planner generated the archived policy actions",
        "policy": {"explicit_trajectory_enabled": cfg["control"]["jump_use_explicit_trajectory"],
                   "knee_height_action_gain": cfg["control"]["jump_knee_height_residual"],
                   "hip_distance_action_gain": cfg["control"]["jump_hip_distance_residual"],
                   "action_correction_profile": cfg["control"]["jump_residual_profile"],
                   "root_vz_reward_gain": cfg["rewards"]["jump_takeoff_vz_command_gain"],
                   "height_command_range_m": [lo_h, hi_h], "distance_command_range_m": [lo_d, hi_d],
                   "abduction_projection": cfg["control"]["jump_project_abad_symmetry"],
                   "hip_knee_full_projection": cfg["control"]["jump_project_leg_symmetry"]},
        "physical_mechanism": "From rest: M*v_x0=integral(F_x)dt; M*v_z0=integral(F_z-M*g)dt. Body pitch alone does not prescribe this impulse.",
        "force_arrows": "Qualitative sagittal ground-force components; no force or trajectory performance data are plotted",
        "policy_loops": "Solid observation feedback in deployment; dashed reward-driven policy updates only during training",
        "font_size_pt": FONT_SIZE, "raster_size_px": [round(dpi * value) for value in FIGURE_SIZE],
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-source", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant", choices=("model", "policy", "both"), default="both")
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    cfg = load_policy(args.config)
    robot = style.CollisionRobot(args.urdf)
    style.validate_robot(robot)
    inertials = load_inertials(args.urdf)
    scenes = make_scenes(robot, inertials)
    corpus = "".join(value for words in TEXT.values() for value in words.values())
    variants = ("model", "policy") if args.variant == "both" else (args.variant,)
    for variant in variants:
        output = args.output_dir / SUBDIRS[variant] if args.variant == "both" else args.output_dir
        style.configure_fonts(output, extra_text=corpus)
        plt.rcParams.update({"font.size": FONT_SIZE, "font.weight": "normal",
                             "axes.labelweight": "normal", "axes.titleweight": "normal"})
        languages = ("zh", "en") if args.language == "both" else (args.language,)
        for language in languages:
            fig = create_figure(robot, inertials, scenes, cfg, variant, language)
            for extension in ("png", "pdf", "svg"):
                target = output / f"{STEMS[variant]}_{language}.{extension}"
                fig.savefig(target, dpi=args.dpi, facecolor="white")
                print(target)
            plt.close(fig)
        write_data(output, cfg, inertials, scenes,
                   {"nominal_urdf": args.urdf, "archived_winner_config": args.config,
                    "archived_policy_source": args.env_source}, variant, args.dpi)
    print("Two distinct paths: ideal centroidal reference planning vs archived trajectory-disabled policy")


if __name__ == "__main__":
    main()
