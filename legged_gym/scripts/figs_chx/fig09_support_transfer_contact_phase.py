#!/usr/bin/env python3
"""Illustrate active swing/load-transfer guidance, not measured contacts."""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
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
DEFAULT_CONFIG = (SNAPSHOT / "clearance_env_cfg.json" if SNAPSHOT.is_dir() else
                  EXPERIMENT / "selected_generalists/clearance/env_cfg.json")
DEFAULT_ENV = (SNAPSHOT / "wheelfoot_flat.py" if SNAPSHOT.is_dir() else
               EXPERIMENT / "code_snapshot/legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py")
DEFAULT_BASE = (SNAPSHOT / "base_task.py" if SNAPSHOT.is_dir() else
                REPO / "legged_gym/envs/base/base_task.py")
DEFAULT_OUTPUT = HERE / "out/support_transfer_contact_phase"
FONT_SIZE = 14
FIGURE_SIZE = (12.8, 7.4)
STANCE_COLOR = "#597FA4"
SWING_COLOR = "#E6C98C"
TRANSFER_COLOR = "#B34959"
ILLUSTRATIVE_ROLL_DEG = 12.0
TEXT = {
    "zh": {
        "scene_title": "(a)  支撑转移引导",
        "phase_title": "(b)  期望支撑／摆动相位",
        "right_support": "右侧支撑、左侧摆动",
        "transfer": "支撑转移",
        "left_support": "左侧支撑、右侧摆动",
        "left_row": "左侧 L",
        "right_row": "右侧 R",
        "force_row": "接触力引导",
        "stance": "支撑",
        "swing": "摆动",
        "load_right": "右侧占比 ↑",
        "load_left": "左侧占比 ↑",
        "phase_axis": "归一化步态相位",
        "caution": "迈步模式的期望引导示意，非实测接触",
    },
    "en": {
        "scene_title": "(a)  Support-transfer guidance",
        "phase_title": "(b)  Intended stance/swing phases",
        "right_support": "Right stance, left swing",
        "transfer": "Support transfer",
        "left_support": "Left stance, right swing",
        "left_row": "Left L",
        "right_row": "Right R",
        "force_row": "Force guidance",
        "stance": "Stance",
        "swing": "Swing",
        "load_right": "Right share ↑",
        "load_left": "Left share ↑",
        "phase_axis": "Normalized gait phase",
        "caution": "Intended stepping guidance, not measured contacts",
    },
}


def assignment(nodes, name):
    return next(node.value for node in nodes if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets))


def compare_threshold(function, name, operator):
    value = assignment(function.body, name)
    if isinstance(value, ast.Call):
        value = value.func.value
    if not isinstance(value, ast.Compare) or not isinstance(value.ops[0], operator):
        raise ValueError("Unexpected phase/gate comparison in archived source")
    return float(ast.literal_eval(value.comparators[0]))


def load_guidance(config_path, env_path):
    config = json.loads(config_path.read_text())
    scales = config["rewards"]["scales"]
    if not config["env"]["use_gait_stepping"] or not config["env"]["use_gait_phase"]:
        raise ValueError("This schematic requires enabled stepping and phase inputs")
    if any(scales[name] <= 0 for name in ("swing_phase", "weight_shift")):
        raise ValueError("Both illustrated guidance terms must be active")
    disabled = ("tracking_contacts_shaped_force", "tracking_contacts_shaped_vel", "foot_clearance")
    if any(scales.get(name, 0.0) != 0 for name in disabled):
        raise ValueError("Do not conflate the illustrated guidance with the legacy contact-probability branch")
    tree = ast.parse(env_path.read_text(encoding="utf-8"))
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    swing = functions["_reward_swing_phase"]
    weight = functions["_reward_weight_shift"]
    switch = compare_threshold(swing, "left_should", ast.Lt)
    if switch != compare_threshold(weight, "left_should_swing", ast.Lt):
        raise ValueError("Swing and load-transfer guidance use different phase boundaries")
    gate = compare_threshold(swing, "step_gate", ast.Gt)
    if gate != compare_threshold(weight, "step_gate", ast.Gt):
        raise ValueError("Swing and load-transfer guidance use different command gates")
    if switch != 0.5:
        raise ValueError("The equal-half illustration requires the archived half-cycle switch")
    return {
        "phase_switch": switch, "step_command_gate_mps": gate,
        "swing_phase_weight": scales["swing_phase"], "weight_shift_weight": scales["weight_shift"],
        "disabled_legacy_weights": {name: scales.get(name, 0.0) for name in disabled},
        "frequency_range_hz": config["gait"]["ranges"]["frequencies"],
        "duration_range": config["gait"]["ranges"]["durations"],
        "first_half": "Left swing, right stance; increase right-side vertical-force share",
        "second_half": "Right swing, left stance; increase left-side vertical-force share",
        "source_kind": "Active swing_phase/weight_shift intent, NOT desired_contact_states array values",
    }


def wheel_bottom(scene, side):
    wheel = next(c for c in scene.collisions if c.link == f"wheel_{side}_Link")
    vertices = np.concatenate(style.collision_faces(wheel))
    zmin = vertices[:, 2].min()
    return vertices[np.isclose(vertices[:, 2], zmin, atol=1e-10, rtol=0)].mean(axis=0)


def rotate_scene(scene, roll_deg):
    rotation = style.rotation_rpy(np.deg2rad([roll_deg, 0.0, 0.0]))
    for frame in scene.frames.values():
        frame[:3, :3] = rotation @ frame[:3, :3]
        frame[:3, 3] = rotation @ frame[:3, 3]
    for center in scene.centers.values():
        center[:] = rotation @ center
    for collision in scene.collisions:
        collision.transform[:3, :3] = rotation @ collision.transform[:3, :3]
        collision.transform[:3, 3] = rotation @ collision.transform[:3, 3]


def make_state(robot, state):
    if state == "right_support":
        pose, roll, stance = {"hip_L_Joint": -0.55, "knee_L_Joint": 0.20}, ILLUSTRATIVE_ROLL_DEG, "R"
    elif state == "left_support":
        pose, roll, stance = {"hip_R_Joint": 0.55, "knee_R_Joint": -0.20}, -ILLUSTRATIVE_ROLL_DEG, "L"
    else:
        pose, roll, stance = {}, 0.0, "R"
    scene = geometry.make_scene(robot, pose)
    rotate_scene(scene, roll)
    contact = wheel_bottom(scene, stance)
    anchor = robot.centers[f"wheel_{stance}_Joint"]
    shift = np.array([anchor[0] - contact[0], anchor[1] - contact[1], -contact[2]])
    geometry.translate_scene(scene, shift)
    clearances = {side: float(wheel_bottom(scene, side)[2]) for side in "LR"}
    if abs(clearances[stance]) > 1e-10:
        raise ValueError("The illustrated support wheel must touch the ground")
    if state == "transfer":
        if max(abs(value) for value in clearances.values()) > 1e-10:
            raise ValueError("The illustrative handover pose must show both wheels on the ground")
    elif clearances["L" if stance == "R" else "R"] <= 0:
        raise ValueError("The illustrated swing wheel must have positive clearance")
    return scene, {"joint_angles_rad": pose, "illustrative_body_roll_deg": roll,
                   "collision_clearances_m": clearances}


def force_arrow(ax, scene, projection, side, length, color):
    contact = wheel_bottom(scene, side)
    start = projection.xy(contact)
    finish = projection.xy(contact + [0, 0, length])
    ax.add_patch(FancyArrowPatch(start, finish, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.8, color=color, zorder=7))
    offset = -0.13 if side == "R" else 0.13
    ax.text(start[0] + offset, start[1] + 0.19, rf"$F_{{{side},z}}$", color=color,
            fontsize=FONT_SIZE, ha="right" if side == "R" else "left", va="center")


def draw_state(ax, robot, state):
    scene, info = make_state(robot, state)
    projection = style.Projection([1, 0, 0])
    ax.fill_between([-0.52, 0.52], -0.025, 0, color="#E6ECEF", zorder=0)
    ax.plot([-0.52, 0.52], [0, 0], color="#91A1AA", linewidth=0.9, zorder=1)
    geometry.draw_robot(ax, scene, projection)
    if state == "transfer":
        force_arrow(ax, scene, projection, "R", 0.20, "#93A3AE")
        force_arrow(ax, scene, projection, "L", 0.30, STANCE_COLOR)
        ax.add_patch(FancyArrowPatch((-0.24, 1.15), (0.24, 1.15), arrowstyle="-|>",
                                     mutation_scale=12, color=TRANSFER_COLOR, linewidth=1.7))
        ax.text(0, 1.25, r"$R\ \rightarrow\ L$", ha="center", va="center",
                color=TRANSFER_COLOR, fontsize=FONT_SIZE)
    else:
        force_arrow(ax, scene, projection, "R" if state == "right_support" else "L", 0.34, STANCE_COLOR)
    for side in "LR":
        foot = projection.xy(wheel_bottom(scene, side))
        ax.text(foot[0], -0.09, side, fontsize=FONT_SIZE, ha="center", va="center", color=style.MUTED)
    ax.set(xlim=(-0.58, 0.58), ylim=(-0.15, 1.34), aspect="equal")
    ax.axis("off")
    return info


def draw_phase_bands(ax, guidance, words):
    switch = guidance["phase_switch"]
    for center_y, stance_first in ((1.40, False), (0.80, True)):
        for left, right, stance in ((0, switch, stance_first), (switch, 1, not stance_first)):
            ax.add_patch(Rectangle((left, center_y - 0.21), right - left, 0.42,
                                    facecolor=STANCE_COLOR if stance else SWING_COLOR,
                                    edgecolor="white", linewidth=0.7, zorder=2))
            ax.text((left + right) / 2, center_y, words["stance" if stance else "swing"],
                    fontsize=FONT_SIZE, ha="center", va="center",
                    color="white" if stance else style.INK, zorder=3)
    ax.axvline(switch, color="#83939F", linewidth=1, linestyle=(0, (4, 3)), zorder=4)
    ax.text(0.25, 0.20, words["load_right"], fontsize=FONT_SIZE, color=TRANSFER_COLOR,
            ha="center", va="center")
    ax.text(0.75, 0.20, words["load_left"], fontsize=FONT_SIZE, color=TRANSFER_COLOR,
            ha="center", va="center")
    ax.set(xlim=(0, 1), ylim=(-0.10, 1.78))
    ax.set_yticks([1.40, 0.80, 0.20], [words["left_row"], words["right_row"], words["force_row"]])
    ax.set_xticks([0, switch, 1], ["0", "0.5", "1"])
    ax.set_xlabel(words["phase_axis"], fontsize=FONT_SIZE, labelpad=8)
    ax.tick_params(axis="y", length=0, pad=12, labelsize=FONT_SIZE, colors=style.INK)
    ax.tick_params(axis="x", length=3, labelsize=FONT_SIZE, colors=style.INK)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#91A1AA")


def create_figure(robot, guidance, language):
    words = TEXT[language]
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=140)
    fig.text(0.04, 0.958, words["scene_title"], fontsize=FONT_SIZE, color=style.INK)
    stages = ("right_support", "transfer", "left_support")
    for left, state in zip((0.045, 0.375, 0.705), stages):
        fig.text(left + 0.125, 0.896, words[state], fontsize=FONT_SIZE,
                 ha="center", va="center", color=style.INK)
        draw_state(fig.add_axes([left, 0.455, 0.25, 0.405]), robot, state)
    for start, end in ((0.307, 0.358), (0.637, 0.688)):
        fig.add_artist(FancyArrowPatch((start, 0.67), (end, 0.67), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#8A9AA5"))
    fig.text(0.04, 0.410, words["phase_title"], fontsize=FONT_SIZE, color=style.INK)
    draw_phase_bands(fig.add_axes([0.14, 0.145, 0.72, 0.225]), guidance, words)
    fig.text(0.52, 0.035, words["caution"], fontsize=FONT_SIZE, color=style.INK, ha="center")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            if not np.isclose(text.get_fontsize(), FONT_SIZE):
                raise ValueError("Every label must use 14 pt")
            if not all(fig.bbox.contains(*corner) for corner in text.get_window_extent(renderer).get_points()):
                raise ValueError("Text falls outside the export: " + text.get_text())
    return fig


def write_data(output, robot, guidance, sources, dpi):
    with (output / "intended_phase_guidance.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["normalized_phase", "left_intended_stance", "left_intended_swing",
                         "right_intended_stance", "right_intended_swing", "preferred_force_share_side", "status"])
        for phase in np.linspace(0, 1, 1000, endpoint=False):
            left_swing = int(phase < guidance["phase_switch"])
            writer.writerow([phase, 1 - left_swing, left_swing, left_swing, 1 - left_swing,
                             "R" if left_swing else "L", "intended_guidance_not_contact_measurement"])
    poses = {state: make_state(robot, state)[1] for state in ("right_support", "transfer", "left_support")}
    manifest = {
        "figure_type": "Intended stepping guidance schematic, NOT measured contacts or ground forces",
        "sources": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for name, path in sources.items()},
        "guidance": guidance, "static_illustration_poses": poses,
        "phase_bands": "Binary intent from active swing_phase/weight_shift phase selection, not legacy desired_contact_states probabilities",
        "support_transfer": "Qualitative R-to-L load-transfer pose; no finite mandatory double-support interval is claimed",
        "actual_contact_timing": "Determined by dynamics, not by these bands",
        "view": "Front view: robot R appears on image left, robot L on image right; feet explicitly labelled",
        "force_arrows": "Ground-reaction direction and qualitative redistribution only; lengths are not in newtons",
        "geometry": "Nominal URDF collision primitives; illustrative roll/poses are not a dynamic equilibrium solution",
        "step_gate": "Guidance enabled only above the archived stepping command threshold",
        "normalization": "Phase is cyclic in [0,1); endpoint 1 is equivalent to 0, not an additional sample",
        "font_size_pt": FONT_SIZE, "raster_size_px": [round(dpi * value) for value in FIGURE_SIZE],
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Active guidance: phase <0.5 left swing/right support; phase >=0.5 right swing/left support")
    print("Legacy contact-probability rewards: disabled in archived policy config")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-source", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--base-source", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    robot = style.CollisionRobot(args.urdf)
    style.validate_robot(robot)
    guidance = load_guidance(args.config, args.env_source)
    corpus = "".join(value for words in TEXT.values() for value in words.values())
    style.configure_fonts(args.output_dir, extra_text=corpus)
    plt.rcParams.update({"font.size": FONT_SIZE, "font.weight": "normal",
                         "axes.labelweight": "normal", "axes.titleweight": "normal"})
    languages = ("zh", "en") if args.language == "both" else (args.language,)
    for language in languages:
        fig = create_figure(robot, guidance, language)
        for extension in ("png", "pdf", "svg"):
            target = args.output_dir / f"support_transfer_contact_phase_{language}.{extension}"
            fig.savefig(target, dpi=args.dpi, facecolor="white")
            print(target)
        plt.close(fig)
    write_data(args.output_dir, robot, guidance,
               {"urdf": args.urdf, "active_policy_config": args.config,
                "active_guidance_source": args.env_source, "legacy_schedule_source": args.base_source}, args.dpi)


if __name__ == "__main__":
    main()
