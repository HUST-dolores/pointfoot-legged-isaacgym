#!/usr/bin/env python3
"""Illustrate task-level rolling/stepping proxies and separate command inputs."""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as DrawingPath
from matplotlib.text import Text
import numpy as np

import fig0_sixd_design_variables as style
import fig03_four_capability_definitions as geometry


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SNAPSHOT = HERE / "source_snapshot"
DEFAULT_URDF = (HERE / "nominal_source.urdf" if (HERE / "nominal_source.urdf").is_file()
                else REPO / "resources/robots/WF_TRON1A/urdf/robot.urdf")
DEFAULT_CONFIG = (SNAPSHOT / "wheelfoot_flat_config.py" if SNAPSHOT.is_dir()
                  else REPO / "legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py")
DEFAULT_ENV = (SNAPSHOT / "wheelfoot_flat.py" if SNAPSHOT.is_dir()
               else REPO / "legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py")
DEFAULT_OUTPUT = HERE / "out/roll_step_velocity_definition"
FONT_SIZE = 14
FIGURE_SIZE = (12.8, 6.1)
ROLL_COLOR = "#487FB7"
STEP_COLOR = "#238B7B"
BODY_COLOR = "#B34959"
COMMANDS = (("pure", 0.9, 0.0), ("mixed", 0.5, 0.4))
MAIN_POSE = {"hip_L_Joint": -0.55, "knee_L_Joint": 0.20}
GHOST_POSE = {"hip_L_Joint": -0.90, "knee_L_Joint": 0.25}
TEXT = {
    "zh": {
        "scene_title": "(a)  运动贡献示意",
        "definition_title": "(b)  任务代理量与命令",
        "body": "机体前进",
        "wheel": "车轮旋转",
        "leg": "腿部摆动",
        "examples": "命令示例 (m/s)",
        "pure": "纯滚动",
        "mixed": "混合运动",
        "caution": "近似贡献解释，非精确接触运动学",
    },
    "en": {
        "scene_title": "(a)  Motion contributions",
        "definition_title": "(b)  Task proxies and commands",
        "body": "Body motion",
        "wheel": "Wheel spin",
        "leg": "Leg swing",
        "examples": "Commands (m/s)",
        "pure": "Pure rolling",
        "mixed": "Mixed motion",
        "caution": "Approximate contribution interpretation, not exact contact kinematics",
    },
}


def assignment(nodes, name):
    return next(node.value for node in nodes if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


def load_definitions(config_path, env_path, robot):
    tree = ast.parse(config_path.read_text(encoding="utf-8"))
    cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BipedCfgWF")
    asset = next(node for node in cfg.body if isinstance(node, ast.ClassDef) and node.name == "asset")
    commands = next(node for node in cfg.body if isinstance(node, ast.ClassDef) and node.name == "commands")
    radius = float(ast.literal_eval(assignment(asset.body, "foot_radius")))
    sign = float(ast.literal_eval(assignment(commands.body, "wheel_roll_sign")))
    if not np.isclose(radius, geometry.wheel_radius(robot)) or sign != 1.0:
        raise ValueError("The illustrated forward rotation requires matching radius and positive wheel sign")
    for side in "LR":
        np.testing.assert_allclose(robot.axis("wheel", side), [0, 1, 0], atol=1e-10)
    source = ast.parse(env_path.read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef)
                    and node.name == "_compute_step_roll_vel")
    wheel_input = assignment(function.body, "wheel_w")
    # Python 3.8 uses ExtSlice/Index nodes for multi-axis subscripts.
    dimensions = (wheel_input.slice.dims if isinstance(wheel_input.slice, ast.ExtSlice)
                  else wheel_input.slice.elts)
    wheel_indices = dimensions[1]
    if isinstance(wheel_indices, ast.Index):
        wheel_indices = wheel_indices.value
    indices = ast.literal_eval(wheel_indices)
    dof_names = [name for name, joint in robot.joints.items() if joint.get("type") != "fixed"]
    if [dof_names[index] for index in indices] != ["wheel_L_Joint", "wheel_R_Joint"]:
        raise ValueError("The source wheel indices no longer match this URDF")
    contact_expression = assignment(function.body, "contact").func.value
    if not isinstance(contact_expression, ast.Compare) or not isinstance(contact_expression.ops[0], ast.Gt):
        raise ValueError("Expected a strict contact-force threshold")
    threshold = float(ast.literal_eval(contact_expression.comparators[0]))
    return {"wheel_radius_m": radius, "wheel_roll_sign": sign,
            "wheel_dof_indices": indices, "contact_vertical_force_threshold_n": threshold,
            "contact_aggregation": "Binary contact-indicator mean, not contact-force magnitude weighting",
            "no_contact_fallback": "Arithmetic mean over both wheel joint speeds"}


def curved_arrow(ax, center, radius, start_deg, end_deg, color):
    angles = np.deg2rad(np.linspace(start_deg, end_deg, 100))
    points = np.asarray(center) + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    ax.add_patch(FancyArrowPatch(path=DrawingPath(points), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=1.8, color=color, zorder=7))
    return points


def draw_scene(ax, robot, words):
    projection = style.Projection([0, -1, 0])
    scene = geometry.make_scene(robot, MAIN_POSE)
    ghost = geometry.make_scene(robot, GHOST_POSE)
    geometry.ground_plane(ax, projection)
    moving = {"hip_L_Link", "knee_L_Link", "wheel_L_Link"}
    ghost_geometry = SimpleNamespace(collisions=[c for c in ghost.collisions if c.link in moving])
    style.draw_collisions(ax, ghost_geometry, projection)
    ax.collections[-1].set_alpha(0.18)
    geometry.draw_robot(ax, scene, projection)
    wheel_center = projection.xy(scene.centers["wheel_R_Joint"])
    curved_arrow(ax, wheel_center, 0.17, 180, 0, ROLL_COLOR)
    wheel_target = wheel_center + 0.17 * np.array([np.cos(np.deg2rad(155)), np.sin(np.deg2rad(155))])
    geometry.leader(ax, wheel_target, (-0.30, 0.19),
                    words["wheel"] + r" $\omega_w$", ROLL_COLOR, align="right")
    hip_center = projection.xy(scene.centers["hip_L_Joint"])
    leg_arc = curved_arrow(ax, hip_center, 0.66, -59, -21, STEP_COLOR)
    geometry.leader(ax, leg_arc[-1], (0.83, 0.56), words["leg"], STEP_COLOR, align="right")
    ax.add_patch(FancyArrowPatch((0.20, 1.015), (0.82, 1.015), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=1.9, color=BODY_COLOR, zorder=7))
    ax.text(0.50, 1.105, words["body"] + r" $v_x$", color=BODY_COLOR,
            ha="center", va="center", fontsize=FONT_SIZE)
    ax.set(xlim=(-0.78, 0.98), ylim=(-0.10, 1.25), aspect="equal")
    ax.axis("off")
    return scene


def draw_definitions(ax, words):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    for y, formula, color in (
        (0.94, r"$v_{\mathrm{roll}} := r\,\bar{\omega}_{\mathcal{C}}$", ROLL_COLOR),
        (0.80, r"$v_{\mathrm{step}} := v_x - v_{\mathrm{roll}}$", STEP_COLOR),
        (0.63, r"$v_x^{\mathrm{cmd}} = c_{\mathrm{roll}} + c_{\mathrm{step}}$", BODY_COLOR),
    ):
        ax.text(0.04, y, formula, fontsize=FONT_SIZE, color=color, va="center")
    ax.plot([0.02, 0.98], [0.52, 0.52], color="#CED9E0", linewidth=0.8)
    ax.text(0.02, 0.43, words["examples"], fontsize=FONT_SIZE, va="center", color=style.INK)
    columns = (0.48, 0.70, 0.92)
    headers = (r"$c_{\mathrm{roll}}$", r"$c_{\mathrm{step}}$", r"$v_x^{\mathrm{cmd}}$")
    colors = (ROLL_COLOR, STEP_COLOR, BODY_COLOR)
    for x, header, color in zip(columns, headers, colors):
        ax.text(x, 0.43, header, ha="center", va="center", color=color, fontsize=FONT_SIZE)
    for y, (mode, roll, step) in zip((0.27, 0.10), COMMANDS):
        ax.text(0.02, y, words[mode], fontsize=FONT_SIZE, va="center", color=style.INK)
        for x, value, color in zip(columns, (roll, step, roll + step), colors):
            ax.text(x, y, f"{value:.1f}", ha="center", va="center", fontsize=FONT_SIZE, color=color)


def create_figure(robot, language):
    words = TEXT[language]
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=140)
    fig.text(0.04, 0.93, words["scene_title"], fontsize=FONT_SIZE, color=style.INK)
    fig.text(0.535, 0.93, words["definition_title"], fontsize=FONT_SIZE, color=style.INK)
    scene = fig.add_axes([0.025, 0.16, 0.465, 0.69])
    definition = fig.add_axes([0.53, 0.18, 0.44, 0.66])
    draw_scene(scene, robot, words)
    draw_definitions(definition, words)
    fig.text(0.52, 0.05, words["caution"], ha="center", fontsize=FONT_SIZE, color=style.INK)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            if not np.isclose(text.get_fontsize(), FONT_SIZE):
                raise ValueError("All labels must use the agreed uniform 14 pt font size")
            if not all(fig.bbox.contains(*corner) for corner in text.get_window_extent(renderer).get_points()):
                raise ValueError("Text falls outside the export: " + text.get_text())
    return fig


def write_data(output, definitions, sources, robot, dpi):
    with (output / "command_examples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mode", "rolling_command_mps", "stepping_command_mps", "total_forward_target_mps", "status"])
        for mode, roll, step in COMMANDS:
            writer.writerow([mode, roll, step, roll + step, "command_example_not_achieved_speed"])
    scene = geometry.make_scene(robot, MAIN_POSE)
    radius = definitions["wheel_radius_m"]
    clearances = {side: float(scene.centers[f"wheel_{side}_Joint"][2] - radius) for side in "LR"}
    if abs(clearances["R"]) > 1e-10 or clearances["L"] <= 0:
        raise ValueError("Illustration must show one grounded wheel and one raised wheel")
    manifest = {
        "figure_type": "Task-definition schematic; neither rollout data nor exact contact kinematics",
        "sources": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for name, path in sources.items()},
        "definitions_from_code": definitions,
        "notation": {
            "vx": "Body-frame forward base velocity, base_lin_vel[:,0]",
            "omega_C": "Sign-corrected wheel joint rate averaged over binary contact indicators; mean of both wheels if no contact",
            "vroll": "Task proxy r*omega_C, not independently measured wheel-ground propulsion",
            "vstep": "Residual vx-vroll; its physical interpretation as leg contribution is approximate",
            "croll": "commands[:,0]", "cstep": "cmd_step[:,0]",
            "vx_cmd": "Target croll+cstep, not actual vx",
        },
        "identity_warning": "vx=vroll+vstep holds by the chosen residual definition, not as a universally valid contact-kinematics law",
        "limitations": ["wheel slip", "airborne wheel spin", "no-contact fallback", "body/leg rotations", "direction projection and uneven terrain"],
        "commands": [{"mode": mode, "roll_mps": roll, "step_mps": step, "total_mps": roll + step}
                     for mode, roll, step in COMMANDS],
        "command_provenance": "Mixed 0.5+0.4 matches the archived payload-motion protocol; pure 0.9+0 is constructed for equal-total-target illustration",
        "robot_visuals": "Colored nominal URDF collision primitives, no brand mesh",
        "main_pose_rad": MAIN_POSE, "ghost_pose_rad": GHOST_POSE,
        "wheel_clearance_m": clearances,
        "arrows": {"wheel": "Clockwise 180-degree arc in +x-right/+z-up side view, consistent with URDF +Y wheel axes",
                   "leg": "Schematic forward swing arc, not a reference trajectory or measured velocity",
                   "body": "Forward body-velocity direction, not command achievement"},
        "font_size_pt": FONT_SIZE, "raster_size_px": [round(dpi * x) for x in FIGURE_SIZE],
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Commands: pure 0.9+0.0, mixed 0.5+0.4 m/s; no measured speeds plotted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-source", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    robot = style.CollisionRobot(args.urdf)
    style.validate_robot(robot)
    definitions = load_definitions(args.config, args.env_source, robot)
    corpus = "".join(value for words in TEXT.values() for value in words.values())
    style.configure_fonts(args.output_dir, extra_text=corpus)
    plt.rcParams.update({"font.size": FONT_SIZE, "font.weight": "normal",
                         "axes.labelweight": "normal", "axes.titleweight": "normal"})
    languages = ("zh", "en") if args.language == "both" else (args.language,)
    for language in languages:
        fig = create_figure(robot, language)
        for extension in ("png", "pdf", "svg"):
            target = args.output_dir / f"roll_step_velocity_definition_{language}.{extension}"
            fig.savefig(target, dpi=args.dpi, facecolor="white")
            print(target)
        plt.close(fig)
    write_data(args.output_dir, definitions,
               {"urdf": args.urdf, "configuration": args.config, "proxy_computation": args.env_source}, robot, args.dpi)


if __name__ == "__main__":
    main()
