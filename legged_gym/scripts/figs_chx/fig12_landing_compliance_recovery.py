#!/usr/bin/env python3
"""Landing phases and conceptual axial compliance, without invented responses."""

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
from matplotlib.text import Text
import numpy as np

import fig0_sixd_design_variables as style
import fig03_four_capability_definitions as geometry
import fig10_jump_goal_takeoff as jump


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "out/landing_compliance_recovery"
FIGURE_SIZE = (14.4, 7.5)
FONT_SIZE = 14
CONTACT_COLOR = "#B34959"
SPRING_COLOR = "#B38A31"
DAMPER_COLOR = "#238B7B"
RECOVERY_COLOR = "#487FB7"
REFERENCE_URL = "https://manipulation.mit.edu/force.html"
TEXT = {
    "zh": {
        "a": "(a)  着陆与恢复的三个环节",
        "b": "(b)  腿向等效阻抗示意",
        "phases": ["首次接触", "压缩缓冲", "恢复站高"],
        "roles": ["接触建立", "腿部缩短、减缓下落", "高度与姿态恢复"],
        "descent": "下落方向", "spring": "虚拟弹簧：弹性恢复", "damper": "虚拟阻尼：阻碍相对运动",
        "height": "站高参考",
        "footer": "机制示意，非实测响应；正式策略采用关节 PD 与学习动作，未启用显式腿向阻抗律",
    },
    "en": {
        "a": "(a)  Contact, compression and recovery",
        "b": "(b)  Conceptual axial compliance",
        "phases": ["First contact", "Compression", "Height recovery"],
        "roles": ["Establish contact", "Shorten legs; slow descent", "Restore height and posture"],
        "descent": "Descent", "spring": "Virtual spring: restore", "damper": "Virtual damper: oppose motion",
        "height": "Standing reference",
        "footer": "Mechanism only, not measured response; archived control is joint PD + learned actions, not an explicit axial law",
    },
}


def make_grounded_scene(robot, knee, pitch_deg=0.0):
    hip = 0.5 * knee
    angles = {"hip_L_Joint": hip, "knee_L_Joint": knee,
              "hip_R_Joint": -hip, "knee_R_Joint": -knee}
    for name, angle in angles.items():
        limits = robot.joints[name].find("limit")
        if not float(limits.get("lower")) <= angle <= float(limits.get("upper")):
            raise ValueError("Illustrative pose outside limits: " + name)
    scene = geometry.make_scene(robot, angles)
    jump.rotate_pitch(scene, pitch_deg)
    wheel_mid = np.mean([scene.centers[f"wheel_{side}_Joint"] for side in "LR"], axis=0)
    bottom = min(jump.wheel_bottom(scene, side)[2] for side in "LR")
    geometry.translate_scene(scene, [-wheel_mid[0], 0, -bottom])
    return scene


def make_phases(robot, target_height):
    lower, upper = 0.0, 1.30
    for _ in range(45):
        knee = (lower + upper) / 2
        height = make_grounded_scene(robot, knee).frames["base_Link"][2, 3]
        if height > target_height:
            lower = knee
        else:
            upper = knee
    standing_knee = (lower + upper) / 2
    scenes = [make_grounded_scene(robot, 0.10, 8.0),
              make_grounded_scene(robot, 1.25, 4.0),
              make_grounded_scene(robot, standing_knee)]
    records = []
    for label, scene, knee, pitch in zip(("first_contact", "compression", "recovery"), scenes,
                                        (0.10, 1.25, standing_knee), (8.0, 4.0, 0.0)):
        hip = scene.centers["hip_L_Joint"]
        wheel = scene.centers["wheel_L_Joint"]
        base = scene.frames["base_Link"][:3, 3]
        records.append({"phase": label, "hip_angle_rad": knee / 2, "knee_angle_rad": knee,
                        "illustrative_body_pitch_deg": pitch,
                        "hip_to_wheel_length_m": float(np.linalg.norm((hip - wheel)[[0, 2]])),
                        "base_to_wheel_sagittal_length_m": float(np.linalg.norm((base - wheel)[[0, 2]])),
                        "base_origin_height_m": float(base[2]),
                        "left_wheel_clearance_m": float(jump.wheel_bottom(scene, "L")[2]),
                        "right_wheel_clearance_m": float(jump.wheel_bottom(scene, "R")[2]),
                        "status": "illustrative_target_geometry_not_measured_response"})
    lengths = [row["hip_to_wheel_length_m"] for row in records]
    heights = [row["base_origin_height_m"] for row in records]
    if not (lengths[1] < lengths[2] < lengths[0] and heights[1] < heights[2] < heights[0]):
        raise ValueError("Compressed geometry must be lower/shorter than touchdown and recovery")
    if abs(heights[2] - target_height) > 1e-10:
        raise ValueError("The illustrated standing reference must match the nominal archived target")
    if max(abs(row[key]) for row in records for key in ("left_wheel_clearance_m", "right_wheel_clearance_m")) > 1e-9:
        raise ValueError("Every illustrated grounded phase must keep both wheels on the ground")
    return scenes, records


def draw_phase(ax, scene, phase_index, words, target_height):
    projection = style.Projection([0, -1, 0])
    ax.fill_between([-0.55, 0.55], -0.025, 0, color="#E6ECEF", zorder=0)
    ax.plot([-0.55, 0.55], [0, 0], color="#93A3AE", linewidth=0.9)
    geometry.draw_robot(ax, scene, projection, side="L")
    hip, wheel = projection.xy([scene.centers["hip_L_Joint"], scene.centers["wheel_L_Joint"]])
    axis = hip - wheel
    normal = np.array([-axis[1], axis[0]]) / np.linalg.norm(axis)
    offset = 0.13 * normal
    ax.plot([wheel[0], hip[0]], [wheel[1], hip[1]], color="#8B9BA4",
            linewidth=0.8, linestyle=(0, (3, 3)), zorder=5)
    jump.arrow(ax, wheel + offset, hip + offset, DAMPER_COLOR, double=True)
    midpoint = (hip + wheel) / 2 + 1.85 * offset
    ax.text(*midpoint, r"$L$", fontsize=FONT_SIZE, color=DAMPER_COLOR, va="center", ha="right")
    contact = projection.xy(jump.wheel_bottom(scene, "L"))
    ax.scatter(*contact, color=CONTACT_COLOR, s=25, zorder=8)
    if phase_index == 0:
        jump.arrow(ax, (-0.42, 0.92), (-0.42, 0.68), CONTACT_COLOR)
        ax.text(-0.42, 1.00, words["descent"], fontsize=FONT_SIZE, color=CONTACT_COLOR, ha="center")
        jump.arrow(ax, contact, contact + [0, 0.24], CONTACT_COLOR)
        ax.text(contact[0] + 0.14, 0.16, r"$F_n$", fontsize=FONT_SIZE, color=CONTACT_COLOR)
    if phase_index == 2:
        base = projection.xy(scene.frames["base_Link"][:3, 3])
        ax.scatter(*base, color=RECOVERY_COLOR, s=18, zorder=8)
        ax.plot([base[0], 0.40], [target_height, target_height], color=RECOVERY_COLOR,
                linewidth=0.9, linestyle=(0, (4, 3)))
        jump.arrow(ax, (0.37, 0), (0.37, target_height), RECOVERY_COLOR, double=True)
        ax.text(0.39, target_height / 2, r"$z_B^\star$", fontsize=FONT_SIZE,
                color=RECOVERY_COLOR, va="center")
        ax.text(0.03, target_height + 0.12, words["height"], fontsize=FONT_SIZE,
                color=RECOVERY_COLOR, ha="center")
    ax.set(xlim=(-0.62, 0.62), ylim=(-0.10, 1.17), aspect="equal")
    ax.axis("off")


def draw_axial_compliance(ax, words):
    wheel = np.array([0.10, 0.17])
    unit = np.array([np.sin(np.deg2rad(18)), np.cos(np.deg2rad(18))])
    normal = np.array([-unit[1], unit[0]])
    length = 0.88
    hip = wheel + length * unit

    def point(fraction, transverse=0.0):
        return wheel + fraction * length * unit + transverse * normal

    def line(points, color, linewidth=1.3, **kwargs):
        data = np.asarray(points)
        ax.plot(data[:, 0], data[:, 1], color=color, linewidth=linewidth, **kwargs)

    line([wheel, hip], "#A4B0B7", 0.8, linestyle=(0, (3, 3)))
    line([point(0, -0.13), point(0, 0.13)], style.MUTED)
    line([point(1, -0.13), point(1, 0.13)], style.MUTED)
    for transverse, color in ((0.11, SPRING_COLOR), (-0.11, DAMPER_COLOR)):
        line([point(0, transverse), point(0.17, transverse)], color)
        line([point(0.82, transverse), point(1, transverse)], color)
    fractions = np.linspace(0.18, 0.82, 13)
    zigzag = [point(0.17, 0.11)]
    zigzag.extend(point(value, 0.11 + (0.045 if i % 2 == 0 else -0.045)) for i, value in enumerate(fractions))
    zigzag.append(point(0.82, 0.11))
    line(zigzag, SPRING_COLOR, 1.55)
    line([point(0.17, -0.11), point(0.34, -0.11)], DAMPER_COLOR, 1.5)
    line([point(0.72, -0.165), point(0.34, -0.165), point(0.34, -0.055), point(0.72, -0.055)], DAMPER_COLOR, 1.5)
    line([point(0.61, -0.16), point(0.61, -0.06)], DAMPER_COLOR, 1.5)
    line([point(0.61, -0.11), point(0.82, -0.11)], DAMPER_COLOR, 1.5)
    for anchor, label in ((hip, r"$H$"), (wheel, r"$W$")):
        ax.scatter(*anchor, s=24, color=style.INK, zorder=5)
        offset = [0.12, 0.035] if label == r"$H$" else [-0.025, -0.09]
        ax.text(*(anchor + offset), label, color=style.INK, fontsize=FONT_SIZE)
    jump.arrow(ax, point(0, 0.40), point(1, 0.40), RECOVERY_COLOR, double=True)
    ax.text(*point(0.48, 0.50), r"$L$", fontsize=FONT_SIZE, color=RECOVERY_COLOR, ha="center")
    ax.text(*point(0.56, 0.23), r"$K_L$", fontsize=FONT_SIZE, color=SPRING_COLOR, ha="right")
    ax.text(*point(0.50, -0.28), r"$D_L$", fontsize=FONT_SIZE, color=DAMPER_COLOR, ha="left")
    ax.set(xlim=(-0.45, 0.78), ylim=(0.0, 1.18), aspect="equal")
    ax.axis("off")


def create_figure(scenes, cfg, language):
    words = TEXT[language]
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=140)
    fig.text(0.035, 0.94, words["a"], fontsize=FONT_SIZE, color=style.INK)
    fig.text(0.755, 0.94, words["b"], fontsize=FONT_SIZE, color=style.INK)
    centers = (0.14, 0.37, 0.60)
    colors = (CONTACT_COLOR, DAMPER_COLOR, RECOVERY_COLOR)
    for i, (center, scene) in enumerate(zip(centers, scenes)):
        fig.text(center, 0.855, words["phases"][i], fontsize=FONT_SIZE, color=colors[i], ha="center")
        draw_phase(fig.add_axes([center - 0.11, 0.30, 0.22, 0.50]), scene, i,
                   words, cfg["rewards"]["base_height_target"])
        fig.text(center, 0.225, words["roles"][i], fontsize=FONT_SIZE, color=style.INK, ha="center")
        if i < 2:
            ax = fig.add_axes([center + 0.105, 0.51, 0.035, 0.05])
            jump.arrow(ax, (0.1, 0.5), (0.9, 0.5), style.MUTED)
            ax.set(xlim=(0, 1), ylim=(0, 1))
            ax.axis("off")
    draw_axial_compliance(fig.add_axes([0.752, 0.34, 0.235, 0.45]), words)
    fig.text(0.87, 0.265, words["spring"], fontsize=FONT_SIZE, color=SPRING_COLOR, ha="center")
    fig.text(0.87, 0.205, words["damper"], fontsize=FONT_SIZE, color=DAMPER_COLOR, ha="center")
    fig.text(0.5, 0.055, words["footer"], fontsize=FONT_SIZE, color=style.MUTED, ha="center")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [item for item in fig.findobj(Text) if item.get_visible() and item.get_text()]
    boxes = [item.get_window_extent(renderer) for item in texts]
    for item, box in zip(texts, boxes):
        if not np.isclose(item.get_fontsize(), FONT_SIZE):
            raise ValueError("All text must use 14 pt")
        if not all(fig.bbox.contains(*corner) for corner in box.get_points()):
            raise ValueError("Text outside page: " + item.get_text())
    for i, box in enumerate(boxes):
        for j, other in enumerate(boxes[:i]):
            if box.padded(5).overlaps(other):
                raise ValueError("Overlapping text: " + texts[i].get_text() + " / " + texts[j].get_text())
    return fig, {"text_count": len(texts), "font_size_pt": FONT_SIZE,
                 "all_text_within_export": True, "text_overlap_pairs": 0}


def validate_archived_control(cfg, source):
    import torch

    tree = ast.parse(source.read_text())
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "BipedWF")
    names = {"_compute_torques", "_reward_jump_stand", "_reward_jump_landing_compress"}
    methods = {item.name: item for item in cls.body if isinstance(item, ast.FunctionDef)}
    cls.bases = []
    cls.body = [methods[name] for name in sorted(names)]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), scope)

    def ns(value):
        return SimpleNamespace(**{key: ns(item) if isinstance(item, dict) else item
                                  for key, item in value.items()})

    dummy = scope["BipedWF"]()
    dummy.num_envs = 3
    dummy.cfg = ns(cfg)
    dummy.jump_state = torch.tensor([2, 3, 0])
    dummy.jumped_flag = torch.ones(3, dtype=torch.bool)
    dummy.jump_height_cmd = torch.full((3, 1), 0.13)
    dummy.jump_dist_cmd = torch.full((3, 1), 0.10)
    dummy._jump_phase_elapsed = lambda: torch.zeros(3)
    dummy.dof_pos = torch.tensor([[0.04, 0.20, -0.10, 0, -0.04, -0.20, 0.10, 0.0]]).repeat(3, 1)
    dummy.dof_vel = torch.full((3, 8), 0.10)
    dummy.default_dof_pos = torch.zeros(3, 8)
    dofs = [f"{name}_{side}_Joint" for side in "LR" for name in ("abad", "hip", "knee", "wheel")]
    dummy.p_gains = torch.tensor([cfg["control"]["stiffness"][name] for name in dofs])
    dummy.d_gains = torch.tensor([cfg["control"]["damping"][name] for name in dofs])
    dummy._motor_available_torque_limits = lambda: torch.full((3, 8), 80.0)
    dummy.torques_scale = torch.ones(3, 8)
    dummy._apply_jump_explicit_trajectory = lambda action: (_ for _ in ()).throw(AssertionError("Disabled trajectory called"))
    actual = dummy._compute_torques(torch.zeros(3, 8))
    expected = -dummy.p_gains * dummy.dof_pos - dummy.d_gains * dummy.dof_vel
    torch.testing.assert_close(actual, expected)
    dummy.contact_forces = torch.zeros(3, 2, 3)
    dummy.contact_forces[:, :, 2] = 20
    dummy.feet_indices = [0, 1]
    dummy.base_lin_vel = torch.zeros(3, 3)
    dummy.dof_vel = torch.zeros(3, 8)
    dummy.base_height = torch.full((3,), cfg["rewards"]["base_height_target"])
    dummy._base_height_target_tensor = lambda: dummy.base_height.clone()
    dummy.projected_gravity = torch.tensor([[0, 0, -1.0]]).repeat(3, 1)
    torch.testing.assert_close(dummy._reward_jump_stand(), torch.tensor([0.0, 0.0, 1.0]))
    dummy.jump_preland_leg_len = torch.full((3,), 0.70)
    dummy.jump_land_leg_len = torch.full((3,), 0.60)
    dummy.jump_land_fire = torch.tensor([True, False, True])
    dummy.jump_takeoff_quality = torch.tensor([1.0, 1.0, 0.0])
    torch.testing.assert_close(dummy._reward_jump_landing_compress(), torch.tensor([1.0, 0.0, 0.0]))
    if cfg["control"]["jump_use_explicit_trajectory"] or cfg["rewards"]["scales"].get("jump_landed_recover", 0) != 0:
        raise ValueError("This figure documents the archived trajectory-disabled control path")
    return {"source_methods_checked": sorted(names),
            "nominal_joint_PD_in_flight_landed_idle": "pass",
            "stand_reward_is_idle_not_landed": "pass", "compression_event_quality_gate": "pass",
            "explicit_trajectory_called": False,
            "scope": "Isolated source functions and schematic geometry; not a landing response or impact-force test"}


def audit_available_logs(args):
    if not args.inspect_log_mat and not args.inspect_event_csv and args.formal_archive is None:
        archived = HERE / "log_availability_audit.json"
        return json.loads(archived.read_text()) if archived.is_file() else {"scope": "No logs supplied", "response_curves_plotted": False}
    evidence = {"scope": "Selected local formal archive and legacy jump artifacts, not a claim about unexamined remote files",
                "response_curves_plotted": False, "response_curve_decision": "Mechanism-only; no verified synchronized landing trace for all requested quantities", "inspected": []}
    if args.formal_archive:
        if not args.formal_archive.is_dir():
            raise FileNotFoundError(args.formal_archive)
        traces = sorted(str(path.relative_to(args.formal_archive)) for path in args.formal_archive.rglob("*")
                        if path.suffix.lower() in (".npz", ".mat"))
        evidence["inspected"].append({"path": str(args.formal_archive), "type": "formal_archive",
                                      "npz_mat_trace_files_found": traces})
    if args.inspect_log_mat:
        from scipy.io import whosmat

        for path in args.inspect_log_mat:
            variables = [{"name": name, "shape": shape, "type": dtype} for name, shape, dtype in whosmat(path)]
            names = {item["name"] for item in variables}
            aliases = {"time": {"t", "time"}, "leg_length": {"leg_length", "leg_len", "effective_leg_length"},
                       "vertical_velocity": {"base_vz", "com_vz", "root_vz"},
                       "pitch_or_reconstructible_orientation": {"pitch", "base_pitch", "body_pitch", "base_quat", "base_euler"}}
            missing = [label for label, options in aliases.items() if not names.intersection(options)]
            evidence["inspected"].append({"path": str(path.resolve()), "type": "legacy_mat",
                                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                          "variables": variables, "missing_requested_trace_channels": missing,
                                          "use_for_figure": False,
                                          "reason": "Legacy policy, not a verified final-policy landing trace; missing channels: " + ", ".join(missing)})
    for path in args.inspect_event_csv:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            header = next(csv.reader(stream))
        evidence["inspected"].append({"path": str(path.resolve()), "type": "per_jump_event_summary_csv",
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "fields": header,
                                      "use_for_figure": False, "reason": "Event summaries are not synchronized within-landing response samples"})
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=jump.DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=jump.DEFAULT_CONFIG)
    parser.add_argument("--env-source", type=Path, default=jump.DEFAULT_ENV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formal-archive", type=Path)
    parser.add_argument("--inspect-log-mat", type=Path, action="append", default=[])
    parser.add_argument("--inspect-event-csv", type=Path, action="append", default=[])
    args = parser.parse_args()
    cfg = jump.load_policy(args.config)
    report = validate_archived_control(cfg, args.env_source)
    robot = style.CollisionRobot(args.urdf)
    style.validate_robot(robot)
    scenes, records = make_phases(robot, cfg["rewards"]["base_height_target"])
    style.configure_fonts(args.output_dir, extra_text="".join(str(words) for words in TEXT.values()))
    plt.rcParams.update({"font.size": FONT_SIZE, "font.weight": "normal"})
    layout = {}
    for language in ("zh", "en") if args.language == "both" else (args.language,):
        fig, layout[language] = create_figure(scenes, cfg, language)
        for extension in ("png", "pdf", "svg"):
            target = args.output_dir / f"landing_compliance_recovery_{language}.{extension}"
            fig.savefig(target, dpi=args.dpi, facecolor="white")
            print(target)
        plt.close(fig)
    with (args.output_dir / "illustrative_phase_geometry.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    log_audit = audit_available_logs(args)
    (args.output_dir / "log_availability_audit.json").write_text(json.dumps(log_audit, indent=2) + "\n")
    report.update({"geometry_order": "contact > recovery > compressed in height and hip-wheel length",
                   "all_poses_within_joint_limits_and_grounded": True, "layout": layout,
                   "response_curves_plotted": False})
    (args.output_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {"figure_type": "Control-mechanism and target-geometry schematic, not measured response",
                "sources": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                            for name, path in {"nominal_urdf": args.urdf, "archived_config": args.config, "archived_source": args.env_source}.items()},
                "primary_theory_reference": REFERENCE_URL, "phase_geometry": records,
                "spring_damper_semantics": "Conceptual parallel virtual spring/damper along sagittal hip H to wheel W; not a physical spring or deployed axial controller",
                "L_definition": "Sagittal hip-center H to wheel-center W distance, distinct from the archived base-origin-to-wheel reward proxy",
                "K_L_D_L_numeric_values_assigned": False, "explicit_axial_impedance_law_deployed": False,
                "joint_PD_gains_are_switched_at_contact": False,
                "reference_height_m": cfg["rewards"]["base_height_target"],
                "recovered_target_pose_is_actual_policy_output": False,
                "response_curves_plotted": False, "synthetic_response_samples_generated": False,
                "position_velocity_pitch_time_curves_interpolated_from_event_summaries": False,
                "geometry_csv_is_time_series": False, "font_size_pt": FONT_SIZE,
                "raster_size_px": [round(value * args.dpi) for value in FIGURE_SIZE]}
    (args.output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
