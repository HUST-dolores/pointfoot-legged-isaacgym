#!/usr/bin/env python3
"""Planar airborne attitude exchange and fixed-CoM wheel-clearance comparison."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as DrawingPath
from matplotlib.text import Text
import numpy as np

import fig0_sixd_design_variables as style
import fig03_four_capability_definitions as geometry
import fig10_jump_goal_takeoff as jump


HERE = Path(__file__).resolve().parent
DEFAULT_URDF = (HERE / "nominal_source.urdf" if (HERE / "nominal_source.urdf").is_file()
                else HERE.parents[2] / "resources/robots/WF_TRON1A/urdf/robot.urdf")
DEFAULT_OUTPUT = HERE / "out/airborne_momentum_exchange"
FIGURE_SIZE = (14.4, 7.2)
FONT_SIZE = 14
BODY_COLOR = "#487FB7"
LEG_COLOR = "#238B7B"
HIP_COLOR = "#B38A31"
COM_COLOR = "#B34959"
Q0, Q1 = -0.35, 0.65
FIXED_KNEE = 0.50
COM_HEIGHT = 1.02
DURATION = 0.35
REFERENCE_URL = "https://underactuated.csail.mit.edu/humanoids.html"
TEXT = {
    "zh": {
        "a": "(a)  腾空调姿：机体—腿部反向转动",
        "b": "(b)  收腿对比：相同质心状态",
        "before": "调姿前", "after": "调姿后",
        "extended": "伸腿构型", "tucked": "收腿构型",
        "body": "机体转动", "legs": "腿部反向转动",
        "hip": "髋关节相对转动",
        "com": "整机质心 C", "same_height": "质心高度相同",
        "exchange": "角动量交换改变机体姿态",
        "clearance": "收腿可增加轮端净空，不增加质心跳高",
        "footer": "矢状面模型示意：无地面接触、仅受重力，忽略空气作用；非策略实测轨迹",
    },
    "en": {
        "a": "(a)  Airborne attitude: opposite body / leg rotations",
        "b": "(b)  Tucking at the same CoM state",
        "before": "Before adjustment", "after": "After adjustment",
        "extended": "Extended legs", "tucked": "Tucked legs",
        "body": "Body rotation", "legs": "Opposite leg rotation",
        "hip": "Relative hip rotation",
        "com": "Whole-robot CoM C", "same_height": "Same CoM height",
        "exchange": "Angular-momentum exchange changes attitude",
        "clearance": "Tucking can raise wheels, not the CoM apex",
        "footer": "Planar model: no contact, gravity only, negligible air effects; not a measured policy trajectory",
    },
}


class PlanarCentroidalModel:
    """Use nominal inertial positions and Iyy in the sagittal projection."""

    def __init__(self, urdf):
        self.robot = style.CollisionRobot(urdf)
        style.validate_robot(self.robot)
        self.inertials = []
        root = ET.parse(urdf).getroot()
        for link in root.findall("link"):
            node = link.find("inertial")
            if node is None or float(node.find("mass").get("value")) <= 0:
                continue
            inertia = node.find("inertia")
            xx, xy, xz, yy, yz, zz = (float(inertia.get(key))
                                      for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"))
            tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            if np.linalg.eigvalsh(tensor).min() <= 0:
                raise ValueError("The illustrative model requires positive link inertias")
            self.inertials.append({"link": link.get("name"), "mass": float(node.find("mass").get("value")),
                                   "origin": style.origin_transform(node.find("origin")), "inertia": tensor})
        self.mass = sum(item["mass"] for item in self.inertials)
        self.leg_owner = {}
        for side in "LR":
            descendants = {f"hip_{side}_Link"}
            changed = True
            while changed:
                children = {joint.find("child").get("link") for joint in self.robot.joints.values()
                            if joint.find("parent").get("link") in descendants}
                changed = not children <= descendants
                descendants.update(children)
            self.leg_owner.update({link: side for link in descendants})
        self.com_inertials = jump.load_inertials(urdf)

    def scene(self, hip, knee=FIXED_KNEE, pitch=0.0, com=(0.0, 0.0, COM_HEIGHT)):
        angles = {"hip_L_Joint": hip, "hip_R_Joint": -hip,
                  "knee_L_Joint": knee, "knee_R_Joint": -knee}
        for name, angle in angles.items():
            limits = self.robot.joints[name].find("limit")
            if not float(limits.get("lower")) <= angle <= float(limits.get("upper")):
                raise ValueError("Illustrative configuration outside limits: " + name)
        scene = geometry.make_scene(self.robot, angles)
        jump.rotate_pitch(scene, np.rad2deg(pitch))
        geometry.translate_scene(scene, np.asarray(com) - jump.center_of_mass(scene, self.com_inertials))
        return scene

    def link_states(self, scene, theta_dot, hip_dot):
        origin = scene.frames["base_Link"][:3, 3]
        states = []
        for item in self.inertials:
            frame = scene.frames[item["link"]] @ item["origin"]
            point = frame[:3, 3]
            velocity = np.cross([0.0, theta_dot, 0.0], point - origin)
            omega_y = theta_dot
            side = self.leg_owner.get(item["link"])
            if side:
                name = f"hip_{side}_Joint"
                axis = (scene.frames[f"hip_{side}_Link"][:3, :3]
                        @ style.values(self.robot.joints[name].find("axis"), "xyz"))
                axis *= hip_dot if side == "L" else -hip_dot
                if not np.allclose(axis[[0, 2]], 0, atol=1e-9):
                    raise ValueError("This one-shape illustration assumes sagittal hip axes")
                velocity += np.cross(axis, point - scene.centers[name])
                omega_y += axis[1]
            tensor = frame[:3, :3] @ item["inertia"] @ frame[:3, :3].T
            states.append({"link": item["link"], "mass": item["mass"], "point": point,
                           "velocity": velocity, "omega_y": omega_y, "Iyy": tensor[1, 1],
                           "group": "legs" if side else "body"})
        center = sum(item["mass"] * item["point"] for item in states) / self.mass
        center_velocity = sum(item["mass"] * item["velocity"] for item in states) / self.mass
        for item in states:
            item["relative_point"] = item["point"] - center
            item["relative_velocity"] = item["velocity"] - center_velocity
        return states

    def momentum(self, scene, theta_dot, hip_dot):
        groups = {"body": 0.0, "legs": 0.0}
        for item in self.link_states(scene, theta_dot, hip_dot):
            orbital = item["mass"] * np.cross(item["relative_point"], item["relative_velocity"])[1]
            groups[item["group"]] += item["Iyy"] * item["omega_y"] + orbital
        return groups

    def coupling(self, hip):
        scene = self.scene(hip)
        A = sum(self.momentum(scene, 1.0, 0.0).values())
        B = sum(self.momentum(scene, 0.0, 1.0).values())
        if A <= 0:
            raise ValueError("Locked whole-robot inertia must be positive")
        return A, B


def reconstruct(model, samples=601):
    q = np.linspace(Q0, Q1, samples)
    coupling = np.array([model.coupling(value) for value in q])
    ratio = -coupling[:, 1] / coupling[:, 0]
    if not np.all((-1 < ratio) & (ratio < 0)):
        raise ValueError("Selected case must show opposite body/leg absolute rotations")
    increment = np.r_[0.0, np.cumsum((ratio[:-1] + ratio[1:]) * np.diff(q) / 2)]
    theta = increment - increment[-1]
    time = np.linspace(0, DURATION, samples)
    unit = time / DURATION
    ease = 3 * unit ** 2 - 2 * unit ** 3
    q_time = Q0 + (Q1 - Q0) * ease
    q_dot = (Q1 - Q0) * (6 * unit - 6 * unit ** 2) / DURATION
    theta_time = np.interp(q_time, q, theta)
    theta_dot = np.interp(q_time, q, ratio) * q_dot
    return {"q_grid": q, "theta_grid": theta, "coupling": coupling, "ratio": ratio,
            "time_s": time, "hip_rad": q_time, "hip_dot_radps": q_dot,
            "body_pitch_rad": theta_time, "body_pitch_dot_radps": theta_dot,
            "leg_pitch_dot_radps": theta_dot + q_dot}


def semicircle(ax, center, radius, start_deg, direction, color):
    angles = np.deg2rad(np.linspace(start_deg, start_deg + direction * 180, 101))
    points = np.asarray(center) + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    ax.add_patch(FancyArrowPatch(path=DrawingPath(points), arrowstyle="-|>",
                                 mutation_scale=12, color=color, linewidth=1.5, zorder=7))


def draw_relative_hip(ax, model, scene, q, pitch):
    hip = scene.centers["hip_L_Joint"][[0, 2]]
    zero = model.scene(0.0, pitch=pitch)
    reference = (zero.centers["knee_L_Joint"] - zero.centers["hip_L_Joint"])[[0, 2]]
    reference /= np.linalg.norm(reference)
    current = (scene.centers["knee_L_Joint"] - scene.centers["hip_L_Joint"])[[0, 2]]
    current /= np.linalg.norm(current)
    angle = np.arctan2(reference[1], reference[0])
    ax.plot([hip[0], hip[0] + 0.31 * reference[0]],
            [hip[1], hip[1] + 0.31 * reference[1]], color=HIP_COLOR,
            linewidth=0.85, linestyle=(0, (3, 3)), zorder=6)
    theta = np.linspace(angle, angle - q, 60)
    points = hip + 0.17 * np.column_stack([np.cos(theta), np.sin(theta)])
    if not np.allclose(points[-1] - hip, 0.17 * current, atol=1e-8):
        raise ValueError("The hip arc must measure relative joint angle, not world attitude")
    ax.add_patch(FancyArrowPatch(path=DrawingPath(points), arrowstyle="-|>", color=HIP_COLOR,
                                 linewidth=1.3, mutation_scale=9, zorder=9))
    ax.scatter(*hip, color=HIP_COLOR, s=17, zorder=8)
    ax.text(hip[0] + 0.26, hip[1] - 0.055, r"$q_h$", fontsize=FONT_SIZE, color=HIP_COLOR,
            va="center")


def create_figure(model, motion, language):
    words = TEXT[language]
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=140)
    fig.text(0.035, 0.94, words["a"], fontsize=FONT_SIZE, color=style.INK)
    fig.text(0.565, 0.94, words["b"], fontsize=FONT_SIZE, color=style.INK)
    projection = style.Projection([0, -1, 0])
    before = model.scene(Q0, pitch=motion["theta_grid"][0])
    after = model.scene(Q1, pitch=motion["theta_grid"][-1])
    ax = fig.add_axes([0.025, 0.20, 0.52, 0.65])
    for scene, offset, label in ((before, -0.60, "before"), (after, 0.60, "after")):
        geometry.translate_scene(scene, [offset, 0, 0])
        geometry.draw_robot(ax, scene, projection, side="L")
        com = jump.center_of_mass(scene, model.com_inertials)[[0, 2]]
        ax.scatter(*com, color=COM_COLOR, s=24, zorder=9)
        if label == "before":
            ax.text(com[0] - 0.10, com[1] - 0.05, r"$C$", fontsize=FONT_SIZE, color=COM_COLOR,
                    ha="right", va="center")
        ax.text(offset, 1.71, words[label], fontsize=FONT_SIZE, ha="center", color=style.INK)
    jump.arrow(ax, (-0.10, 1.19), (0.15, 1.19), style.MUTED)
    body_center = next(c for c in after.collisions if c.link == "base_Link").transform[:3, 3][[0, 2]]
    semicircle(ax, body_center, 0.23, 20, 1, BODY_COLOR)
    hip = after.centers["hip_L_Joint"][[0, 2]]
    semicircle(ax, hip, 0.58, 60, -1, LEG_COLOR)
    ax.text(0.27, 1.56, words["body"], fontsize=FONT_SIZE, color=BODY_COLOR,
            va="center", ha="center")
    ax.text(0.98, 0.43, words["legs"], fontsize=FONT_SIZE, color=LEG_COLOR, va="center", ha="right")
    draw_relative_hip(ax, model, after, Q1, 0.0)
    ax.text(0.60, 0.25, words["hip"], fontsize=FONT_SIZE, color=HIP_COLOR, ha="center")
    ax.set(xlim=(-1.12, 1.42), ylim=(0.23, 1.86), aspect="equal")
    ax.axis("off")
    fig.text(0.285, 0.155, words["exchange"], fontsize=FONT_SIZE, color=style.INK, ha="center")

    extended = model.scene(0.25, knee=0.10)
    tucked = model.scene(0.425, knee=1.15)
    ax2 = fig.add_axes([0.56, 0.20, 0.42, 0.65])
    ax2.fill_between([-0.55, 1.45], -0.025, 0, color="#E6ECEF")
    ax2.plot([-0.55, 1.45], [0, 0], color="#93A3AE", linewidth=0.85)
    ax2.plot([-0.52, 1.43], [COM_HEIGHT, COM_HEIGHT], color=COM_COLOR,
             linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
    for i, (scene, offset, label) in enumerate(((extended, 0.0, "extended"), (tucked, 1.0, "tucked"))):
        geometry.translate_scene(scene, [offset, 0, 0])
        geometry.draw_robot(ax2, scene, projection, side="L")
        c = jump.center_of_mass(scene, model.com_inertials)[[0, 2]]
        ax2.scatter(*c, color=COM_COLOR, s=24, zorder=9)
        ax2.text(c[0] - 0.075, c[1] + 0.05, r"$C$", fontsize=FONT_SIZE, color=COM_COLOR, ha="right")
        bottom = min((jump.wheel_bottom(scene, side) for side in "LR"), key=lambda p: p[2])[[0, 2]]
        x_dim = offset + 0.35
        ax2.plot([bottom[0], x_dim], [bottom[1], bottom[1]], color=LEG_COLOR, linewidth=0.85)
        ax2.scatter(*bottom, color=LEG_COLOR, s=16, zorder=9)
        jump.arrow(ax2, (x_dim, 0), (x_dim, bottom[1]), LEG_COLOR, double=True)
        ax2.text(x_dim + 0.04, bottom[1] / 2, rf"$h_{{w,{i + 1}}}$", color=LEG_COLOR,
                 fontsize=FONT_SIZE, va="center")
        ax2.text(offset, 1.71, words[label], fontsize=FONT_SIZE, color=style.INK, ha="center")
    ax2.text(0.50, 1.48, words["same_height"], color=COM_COLOR, fontsize=FONT_SIZE, ha="center")
    ax2.set(xlim=(-0.60, 1.58), ylim=(-0.08, 1.86), aspect="equal")
    ax2.axis("off")
    fig.text(0.77, 0.155, words["clearance"], fontsize=FONT_SIZE, ha="center", color=style.INK)
    fig.text(0.5, 0.055, words["footer"], fontsize=FONT_SIZE, ha="center", color=style.MUTED)
    return fig, {"before": before, "after": after, "extended": extended, "tucked": tucked}


def validate_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [text for text in fig.findobj(Text) if text.get_visible() and text.get_text()]
    boxes = [text.get_window_extent(renderer) for text in texts]
    for text, box in zip(texts, boxes):
        if not np.isclose(text.get_fontsize(), FONT_SIZE):
            raise ValueError("All labels must use the same 14 pt size")
        if not all(fig.bbox.contains(*corner) for corner in box.get_points()):
            raise ValueError("Text outside page: " + text.get_text())
    for i, box in enumerate(boxes):
        for j, other in enumerate(boxes[:i]):
            if box.padded(5).overlaps(other):
                raise ValueError("Text overlap: " + texts[i].get_text() + " / " + texts[j].get_text())
    return {"font_size_pt": FONT_SIZE, "text_count": len(texts), "text_overlap_pairs": 0,
            "all_text_within_export": True}


def validate_physics(model, motion):
    residual = []
    group_samples = []
    velocity_error = []
    eps = 1e-6
    for i in np.linspace(1, len(motion["time_s"]) - 2, 11, dtype=int):
        q = motion["hip_rad"][i]
        theta = motion["body_pitch_rad"][i]
        q_dot = motion["hip_dot_radps"][i]
        # Exact coupling at each test pose, independent of interpolation error.
        A, B = model.coupling(q)
        theta_dot = -B / A * q_dot
        scene = model.scene(q, pitch=theta)
        groups = model.momentum(scene, theta_dot, q_dot)
        residual.append(abs(sum(groups.values())))
        group_samples.append(groups)
        if not (groups["body"] < 0 < groups["legs"]):
            raise ValueError("Groups must exchange opposite centroidal angular momentum")
        low = model.scene(q - eps * q_dot, pitch=theta - eps * theta_dot)
        high = model.scene(q + eps * q_dot, pitch=theta + eps * theta_dot)
        states = model.link_states(scene, theta_dot, q_dot)
        for item in states:
            origin = next(node["origin"] for node in model.inertials if node["link"] == item["link"])
            finite_difference = ((high.frames[item["link"]] @ origin)[:3, 3]
                                 - (low.frames[item["link"]] @ origin)[:3, 3]) / (2 * eps)
            velocity_error.append(float(np.max(np.abs(finite_difference - item["relative_velocity"]))))
    if max(residual) > 1e-9 or max(velocity_error) > 1e-7:
        raise ValueError("Momentum or analytic velocity validation failed")
    q0_scene = model.scene(Q0, pitch=motion["theta_grid"][0])
    q1_scene = model.scene(Q1, pitch=0.0)
    all_scenes = [q0_scene, q1_scene, model.scene(0.25, knee=0.10), model.scene(0.425, knee=1.15)]
    com_errors = [np.linalg.norm(jump.center_of_mass(scene, model.com_inertials) - [0, 0, COM_HEIGHT])
                  for scene in all_scenes]
    heights = [min(jump.wheel_bottom(scene, side)[2] for side in "LR") for scene in all_scenes]
    if min(heights) <= 0 or not heights[3] > heights[2] or max(com_errors) > 1e-12:
        raise ValueError("Both comparisons must be airborne at the prescribed whole-CoM state")
    fine = reconstruct(model, 1201)
    convergence = float(abs(fine["theta_grid"][0] - motion["theta_grid"][0]))
    if convergence > 1e-6:
        raise ValueError("Shape integration has not converged")
    return {"scope": "Projected planar model and collision geometry, NOT full dynamics feasibility or policy rollout",
            "momentum_and_velocity_test_pose_count": len(group_samples),
            "maximum_angular_momentum_residual_kg_m2ps": max(residual),
            "maximum_link_velocity_finite_difference_error_mps": max(velocity_error),
            "opposite_body_leg_momentum_in_test_cases": True,
            "integration_refinement_error_rad": convergence,
            "whole_com_position_maximum_error_m": max(com_errors),
            "extended_wheel_clearance_m": heights[2], "tucked_wheel_clearance_m": heights[3],
            "clearance_gain_at_fixed_com_m": heights[3] - heights[2],
            "airborne_all_poses": True, "joint_limits_checked": True}


def write_data(output, model, motion, report, urdf, layout, dpi):
    with (output / "planar_momentum_exchange.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "hip_angle_rad", "body_pitch_rad", "hip_rate_radps",
                         "body_pitch_rate_radps", "leg_absolute_pitch_rate_radps",
                         "body_centroidal_Hy_kg_m2ps", "legs_centroidal_Hy_kg_m2ps", "total_Hy_kg_m2ps", "status"])
        for t, q, theta, q_dot in zip(motion["time_s"], motion["hip_rad"], motion["body_pitch_rad"], motion["hip_dot_radps"]):
            A, B = model.coupling(q)
            theta_dot = -B / A * q_dot
            groups = model.momentum(model.scene(q, pitch=theta), theta_dot, q_dot)
            writer.writerow([t, q, theta, q_dot, theta_dot, theta_dot + q_dot,
                             groups["body"], groups["legs"], sum(groups.values()), "illustrative_planar_model_not_measurement"])
    with (output / "fixed_com_clearance.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["configuration", "whole_com_height_m", "wheel_minimum_clearance_m", "status"])
        writer.writerow(["extended", COM_HEIGHT, report["extended_wheel_clearance_m"], "fixed_com_kinematic_comparison_not_rollout"])
        writer.writerow(["tucked", COM_HEIGHT, report["tucked_wheel_clearance_m"], "fixed_com_kinematic_comparison_not_rollout"])
    report["layout"] = layout
    (output / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {"figure_type": "Physics/kinematic model illustration", "measured_policy_results": False,
                "nominal_urdf": {"path": str(urdf.resolve()), "sha256": hashlib.sha256(urdf.read_bytes()).hexdigest()},
                "primary_reference": REFERENCE_URL, "nominal_total_mass_kg": model.mass,
                "assumptions": ["No ground contact", "Uniform gravity", "Negligible air forces/torques",
                                "Sagittal projection, only Hy reconstructed", "Both hips mirrored, knees and wheels locked in panel a",
                                "Zero initial total Hy as a transparent example; not a universal opposite-velocity law"],
                "hip_range_rad": [Q0, Q1], "fixed_knee_rad_panel_a": FIXED_KNEE,
                "initial_body_pitch_deg": float(np.rad2deg(motion["theta_grid"][0])), "final_body_pitch_deg": 0.0,
                "model_motion_duration_s": DURATION, "illustrative_com_height_m": COM_HEIGHT,
                "body_group": sorted(item["link"] for item in model.inertials if item["link"] not in model.leg_owner),
                "leg_group": sorted(model.leg_owner), "momentum_about": "Whole-robot CoM; spin plus orbital contributions",
                "body_angles_in_a_are_momentum_reconstructed": True,
                "panel_a_display": "CoM translations removed for before/after configuration comparison",
                "panel_b_display": "Two hypothetical configurations placed side by side at the SAME CoM state, not sequential flight snapshots",
                "arrow_semantics": "Half-circle arrows show opposite rotations during adjustment, not instantaneous endpoint velocities or quantitative torque magnitudes",
                "wheel_clearance_warning": "More wheel clearance does not add whole-CoM flight height for unchanged takeoff CoM position/velocity",
                "font_size_pt": FONT_SIZE, "raster_size_px": [round(value * dpi) for value in FIGURE_SIZE]}
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    model = PlanarCentroidalModel(args.urdf)
    motion = reconstruct(model)
    report = validate_physics(model, motion)
    style.configure_fonts(args.output_dir, extra_text="".join(str(words) for words in TEXT.values()))
    plt.rcParams.update({"font.size": FONT_SIZE, "font.weight": "normal"})
    layout = {}
    for language in ("zh", "en") if args.language == "both" else (args.language,):
        fig, _ = create_figure(model, motion, language)
        layout[language] = validate_layout(fig)
        for extension in ("png", "pdf", "svg"):
            target = args.output_dir / f"airborne_momentum_exchange_{language}.{extension}"
            fig.savefig(target, dpi=args.dpi, facecolor="white")
            print(target)
        plt.close(fig)
    write_data(args.output_dir, model, motion, report, args.urdf, layout, args.dpi)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
