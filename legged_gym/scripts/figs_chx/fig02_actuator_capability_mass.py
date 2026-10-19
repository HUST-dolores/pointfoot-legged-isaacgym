#!/usr/bin/env python3
"""Plot actuator capability/mass coupling and the uphill wheel torque envelope."""

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

import fig0_sixd_design_variables as style


REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "out/actuator_capability_mass"
CONFIG = REPO / "legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py"
URDF = REPO / "resources/robots/WF_TRON1A/urdf/robot.urdf"
MOTOR_MODEL = REPO / "legged_gym/utils/motor_cost.py"
SCALES = (0.6, 1.0, 1.6)
CURVE_COLORS = ("#C39137", "#238B7B", "#487FB7")
FONT_SIZE = 14
FIGURE_SIZE = (12.8, 5.2)
TEXT = {
    "zh": {
        "flow_title": "(a)  电机参数映射",
        "envelope_title": "(b)  上坡工况：力矩—转速包络",
        "scale": "驱动尺度",
        "torque": "电机输出力矩",
        "mass": "电机质量",
        "host": "连杆质量与惯量",
        "dynamics": "机器人动力学",
        "torque_region": "低速\n恒转矩区",
        "power_region": "高速恒功率区",
        "speed_axis": "电机输出角速度 (rad/s)",
        "torque_axis": "电机输出力矩 (N·m)",
        "curve_06": "尺度 0.6",
        "curve_10": "尺度 1.0（标称）",
        "curve_16": "尺度 1.6",
    },
    "en": {
        "flow_title": "(a)  Motor parameter mapping",
        "envelope_title": "(b)  Uphill torque-speed envelope",
        "scale": "Motor scale",
        "torque": "Motor output torque",
        "mass": "Motor mass",
        "host": "Link mass and inertia",
        "dynamics": "Robot dynamics",
        "torque_region": "Constant\ntorque",
        "power_region": "High-speed\nconstant power",
        "speed_axis": "Motor output angular speed (rad/s)",
        "torque_axis": "Motor output torque (N m)",
        "curve_06": "Scale 0.6",
        "curve_10": "Scale 1.0 (nominal)",
        "curve_16": "Scale 1.6",
    },
}


def load_model():
    path = MOTOR_MODEL if MOTOR_MODEL.is_file() else HERE / "motor_cost.py"
    spec = importlib.util.spec_from_file_location("figure_motor_cost", path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model, path


def anchors_from_config():
    tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
    cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "BipedCfgWF")
    domain = next(node for node in cfg.body if isinstance(node, ast.ClassDef)
                  and node.name == "domain_rand")
    assignments = {}
    for node in domain.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = node.value
    names = ast.literal_eval(assignments["motor_group_names"])
    masses = ast.literal_eval(assignments["motor_nominal_masses_kg"])
    tree = ET.parse(URDF).getroot()
    effort = {
        name: float(tree.find(f"joint[@name='{name}_L_Joint']/limit").get("effort"))
        for name in names
    }
    return {
        "groups": names,
        "nominal_module_mass_kg": dict(zip(names, masses)),
        "nominal_peak_torque_nm": effort,
        "mass_exponent": ast.literal_eval(assignments["motor_mass_exponent"]),
        "power_exponent": ast.literal_eval(assignments["motor_power_exponent"]),
        "specific_power_w_per_kg": ast.literal_eval(assignments["motor_specific_power_w_per_kg"]),
    }


def node(ax, center, width, height, label, face, edge):
    x, y = center
    ax.add_patch(Rectangle(
        (x - width / 2, y - height / 2), width, height,
        linewidth=1.0,
        facecolor=face, edgecolor=edge, zorder=3,
    ))
    ax.text(x, y, label, ha="center", va="center", fontsize=FONT_SIZE,
            color=style.INK, zorder=4)


def arrow(ax, start, end, color=style.MUTED):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                linewidth=1.5, color=color, shrinkA=1.5,
                                shrinkB=3, zorder=2))


def draw_flow(ax, words):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    node(ax, (0.465, 0.88), 0.40, 0.105,
         words["scale"] + r"  $k_i$", "#F2F6FA", "#7199BD")
    node(ax, (0.23, 0.615), 0.39, 0.125, words["torque"],
         "#F0F7F4", "#67A496")
    node(ax, (0.71, 0.615), 0.365, 0.125, words["mass"],
         "#FAF6EA", "#C5A467")
    node(ax, (0.71, 0.365), 0.44, 0.125, words["host"],
         "#F3F5F9", "#7E98B5")
    node(ax, (0.465, 0.125), 0.725, 0.125, words["dynamics"],
         "#F6F3F9", "#A594BC")

    ax.plot([0.465, 0.465], [0.82, 0.758], color=style.MUTED, linewidth=1.5)
    ax.plot([0.23, 0.71], [0.758, 0.758], color=style.MUTED, linewidth=1.5)
    arrow(ax, (0.23, 0.758), (0.23, 0.685))
    arrow(ax, (0.71, 0.758), (0.71, 0.685))
    arrow(ax, (0.71, 0.545), (0.71, 0.435), "#B09563")
    arrow(ax, (0.23, 0.545), (0.23, 0.195), "#4F9B89")
    arrow(ax, (0.71, 0.295), (0.71, 0.195), "#718FAC")


def envelope_samples(model, anchors):
    peak = anchors["nominal_peak_torque_nm"]["wheel"]
    mass = anchors["nominal_module_mass_kg"]["wheel"]
    density = anchors["specific_power_w_per_kg"]
    power = model.relative_motor_power_w(mass, 1, density, anchors["power_exponent"])
    corner = power / peak
    speed = np.unique(np.concatenate([np.linspace(0, 25, 601), [corner]]))
    curves = {}
    records = []
    for scale in SCALES:
        torque = np.asarray([
            model.available_motor_torque_nm(
                peak, mass, scale, omega, density, anchors["power_exponent"]
            ) for omega in speed
        ])
        curves[scale] = torque
        records.append({
            "scale": scale,
            "peak_torque_nm": peak * scale,
            "module_mass_kg": model.relative_motor_mass_kg(mass, scale, anchors["mass_exponent"]),
            "power_limit_w": model.relative_motor_power_w(mass, scale, density, anchors["power_exponent"]),
            "corner_speed_rad_s": model.relative_motor_power_w(mass, scale, density,
                                                               anchors["power_exponent"]) / (peak * scale),
        })
        if not np.allclose(torque[speed <= corner], peak * scale):
            raise ValueError("Constant-torque branch does not match the source model")
        if not np.allclose(torque[speed > corner] * speed[speed > corner], records[-1]["power_limit_w"]):
            raise ValueError("Constant-power branch does not match the source model")
    return speed, curves, power, corner, records


def draw_envelope(ax, words, anchors, speed, curves, corner):
    ax.set(xlim=(0, 25), ylim=(0, 78))
    ax.axvspan(0, corner, facecolor="#FCF1D7", zorder=0)
    ax.axvspan(corner, 25, facecolor="#EDF4F7", zorder=0)
    ax.axvline(corner, color="#85929B", linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    for scale, color, key in zip(SCALES, CURVE_COLORS, ("curve_06", "curve_10", "curve_16")):
        ax.plot(speed, curves[scale], color=color, linewidth=2.9 if scale == 1 else 2.2,
                label=words[key], zorder=4)
    ax.text(corner / 2, 72.0, words["torque_region"], ha="center", va="center",
            fontsize=FONT_SIZE, color="#8E7138", linespacing=1.2)
    ax.text((corner + 25) / 2, 72, words["power_region"], ha="center", va="center",
            fontsize=FONT_SIZE, color="#4D6E81", linespacing=1.2)
    ax.set_xlabel(words["speed_axis"], fontsize=FONT_SIZE, labelpad=7)
    ax.set_ylabel(words["torque_axis"], fontsize=FONT_SIZE, labelpad=7)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_yticks([0, 20, 40, 60])
    ax.tick_params(labelsize=FONT_SIZE, colors=style.MUTED, length=4, width=0.8)
    ax.grid(axis="y", linewidth=0.7, color="#DCE4E8", zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color("#8C99A1")
        ax.spines[name].set_linewidth(0.9)
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.85), frameon=False,
              fontsize=FONT_SIZE, handlelength=2.1, labelspacing=0.65)


def create_figure(language, anchors, speed, curves, power, corner):
    words = TEXT[language]
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=140)
    fig.text(0.05, 0.925, words["flow_title"], fontsize=FONT_SIZE, color=style.INK)
    fig.text(0.555, 0.925, words["envelope_title"], fontsize=FONT_SIZE, color=style.INK)
    flow = fig.add_axes([0.025, 0.095, 0.485, 0.78])
    plot = fig.add_axes([0.600, 0.165, 0.377, 0.710])
    draw_flow(flow, words)
    draw_envelope(plot, words, anchors, speed, curves, corner)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--anchors-json", type=Path)
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extra_text = "".join(word for language in TEXT.values() for word in language.values())
    style.configure_fonts(args.output_dir, extra_text=extra_text)
    anchors = (json.loads(args.anchors_json.read_text(encoding="utf-8"))
               if args.anchors_json else anchors_from_config())
    model, model_path = load_model()
    speed, curves, power, corner, records = envelope_samples(model, anchors)
    languages = ("zh", "en") if args.language == "both" else (args.language,)
    for language in languages:
        fig = create_figure(language, anchors, speed, curves, power, corner)
        for suffix in ("png", "pdf", "svg"):
            target = args.output_dir / f"actuator_capability_mass_{language}.{suffix}"
            fig.savefig(target, dpi=args.dpi, facecolor="white")
            print(target)
        plt.close(fig)
    (args.output_dir / "anchor_parameters.json").write_text(
        json.dumps(anchors, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "wheel_torque_speed_envelopes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["output_speed_rad_s", "torque_scale_0p6_nm", "torque_scale_1p0_nm", "torque_scale_1p6_nm"])
        writer.writerows(zip(speed, *(curves[scale] for scale in SCALES)))
    manifest = {
        "figure": "Actuator capability and mass cost mapping",
        "source_model": str(model_path),
        "source_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "source_config": str(CONFIG),
        "source_urdf": str(URDF),
        "anchors": anchors,
        "envelope_records": records,
        "displayed_speed_interval_rad_s": [0, 25],
        "speed_interval_is_display_window_not_maximum_speed": True,
        "hosting_mass_rule": "Add each module mass change to its corresponding left/right hosting link",
        "inertia_rule": "Isaac Gym set_actor_rigid_body_properties(..., recomputeInertia=True)",
        "ability_path": "Actuator scale -> torque limit -> drive output constraint -> dynamics",
        "mass_path": "Actuator scale -> module mass -> hosting link mass -> recomputed inertia -> dynamics",
        "curve_type": "Evaluation-model envelope, using 450 W/kg baseline-anchored mechanical power",
        "slope_deg": 10,
        "languages": list(languages),
        "raster_size_px": [round(size * args.dpi) for size in FIGURE_SIZE],
        "visible_relationship_equations": False,
        "display_mass_label": "Motor mass; numeric definition retains the original motor-plus-reduction assembly mass",
        "uniform_font_size_pt": FONT_SIZE,
        "anchor_annotations_visible": False,
        "figure_revision": 2,
    }
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
