#!/usr/bin/env python3
"""Calculate local inversion sensitivities, not measured estimator errors."""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import NullLocator, ScalarFormatter
import numpy as np

import fig0_sixd_design_variables as style


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SNAPSHOT = HERE / "source_snapshot"
DEFAULT_CONFIG = (SNAPSHOT / "wheelfoot_flat_config.py" if SNAPSHOT.is_dir() else
                  REPO / "legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py")
DEFAULT_ENV = (SNAPSHOT / "wheelfoot_flat.py" if SNAPSHOT.is_dir() else
               REPO / "legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py")
DEFAULT_POSITION = (SNAPSHOT / "analyze_load_experiment.py" if SNAPSHOT.is_dir() else
                    REPO / "legged_gym/scripts/analyze_load_experiment.py")
DEFAULT_OUTPUT = HERE / "out/analytical_load_sensitivity"
FONT_SIZE = 14
FIGURE_SIZE = (12.8, 5.8)
ANGLE_RANGE_DEG = (30.0, 140.0)
SINGULAR_GAP_DEG = 0.002
MASS_RANGE_KG = (0.1, 20.0)
OFFSETS_M = (0.05, 0.10, 0.20)
CURVE_COLORS = ("#C39137", "#238B7B", "#B34959")
TEXT = {
    "zh": {
        "mass_title": "(a)  质量反演：构型敏感性",
        "position_title": "(b)  质心到位置反解：低负载敏感性",
        "angle_axis": r"构型角 $\varphi$ (°)",
        "mass_gain_axis": "质量误差增益 (kg/(N·m))",
        "payload_axis": "负载质量 (kg)",
        "position_gain_axis": "位置误差增益 (cm/kg)",
        "singularity": "力臂为零",
        "offset_title": "负载偏心距",
        "analysis_label": "模型分析（非测量结果）",
    },
    "en": {
        "mass_title": "(a)  Mass inversion vs. configuration",
        "position_title": "(b)  CoM-to-position inversion at low payload",
        "angle_axis": r"Configuration angle $\varphi$ (°)",
        "mass_gain_axis": "Mass-error gain (kg/(N m))",
        "payload_axis": "Payload mass (kg)",
        "position_gain_axis": "Position-error gain (cm/kg)",
        "singularity": "Zero moment arm",
        "offset_title": "Payload eccentricity",
        "analysis_label": "Model analysis (not measurements)",
    },
}


def literal_assignments(nodes, names):
    values = {}
    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[target.id] = ast.literal_eval(node.value)
    if values.keys() != set(names):
        raise ValueError("Model source is missing expected literal parameters")
    return values


def load_parameters(config_path, env_path, position_path):
    config = ast.parse(config_path.read_text(encoding="utf-8"))
    cfg = next(node for node in config.body if isinstance(node, ast.ClassDef) and node.name == "BipedCfgWF")
    asset = next(node for node in cfg.body if isinstance(node, ast.ClassDef) and node.name == "asset")
    sim = next(node for node in cfg.body if isinstance(node, ast.ClassDef) and node.name == "sim")
    values = literal_assignments(asset.body, (
        "load_estimation_thigh_length", "load_estimation_alpha_L",
        "load_estimation_alpha_R", "load_estimation_body_mass"))
    gravity = literal_assignments(sim.body, ("gravity",))["gravity"]
    if gravity[:2] != [0.0, 0.0] or gravity[2] >= 0:
        raise ValueError("This analysis assumes vertical downward gravity")
    environment = ast.parse(env_path.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(environment) if isinstance(node, ast.FunctionDef)
                  and node.name == "_compute_load_estimates")
    offsets = []
    for node in method.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                t.id in ("mass_moment_arm_L", "mass_moment_arm_R") for t in node.targets):
            if not isinstance(node.value, ast.BinOp) or not isinstance(node.value.op, ast.Add):
                raise ValueError("Effective moment-arm formula no longer has the expected form")
            offsets.append(float(ast.literal_eval(node.value.right)))
    if len(offsets) != 2 or offsets[0] != offsets[1]:
        raise ValueError("The symmetric configuration slice requires matching arm offsets")
    position = ast.parse(position_path.read_text(encoding="utf-8"))
    position_mass = literal_assignments(position.body, ("BODY_MASS",))["BODY_MASS"]
    if position_mass != values["load_estimation_body_mass"]:
        raise ValueError("Position inversion and the configuration disagree on host-body mass")
    params = {
        "thigh_length_m": float(values["load_estimation_thigh_length"]),
        "moment_arm_offset_m": offsets[0],
        "gravity_mps2": abs(float(gravity[2])),
        "alpha_left": float(values["load_estimation_alpha_L"]),
        "alpha_right": float(values["load_estimation_alpha_R"]),
        "host_body_mass_kg": float(position_mass),
        "abad_angle_deg": 0.0,
    }
    if not all(np.isfinite(value) for value in params.values()):
        raise ValueError("Model parameters must be finite")
    if min(params[key] for key in ("thigh_length_m", "gravity_mps2", "host_body_mass_kg")) <= 0:
        raise ValueError("Lengths, gravity and mass must be positive")
    if abs(params["moment_arm_offset_m"]) >= params["thigh_length_m"]:
        raise ValueError("The selected geometry has no interior zero-arm configuration")
    return params


def moment_arm(phi_rad, params):
    return params["thigh_length_m"] * np.cos(phi_rad) + params["moment_arm_offset_m"]


def torque_to_mass_gain(phi_rad, params):
    arm = np.asarray(moment_arm(phi_rad, params), dtype=float)
    numerator = np.hypot(params["alpha_left"], params["alpha_right"])
    denominator = params["gravity_mps2"] * np.abs(arm) * abs(np.cos(np.deg2rad(params["abad_angle_deg"])))
    # The singular point is undefined, not a finite clipped maximum.
    return np.divide(numerator, denominator, out=np.full_like(denominator, np.nan),
                     where=denominator > 1e-12)


def position_mass_gain(mass_kg, offset_m, host_mass_kg):
    mass = np.asarray(mass_kg, dtype=float)
    if (not np.isfinite(mass).all() or np.any(mass <= 0) or
            not np.isfinite(offset_m) or not np.isfinite(host_mass_kg) or
            offset_m < 0 or host_mass_kg <= 0):
        raise ValueError("Position sensitivity requires positive finite mass and nonnegative eccentricity")
    # Different operating masses have a physically consistent CoM shift. Each
    # partial derivative then holds that operating point's CoM input fixed.
    return host_mass_kg * offset_m / (mass * (host_mass_kg + mass))


def model_samples(params):
    singular_deg = float(np.rad2deg(np.arccos(-params["moment_arm_offset_m"] / params["thigh_length_m"])))
    left = singular_deg - np.geomspace(singular_deg - ANGLE_RANGE_DEG[0], SINGULAR_GAP_DEG, 600)
    right = singular_deg + np.geomspace(SINGULAR_GAP_DEG, ANGLE_RANGE_DEG[1] - singular_deg, 600)
    angles = np.concatenate([left, [singular_deg], right])
    phi = np.deg2rad(angles)
    arms = moment_arm(phi, params)
    gains = torque_to_mass_gain(phi, params)
    if np.isfinite(gains[600]) or np.count_nonzero(~np.isfinite(gains)) != 1:
        raise ValueError("The curve must leave exactly the zero-arm sample undefined")
    masses = np.geomspace(*MASS_RANGE_KG, 650)
    position_gains = np.array([position_mass_gain(masses, offset, params["host_body_mass_kg"])
                               for offset in OFFSETS_M])
    return singular_deg, angles, arms, gains, masses, position_gains


def prepare_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color("#84949F")
        ax.spines[name].set_linewidth(0.9)
    ax.grid(which="major", color="#E7EDF0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FONT_SIZE, colors=style.INK, length=3.5)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())


def create_figure(samples, language):
    singular, angles, arms, mass_gains, masses, position_gains = samples
    words = TEXT[language]
    fig, (left, right) = plt.subplots(1, 2, figsize=FIGURE_SIZE, dpi=140)
    fig.subplots_adjust(left=0.09, right=0.973, bottom=0.23, top=0.85, wspace=0.32)
    left.set_yscale("log")
    left.plot(angles, mass_gains, color="#487FB7", linewidth=2)
    left.axvline(singular, color="#82929E", linewidth=1, linestyle=(0, (4, 3)))
    left.annotate(words["singularity"], xy=(singular, 400), xytext=(104, 400),
                   fontsize=FONT_SIZE, color=style.MUTED, ha="left", va="center",
                   arrowprops={"arrowstyle": "-", "color": style.MUTED, "lw": 0.9,
                               "shrinkA": 4, "shrinkB": 3})
    left.set(xlim=ANGLE_RANGE_DEG, ylim=(0.1, 1000))
    left.set_xticks([40, 60, 80, 100, 120, 140])
    left.set_yticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"])
    left.set_xlabel(words["angle_axis"], fontsize=FONT_SIZE, labelpad=10)
    left.set_ylabel(words["mass_gain_axis"], fontsize=FONT_SIZE, labelpad=8)
    left.set_title(words["mass_title"], fontsize=FONT_SIZE, fontweight="normal", loc="left", pad=17)
    right.set(xscale="log", yscale="log", xlim=MASS_RANGE_KG, ylim=(0.05, 300))
    for offset, color, gain in zip(OFFSETS_M, CURVE_COLORS, position_gains):
        right.plot(masses, 100 * gain, color=color, linewidth=1.9, label=f"{offset * 100:g} cm")
    right.set_xticks([0.1, 0.3, 1, 3, 10, 20])
    right.xaxis.set_major_formatter(ScalarFormatter())
    right.set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    right.set_xlabel(words["payload_axis"], fontsize=FONT_SIZE, labelpad=10)
    right.set_ylabel(words["position_gain_axis"], fontsize=FONT_SIZE, labelpad=8)
    right.set_title(words["position_title"], fontsize=FONT_SIZE, fontweight="normal", loc="left", pad=17)
    legend = right.legend(title=words["offset_title"], fontsize=FONT_SIZE,
                          title_fontsize=FONT_SIZE, loc="upper right", frameon=False)
    legend.get_title().set_fontweight("normal")
    for ax in (left, right):
        prepare_axis(ax)
    fig.text(0.535, 0.045, words["analysis_label"], ha="center", fontsize=FONT_SIZE, color=style.INK)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            if not np.isclose(text.get_fontsize(), FONT_SIZE):
                raise ValueError("Every label must use the agreed 14 pt font size")
            if not all(fig.bbox.contains(*corner) for corner in text.get_window_extent(renderer).get_points()):
                raise ValueError("Text falls outside the export: " + text.get_text())
    return fig


def write_data(output, params, samples, sources, dpi):
    singular, angles, arms, mass_gains, masses, position_gains = samples
    with (output / "model_mass_sensitivity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["configuration_angle_deg", "effective_moment_arm_m", "gain_kg_per_nm", "status"])
        for angle, arm, gain in zip(angles, arms, mass_gains):
            writer.writerow([angle, arm, gain if np.isfinite(gain) else "",
                             "defined" if np.isfinite(gain) else "undefined_zero_arm"])
    with (output / "model_position_sensitivity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["payload_mass_kg", "payload_eccentricity_m", "operating_com_shift_m",
                         "position_mass_gain_m_per_kg", "position_mass_gain_cm_per_kg"])
        for offset, gains in zip(OFFSETS_M, position_gains):
            shifts = masses / (params["host_body_mass_kg"] + masses) * offset
            for mass, shift, gain in zip(masses, shifts, gains):
                writer.writerow([mass, offset, shift, gain, 100 * gain])
    manifest = {
        "figure_type": "Deterministic local-Jacobian model analysis; NOT measured errors, rollouts or estimator RMSE",
        "source_files": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for name, path in sources.items()},
        "parameters": params,
        "mass_gain_definition": "Euclidean norm of d(total raw analytical mass)/d(tau_load_left,tau_load_right); fixed pose",
        "mass_gain_formula": "hypot(alpha_L,alpha_R) / (g * abs(L*cos(phi)+c) * abs(cos(abad)))",
        "mass_configuration_slice": "Both sagittal configuration angles equal; abad=0; h(q) held fixed in torque derivative",
        "mass_formula_regularization": "Underlying unregularized formula. Runtime individual-cosine floors do not protect the effective-arm cancellation.",
        "singular_configuration_deg": singular,
        "angle_domain_deg": list(ANGLE_RANGE_DEG),
        "singular_sampling_gap_each_side_deg": SINGULAR_GAP_DEG,
        "mass_display_window_kg_per_nm": [0.1, 1000],
        "mass_curve_above_display_window": "Retained in CSV; mathematical gain continues to infinity, not a finite maximum",
        "undefined_sample": "Zero effective arm has no finite derivative; CSV gain left blank and plot broken there",
        "position_inversion": "r_payload = c0 + (host_mass + payload_mass)/payload_mass * delta_com",
        "position_gain_definition": "Norm of partial d(r_payload)/d(payload_mass) holding delta_com fixed at each operating point",
        "position_gain_formula": "host_mass * eccentricity / (payload_mass * (host_mass + payload_mass))",
        "operating_com_shift": "payload_mass/(host_mass+payload_mass) * eccentricity; fixed physical eccentricity across the mass sweep",
        "position_inputs_held_fixed_for_derivative": ["host body mass", "reference body CoM", "operating CoM-shift observation"],
        "position_domain_kg": list(MASS_RANGE_KG), "eccentricities_m": list(OFFSETS_M),
        "position_plot_unit_conversion": "m/kg times 100 -> cm/kg",
        "not_applied": ["simulation", "real log samples", "noise injection", "filtering", "position gates", "output clipping"],
        "limitations": "First-order input sensitivities only, not total estimation error or a validation of the quasi-static assumptions",
        "font_size_pt": FONT_SIZE, "raster_size_px": [round(dpi * x) for x in FIGURE_SIZE],
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Singular configuration:", singular, "deg; position curves: 5, 10, 20 cm eccentricity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-source", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--position-source", type=Path, default=DEFAULT_POSITION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    params = load_parameters(args.config, args.env_source, args.position_source)
    samples = model_samples(params)
    corpus = "".join(value for words in TEXT.values() for value in words.values()) + "5 cm 10 cm 20 cm"
    style.configure_fonts(args.output_dir, extra_text=corpus)
    plt.rcParams.update({"font.size": FONT_SIZE, "font.weight": "normal",
                         "axes.labelweight": "normal", "axes.titleweight": "normal"})
    languages = ("zh", "en") if args.language == "both" else (args.language,)
    for language in languages:
        figure = create_figure(samples, language)
        for extension in ("png", "pdf", "svg"):
            target = args.output_dir / f"analytical_load_sensitivity_{language}.{extension}"
            figure.savefig(target, dpi=args.dpi, facecolor="white")
            print(target)
        plt.close(figure)
    write_data(args.output_dir, params, samples,
               {"configuration": args.config, "mass_inversion": args.env_source,
                "com_to_position_inversion": args.position_source}, args.dpi)


if __name__ == "__main__":
    main()
