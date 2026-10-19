#!/usr/bin/env python3
"""Plot physical capability projections from one compatible nested BO search."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

import fig0_sixd_design_variables as style


HERE = Path(__file__).resolve().parent
ARCHIVED_DATA = HERE / "source_data"
EXPERIMENT = Path("/home/xu/Desktop/codesign_worklog/nested_four_objective_sixd_2026-07-30/results")
DEFAULT_SOURCE = ARCHIVED_DATA if ARCHIVED_DATA.is_dir() else EXPERIMENT
DEFAULT_OUTPUT = HERE / "out/candidate_tradeoffs"
TASKS = ("payload", "speed", "clearance", "jump")
DESIGN_KEYS = ("thigh", "shank", "abad", "hip", "knee", "wheel")
FONT_SIZE = 14
FIGURE_SIZE = (12.8, 5.8)
FRONT_COLOR = "#287FAD"
DOMINATED_COLOR = "#BAC3CB"
REFERENCE_COLOR = "#A17920"
COMPROMISE_COLOR = "#B34959"
INPUT_FILES = (
    "live_nested_bo/SEARCH_ALL.json",
    "live_nested_bo/SEARCH_PARETO_FRONT.json",
    "live_nested_bo/RUN_MANIFEST.json",
    "live_nested_bo/NORMALIZATION.json",
    "final_nested_bo/FINAL_RESULTS.json",
)
TEXT = {
    "zh": {
        "payload_jump_title": "(a)  承载—跳跃面积",
        "speed_clearance_title": "(b)  上坡速度—迈步净空",
        "payload_axis": "可靠承载 (kg)",
        "jump_axis": r"跳跃可靠面积 ($10^{-3}$ m$^2$)",
        "speed_axis": "10° 上坡实际速度 (m/s)",
        "clearance_axis": "迈步可靠净空 (cm)",
        "dominated": "被支配候选",
        "nondominated": "非支配候选（四指标）",
        "nominal": "标称",
        "reference": "单项参考",
        "compromise": "折中设计",
        "payload_reference": "承载参考",
        "jump_reference": "跳跃参考",
        "speed_reference": "速度参考",
        "clearance_reference": "迈步参考",
    },
    "en": {
        "payload_jump_title": "(a)  Payload vs. jump area",
        "speed_clearance_title": "(b)  Uphill speed vs. clearance",
        "payload_axis": "Reliable payload (kg)",
        "jump_axis": r"Reliable jump area ($10^{-3}$ m$^2$)",
        "speed_axis": "Actual uphill speed at 10° (m/s)",
        "clearance_axis": "Reliable stepping clearance (cm)",
        "dominated": "Dominated",
        "nondominated": "Nondominated (4D)",
        "nominal": "Nominal",
        "reference": "Task reference",
        "compromise": "Compromise",
        "payload_reference": "Payload ref.",
        "jump_reference": "Jump ref.",
        "speed_reference": "Speed ref.",
        "clearance_reference": "Clearance ref.",
    },
}


def nondominated_mask(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Dominance requires a nonempty finite objective matrix")
    # All objectives are maximized; duplicate vectors do not dominate each other.
    result = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        dominates = np.all(values >= point, axis=1) & np.any(values > point, axis=1)
        result[index] = not dominates.any()
    return result


def load_data(source):
    documents = {name: json.loads((source / name).read_text()) for name in INPUT_FILES}
    rows = documents[INPUT_FILES[0]]
    manifest = documents["live_nested_bo/RUN_MANIFEST.json"]
    normalization = documents["live_nested_bo/NORMALIZATION.json"]
    final = documents["final_nested_bo/FINAL_RESULTS.json"]
    ids = [row["design_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated design IDs cannot be plotted as independent candidates")
    seeds = {tuple(row[key] for key in ("train_seed", "validation_seed", "test_seed"))
             for row in rows}
    if len(seeds) != 1 or len(set(next(iter(seeds)))) != 3:
        raise ValueError("Search candidates must share independent train/validation/test seeds")
    if manifest["inner_updates"] != 400:
        raise ValueError("This figure's archived protocol expects a 400-update inner budget")
    allowed_offsets = set(manifest["inner_checkpoint_offsets"])
    values = np.array([[row["raw_by_task"][task] for task in TASKS] for row in rows])
    front = nondominated_mask(values)
    stored = {row["design_id"] for row in documents[INPUT_FILES[1]]}
    if stored != {ids[i] for i in np.flatnonzero(front)}:
        raise ValueError("Recomputed four-objective front differs from the archived front")
    nominal_ids = {normalization[task]["nominal_design_id"] for task in TASKS}
    if len(nominal_ids) != 1:
        raise ValueError("Task normalizations disagree on nominal hardware")
    roles = {"nominal": nominal_ids.pop(), "compromise": final["winner"]["design_id"]}
    roles.update({task + "_reference": normalization[task]["official_specialist_design_id"]
                  for task in TASKS})
    by_id = dict(zip(ids, rows))
    if not set(roles.values()) <= set(ids):
        raise ValueError("Every highlighted design must have a same-protocol search observation")
    if not np.array_equal(by_id[roles["nominal"]]["design_vector"], np.ones(6)):
        raise ValueError("The nominal design must have six unit scales")
    for task in TASKS:
        expected = manifest["single_task_specialists_are_fixed_calibration_designs"][task]
        observed = by_id[roles[task + "_reference"]]["design"]
        if any(observed[key] != expected[key] for key in DESIGN_KEYS):
            raise ValueError("Reference hardware differs from the run manifest")
    for row in rows:
        for task in TASKS:
            selected = row["selected_checkpoint_by_task"][task]
            if selected["checkpoint_offset"] not in allowed_offsets:
                raise ValueError("Checkpoint selection exceeds the shared inner budget")
            if selected["validation_seed"] != row["validation_seed"]:
                raise ValueError("Checkpoint selection was not based on the common validation seed")
    return rows, values, front, roles, manifest


def plot_candidates(ax, rows, values, front, roles, columns, factors):
    projected = values[:, columns] * np.asarray(factors)
    for mask, color, size in ((~front, DOMINATED_COLOR, 37), (front, FRONT_COLOR, 57)):
        ax.scatter(projected[mask, 0], projected[mask, 1], s=size, c=color,
                   edgecolors="white", linewidths=0.55, zorder=3)
    ids = [row["design_id"] for row in rows]
    positions = {role: projected[ids.index(design_id)] for role, design_id in roles.items()}
    for role, position in positions.items():
        if role == "nominal":
            marker, color, size = "s", style.INK, 170
        elif role == "compromise":
            marker, color, size = "*", COMPROMISE_COLOR, 360
        else:
            marker, color, size = "D", REFERENCE_COLOR, 190
        ax.scatter(*position, s=size, marker=marker, facecolors="none",
                   edgecolors=color, linewidths=1.45, zorder=5)
    ax.grid(color="#E8EDF0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color("#84949F")
        ax.spines[name].set_linewidth(0.9)
    ax.tick_params(labelsize=FONT_SIZE, colors=style.INK, length=3.5)
    return positions


def annotate_design(ax, role, position, destination, words, align="left"):
    color = (COMPROMISE_COLOR if role == "compromise" else
             style.INK if role == "nominal" else REFERENCE_COLOR)
    ax.annotate(words[role], xy=position, xytext=destination, color=color,
                fontsize=FONT_SIZE, ha=align, va="center", zorder=7,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9,
                            "shrinkA": 4, "shrinkB": 9},
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.6})


def create_figure(rows, values, front, roles, language):
    words = TEXT[language]
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, dpi=140)
    fig.subplots_adjust(left=0.076, right=0.98, bottom=0.23, top=0.87, wspace=0.27)
    left, right = axes
    positions = plot_candidates(left, rows, values, front, roles, (0, 3), (1, 1000))
    left.set(xlim=(-1.8, 30.5), ylim=(-0.65, 12.25))
    left.set_xticks([0, 5, 10, 15, 20, 25, 30])
    left.set_yticks([0, 2, 4, 6, 8, 10, 12])
    left.set_xlabel(words["payload_axis"], fontsize=FONT_SIZE, labelpad=9)
    left.set_ylabel(words["jump_axis"], fontsize=FONT_SIZE, labelpad=8)
    left.set_title(words["payload_jump_title"], fontsize=FONT_SIZE, fontweight="normal",
                   loc="left", pad=15)
    for role, destination in (("payload_reference", (19.0, 9.0)),
                               ("jump_reference", (18.4, 11.0)),
                               ("nominal", (8.2, 1.05)),
                               ("compromise", (22.2, 6.15))):
        annotate_design(left, role, positions[role], destination, words)

    positions = plot_candidates(right, rows, values, front, roles, (1, 2), (1, 100))
    right.set(xlim=(1.27, 2.18), ylim=(15.4, 22.65))
    right.set_xticks([1.4, 1.6, 1.8, 2.0])
    right.set_yticks([16, 18, 20, 22])
    right.set_xlabel(words["speed_axis"], fontsize=FONT_SIZE, labelpad=9)
    right.set_ylabel(words["clearance_axis"], fontsize=FONT_SIZE, labelpad=8)
    right.set_title(words["speed_clearance_title"], fontsize=FONT_SIZE, fontweight="normal",
                    loc="left", pad=15)
    for role, destination in (("clearance_reference", (1.53, 22.05)),
                               ("speed_reference", (2.045, 20.55)),
                               ("nominal", (1.365, 17.45)),
                               ("compromise", (1.955, 16.65))):
        annotate_design(right, role, positions[role], destination, words)

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=DOMINATED_COLOR,
               markersize=7, label=words["dominated"]),
        Line2D([], [], marker="o", linestyle="none", color=FRONT_COLOR,
               markersize=7, label=words["nondominated"]),
        Line2D([], [], marker="s", linestyle="none", color=style.INK,
               markerfacecolor="none", markersize=9, label=words["nominal"]),
        Line2D([], [], marker="D", linestyle="none", color=REFERENCE_COLOR,
               markerfacecolor="none", markersize=9, label=words["reference"]),
        Line2D([], [], marker="*", linestyle="none", color=COMPROMISE_COLOR,
               markerfacecolor="none", markersize=13, label=words["compromise"]),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.025),
               frameon=False, ncol=5, fontsize=FONT_SIZE, columnspacing=1.35,
               handletextpad=0.5, handlelength=1.2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            if not np.isclose(text.get_fontsize(), FONT_SIZE):
                raise ValueError("All figure text must use the agreed uniform font size")
            if not fig.bbox.contains(*text.get_window_extent(renderer).get_points()[0]) or not \
                    fig.bbox.contains(*text.get_window_extent(renderer).get_points()[1]):
                raise ValueError("Text lies outside the export: " + text.get_text())
    return fig


def write_analysis(output, source, rows, values, front, roles, manifest, dpi):
    pair_masks = {
        "payload_jump": nondominated_mask(values[:, (0, 3)]),
        "speed_clearance": nondominated_mask(values[:, (1, 2)]),
    }
    ids = [row["design_id"] for row in rows]
    fields = ["design_id", "phase", "role", *DESIGN_KEYS,
              "payload_kg", "uphill_speed_mps", "clearance_m", "jump_area_m2",
              "nondominated_4d", "nondominated_payload_jump_2d",
              "nondominated_speed_clearance_2d", "dominated_by_ids",
              "train_seed", "validation_seed", "test_seed", "equal_weight_score"]
    with (output / "candidate_points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            dominators = np.flatnonzero(np.all(values >= values[index], axis=1) &
                                        np.any(values > values[index], axis=1))
            writer.writerow({
                "design_id": row["design_id"], "phase": row["phase"],
                "role": ";".join(role for role, id_ in roles.items() if id_ == row["design_id"]),
                **row["design"], "payload_kg": values[index, 0],
                "uphill_speed_mps": values[index, 1], "clearance_m": values[index, 2],
                "jump_area_m2": values[index, 3], "nondominated_4d": bool(front[index]),
                "nondominated_payload_jump_2d": bool(pair_masks["payload_jump"][index]),
                "nondominated_speed_clearance_2d": bool(pair_masks["speed_clearance"][index]),
                "dominated_by_ids": ";".join(ids[i] for i in dominators),
                **{key: row[key] for key in ("train_seed", "validation_seed", "test_seed")},
                "equal_weight_score": row["equal_weight_score"],
            })
    result = {
        "source_dir": str(source.resolve()),
        "source_files_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                for name in INPUT_FILES},
        "candidate_count": len(rows), "nondominated_4d_count": int(front.sum()),
        "nondominated_4d_ids": [ids[i] for i in np.flatnonzero(front)],
        "pairwise_front_ids_not_used_for_point_color": {
            pair: [ids[i] for i in np.flatnonzero(mask)] for pair, mask in pair_masks.items()},
        "highlighted_roles": roles,
        "highlighted_values_search_test_only": {
            role: {"design_id": id_, "design": rows[ids.index(id_)]["design"],
                   "raw_by_task": rows[ids.index(id_)]["raw_by_task"],
                   "nondominated_4d": bool(front[ids.index(id_)])} for role, id_ in roles.items()},
        "objective_directions": {task: "maximize" for task in TASKS},
        "dominance": "Other candidate >= in all four physical objectives and > in at least one; no weights, clipping or epsilon",
        "statistics": "One common search train seed; validation-selected checkpoints; independent common search test seed; no error bars or inferential significance claim",
        "seeds": {key: rows[0][key] for key in ("train_seed", "validation_seed", "test_seed")},
        "inner_update_budget": manifest["inner_updates"],
        "checkpoint_offsets": manifest["inner_checkpoint_offsets"],
        "compromise_identity": "Final formal-retest winner hardware, plotted at its search-test observation, NOT its three-seed formal mean",
        "task_references": "Earlier single-task calibration hardware, re-evaluated under this same nested search protocol; not new single-task optima",
        "display_conversions": {"jump_area_m2_to_10_minus_3_m2": 1000,
                                "clearance_m_to_cm": 100},
        "point_adjustments": "None: no outlier deletion, jitter, smoothing or aggregation",
        "projection_warning": "A 4D nondominated point can be dominated in either 2D projection. No connecting front line is drawn.",
        "font_size_pt": FONT_SIZE, "raster_size_px": [round(dpi * x) for x in FIGURE_SIZE],
    }
    (output / "figure_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Candidates:", len(rows), "four-objective nondominated:", int(front.sum()))
    print("Two-objective front counts:", {pair: int(mask.sum()) for pair, mask in pair_masks.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    rows, values, front, roles, manifest = load_data(args.source_dir)
    corpus = "".join(value for words in TEXT.values() for value in words.values())
    style.configure_fonts(args.output_dir, extra_text=corpus)
    plt.rcParams.update({"font.size": FONT_SIZE, "font.weight": "normal",
                         "axes.labelweight": "normal", "axes.titleweight": "normal"})
    languages = ("zh", "en") if args.language == "both" else (args.language,)
    for language in languages:
        figure = create_figure(rows, values, front, roles, language)
        for extension in ("png", "pdf", "svg"):
            target = args.output_dir / f"candidate_tradeoffs_{language}.{extension}"
            figure.savefig(target, dpi=args.dpi, facecolor="white")
            print(target)
        plt.close(figure)
    write_analysis(args.output_dir, args.source_dir, rows, values, front, roles, manifest, args.dpi)


if __name__ == "__main__":
    main()
