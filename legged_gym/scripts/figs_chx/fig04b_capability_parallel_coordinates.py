#!/usr/bin/env python3
"""Show four normalized capabilities without replacing the pairwise scatter."""

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
import fig04_candidate_tradeoffs as candidates


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "out/capability_parallel_coordinates"
FONT_SIZE = 14
FIGURE_SIZE = (12.8, 6.2)
BACKGROUND_COLOR = "#BBC5CD"
ROLE_STYLE = {
    "nominal": {"color": "#293B46", "marker": "s", "linestyle": (0, (5, 3))},
    "payload_reference": {"color": "#A17920", "marker": "o", "linestyle": "-"},
    "speed_reference": {"color": "#287FAD", "marker": "^", "linestyle": "-"},
    "clearance_reference": {"color": "#238B7B", "marker": "D", "linestyle": "-"},
    "jump_reference": {"color": "#8272AB", "marker": "v", "linestyle": "-"},
    "compromise": {"color": "#B34959", "marker": "*", "linestyle": "-"},
}
TEXT = {
    "zh": {
        "title": "四指标能力取舍",
        "axis_labels": ["可靠承载", "10° 上坡速度", "迈步净空", "跳跃可靠面积"],
        "y_label": "归一化能力",
        "other": "其他候选",
    },
    "en": {
        "title": "Four-capability trade-offs",
        "axis_labels": ["Reliable payload", "Uphill speed at 10°", "Stepping clearance", "Reliable jump area"],
        "y_label": "Normalized capability",
        "other": "Other candidates",
    },
}


def normalize_capabilities(raw, calibration):
    raw = np.asarray(raw, dtype=float)
    nominal = np.array([calibration[task]["nominal"] for task in candidates.TASKS])
    span = np.array([calibration[task]["span"] for task in candidates.TASKS])
    if raw.ndim != 2 or raw.shape[1] != 4 or not np.isfinite(raw).all():
        raise ValueError("Expected a finite matrix of four physical capabilities")
    if not np.isfinite(nominal).all() or not np.isfinite(span).all() or np.any(span <= 0):
        raise ValueError("Fixed normalization requires finite anchors and positive spans")
    return (raw - nominal) / span


def load_data(source):
    rows, raw, front, roles, protocol = candidates.load_data(source)
    calibration = json.loads((source / "live_nested_bo/NORMALIZATION.json").read_text())
    normalized = normalize_capabilities(raw, calibration)
    archived = np.array([[row["normalized_by_task"][task] for task in candidates.TASKS]
                         for row in rows])
    np.testing.assert_allclose(normalized, archived, rtol=0, atol=1e-12)
    np.testing.assert_allclose(normalized.mean(axis=1),
                               [row["equal_weight_score"] for row in rows], rtol=0, atol=1e-12)
    ids = [row["design_id"] for row in rows]
    np.testing.assert_array_equal(normalized[ids.index(roles["nominal"])], np.zeros(4))
    return rows, raw, normalized, front, roles, protocol, calibration


def create_figure(rows, normalized, roles, language):
    words = {**candidates.TEXT[language], **TEXT[language]}
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=140)
    fig.subplots_adjust(left=0.092, right=0.956, bottom=0.265, top=0.88)
    x = np.arange(4)
    ids = [row["design_id"] for row in rows]
    highlighted_ids = set(roles.values())
    for index, id_ in enumerate(ids):
        if id_ not in highlighted_ids:
            ax.plot(x, normalized[index], color=BACKGROUND_COLOR, linewidth=0.75,
                    alpha=0.33, zorder=1)
    for column in x:
        ax.axvline(column, color="#80949F", linewidth=0.85, zorder=2)
    for role, settings in ROLE_STYLE.items():
        is_compromise = role == "compromise"
        ax.plot(x, normalized[ids.index(roles[role])], **settings,
                linewidth=2.6 if is_compromise else 1.7,
                markersize=11 if is_compromise else 6.5,
                markerfacecolor="white", markeredgewidth=1.3,
                zorder=6 if is_compromise else 5 if role == "nominal" else 4)
    ax.set(xlim=(-0.15, 3.15), ylim=(-1.95, 1.7))
    if np.min(normalized) <= ax.get_ylim()[0] or np.max(normalized) >= ax.get_ylim()[1]:
        raise ValueError("Plot range must retain all candidate capabilities")
    ax.set_xticks(x, words["axis_labels"], fontsize=FONT_SIZE)
    ax.set_yticks(np.arange(-1.5, 1.51, 0.5))
    ax.set_ylabel(words["y_label"], fontsize=FONT_SIZE, labelpad=10)
    ax.set_title(words["title"], fontsize=FONT_SIZE, fontweight="normal", loc="left", pad=17)
    ax.tick_params(axis="x", length=0, pad=12, labelsize=FONT_SIZE, colors=style.INK)
    ax.tick_params(axis="y", length=3, labelsize=FONT_SIZE, colors=style.INK)
    ax.grid(axis="y", color="#E6ECEF", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#81949F")
    ax.spines["left"].set_linewidth(0.8)

    def handle(role):
        return Line2D([], [], **ROLE_STYLE[role], linewidth=2 if role == "compromise" else 1.7,
                      markersize=11 if role == "compromise" else 6.5,
                      markerfacecolor="white", markeredgewidth=1.3, label=words[role])

    main_handles = [Line2D([], [], color=BACKGROUND_COLOR, linewidth=1, label=words["other"]),
                    handle("nominal"), handle("compromise")]
    fig.legend(handles=main_handles, loc="lower center", bbox_to_anchor=(0.53, 0.095),
               ncol=3, frameon=False, fontsize=FONT_SIZE, columnspacing=2.5, handlelength=2.4)
    reference_handles = [handle(task + "_reference") for task in candidates.TASKS]
    fig.legend(handles=reference_handles, loc="lower center", bbox_to_anchor=(0.53, 0.025),
               ncol=4, frameon=False, fontsize=FONT_SIZE, columnspacing=2, handlelength=2.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            if not np.isclose(text.get_fontsize(), FONT_SIZE):
                raise ValueError("All figure text must use 14 pt")
            corners = text.get_window_extent(renderer).get_points()
            if not all(fig.bbox.contains(*corner) for corner in corners):
                raise ValueError("Text falls outside the export: " + text.get_text())
    return fig


def write_data(output, source, rows, raw, normalized, front, roles, protocol, calibration, dpi):
    fields = ["design_id", "phase", "role", *candidates.DESIGN_KEYS,
              "payload_kg", "uphill_speed_mps", "clearance_m", "jump_area_m2",
              *[task + "_normalized" for task in candidates.TASKS],
              "equal_weight_score", "nondominated_4d", "train_seed", "validation_seed", "test_seed"]
    with (output / "parallel_coordinates_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({
                "design_id": row["design_id"], "phase": row["phase"],
                "role": ";".join(role for role, id_ in roles.items() if id_ == row["design_id"]),
                **row["design"],
                **dict(zip(("payload_kg", "uphill_speed_mps", "clearance_m", "jump_area_m2"), raw[index])),
                **dict(zip((task + "_normalized" for task in candidates.TASKS), normalized[index])),
                "equal_weight_score": row["equal_weight_score"], "nondominated_4d": bool(front[index]),
                **{key: row[key] for key in ("train_seed", "validation_seed", "test_seed")},
            })
    ids = [row["design_id"] for row in rows]
    manifest = {
        "source_dir": str(source.resolve()),
        "source_files_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                for name in candidates.INPUT_FILES},
        "candidate_count": len(rows), "background_candidate_count": len(rows) - len(set(roles.values())),
        "axis_order": list(candidates.TASKS),
        "normalization": "(raw physical capability - fixed nominal search value) / fixed initial calibration span",
        "calibration": calibration,
        "normalization_clipped": False,
        "normalized_value_ranges": {task: [float(normalized[:, j].min()), float(normalized[:, j].max())]
                                    for j, task in enumerate(candidates.TASKS)},
        "highlighted_designs": {
            role: {"design_id": id_, "design": rows[ids.index(id_)]["design"],
                   "raw_by_task": rows[ids.index(id_)]["raw_by_task"],
                   "normalized_by_task": dict(zip(candidates.TASKS, normalized[ids.index(id_)]))}
            for role, id_ in roles.items()},
        "nominal_normalized_value": 0,
        "unit_value_meaning": "One fixed calibration span above nominal, not a universal maximum or a 100-percent gain",
        "statistics": "Same search-test point estimates as Figure 04 scatter; not formal three-seed means",
        "seeds": {key: rows[0][key] for key in ("train_seed", "validation_seed", "test_seed")},
        "inner_update_budget": protocol["inner_updates"],
        "objective_direction": "All four axes increase upward; higher is better",
        "gray_lines": "Other 42 candidates, NOT uncertainty bands and NOT an indication of dominance",
        "connecting_segments": "Visual links between a candidate's four task values, not interpolation or task transitions",
        "point_adjustments": "None: zero/failed/negative-normalized results retained; no smoothing, jitter or clipping",
        "scatter_figures": "Retained unchanged as a separate figure",
        "font_size_pt": FONT_SIZE, "raster_size_px": [round(dpi * size) for size in FIGURE_SIZE],
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=candidates.DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    rows, raw, normalized, front, roles, protocol, calibration = load_data(args.source_dir)
    corpus = "".join("".join(words.values()) for words in candidates.TEXT.values())
    corpus += "".join(value if isinstance(value, str) else "".join(value)
                      for words in TEXT.values() for value in words.values())
    style.configure_fonts(args.output_dir, extra_text=corpus)
    plt.rcParams.update({"font.size": FONT_SIZE, "font.weight": "normal",
                         "axes.labelweight": "normal", "axes.titleweight": "normal"})
    languages = ("zh", "en") if args.language == "both" else (args.language,)
    for language in languages:
        figure = create_figure(rows, normalized, roles, language)
        for extension in ("png", "pdf", "svg"):
            target = args.output_dir / f"capability_parallel_coordinates_{language}.{extension}"
            figure.savefig(target, dpi=args.dpi, facecolor="white")
            print(target)
        plt.close(figure)
    write_data(args.output_dir, args.source_dir, rows, raw, normalized, front, roles, protocol, calibration, args.dpi)
    print("Candidates:", len(rows), "highlighted:", len(roles), "background:", len(rows) - len(roles))
    print("PASS: normalized values and equal-weight scores match the archived BO records")


if __name__ == "__main__":
    main()
