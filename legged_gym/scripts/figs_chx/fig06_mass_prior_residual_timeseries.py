#!/usr/bin/env python3
"""Plot recorded mass estimates and reconstruct the model-guided residual head."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.io import loadmat

import fig0_sixd_design_variables as style


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE_NAME = "play_data_20260706-102219_A2_mg.mat"
DEFAULT_SOURCE = (HERE / "raw_source.mat" if (HERE / "raw_source.mat").is_file()
                  else REPO / "logs/wheelfoot_flat/WF_TRON1A/exported" / SOURCE_NAME)
DEFAULT_OUTPUT = HERE / "out/mass_prior_residual_timeseries"
FONT_SIZE = 14
FIGURE_SIZE = (10.8, 6.4)
TRUE_COLOR = "#222222"
PRIOR_COLOR = "#487FB7"
FUSED_COLOR = "#BE583C"
TEXT = {
    "zh": {
        "mass_title": "(a)  质量估计",
        "residual_title": "(b)  残差修正",
        "truth": "真实质量",
        "prior": "解析估计",
        "fused": "融合估计",
        "target_residual": "目标残差（真实−解析）",
        "predicted_residual": "网络预测残差",
        "mass_axis": "负载质量 (kg)",
        "residual_axis": "质量残差 (kg)",
        "time_axis": "时间 (s)",
        "load": "加载",
        "unload": "卸载",
        "event": "负载状态变化",
    },
    "en": {
        "mass_title": "(a)  Mass estimates",
        "residual_title": "(b)  Residual correction",
        "truth": "True mass",
        "prior": "Analytical prior",
        "fused": "Fused estimate",
        "target_residual": "Target residual (true minus prior)",
        "predicted_residual": "Predicted residual",
        "mass_axis": "Payload mass (kg)",
        "residual_axis": "Mass residual (kg)",
        "time_axis": "Time (s)",
        "load": "Load",
        "unload": "Unload",
        "event": "Payload state change",
    },
}


def load_record(source, env_cfg, env_index):
    data = loadmat(source, simplify_cells=True)
    meta = data["meta"]
    if not bool(meta.get("use_load_residual_estimation", False)):
        raise ValueError("This record does not use residual-based mass estimation")
    if str(meta.get("corrupt_estimate", "none")) != "none":
        raise ValueError("Corrupted estimates cannot be treated as raw fused predictions")
    if env_cfg is None:
        archived = HERE / "source_snapshot/env_cfg.json"
        env_cfg = (archived if archived.is_file() else
                   REPO / "logs/wheelfoot_flat/WF_TRON1A" / meta["load_run"] / "env_cfg.json")
    config = json.loads(env_cfg.read_text(encoding="utf-8"))
    threshold = float(config["asset"]["load_estimation_position_zero_mass_threshold"])
    fields = ("payload_mass_ref_all", "payload_mass_all", "encoder_mass_all")
    arrays = [np.asarray(data[name], dtype=np.float64) for name in fields]
    if any(array.ndim != 2 or array.shape != arrays[0].shape for array in arrays):
        raise ValueError("Expected matching time-by-environment matrices")
    if not 0 <= env_index < arrays[0].shape[1]:
        raise ValueError("Environment index is outside the recorded matrix")
    dt = float(data["dt"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Record has invalid sampling period")
    truth, raw_prior, fused = [array[:, env_index] for array in arrays]
    if not all(np.isfinite(array).all() for array in (truth, raw_prior, fused)):
        raise ValueError("Selected environment contains non-finite estimates")

    # PPO adds the payload-present-gated QS baseline, not the ungated mass signal.
    prior = np.where(raw_prior >= threshold, raw_prior, 0.0)
    mass_scale = float(config["normalization"]["obs_scales"]["mass_scale"])
    obs_clip = float(config["normalization"]["clip_observations"])
    if np.max(np.abs(prior * mass_scale)) > obs_clip:
        raise ValueError("The prior hits observation clipping; subtracting it cannot recover the network residual")
    predicted = fused - prior
    target = truth - prior
    if not np.allclose(prior + predicted, fused, rtol=0, atol=1e-10):
        raise ValueError("Residual reconstruction is inconsistent with the fused estimate")
    time = np.arange(truth.size) * dt
    loaded = truth > threshold
    changes = np.flatnonzero(loaded[1:] != loaded[:-1]) + 1
    events = [{
        "sample_index": int(index), "time_s": float(time[index]),
        "type": "load" if loaded[index] else "unload",
        "mass_before_kg": float(truth[index - 1]),
        "mass_after_kg": float(truth[index]),
    } for index in changes]
    signals = {
        "time_s": time,
        "true_mass_kg": truth,
        "analytical_mass_ungated_kg": raw_prior,
        "analytical_prior_kg": prior,
        "fused_mass_kg": fused,
        "target_residual_kg": target,
        "predicted_residual_kg": predicted,
        "load_on_body": loaded.astype(int),
    }
    return signals, events, meta, config, env_cfg, dt, arrays[0].shape[1]


def window_mask(signals, time_range):
    time = signals["time_s"]
    mask = (time >= time_range[0] - 1e-6) & (time <= time_range[1] + 1e-6)
    if mask.sum() < 2:
        raise ValueError("Display window contains too few samples")
    return mask


def data_limits(series):
    values = np.concatenate(series)
    low, high = min(0.0, float(values.min())), max(0.0, float(values.max()))
    margin = max(1.0, (high - low) * 0.17)
    return low - margin, high + margin


def draw_events(axes, signals, events, time_range, words):
    for event in events:
        time = event["time_s"]
        if not time_range[0] <= time <= time_range[1]:
            continue
        for ax in axes:
            ax.axvline(time, color="#73818A", linewidth=1.0,
                       linestyle=(0, (3, 3)), zorder=1)
    shown = [event for event in events if time_range[0] <= event["time_s"] <= time_range[1]]
    # Label isolated transitions; tightly spaced contact toggles remain marked by lines.
    for index, event in enumerate(shown):
        distances = [abs(event["time_s"] - other["time_s"])
                     for j, other in enumerate(shown) if j != index]
        if distances and min(distances) < (time_range[1] - time_range[0]) * 0.065:
            continue
        axes[0].text(event["time_s"], 0.98, words[event["type"]],
                     transform=axes[0].get_xaxis_transform(), fontsize=FONT_SIZE,
                     color="#52616A", ha="center", va="top", zorder=5)


def create_figure(language, signals, events, time_range):
    words = TEXT[language]
    mask = window_mask(signals, time_range)
    time = signals["time_s"][mask]
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, dpi=140, sharex=True)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.115, top=0.89, hspace=0.43)
    top, bottom = axes
    top.plot(time, signals["analytical_prior_kg"][mask], color=PRIOR_COLOR,
             linewidth=1.45, linestyle=(0, (4, 2)), label=words["prior"], zorder=3)
    top.plot(time, signals["fused_mass_kg"][mask], color=FUSED_COLOR,
             linewidth=1.9, label=words["fused"], zorder=4)
    top.plot(time, signals["true_mass_kg"][mask], color=TRUE_COLOR,
             linewidth=2.0, label=words["truth"], zorder=4)
    bottom.plot(time, signals["target_residual_kg"][mask], color=TRUE_COLOR,
                linewidth=1.45, linestyle=(0, (4, 2)), label=words["target_residual"], zorder=3)
    bottom.plot(time, signals["predicted_residual_kg"][mask], color=FUSED_COLOR,
                linewidth=1.8, label=words["predicted_residual"], zorder=4)
    bottom.axhline(0, color="#A8B3BA", linewidth=0.8, zorder=1)

    for ax, title in zip(axes, ("mass_title", "residual_title")):
        ax.set_xlim(time_range)
        ax.text(0, 1.19, words[title], transform=ax.transAxes,
                fontsize=FONT_SIZE, va="bottom", color=style.INK)
        ax.tick_params(labelsize=FONT_SIZE, colors=style.MUTED, length=4, width=0.8)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="y", color="#DEE5E9", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for name in ("left", "bottom"):
            ax.spines[name].set_color("#8A979F")
            ax.spines[name].set_linewidth(0.8)
    top.set_ylim(data_limits([signals[key][mask] for key in
                              ("true_mass_kg", "analytical_prior_kg", "fused_mass_kg")]))
    bottom.set_ylim(data_limits([signals[key][mask] for key in
                                 ("target_residual_kg", "predicted_residual_kg")]))
    top.set_ylabel(words["mass_axis"], fontsize=FONT_SIZE, labelpad=7)
    bottom.set_ylabel(words["residual_axis"], fontsize=FONT_SIZE, labelpad=7)
    bottom.set_xlabel(words["time_axis"], fontsize=FONT_SIZE, labelpad=7)
    bottom.xaxis.set_major_locator(MaxNLocator(6))
    top_handles, top_labels = top.get_legend_handles_labels()
    order = (2, 0, 1)
    top.legend([top_handles[i] for i in order], [top_labels[i] for i in order],
               loc="lower right", bbox_to_anchor=(1, 1.065), ncol=3,
               frameon=False, fontsize=FONT_SIZE, handlelength=2.0,
               columnspacing=1.4, borderaxespad=0)
    bottom.legend(loc="lower right", bbox_to_anchor=(1, 1.065), ncol=2,
                  frameon=False, fontsize=FONT_SIZE, handlelength=2.0,
                  columnspacing=1.4, borderaxespad=0)
    draw_events(axes, signals, events, time_range, words)
    return fig


def write_csv(path, signals, mask):
    keys = list(signals)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        writer.writerows(zip(*(signals[key][mask] for key in keys)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mat", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--env-cfg", type=Path)
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument("--time-range", type=float, nargs=2, default=(10, 20))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = "".join(value for language in TEXT.values() for value in language.values())
    style.configure_fonts(args.output_dir, extra_text=corpus)
    signals, events, meta, cfg, cfg_path, dt, num_envs = load_record(
        args.input_mat, args.env_cfg, args.env_index
    )
    languages = ("zh", "en") if args.language == "both" else (args.language,)
    views = {"main": args.time_range, "full": (0.0, 40.0)}
    metrics = {}
    for view, bounds in views.items():
        mask = window_mask(signals, bounds)
        write_csv(args.output_dir / f"timeseries_{view}.csv", signals, mask)
        truth = signals["true_mass_kg"][mask]
        metrics[view] = {
            "time_range_s": list(bounds),
            "samples": int(mask.sum()),
            "prior_rmse_kg": float(np.sqrt(np.mean((signals["analytical_prior_kg"][mask] - truth) ** 2))),
            "fused_rmse_kg": float(np.sqrt(np.mean((signals["fused_mass_kg"][mask] - truth) ** 2))),
        }
        for language in languages:
            fig = create_figure(language, signals, events, bounds)
            suffix = "" if view == "main" else "_full"
            for extension in ("png", "pdf", "svg"):
                path = args.output_dir / f"mass_prior_residual{suffix}_{language}.{extension}"
                fig.savefig(path, facecolor="white", dpi=args.dpi)
                print(path)
            plt.close(fig)
    (args.output_dir / "events.json").write_text(json.dumps(events, indent=2) + "\n")
    summary = {
        "source_mat": str(args.input_mat.resolve()),
        "source_mat_sha256": hashlib.sha256(args.input_mat.read_bytes()).hexdigest(),
        "environment_index_zero_based": args.env_index,
        "recorded_environment_count": num_envs,
        "environment_selection": "Environment 0 fixed before comparing estimator errors",
        "display_window": "Second nominal payload cycle, 10-20 s; complete 0-40 s record also exported",
        "dt_s": dt,
        "run": meta["load_run"],
        "checkpoint": meta["checkpoint"],
        "test_seed": meta.get("verify", {}).get("play_cfg", {}).get("seed"),
        "forward_command_mps": meta["cmd_vx"],
        "load_mass_range_kg": np.asarray(meta["load_mass_range_used"]).tolist(),
        "prior_threshold_kg": cfg["asset"]["load_estimation_position_zero_mass_threshold"],
        "source_config": str(cfg_path.resolve()),
        "truth_field": "payload_mass_ref_all[:, env_index]",
        "raw_analytical_field": "payload_mass_all[:, env_index]",
        "fusion_field": "encoder_mass_all[:, env_index]",
        "prior_definition": "where(analytical mass >= payload-present threshold, analytical mass, 0)",
        "target_residual": "true mass minus the gated analytical prior",
        "predicted_residual": "fused mass minus the same gated analytical prior",
        "prediction_reconstruction": "Exact additive-fusion decomposition, not a separately saved raw-head signal",
        "timestamp_alignment": "Historical play.py recomputes fused output after env.step, matching analytical estimate timestamp",
        "event_definition": "Transitions of the recorded payload-on-body mass, not the planned load schedule",
        "processing": "No additional smoothing, lag shifting, sample omission within either display window, clipping, or outlier replacement",
        "prior_observation_clipping": "Verified inactive in the selected environment over the complete recorded interval",
        "residual_baseline_observation_noise": "Zero according to the archived environment implementation",
        "metrics": metrics,
        "font_size_pt": FONT_SIZE,
        "raster_size_px": [round(size * args.dpi) for size in FIGURE_SIZE],
    }
    (args.output_dir / "figure_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RMSE summary:", json.dumps(metrics))


if __name__ == "__main__":
    main()
