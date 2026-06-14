#!/usr/bin/env python3
"""Export selected TensorBoard scalars for thesis training-curve plots.

Run with the training conda env, for example:
  conda run -n pointfoot_legged_gym python legged_gym/scripts/figs_ch4/export_training_scalars.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "wheelfoot_flat" / "WF_TRON1A"
DEFAULT_CH4_OUT_ROOT = DEFAULT_LOG_ROOT / "exported_paper" / "ch4_training_curves"
DEFAULT_CH5_OUT_ROOT = DEFAULT_LOG_ROOT / "exported_paper" / "ch5_training_curves"


PRESETS = {
    "ch4_narrow": OrderedDict(
        [
            (
                "Model-guided",
                ["exper_qs_resi_load_boost_3_seed_45_pemass"],
            ),
            (
                "Estimate-guided",
                ["exper_qs_noresi_load_boost_3_seed_45_pemass"],
            ),
            (
                "Source-guided",
                ["exper_history_only_load_boost_3_seed_45_pemass"],
            ),
            (
                "RL-only",
                [
                    "May24_19-30-38_exper_history_only_no_torq_load_boost_3_seed_45_pemass"
                ],
            ),
        ]
    ),
    "ch5_wide": OrderedDict(
        [
            (
                "Model-guided",
                [
                    "Jun04_23-18-34_wide2-30_model_guided_seed1",
                    "Jun07_09-20-20_wide2-30_model_guided_seed2",
                ],
            ),
            (
                "Estimate-guided",
                [
                    "Jun06_00-18-09_wide2-30_estimate_guided_seed1",
                    "Jun07_22-14-09_wide2-30_estimate_guided_seed2",
                ],
            ),
            (
                "Source-guided",
                [
                    "Jun05_17-50-01_wide2-30_source_guided_seed42",
                    "Jun07_00-14-20_wide2-30_source_guided_seed2",
                ],
            ),
            (
                "RL-only",
                [
                    "Jun05_03-01-38_wide2-30_rl_only_seed1",
                    "Jun09_00-29-29_wide2-30_rl_only_seed2",
                ],
            ),
        ]
    ),
}


METRICS = OrderedDict(
    [
        (
            "mean_reward",
            ["Train/mean_reward", "Episode/rew_total", "Episode/reward"],
        ),
        (
            "mean_episode_length",
            [
                "Train/mean_episode_length",
                "Episode/episode_length",
                "Episode/episode_length_s",
            ],
        ),
        (
            "payload_loss",
            [
                "Loss/encoder",
                "L_payload",
                "Loss/L_payload",
                "Loss/payload",
                "Metric/mass_mse",
            ],
        ),
    ]
)

OPTIONAL_METRICS = OrderedDict(
    [
        (
            "payload_mass_loss",
            ["Metric/mass_mse", "Metric/payload_mass_mse", "Metric/load_mass_mse"],
        ),
        (
            "payload_x_loss",
            [
                "Metric/com_x_mse",
                "Metric/load_x_mse",
                "Metric/payload_x_mse",
                "Metric/extra_loss_com_x",
                "extra_loss_com_x",
            ],
        ),
        (
            "payload_y_loss",
            [
                "Metric/com_y_mse",
                "Metric/load_y_mse",
                "Metric/payload_y_mse",
                "Metric/extra_loss_com_y",
                "extra_loss_com_y",
            ],
        ),
        ("mass_mse", ["Metric/mass_mse"]),
        ("com_mse", ["Metric/com_mse"]),
        ("vel_mse", ["Metric/vel_mse"]),
    ]
)


def safe_name(text: str) -> str:
    text = text.replace("/", "_").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "", text)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_event_accumulator(run_dir: Path):
    acc = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    acc.Reload()
    return acc


def pick_tag(tags: list[str], candidates: list[str]) -> str | None:
    for tag in candidates:
        if tag in tags:
            return tag
    return None


def add_run_rows(log_root: Path, variant: str, run_name: str, metric_map):
    run_dir = log_root / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")

    acc = load_event_accumulator(run_dir)
    tags = sorted(acc.Tags().get("scalars", []))
    env_cfg = load_json(run_dir / "env_cfg.json")
    train_cfg = load_json(run_dir / "train_cfg.json")
    load_range = (
        env_cfg.get("domain_rand", {}).get("add_load_range")
        if isinstance(env_cfg.get("domain_rand"), dict)
        else None
    )
    seed = train_cfg.get("seed", env_cfg.get("seed", ""))
    max_iterations = train_cfg.get("runner", {}).get("max_iterations", "")

    rows = []
    manifest_rows = []
    for metric, candidates in metric_map.items():
        tag = pick_tag(tags, candidates)
        if tag is None:
            manifest_rows.append(
                {
                    "variant": variant,
                    "run_name": run_name,
                    "metric": metric,
                    "tag": "",
                    "status": "missing",
                    "num_points": 0,
                    "min_step": "",
                    "max_step": "",
                    "seed": seed,
                    "max_iterations": max_iterations,
                    "load_range": json.dumps(load_range),
                }
            )
            continue

        events = acc.Scalars(tag)
        if events:
            min_step = min(e.step for e in events)
            max_step = max(e.step for e in events)
        else:
            min_step = ""
            max_step = ""

        manifest_rows.append(
            {
                "variant": variant,
                "run_name": run_name,
                "metric": metric,
                "tag": tag,
                "status": "ok",
                "num_points": len(events),
                "min_step": min_step,
                "max_step": max_step,
                "seed": seed,
                "max_iterations": max_iterations,
                "load_range": json.dumps(load_range),
            }
        )
        for e in events:
            rows.append(
                {
                    "variant": variant,
                    "run_name": run_name,
                    "metric": metric,
                    "tag": tag,
                    "wall_time": f"{e.wall_time:.6f}",
                    "step": e.step,
                    "value": f"{float(e.value):.10g}",
                }
            )

    return rows, manifest_rows, tags


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="ch4_narrow")
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also export diagnostic Metric/mass_mse, Metric/com_mse, and Metric/vel_mse.",
    )
    args = parser.parse_args()

    out_root = args.out_root
    if out_root is None:
        out_root = (
            DEFAULT_CH5_OUT_ROOT if args.preset == "ch5_wide" else DEFAULT_CH4_OUT_ROOT
        )

    metric_map = OrderedDict(METRICS)
    if args.include_optional:
        metric_map.update(OPTIONAL_METRICS)

    csv_root = out_root / args.preset / "csv"
    tag_root = out_root / args.preset / "tags"
    all_rows = []
    manifest_rows = []

    for variant, runs in PRESETS[args.preset].items():
        for run_name in runs:
            rows, run_manifest, tags = add_run_rows(
                args.log_root, variant, run_name, metric_map
            )
            all_rows.extend(rows)
            manifest_rows.extend(run_manifest)

            tag_path = tag_root / f"{safe_name(run_name)}_tags.txt"
            tag_path.parent.mkdir(parents=True, exist_ok=True)
            tag_path.write_text("\n".join(tags) + "\n", encoding="utf-8")

    long_fields = [
        "variant",
        "run_name",
        "metric",
        "tag",
        "wall_time",
        "step",
        "value",
    ]
    manifest_fields = [
        "variant",
        "run_name",
        "metric",
        "tag",
        "status",
        "num_points",
        "min_step",
        "max_step",
        "seed",
        "max_iterations",
        "load_range",
    ]
    write_csv(csv_root / "all_scalars_long.csv", all_rows, long_fields)
    write_csv(csv_root / "manifest.csv", manifest_rows, manifest_fields)

    for metric in metric_map:
        metric_rows = [r for r in all_rows if r["metric"] == metric]
        write_csv(csv_root / f"{metric}.csv", metric_rows, long_fields)
        for variant in PRESETS[args.preset]:
            variant_rows = [r for r in metric_rows if r["variant"] == variant]
            write_csv(
                csv_root / metric / f"{safe_name(variant)}.csv",
                variant_rows,
                long_fields,
            )

    print(f"[export_training_scalars] preset={args.preset}")
    print(f"[export_training_scalars] csv={csv_root}")
    print(f"[export_training_scalars] manifest={csv_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
