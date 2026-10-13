#!/usr/bin/env python3
"""Extend speed-policy command coverage until measured gains converge.

Each round trains two equal-budget curriculum branches and validates all
intermediate checkpoints on the same three mechanical designs. Once the
matched-design gain is negligible for two consecutive extensions, or every
extended checkpoint is worse than the base policy, one fresh six-dimensional
BO and three-seed formal evaluation is run with the selected frozen policy.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(
    os.environ.get(
        "SPEED_LIMIT_ROOT",
        "/home/ps/mydrive/xu/pointfoot-legged-gym",
    )
)
PYTHON = os.environ.get(
    "SPEED_LIMIT_PYTHON",
    "/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python",
)
OUTPUT = Path(
    os.environ.get(
        "SPEED_LIMIT_OUT",
        "/home/ps/mydrive/xu/codesign_runs/"
        "sixd_speed_limit_convergence_v1_2026-07-26",
    )
)
BASE_RUN = os.environ.get(
    "SPEED_LIMIT_BASE_RUN",
    "Jul26_18-04-36_physadapt_sixdgen_speed_ext_"
    "wide_stage2_3p8_s20260726_u200",
)
BASE_CHECKPOINT = int(
    os.environ.get("SPEED_LIMIT_BASE_CKPT", "28700")
)
BASE_COMMAND_MAX = float(
    os.environ.get("SPEED_LIMIT_BASE_COMMAND_MAX", "4.6")
)
ROUND_STEP_MPS = float(
    os.environ.get("SPEED_LIMIT_ROUND_STEP_MPS", "0.8")
)
MAX_ROUNDS = int(os.environ.get("SPEED_LIMIT_MAX_ROUNDS", "6"))
STAGE_UPDATES = int(
    os.environ.get("SPEED_LIMIT_STAGE_UPDATES", "200")
)
ABS_GAIN_THRESHOLD_MPS = float(
    os.environ.get("SPEED_LIMIT_ABS_GAIN_MPS", "0.05")
)
REL_GAIN_THRESHOLD = float(
    os.environ.get("SPEED_LIMIT_REL_GAIN", "0.01")
)
SMALL_GAIN_ROUNDS = int(
    os.environ.get("SPEED_LIMIT_SMALL_GAIN_ROUNDS", "2")
)
GPUS = os.environ.get("SPEED_LIMIT_GPUS", "0,1")

VALIDATION_DESIGNS = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [
        0.8538561291992665,
        0.830804542452097,
        0.6316521329805255,
        0.6768537418916821,
        0.8832088768482208,
        1.5876056658104063,
    ],
    [
        0.845273,
        0.843658,
        0.716739,
        0.658101,
        0.760218,
        1.597946,
    ],
]


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def command_levels(maximum):
    span = float(maximum) - 0.5
    levels = [
        round(0.5 + span * index / 7.0, 2)
        for index in range(8)
    ]
    levels[-1] = round(float(maximum), 2)
    if any(right <= left for left, right in zip(levels, levels[1:])):
        raise ValueError(f"invalid curriculum levels: {levels}")
    return levels


def comma_values(values):
    return ",".join(f"{float(value):g}" for value in values)


def run_child(name, environment, directory):
    directory.mkdir(parents=True, exist_ok=True)
    console_path = directory / "driver_console.log"
    child_environment = os.environ.copy()
    child_environment.update(environment)
    started = time.time()
    with console_path.open("a", encoding="utf-8") as console:
        console.write(f"\n[{now_iso()}] START {name}\n")
        console.flush()
        process = subprocess.run(
            [
                PYTHON,
                "legged_gym/scripts/"
                "run_sixd_speed_extended_search.py",
            ],
            cwd=str(ROOT),
            env=child_environment,
            stdout=console,
            stderr=subprocess.STDOUT,
            check=False,
        )
        console.write(
            f"[{now_iso()}] EXIT {process.returncode} "
            f"elapsed_s={time.time() - started:.1f}\n"
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {process.returncode}; "
            f"see {console_path}"
        )
    return time.time() - started


def round_environment(
    output,
    source,
    base_run,
    base_checkpoint,
    base_command_max,
    target_command_max,
):
    first_max = base_command_max + 0.40 * ROUND_STEP_MPS
    second_max = base_command_max + 0.75 * ROUND_STEP_MPS
    return {
        "SPEED_EXT_SOURCE": str(source),
        "SPEED_EXT_OUT": str(output),
        "SPEED_EXT_BASE_RUN": str(base_run),
        "SPEED_EXT_BASE_CKPT": str(base_checkpoint),
        "SPEED_EXT_BASE_COMMAND_MAX": f"{base_command_max:g}",
        "SPEED_EXT_STAGE_UPDATES": str(STAGE_UPDATES),
        "SPEED_EXT_SELECTION_ONLY": "1",
        "SPEED_EXT_VALIDATION_DESIGNS_JSON": json.dumps(
            VALIDATION_DESIGNS
        ),
        "SPEED_EXT_GRADUAL_STAGE1_LEVELS": comma_values(
            command_levels(first_max)
        ),
        "SPEED_EXT_DIRECT_STAGE1_LEVELS": comma_values(
            command_levels(second_max)
        ),
        "SPEED_EXT_GRADUAL_STAGE2_LEVELS": comma_values(
            command_levels(second_max)
        ),
        "SPEED_EXT_WIDE_STAGE2_LEVELS": comma_values(
            command_levels(target_command_max)
        ),
        "SIXD_GPUS": GPUS,
    }


def policy_row(selection, label):
    for row in selection["policies"]:
        if row["label"] == label:
            return row
    raise KeyError(f"missing policy row {label}")


def write_report(history, stop_reason, final):
    lines = [
        "# Stable actual-speed convergence",
        "",
        f"Stop reason: `{stop_reason}`.",
        "",
        "| round | base max | target max | selected max | "
        "base mean | selected mean | gain | relative gain | "
        "right-censored designs |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in history:
        lines.append(
            f"| {row['round']} | {row['base_command_max_mps']:.2f} | "
            f"{row['target_command_max_mps']:.2f} | "
            f"{row['selected_command_max_mps']:.2f} | "
            f"{row['base_mean_mps']:.5f} | "
            f"{row['selected_mean_mps']:.5f} | "
            f"{row['absolute_gain_mps']:.5f} | "
            f"{100.0 * row['relative_gain']:.3f}% | "
            f"{row['right_censored_count']}/"
            f"{row['validation_design_count']} |"
        )
    winner = final["speed"]["winner"]
    lines.extend(
        [
            "",
            "## Final formal result",
            "",
            f"- Design: `{winner['design_vector']}`",
            f"- Stable actual speed: "
            f"`{winner['objective_mean']:.6f} +/- "
            f"{winner['objective_std']:.6f} m/s`",
            f"- Commands by seed: "
            f"`{winner['command_at_metric_by_seed']}`",
            f"- Right-censored seeds: "
            f"`{winner['right_censored_by_training_domain_by_seed']}`",
        ]
    )
    (OUTPUT / "FINAL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "STARTED").write_text(now_iso() + "\n", encoding="ascii")
    (OUTPUT / "DONE").unlink(missing_ok=True)
    (OUTPUT / "EXIT_CODE").unlink(missing_ok=True)
    atomic_json(
        OUTPUT / "RUN_MANIFEST.json",
        {
            "started_at": now_iso(),
            "base_policy": {
                "run_name": BASE_RUN,
                "checkpoint": BASE_CHECKPOINT,
                "training_command_max_mps": BASE_COMMAND_MAX,
            },
            "round_step_mps": ROUND_STEP_MPS,
            "max_rounds": MAX_ROUNDS,
            "stage_updates": STAGE_UPDATES,
            "absolute_gain_threshold_mps": ABS_GAIN_THRESHOLD_MPS,
            "relative_gain_threshold": REL_GAIN_THRESHOLD,
            "required_consecutive_small_gain_rounds": SMALL_GAIN_ROUNDS,
            "validation_designs": VALIDATION_DESIGNS,
            "gpus": GPUS,
        },
    )

    base_run = BASE_RUN
    base_checkpoint = BASE_CHECKPOINT
    base_command_max = BASE_COMMAND_MAX
    source = (
        "/home/ps/mydrive/xu/codesign_runs/"
        "sixd_speed_extended_v2_2026-07-26"
    )
    history = []
    small_gain_streak = 0
    stop_reason = None

    for round_index in range(1, MAX_ROUNDS + 1):
        target = round(base_command_max + ROUND_STEP_MPS, 2)
        round_dir = (
            OUTPUT
            / f"round_{round_index:02d}_to_"
            f"{target:g}".replace(".", "p")
        )
        atomic_json(
            OUTPUT / "STATUS.json",
            {
                "updated_at": now_iso(),
                "stage": "training_and_checkpoint_validation",
                "round": round_index,
                "base_command_max_mps": base_command_max,
                "target_command_max_mps": target,
                "small_gain_streak": small_gain_streak,
            },
        )
        elapsed = run_child(
            f"convergence-round-{round_index}",
            round_environment(
                round_dir,
                source,
                base_run,
                base_checkpoint,
                base_command_max,
                target,
            ),
            round_dir,
        )
        selection = json.loads(
            (round_dir / "CHECKPOINT_SELECTION.json").read_text()
        )
        selected = selection["selected"]
        base_row = policy_row(selection, "base_policy")
        gain = (
            float(selected["mean_objective"])
            - float(base_row["mean_objective"])
        )
        relative_gain = gain / max(
            float(base_row["mean_objective"]), 1.0e-9
        )
        selected_extends_domain = (
            float(selected["training_command_max_mps"])
            > base_command_max + 1.0e-9
        )
        small_gain = (
            gain < ABS_GAIN_THRESHOLD_MPS
            and relative_gain < REL_GAIN_THRESHOLD
        )
        small_gain_streak = (
            small_gain_streak + 1 if small_gain else 0
        )
        row = {
            "round": round_index,
            "elapsed_s": elapsed,
            "base_run": base_run,
            "base_checkpoint": base_checkpoint,
            "base_command_max_mps": base_command_max,
            "target_command_max_mps": target,
            "base_mean_mps": float(base_row["mean_objective"]),
            "selected_run": selected["run_name"],
            "selected_checkpoint": int(selected["checkpoint"]),
            "selected_command_max_mps": float(
                selected["training_command_max_mps"]
            ),
            "selected_mean_mps": float(selected["mean_objective"]),
            "selected_minimum_mps": float(
                selected["minimum_objective"]
            ),
            "absolute_gain_mps": gain,
            "relative_gain": relative_gain,
            "small_gain": small_gain,
            "small_gain_streak": small_gain_streak,
            "selected_extends_domain": selected_extends_domain,
            "right_censored_count": int(
                selected["right_censored_count"]
            ),
            "validation_design_count": int(
                selected["design_count"]
            ),
            "round_directory": str(round_dir),
        }
        history.append(row)
        atomic_json(OUTPUT / "CONVERGENCE_HISTORY.json", history)

        if not selected_extends_domain:
            stop_reason = "all_extended_checkpoints_no_better_than_base"
            break

        base_run = selected["run_name"]
        base_checkpoint = int(selected["checkpoint"])
        base_command_max = float(
            selected["training_command_max_mps"]
        )
        source = round_dir

        if small_gain_streak >= SMALL_GAIN_ROUNDS:
            stop_reason = "two_consecutive_negligible_gain_rounds"
            break

    if stop_reason is None:
        stop_reason = "maximum_round_guard_reached"

    final_dir = OUTPUT / "final_search"
    atomic_json(
        OUTPUT / "STATUS.json",
        {
            "updated_at": now_iso(),
            "stage": "final_sixd_search",
            "stop_reason": stop_reason,
            "selected_run": base_run,
            "selected_checkpoint": base_checkpoint,
            "selected_command_max_mps": base_command_max,
        },
    )
    final_environment = {
        "SPEED_EXT_SOURCE": str(source),
        "SPEED_EXT_OUT": str(final_dir),
        "SPEED_EXT_FROZEN_RUN": str(base_run),
        "SPEED_EXT_FROZEN_CKPT": str(base_checkpoint),
        "SPEED_EXT_FROZEN_COMMAND_MAX": f"{base_command_max:g}",
        "SPEED_EXT_VALIDATION_DESIGNS_JSON": json.dumps(
            VALIDATION_DESIGNS
        ),
        "SIXD_GPUS": GPUS,
    }
    run_child("final-sixd-search", final_environment, final_dir)
    final = json.loads((final_dir / "FINAL_RESULTS.json").read_text())
    shutil.copy2(
        final_dir / "FINAL_RESULTS.json",
        OUTPUT / "FINAL_RESULTS.json",
    )
    shutil.copy2(
        final_dir / "FINAL_RESULTS.csv",
        OUTPUT / "FINAL_RESULTS.csv",
    )
    atomic_json(
        OUTPUT / "FINAL_SELECTION.json",
        {
            "stop_reason": stop_reason,
            "run_name": base_run,
            "checkpoint": base_checkpoint,
            "training_command_max_mps": base_command_max,
            "convergence_round_count": len(history),
        },
    )
    write_report(history, stop_reason, final)
    atomic_json(
        OUTPUT / "LIFECYCLE.json",
        {
            "completed_at": now_iso(),
            "stop_reason": stop_reason,
            "round_count": len(history),
            "final_search_directory": str(final_dir),
        },
    )
    atomic_json(
        OUTPUT / "STATUS.json",
        {
            "updated_at": now_iso(),
            "stage": "complete",
            "stop_reason": stop_reason,
            "round_count": len(history),
            "eta_seconds": 0.0,
        },
    )
    (OUTPUT / "EXIT_CODE").write_text("0\n", encoding="ascii")
    (OUTPUT / "DONE").write_text(now_iso() + "\n", encoding="ascii")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "EXIT_CODE").write_text("1\n", encoding="ascii")
        raise
