#!/usr/bin/env python3
"""Resume the uphill power study after the currently running training stage.

The driver waits for checkpoint selection, runs the frozen calibration gate,
performs one equal-budget retraining attempt if needed, and launches BO only
after a passing calibration.
"""

import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(
    os.environ.get(
        "POWER_PIPE_ROOT",
        "/home/ps/mydrive/xu/pointfoot-legged-gym",
    )
)
PYTHON = os.environ.get(
    "POWER_PIPE_PYTHON",
    "/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python",
)
OUTPUT = Path(
    os.environ.get(
        "POWER_PIPE_OUT",
        "/home/ps/mydrive/xu/codesign_runs/"
        "sixd_uphill_speed_power_v2_pipeline_2026-07-27",
    )
)
INITIAL_TRAINING = Path(
    os.environ.get(
        "POWER_PIPE_INITIAL_TRAINING",
        "/home/ps/mydrive/xu/codesign_runs/"
        "sixd_uphill_speed_power_v2_training_2026-07-27",
    )
)
GPUS = os.environ.get("POWER_PIPE_GPUS", "0,1")
POLL_SECONDS = float(os.environ.get("POWER_PIPE_POLL_SECONDS", "30"))
WAIT_TIMEOUT_SECONDS = float(
    os.environ.get("POWER_PIPE_WAIT_TIMEOUT_SECONDS", str(3 * 3600))
)
RETRY_UPDATES = int(os.environ.get("POWER_PIPE_RETRY_UPDATES", "400"))
RETRY_ENABLED = os.environ.get(
    "POWER_PIPE_RETRY_ENABLED", "1"
).strip().lower() in ("1", "true", "yes", "on")
NOMINAL_SPEED_GATE_ENABLED = os.environ.get(
    "POWER_PIPE_NOMINAL_SPEED_GATE_ENABLED", "1"
).strip().lower() in ("1", "true", "yes", "on")
CALIBRATION_PROTOCOL = os.environ.get(
    "POWER_PIPE_CALIBRATION_PROTOCOL",
    "sixd_uphill_speed_power_v2_calibration_2026-07-27",
)
GATE_REVISION_REASON = os.environ.get(
    "POWER_PIPE_GATE_REVISION_REASON", ""
).strip()
STEP_COMPONENT = float(
    os.environ.get("POWER_PIPE_STEP_COMPONENT", "0.40")
)
if STEP_COMPONENT < 0.0:
    raise ValueError("POWER_PIPE_STEP_COMPONENT must be non-negative")
POWER_ENVIRONMENT = {
    "PHYS_MOTOR_POWER_ENVELOPE": "1",
    "PHYS_MOTOR_SPECIFIC_POWER_W_PER_KG": "450",
    "PHYS_MOTOR_POWER_EXPONENT": "1",
    "PHYS_MOTOR_POWER_MIN_SPEED_RAD_S": "0.001",
    "PHYS_EVAL_STEP_COMPONENT": str(STEP_COMPONENT),
    "PHYS_TRAIN_SPEED_STEP_COMPONENT": str(STEP_COMPONENT),
}
VALIDATION_DESIGNS = [
    [1, 1, 1, 1, 1, 0.6],
    [1, 1, 1, 1, 1, 1.0],
    [1, 1, 1, 1, 1, 1.6],
]
FORMAL_CONTROLS = [
    [1, 1, 1, 1, 1, 1],
    [0.9, 0.9, 1, 1, 1, 1],
    [
        0.9165454473346472,
        0.8298767622560264,
        0.807666621543467,
        1.1777029590681196,
        0.9549397187307477,
        1.5218089999631048,
    ],
    [1, 1, 1, 1, 1, 0.6],
    [1, 1, 1, 1, 1, 1.6],
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


def update_status(**values):
    status_path = OUTPUT / "STATUS.json"
    current = {}
    if status_path.is_file():
        current = json.loads(status_path.read_text(encoding="utf-8"))
    current.update(values)
    current["updated_at"] = now_iso()
    atomic_json(status_path, current)


def child_environment(extra):
    environment = os.environ.copy()
    environment.update(POWER_ENVIRONMENT)
    environment.update(extra)
    environment["PATH"] = (
        str(Path(PYTHON).parent)
        + os.pathsep
        + environment.get("PATH", "")
    )
    environment["PYTHONUNBUFFERED"] = "1"
    environment["ROBOT_TYPE"] = "WF_TRON1A"
    environment["SIXD_GPUS"] = GPUS
    return environment


def run_child(label, script, environment):
    log_path = OUTPUT / f"{label}.log"
    update_status(stage=label, child_log=str(log_path))
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now_iso()}] START {label}\n")
        log.flush()
        process = subprocess.run(
            [PYTHON, str(ROOT / script)],
            cwd=str(ROOT),
            env=child_environment(environment),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        elapsed = time.time() - started
        log.write(
            f"[{now_iso()}] EXIT {process.returncode} "
            f"elapsed_s={elapsed:.1f}\n"
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {process.returncode}; "
            f"see {log_path}"
        )
    return elapsed


def wait_for_training(training_dir):
    started = time.time()
    update_status(
        stage="waiting_for_initial_training",
        waiting_for=str(training_dir / "DONE"),
    )
    while True:
        exit_path = training_dir / "EXIT_CODE"
        if exit_path.is_file():
            exit_code = int(exit_path.read_text().strip())
            if exit_code != 0:
                raise RuntimeError(
                    f"training failed with exit code {exit_code}: "
                    f"{training_dir}"
                )
        if (training_dir / "DONE").is_file():
            selection = training_dir / "SELECTED_POLICY.json"
            if not selection.is_file():
                raise RuntimeError(
                    f"training completed without {selection}"
                )
            return selection
        elapsed = time.time() - started
        if elapsed > WAIT_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"timed out waiting for training: {training_dir}"
            )
        update_status(
            stage="waiting_for_initial_training",
            wait_elapsed_s=elapsed,
        )
        time.sleep(POLL_SECONDS)


def run_calibration(round_index, selection_path):
    calibration_dir = OUTPUT / f"calibration_round_{round_index}"
    run_child(
        f"calibration_round_{round_index}",
        "legged_gym/scripts/run_uphill_speed_power_calibration.py",
        {
            "POWER_CAL_OUT": str(calibration_dir),
            "POWER_CAL_SELECTION": str(selection_path),
            "POWER_CAL_SEEDS": "20260727,20260728,20260729",
            "POWER_CAL_COMMAND_MIN": "0.5",
            "POWER_CAL_COMMAND_MAX": "5.0",
            "POWER_CAL_COMMAND_STEP": "0.2",
            "POWER_CAL_SLOPE_DEG": "10",
            "POWER_CAL_SLOPE_LENGTH": "110",
            "POWER_CAL_NOMINAL_SPEED_GATE_ENABLED": (
                "1" if NOMINAL_SPEED_GATE_ENABLED else "0"
            ),
            "POWER_CAL_PROTOCOL": CALIBRATION_PROTOCOL,
            "POWER_CAL_GATE_REVISION_REASON": GATE_REVISION_REASON,
        },
    )
    report_path = calibration_dir / "CALIBRATION.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report, report_path


def run_retry(selection):
    retry_dir = OUTPUT / "training_retry"
    run_child(
        "training_retry",
        "legged_gym/scripts/run_sixd_speed_extended_search.py",
        {
            "SPEED_EXT_SOURCE": str(INITIAL_TRAINING),
            "SPEED_EXT_OUT": str(retry_dir),
            "SPEED_EXT_BASE_RUN": selection["run_name"],
            "SPEED_EXT_BASE_CKPT": str(selection["checkpoint"]),
            "SPEED_EXT_BASE_COMMAND_MAX": str(
                selection["training_command_max_mps"]
            ),
            "SPEED_EXT_BASE_TRAINING_SLOPE_DEG": "10",
            "SPEED_EXT_FIXED_SLOPE_DEG": "10",
            "SPEED_EXT_STAGE_UPDATES": str(RETRY_UPDATES),
            "SPEED_EXT_STEP_COMPONENT": str(STEP_COMPONENT),
            "SPEED_EXT_SELECTION_ONLY": "1",
            "SPEED_EXT_VALIDATION_DESIGNS_JSON": json.dumps(
                VALIDATION_DESIGNS
            ),
            "SPEED_EXT_GRADUAL_STAGE1_LEVELS": (
                "0.5,0.9,1.3,1.7,2.1,2.5,2.9,3.3"
            ),
            "SPEED_EXT_DIRECT_STAGE1_LEVELS": (
                "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0"
            ),
            "SPEED_EXT_GRADUAL_STAGE2_LEVELS": (
                "0.5,1.0,1.5,2.0,2.6,3.2,3.8,4.4"
            ),
            "SPEED_EXT_WIDE_STAGE2_LEVELS": (
                "0.5,1.0,1.5,2.0,2.6,3.3,4.1,5.0"
            ),
            "SPEED_EXT_EVAL_SLOPE_LENGTH": "110",
            "SPEED_EXT_TRAIN_SLOPE_LENGTH": "240",
            "SPEED_EXT_SLOPE_HORIZONTAL_SCALE": "0.1",
        },
    )
    return retry_dir / "SELECTED_POLICY.json"


def run_search(selection, source):
    search_dir = OUTPUT / "full_search"
    run_child(
        "full_search",
        "legged_gym/scripts/run_sixd_speed_extended_search.py",
        {
            "SPEED_EXT_SOURCE": str(source),
            "SPEED_EXT_OUT": str(search_dir),
            "SPEED_EXT_SELECTION_ONLY": "0",
            "SPEED_EXT_FROZEN_RUN": selection["run_name"],
            "SPEED_EXT_FROZEN_CKPT": str(selection["checkpoint"]),
            "SPEED_EXT_FROZEN_COMMAND_MAX": str(
                selection["training_command_max_mps"]
            ),
            "SPEED_EXT_FROZEN_TRAINING_SLOPE_DEG": "10",
            "SPEED_EXT_FIXED_SLOPE_DEG": "10",
            "SPEED_EXT_STEP_COMPONENT": str(STEP_COMPONENT),
            "SPEED_EXT_EVAL_SLOPE_LENGTH": "110",
            "SPEED_EXT_SLOPE_HORIZONTAL_SCALE": "0.1",
            "SPEED_EXT_BO1": "8",
            "SPEED_EXT_BO2": "8",
            "SPEED_EXT_FINALIST_COUNT": "8",
            "SPEED_EXT_FINALIST_SEEDS": (
                "20260727,20260728,20260729"
            ),
            "SPEED_EXT_FORMAL_TOP_COUNT": "3",
            "SPEED_EXT_FORMAL_CONTROLS_JSON": json.dumps(
                FORMAL_CONTROLS
            ),
            "SIXD_INITIAL_LHS": "12",
            "SIXD_INITIAL_SAMPLER": "sobol",
            "SIXD_FORMAL_SEEDS": (
                "20260727,20260728,20260729,20260730,20260731"
            ),
        },
    )
    return search_dir


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "STARTED").write_text(now_iso() + "\n", encoding="ascii")
    (OUTPUT / "DONE").unlink(missing_ok=True)
    (OUTPUT / "EXIT_CODE").unlink(missing_ok=True)
    update_status(
        stage="starting",
        started_at=now_iso(),
        initial_training=str(INITIAL_TRAINING),
        gpus=GPUS,
        retry_updates=RETRY_UPDATES,
        retry_enabled=RETRY_ENABLED,
        nominal_speed_gate_enabled=NOMINAL_SPEED_GATE_ENABLED,
        calibration_protocol=CALIBRATION_PROTOCOL,
        gate_revision_reason=GATE_REVISION_REASON,
        speed_step_component_mps=STEP_COMPONENT,
    )

    selection_path = wait_for_training(INITIAL_TRAINING)
    report, report_path = run_calibration(1, selection_path)
    calibration_history = [str(report_path)]
    source = INITIAL_TRAINING
    if not report["passed"] and RETRY_ENABLED:
        selection = json.loads(
            selection_path.read_text(encoding="utf-8")
        )
        selection_path = run_retry(selection)
        source = selection_path.parent
        report, report_path = run_calibration(2, selection_path)
        calibration_history.append(str(report_path))

    if not report["passed"]:
        update_status(
            stage="blocked_by_calibration",
            calibration_history=calibration_history,
            failed_checks=[
                name
                for name, check in report["checks"].items()
                if not check["passed"]
            ],
        )
        (OUTPUT / "CALIBRATION_BLOCKED").write_text(
            now_iso() + "\n", encoding="ascii"
        )
        (OUTPUT / "EXIT_CODE").write_text("2\n", encoding="ascii")
        (OUTPUT / "DONE").write_text(now_iso() + "\n", encoding="ascii")
        return

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    search_dir = run_search(selection, source)
    update_status(
        stage="complete",
        calibration_history=calibration_history,
        selected_policy=selection,
        search_directory=str(search_dir),
    )
    (OUTPUT / "EXIT_CODE").write_text("0\n", encoding="ascii")
    (OUTPUT / "DONE").write_text(now_iso() + "\n", encoding="ascii")
    print(f"[power-pipeline] COMPLETE {search_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        update_status(stage="failed")
        (OUTPUT / "EXIT_CODE").write_text("1\n", encoding="ascii")
        raise
