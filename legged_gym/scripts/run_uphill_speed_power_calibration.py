#!/usr/bin/env python3
"""Run the frozen BO gate for the 10-degree uphill power-envelope study."""

import json
import os
from pathlib import Path
import statistics
import time

import numpy as np


ROOT = Path(
    os.environ.get(
        "POWER_CAL_ROOT",
        "/home/ps/mydrive/xu/pointfoot-legged-gym",
    )
)
OUTPUT = Path(
    os.environ.get(
        "POWER_CAL_OUT",
        "/home/ps/mydrive/xu/codesign_runs/"
        "sixd_uphill_speed_power_v2_calibration_2026-07-27",
    )
)
SELECTION_PATH = Path(
    os.environ.get(
        "POWER_CAL_SELECTION",
        "/home/ps/mydrive/xu/codesign_runs/"
        "sixd_uphill_speed_power_v2_training_2026-07-27/"
        "SELECTED_POLICY.json",
    )
)
SEEDS = tuple(
    int(value)
    for value in os.environ.get(
        "POWER_CAL_SEEDS", "20260727,20260728,20260729"
    ).split(",")
    if value.strip()
)
COMMAND_MIN = float(os.environ.get("POWER_CAL_COMMAND_MIN", "0.5"))
COMMAND_MAX = float(os.environ.get("POWER_CAL_COMMAND_MAX", "5.0"))
COMMAND_STEP = float(os.environ.get("POWER_CAL_COMMAND_STEP", "0.2"))
MIN_NOMINAL_SPEED = float(
    os.environ.get("POWER_CAL_MIN_NOMINAL_SPEED", "2.5")
)
NOMINAL_SPEED_GATE_ENABLED = os.environ.get(
    "POWER_CAL_NOMINAL_SPEED_GATE_ENABLED", "1"
).strip().lower() in ("1", "true", "yes", "on")
CALIBRATION_PROTOCOL = os.environ.get(
    "POWER_CAL_PROTOCOL",
    "sixd_uphill_speed_power_v2_calibration_2026-07-27",
)
GATE_REVISION_REASON = os.environ.get(
    "POWER_CAL_GATE_REVISION_REASON", ""
).strip()
MIN_STRONG_ABS_GAIN = float(
    os.environ.get("POWER_CAL_MIN_STRONG_ABS_GAIN", "0.15")
)
MIN_STRONG_REL_GAIN = float(
    os.environ.get("POWER_CAL_MIN_STRONG_REL_GAIN", "0.10")
)
MIN_POSITIVE_PAIRS = int(
    os.environ.get("POWER_CAL_MIN_POSITIVE_PAIRS", "2")
)
MIN_WEAK_POWER_LIMITED = float(
    os.environ.get("POWER_CAL_MIN_WEAK_POWER_LIMITED", "0.10")
)
MAX_FINAL_COMMAND_GAIN = float(
    os.environ.get("POWER_CAL_MAX_FINAL_COMMAND_GAIN", "0.05")
)
EXPECTED_SPECIFIC_POWER = float(
    os.environ.get("PHYS_MOTOR_SPECIFIC_POWER_W_PER_KG", "450.0")
)
STEP_COMPONENT = float(
    os.environ.get("PHYS_EVAL_STEP_COMPONENT", "0.40")
)
if STEP_COMPONENT < 0.0:
    raise ValueError("PHYS_EVAL_STEP_COMPONENT must be non-negative")

os.environ["SIXD_OUT"] = str(OUTPUT)
os.environ["PHYS_EVAL_SPEED_SUCCESS_MODE"] = "stable_actual"
os.environ["PHYS_EVAL_SPEED_SLOPE_DEG"] = os.environ.get(
    "POWER_CAL_SLOPE_DEG", "10"
)
os.environ["PHYS_EVAL_SLOPE_LENGTH"] = os.environ.get(
    "POWER_CAL_SLOPE_LENGTH", "110"
)
os.environ["PHYS_EVAL_SLOPE_HORIZONTAL_SCALE"] = os.environ.get(
    "POWER_CAL_SLOPE_HORIZONTAL_SCALE", "0.1"
)

import run_sixd_single_objective_search as base  # noqa: E402


DESIGNS = {
    "weak": np.asarray([1, 1, 1, 1, 1, 0.6], dtype=np.float64),
    "nominal": np.asarray([1, 1, 1, 1, 1, 1.0], dtype=np.float64),
    "strong": np.asarray([1, 1, 1, 1, 1, 1.6], dtype=np.float64),
}


def command_grid():
    values = list(
        np.round(
            np.arange(
                COMMAND_MIN,
                COMMAND_MAX + 1.0e-9,
                COMMAND_STEP,
            ),
            6,
        )
    )
    if not values or not np.isclose(values[-1], COMMAND_MAX):
        values.append(COMMAND_MAX)
    return sorted(set(float(value) for value in values))


def _power_limited_high_command(result):
    threshold = COMMAND_MAX - 1.0
    values = [
        float(row["wheel_power_limited_fraction_mean"])
        for row in result["per_level"]
        if float(row["level"]) >= threshold
    ]
    return float(np.mean(values)) if values else 0.0


def _curve_converged(result):
    training_limit = float(
        result["summary"]["training_command_max_mps"]
    )
    rows = [
        row
        for row in result["per_level"]
        if float(row["level"]) <= training_limit + 1.0e-9
    ]
    values = np.asarray(
        [row["stable_actual_speed_q10_mps"] for row in rows],
        dtype=np.float64,
    )
    best_index = int(np.argmax(values))
    if best_index < len(values) - 1:
        return True, {
            "reason": "best command is below the tested upper boundary",
            "best_index": best_index,
            "level_count": len(values),
        }
    if len(values) < 3:
        return False, {
            "reason": "fewer than three in-domain command levels",
            "best_index": best_index,
            "level_count": len(values),
        }
    gains = [
        float(values[-2] - values[-3]),
        float(values[-1] - values[-2]),
    ]
    passed = all(gain < MAX_FINAL_COMMAND_GAIN for gain in gains)
    return passed, {
        "reason": "last two stable-speed gains checked",
        "best_index": best_index,
        "level_count": len(values),
        "last_two_gains_mps": gains,
        "threshold_mps": MAX_FINAL_COMMAND_GAIN,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "STARTED").write_text(base.now_iso() + "\n", encoding="ascii")
    (OUTPUT / "DONE").unlink(missing_ok=True)
    (OUTPUT / "EXIT_CODE").unlink(missing_ok=True)
    selected = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    levels = command_grid()
    base.FORMAL_LEVELS["speed"] = ",".join(
        f"{value:g}" for value in levels
    )
    selected_by_task = {"speed": selected}
    records = base.evaluate_design_stage(
        "uphill_power_endpoint_calibration",
        {"speed": list(DESIGNS.values())},
        selected_by_task,
        fidelity="formal",
        seeds=SEEDS,
    )
    base.save_records(OUTPUT / "FORMAL_RECORDS.json", records)

    labels = {
        base.design_key(design): label
        for label, design in DESIGNS.items()
    }
    rows_by_label = {label: [] for label in DESIGNS}
    for record in records:
        result = base.read_json(Path(record["result_path"]))
        power_model = result.get("motor_power_model", {})
        if (
            not bool(power_model.get("enabled", False))
            or not np.isclose(
                float(
                    power_model.get(
                        "specific_power_w_per_kg", float("nan")
                    )
                ),
                EXPECTED_SPECIFIC_POWER,
            )
        ):
            raise RuntimeError(
                f"power-model provenance mismatch: {record['result_path']}"
            )
        actual_step = float(
            result.get("command_decomposition", {}).get(
                "step_component_mps", float("nan")
            )
        )
        if not np.isclose(actual_step, STEP_COMPONENT):
            raise RuntimeError(
                f"speed-step mismatch: expected {STEP_COMPONENT}, "
                f"got {actual_step}: {record['result_path']}"
            )
        label = labels[record["design_id"]]
        converged, convergence_detail = _curve_converged(result)
        rows_by_label[label].append(
            {
                "seed": int(record["seed"]),
                "objective_mps": float(record["objective"]),
                "command_at_metric_mps": float(
                    record["summary"]["command_at_metric_mps"]
                ),
                "no_reset_rate_at_metric": float(
                    record["summary"]["no_reset_rate_at_command"]
                ),
                "power_limited_fraction_at_metric": float(
                    record["summary"][
                        "wheel_power_limited_fraction_at_command"
                    ]
                ),
                "high_command_power_limited_fraction": (
                    _power_limited_high_command(result)
                ),
                "curve_converged": converged,
                "convergence_detail": convergence_detail,
                "result_path": record["result_path"],
            }
        )
    for values in rows_by_label.values():
        values.sort(key=lambda row: row["seed"])

    weak = rows_by_label["weak"]
    nominal = rows_by_label["nominal"]
    strong = rows_by_label["strong"]
    weak_objective = [row["objective_mps"] for row in weak]
    nominal_objective = [row["objective_mps"] for row in nominal]
    strong_objective = [row["objective_mps"] for row in strong]
    weak_mean = statistics.mean(weak_objective)
    nominal_mean = statistics.mean(nominal_objective)
    strong_mean = statistics.mean(strong_objective)
    required_gain = max(
        MIN_STRONG_ABS_GAIN,
        MIN_STRONG_REL_GAIN * weak_mean,
    )
    paired_differences = [
        strong_row["objective_mps"] - weak_row["objective_mps"]
        for weak_row, strong_row in zip(weak, strong)
    ]
    positive_pairs = sum(value > 0.0 for value in paired_differences)
    weak_power_limited = statistics.mean(
        row["high_command_power_limited_fraction"]
        for row in weak
    )
    converged_nominal_seeds = sum(
        row["curve_converged"] for row in nominal
    )

    nominal_speed_criterion_met = nominal_mean >= MIN_NOMINAL_SPEED
    checks = {
        "nominal_speed": {
            "passed": (
                nominal_speed_criterion_met
                if NOMINAL_SPEED_GATE_ENABLED
                else True
            ),
            "enabled_as_gate": NOMINAL_SPEED_GATE_ENABLED,
            "criterion_met": nominal_speed_criterion_met,
            "role": (
                "required"
                if NOMINAL_SPEED_GATE_ENABLED
                else "informational"
            ),
            "observed_mps": nominal_mean,
            "threshold_mps": MIN_NOMINAL_SPEED,
        },
        "command_domain_convergence": {
            "passed": converged_nominal_seeds >= 2,
            "converged_seed_count": converged_nominal_seeds,
            "required_seed_count": 2,
        },
        "strong_vs_weak_mean": {
            "passed": strong_mean - weak_mean >= required_gain,
            "observed_gain_mps": strong_mean - weak_mean,
            "required_gain_mps": required_gain,
            "weak_mean_mps": weak_mean,
            "strong_mean_mps": strong_mean,
        },
        "paired_seed_order": {
            "passed": positive_pairs >= MIN_POSITIVE_PAIRS,
            "positive_pair_count": positive_pairs,
            "required_pair_count": MIN_POSITIVE_PAIRS,
            "paired_differences_mps": paired_differences,
        },
        "weak_motor_power_activation": {
            "passed": weak_power_limited >= MIN_WEAK_POWER_LIMITED,
            "observed_fraction": weak_power_limited,
            "threshold_fraction": MIN_WEAK_POWER_LIMITED,
        },
    }
    passed = all(check["passed"] for check in checks.values())
    report = {
        "protocol": CALIBRATION_PROTOCOL,
        "created_at": base.now_iso(),
        "gate_revision_reason": GATE_REVISION_REASON,
        "selection_path": str(SELECTION_PATH),
        "selected_policy": selected,
        "slope_deg": float(os.environ["PHYS_EVAL_SPEED_SLOPE_DEG"]),
        "specific_power_w_per_kg": EXPECTED_SPECIFIC_POWER,
        "speed_step_component_mps": STEP_COMPONENT,
        "command_levels_mps": levels,
        "seeds": list(SEEDS),
        "designs": {
            label: base.design_dict(design)
            for label, design in DESIGNS.items()
        },
        "results_by_design": rows_by_label,
        "objective_mean_mps": {
            "weak": weak_mean,
            "nominal": nominal_mean,
            "strong": strong_mean,
        },
        "checks": checks,
        "passed": passed,
        "next_action": (
            "six-dimensional BO is allowed"
            if passed
            else "do not run BO; continue policy/model calibration"
        ),
    }
    base.atomic_json(OUTPUT / "CALIBRATION.json", report)
    lines = [
        "# Uphill power-envelope calibration",
        "",
        f"Overall gate: `{'PASS' if passed else 'FAIL'}`.",
        "",
        "| check | pass | observed | threshold |",
        "|---|---|---|---|",
    ]
    for name, check in checks.items():
        observed = {
            key: value
            for key, value in check.items()
            if key != "passed" and "threshold" not in key
            and "required" not in key
        }
        threshold = {
            key: value
            for key, value in check.items()
            if "threshold" in key or "required" in key
        }
        lines.append(
            f"| {name} | {check['passed']} | `{observed}` | "
            f"`{threshold}` |"
        )
    (OUTPUT / "CALIBRATION_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / ("PASS" if passed else "FAIL")).write_text(
        base.now_iso() + "\n", encoding="ascii"
    )
    (OUTPUT / "EXIT_CODE").write_text("0\n", encoding="ascii")
    (OUTPUT / "DONE").write_text(base.now_iso() + "\n", encoding="ascii")
    time.sleep(1.0)
    print(
        f"[power-calibration] {'PASS' if passed else 'FAIL'} "
        f"{OUTPUT / 'CALIBRATION.json'}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "EXIT_CODE").write_text("1\n", encoding="ascii")
        raise
