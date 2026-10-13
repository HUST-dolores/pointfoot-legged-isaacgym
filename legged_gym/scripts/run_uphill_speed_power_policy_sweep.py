#!/usr/bin/env python3
"""Compare saved uphill-speed policies with one frozen formal endpoint sweep."""

import json
import os
from pathlib import Path
import re

import numpy as np


OUTPUT = Path(
    os.environ.get(
        "POWER_SWEEP_OUT",
        "/home/ps/mydrive/xu/codesign_runs/"
        "sixd_uphill_speed_power_v2_policy_sweep_2026-07-27",
    )
)
POLICIES_PATH = Path(
    os.environ.get("POWER_SWEEP_POLICIES", "")
)
SEED = int(os.environ.get("POWER_SWEEP_SEED", "20260726"))
COMMAND_MAX = float(os.environ.get("POWER_SWEEP_COMMAND_MAX", "5.0"))
COMMAND_STEP = float(os.environ.get("POWER_SWEEP_COMMAND_STEP", "0.2"))
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
os.environ["PHYS_EVAL_SPEED_SLOPE_DEG"] = "10"
os.environ["PHYS_EVAL_SLOPE_LENGTH"] = os.environ.get(
    "POWER_SWEEP_SLOPE_LENGTH", "110"
)
os.environ["PHYS_EVAL_SLOPE_HORIZONTAL_SCALE"] = os.environ.get(
    "POWER_SWEEP_SLOPE_HORIZONTAL_SCALE", "0.1"
)

import run_sixd_single_objective_search as base  # noqa: E402


DESIGNS = (
    np.asarray([1, 1, 1, 1, 1, 0.6], dtype=np.float64),
    np.asarray([1, 1, 1, 1, 1, 1.0], dtype=np.float64),
    np.asarray([1, 1, 1, 1, 1, 1.6], dtype=np.float64),
)


def slug(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def command_grid():
    values = list(
        np.round(
            np.arange(0.5, COMMAND_MAX + 1.0e-9, COMMAND_STEP),
            6,
        )
    )
    if not values or not np.isclose(values[-1], COMMAND_MAX):
        values.append(COMMAND_MAX)
    return sorted(set(float(value) for value in values))


def load_policies():
    if not POLICIES_PATH.is_file():
        raise ValueError("POWER_SWEEP_POLICIES must point to a JSON file")
    payload = json.loads(POLICIES_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("policy file must contain a non-empty JSON list")
    policies = []
    seen = set()
    for index, value in enumerate(payload):
        run_name = str(value["run_name"])
        checkpoint = int(value["checkpoint"])
        command_max = float(value["training_command_max_mps"])
        key = (run_name, checkpoint)
        if key in seen:
            continue
        seen.add(key)
        policies.append(
            {
                "label": str(
                    value.get("label", f"policy_{index:03d}")
                ),
                "run_name": run_name,
                "checkpoint": checkpoint,
                "training_command_max_mps": command_max,
                "training_slope_deg": float(
                    value.get("training_slope_deg", 10.0)
                ),
            }
        )
    return policies


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "STARTED").write_text(base.now_iso() + "\n", encoding="ascii")
    (OUTPUT / "DONE").unlink(missing_ok=True)
    (OUTPUT / "EXIT_CODE").unlink(missing_ok=True)
    policies = load_policies()
    levels = command_grid()
    base.FORMAL_LEVELS["speed"] = ",".join(
        f"{value:g}" for value in levels
    )

    requests = []
    request_policy = {}
    for policy in policies:
        for design in DESIGNS:
            request = base.EvalRequest(
                "speed",
                policy["run_name"],
                policy["checkpoint"],
                design,
                SEED,
                f"power_policy_sweep_{slug(policy['label'])}",
                "formal",
            )
            requests.append(request)
            request_policy[id(request)] = policy
    base.run_jobs(
        "uphill_power_policy_sweep",
        [request.job for request in requests],
    )

    rows_by_policy = {policy["label"]: [] for policy in policies}
    records = []
    for request in requests:
        policy = request_policy[id(request)]
        result = base.read_json(request.job.expected_result)
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
                f"power-model mismatch: {request.job.expected_result}"
            )
        actual_step = float(
            result.get("command_decomposition", {}).get(
                "step_component_mps", float("nan")
            )
        )
        if not np.isclose(actual_step, STEP_COMPONENT):
            raise RuntimeError(
                f"speed-step mismatch: expected {STEP_COMPONENT}, "
                f"got {actual_step}: {request.job.expected_result}"
            )
        record = base.record_from_request(request)
        record["policy_label"] = policy["label"]
        records.append(record)
        rows_by_policy[policy["label"]].append(record)
    base.save_records(OUTPUT / "FORMAL_RECORDS.json", records)

    rows = []
    for policy in policies:
        values = rows_by_policy[policy["label"]]
        objective = [
            float(value["objective"]) for value in values
        ]
        rows.append(
            {
                **policy,
                "endpoint_objective_mps": objective,
                "mean_objective_mps": float(np.mean(objective)),
                "minimum_objective_mps": float(np.min(objective)),
                "result_paths": [
                    value["result_path"] for value in values
                ],
            }
        )
    rows.sort(
        key=lambda row: (
            -row["mean_objective_mps"],
            -row["minimum_objective_mps"],
            row["checkpoint"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    selected = {
        key: rows[0][key]
        for key in (
            "run_name",
            "checkpoint",
            "training_command_max_mps",
            "training_slope_deg",
        )
    }
    selected["validation"] = rows
    base.atomic_json(
        OUTPUT / "POLICY_SWEEP.json",
        {
            "protocol": (
                "sixd_uphill_speed_power_v2_policy_sweep_2026-07-27"
            ),
            "seed": SEED,
            "speed_step_component_mps": STEP_COMPONENT,
            "command_levels_mps": levels,
            "policies": rows,
            "selected": selected,
        },
    )
    base.atomic_json(OUTPUT / "SELECTED_POLICY.json", selected)
    (OUTPUT / "EXIT_CODE").write_text("0\n", encoding="ascii")
    (OUTPUT / "DONE").write_text(base.now_iso() + "\n", encoding="ascii")
    print(
        f"[power-policy-sweep] selected={rows[0]['label']} "
        f"mean={rows[0]['mean_objective_mps']:.6f}",
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
