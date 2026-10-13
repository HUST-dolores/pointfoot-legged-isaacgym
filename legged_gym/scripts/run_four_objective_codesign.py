#!/usr/bin/env python3
"""Run equal-weight four-task six-dimensional co-design optimization.

The four frozen task policies evaluate every shared mechanical candidate for
clearance, payload, 10-degree uphill speed, and jump area. The speed task uses
the frozen pure-wheel, motor-power-limited protocol; the other three tasks
retain their previous protocols. Search objectives are normalized so the
matched nominal design is zero and the corresponding single-task specialist
is one. A scalar equal-weight score drives GP-EI while the raw four-objective
Pareto front is retained for interpretation.
"""

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np


OUTPUT = Path(
    os.environ.get(
        "FOUR_OBJ_OUT",
        "/home/ps/mydrive/xu/codesign_runs/"
        "four_objective_sixd_v2_uphill_speed_2026-07-27",
    )
)
SPEED_ROOT = Path(
    os.environ.get(
        "FOUR_OBJ_SPEED_ROOT",
        "/home/ps/mydrive/xu/codesign_runs/"
        "sixd_uphill_speed_power_v4_bo_pipeline_2026-07-27/"
        "full_search",
    )
)
os.environ["SIXD_OUT"] = str(OUTPUT)
os.environ["PHYS_EVAL_SPEED_SUCCESS_MODE"] = "stable_actual"
os.environ.setdefault("SIXD_SEED", "20260727")
os.environ.setdefault(
    "SIXD_FORMAL_SEEDS",
    "20260727,20260728,20260729,20260730,20260731",
)
os.environ.setdefault("SIXD_INITIAL_SAMPLER", "sobol")

import run_sixd_single_objective_search as base  # noqa: E402


TASKS = ("payload", "speed", "clearance", "jump")
WEIGHTS = {task: 0.25 for task in TASKS}
BO1_COUNT = int(os.environ.get("FOUR_OBJ_BO1", "8"))
BO2_COUNT = int(os.environ.get("FOUR_OBJ_BO2", "8"))
SEARCH_TOP_COUNT = int(os.environ.get("FOUR_OBJ_FORMAL_TOP", "5"))
UNIT = {
    "payload": "kg",
    "speed": "m/s",
    "clearance": "m",
    "jump": "m^2",
}

STATIC_POLICIES = {
    "payload": {
        "run_name": (
            "Jul26_01-11-49_physadapt_sixdgen_payload_"
            "s20260726_u400"
        ),
        "checkpoint": 27400,
    },
    "clearance": {
        "run_name": (
            "Jul26_01-02-28_physadapt_sixdgen_clearance_"
            "s20260726_u400"
        ),
        "checkpoint": 27400,
    },
    "jump": {
        "run_name": "Jul11_21-43-33_jumpfwd23_residual_s79",
        "checkpoint": 4000,
    },
}

SPEED_PROTOCOL = {
    "fixed_slope_deg": 10.0,
    "positive_speed_direction": (
        "+world_x and +world_z along the ascending terrain tangent"
    ),
    "step_component_mps": 0.0,
    "slope_length_m": 110.0,
    "slope_horizontal_scale_m": 0.1,
    "motor_power_envelope": True,
    "specific_power_w_per_kg": 450.0,
    "power_exponent": 1.0,
    "power_min_speed_rad_s": 0.001,
    "metric_name": "V_uphill_stable_actual_90",
}

# This is the best point newly proposed by the completed v4 BO. The
# preregistered historical control is deliberately not used as the anchor.
OFFICIAL_SPEED_SPECIALIST = np.asarray(
    [
        0.9648397047072649,
        0.9025079853832721,
        0.9153309095650911,
        0.6738600976765156,
        0.7679443968459964,
        1.5596084440127016,
    ],
    dtype=np.float64,
)

STATIC_SPECIALISTS = {
    "payload": np.asarray(
        [
            1.157807829976082,
            1.0520169399678707,
            0.9914530286565423,
            0.8961383629590273,
            1.5578197859227658,
            0.607684264332056,
        ],
        dtype=np.float64,
    ),
    "clearance": np.asarray(
        [
            1.1970247339457272,
            1.198773255571723,
            1.566337169520557,
            1.1520810587331654,
            1.1318710934370757,
            0.8972766384482384,
        ],
        dtype=np.float64,
    ),
    "jump": np.asarray(
        [
            1.0835728153586388,
            1.0553055875003339,
            0.8760620675981045,
            0.606938312202692,
            1.0510720726102591,
            0.769737096875906,
        ],
        dtype=np.float64,
    ),
}


def command_grid(maximum, step):
    values = list(
        np.round(np.arange(0.5, maximum + 1.0e-9, step), 6)
    )
    if not values or not np.isclose(values[-1], maximum):
        values.append(float(maximum))
    return sorted(set(float(value) for value in values))


def comma_values(values):
    return ",".join(f"{float(value):g}" for value in values)


def load_speed_contract():
    selection_path = SPEED_ROOT / "SELECTED_POLICY.json"
    if not selection_path.is_file():
        selection_path = SPEED_ROOT / "FINAL_SELECTION.json"
    selection = base.read_json(selection_path)
    final = base.read_json(SPEED_ROOT / "FINAL_RESULTS.json")
    candidates = final["speed"].get("finalists", [])
    matches = [
        candidate
        for candidate in candidates
        if np.allclose(
            np.asarray(candidate["design_vector"], dtype=np.float64),
            OFFICIAL_SPEED_SPECIALIST,
            rtol=0.0,
            atol=1.0e-10,
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "completed speed BO does not contain exactly one official "
            "new-BO specialist"
        )
    winner = matches[0]
    policy = {
        "run_name": selection["run_name"],
        "checkpoint": int(selection["checkpoint"]),
        "training_command_max_mps": float(
            selection["training_command_max_mps"]
        ),
    }
    if not np.isclose(
        float(selection.get("evaluation_slope_deg", 10.0)),
        SPEED_PROTOCOL["fixed_slope_deg"],
    ):
        raise RuntimeError("speed policy slope does not match protocol")
    specialist = OFFICIAL_SPEED_SPECIALIST.copy()
    return policy, specialist, winner


SPEED_POLICY, SPEED_SPECIALIST, SPEED_SINGLE_RESULT = (
    load_speed_contract()
)
POLICIES = {
    **STATIC_POLICIES,
    "speed": SPEED_POLICY,
}
SPECIALISTS = {
    **STATIC_SPECIALISTS,
    "speed": SPEED_SPECIALIST,
}

if set(TASKS) != set(POLICIES) or "slope" in POLICIES:
    raise RuntimeError("four-task contract must exclude slope")

base.SEARCH_LEVELS["speed"] = comma_values(
    command_grid(SPEED_POLICY["training_command_max_mps"], 0.2)
)
base.FORMAL_LEVELS["speed"] = comma_values(
    command_grid(SPEED_POLICY["training_command_max_mps"], 0.1)
)


def snapshot_sources():
    for relative in (
        "legged_gym/scripts/physical_capability_eval.py",
        "legged_gym/scripts/measure_jumpfwd.py",
        "legged_gym/scripts/run_sixd_single_objective_search.py",
        "legged_gym/scripts/run_four_objective_codesign.py",
        "legged_gym/envs/base/base_task.py",
        "legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py",
        "legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py",
        "legged_gym/utils/motor_cost.py",
        "legged_gym/utils/terrain.py",
    ):
        source = base.ROOT / relative
        if not source.is_file():
            continue
        target = OUTPUT / "code_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def archive_policies():
    for task, policy in POLICIES.items():
        environment_name = (
            "wheelfoot_jump_forward"
            if task == "jump"
            else "wheelfoot_flat"
        )
        run_dir = (
            base.ROOT
            / "logs"
            / environment_name
            / "WF_TRON1A"
            / policy["run_name"]
        )
        target = OUTPUT / "selected_policies" / task
        target.mkdir(parents=True, exist_ok=True)
        model_name = f"model_{policy['checkpoint']}.pt"
        if not (run_dir / model_name).is_file():
            raise FileNotFoundError(
                f"missing frozen {task} policy: {run_dir / model_name}"
            )
        for name in (
            model_name,
            "env_cfg.json",
            "train_cfg.json",
            "physical_training_manifest.json",
            "physical_training_result.json",
        ):
            source = run_dir / name
            if source.is_file():
                shutil.copy2(source, target / name)
        base.atomic_json(target / "selection.json", policy)


def archive_speed_contract():
    target = OUTPUT / "speed_contract"
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "SELECTED_POLICY.json",
        "FINAL_SELECTION.json",
        "FINAL_RESULTS.json",
        "FORMAL_RECORDS.json",
        "RUN_MANIFEST.json",
    ):
        source = SPEED_ROOT / name
        if source.is_file():
            shutil.copy2(source, target / name)


def apply_task_protocol(request):
    environment = request.job.environment
    environment["PHYS_EVAL_SPEED_SUCCESS_MODE"] = "stable_actual"
    if request.task == "speed":
        environment.update(
            {
                "PHYS_EVAL_SPEED_SLOPE_DEG": str(
                    SPEED_PROTOCOL["fixed_slope_deg"]
                ),
                "PHYS_EVAL_STEP_COMPONENT": str(
                    SPEED_PROTOCOL["step_component_mps"]
                ),
                "PHYS_EVAL_SLOPE_LENGTH": str(
                    SPEED_PROTOCOL["slope_length_m"]
                ),
                "PHYS_EVAL_SLOPE_HORIZONTAL_SCALE": str(
                    SPEED_PROTOCOL["slope_horizontal_scale_m"]
                ),
                "PHYS_MOTOR_POWER_ENVELOPE": "1",
                "PHYS_MOTOR_SPECIFIC_POWER_W_PER_KG": str(
                    SPEED_PROTOCOL["specific_power_w_per_kg"]
                ),
                "PHYS_MOTOR_POWER_EXPONENT": str(
                    SPEED_PROTOCOL["power_exponent"]
                ),
                "PHYS_MOTOR_POWER_MIN_SPEED_RAD_S": str(
                    SPEED_PROTOCOL["power_min_speed_rad_s"]
                ),
            }
        )
    else:
        environment.update(
            {
                "PHYS_EVAL_SPEED_SLOPE_DEG": "0",
                "PHYS_EVAL_STEP_COMPONENT": "0.40",
                "PHYS_MOTOR_POWER_ENVELOPE": "0",
            }
        )


def validate_task_protocol(request):
    if request.task == "jump":
        return
    result = base.read_json(request.job.expected_result)
    power = result.get("motor_power_model", {})
    if request.task == "speed":
        summary = result.get("summary", {})
        decomposition = result.get("command_decomposition", {})
        checks = (
            np.isclose(
                float(result.get("speed_slope_deg", float("nan"))),
                SPEED_PROTOCOL["fixed_slope_deg"],
            ),
            np.isclose(
                float(
                    decomposition.get(
                        "step_component_mps", float("nan")
                    )
                ),
                SPEED_PROTOCOL["step_component_mps"],
            ),
            bool(power.get("enabled", False)),
            np.isclose(
                float(
                    power.get(
                        "specific_power_w_per_kg", float("nan")
                    )
                ),
                SPEED_PROTOCOL["specific_power_w_per_kg"],
            ),
            summary.get("metric_name") == SPEED_PROTOCOL["metric_name"],
            result.get("positive_speed_direction")
            == SPEED_PROTOCOL["positive_speed_direction"],
        )
        if not all(checks):
            raise RuntimeError(
                "speed result violates the frozen uphill/pure-wheel/power "
                f"protocol: {request.job.expected_result}"
            )
    elif (
        not np.isclose(float(result.get("speed_slope_deg", 0.0)), 0.0)
        or bool(power.get("enabled", False))
    ):
        raise RuntimeError(
            f"{request.task} result was contaminated by speed protocol: "
            f"{request.job.expected_result}"
        )


def shared_initial_designs():
    return base.unique_designs(
        base.initial_designs()
        + [SPECIALISTS[task] for task in TASKS]
    )


def evaluate_raw(phase, designs, fidelity="search", seeds=None):
    if seeds is None:
        seeds = (base.SEED,)
    shared = base.unique_designs(designs)
    requests = []
    for task in TASKS:
        policy = POLICIES[task]
        for design in shared:
            for seed in seeds:
                request = base.EvalRequest(
                    task,
                    policy["run_name"],
                    policy["checkpoint"],
                    design,
                    seed,
                    phase,
                    fidelity,
                )
                apply_task_protocol(request)
                requests.append(request)
    base.run_jobs(phase, [request.job for request in requests])
    for request in requests:
        validate_task_protocol(request)
    records = [base.record_from_request(request) for request in requests]
    expected = len(shared) * len(TASKS) * len(seeds)
    if len(records) != expected:
        raise RuntimeError(
            f"{phase} expected {expected} records, got {len(records)}"
        )
    for record in records:
        policy = POLICIES[record["task"]]
        if (
            record["run_name"] != policy["run_name"]
            or int(record["checkpoint"]) != int(policy["checkpoint"])
        ):
            raise RuntimeError(
                f"{phase} policy provenance mismatch for "
                f"{record['task']}/{record['design_id']}"
            )
    return records


def task_design_records(records):
    result = {}
    for record in records:
        key = (record["task"], record["design_id"])
        result.setdefault(key, []).append(record)
    return result


def search_anchors(initial_records):
    indexed = task_design_records(initial_records)
    nominal_id = base.design_key(base.NOMINAL)
    anchors = {}
    for task in TASKS:
        specialist_id = base.design_key(SPECIALISTS[task])
        nominal = indexed[(task, nominal_id)][0]["objective"]
        specialist = indexed[(task, specialist_id)][0]["objective"]
        span = float(specialist - nominal)
        if span <= 1.0e-12:
            raise RuntimeError(
                f"{task} specialist does not exceed nominal: "
                f"{specialist} <= {nominal}"
            )
        anchors[task] = {
            "nominal": float(nominal),
            "specialist": float(specialist),
            "span": span,
            "nominal_design_id": nominal_id,
            "specialist_design_id": specialist_id,
        }
    return anchors


def normalized_value(task, value, anchors):
    anchor = anchors[task]
    return (
        float(value) - float(anchor["nominal"])
    ) / float(anchor["span"])


def aggregate_search(records, anchors):
    grouped = {}
    for record in records:
        grouped.setdefault(record["design_id"], []).append(record)
    rows = []
    for design_id, values in grouped.items():
        by_task = {record["task"]: record for record in values}
        if set(by_task) != set(TASKS):
            raise RuntimeError(
                f"search design {design_id} lacks one or more tasks"
            )
        raw = {
            task: float(by_task[task]["objective"]) for task in TASKS
        }
        normalized = {
            task: normalized_value(task, raw[task], anchors)
            for task in TASKS
        }
        objective = float(
            sum(WEIGHTS[task] * normalized[task] for task in TASKS)
        )
        first = values[0]
        rows.append(
            {
                "design_id": design_id,
                "design": first["design"],
                "design_vector": first["design_vector"],
                "objective": objective,
                "equal_weight_score": objective,
                "raw_by_task": raw,
                "normalized_by_task": normalized,
                "motor_module_mass_kg": first[
                    "motor_module_mass_kg"
                ],
                "result_paths": {
                    task: by_task[task]["result_path"] for task in TASKS
                },
            }
        )
    rows.sort(
        key=lambda row: (
            -row["equal_weight_score"],
            row["motor_module_mass_kg"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def pareto_front(rows, value_key):
    front = []
    for index, row in enumerate(rows):
        value = np.asarray(
            [row[value_key][task] for task in TASKS],
            dtype=np.float64,
        )
        dominated = False
        for other_index, other in enumerate(rows):
            if other_index == index:
                continue
            other_value = np.asarray(
                [other[value_key][task] for task in TASKS],
                dtype=np.float64,
            )
            if np.all(other_value >= value) and np.any(
                other_value > value
            ):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front


def formal_anchors(records):
    indexed = task_design_records(records)
    nominal_id = base.design_key(base.NOMINAL)
    anchors = {}
    for task in TASKS:
        specialist_id = base.design_key(SPECIALISTS[task])
        nominal_values = [
            float(record["objective"])
            for record in indexed[(task, nominal_id)]
        ]
        specialist_values = [
            float(record["objective"])
            for record in indexed[(task, specialist_id)]
        ]
        nominal = float(np.mean(nominal_values))
        specialist = float(np.mean(specialist_values))
        span = specialist - nominal
        if span <= 1.0e-12:
            raise RuntimeError(
                f"formal {task} specialist does not exceed nominal"
            )
        anchors[task] = {
            "nominal": nominal,
            "specialist": specialist,
            "span": span,
            "nominal_by_seed": nominal_values,
            "specialist_by_seed": specialist_values,
            "nominal_design_id": nominal_id,
            "specialist_design_id": specialist_id,
        }
    return anchors


def aggregate_formal(records, anchors):
    indexed = task_design_records(records)
    design_ids = sorted({record["design_id"] for record in records})
    rows = []
    for design_id in design_ids:
        task_records = {
            task: sorted(
                indexed[(task, design_id)],
                key=lambda record: record["seed"],
            )
            for task in TASKS
        }
        seeds = [record["seed"] for record in task_records[TASKS[0]]]
        if any(
            [record["seed"] for record in task_records[task]] != seeds
            for task in TASKS
        ):
            raise RuntimeError(
                f"formal seed mismatch for design {design_id}"
            )
        raw_mean = {
            task: float(
                np.mean(
                    [
                        record["objective"]
                        for record in task_records[task]
                    ]
                )
            )
            for task in TASKS
        }
        raw_std = {
            task: float(
                np.std(
                    [
                        record["objective"]
                        for record in task_records[task]
                    ],
                    ddof=1,
                )
            )
            for task in TASKS
        }
        normalized = {
            task: normalized_value(task, raw_mean[task], anchors)
            for task in TASKS
        }
        score_by_seed = []
        for seed_index in range(len(seeds)):
            score_by_seed.append(
                float(
                    sum(
                        WEIGHTS[task]
                        * normalized_value(
                            task,
                            task_records[task][seed_index]["objective"],
                            anchors,
                        )
                        for task in TASKS
                    )
                )
            )
        first = task_records[TASKS[0]][0]
        rows.append(
            {
                "design_id": design_id,
                "design": first["design"],
                "design_vector": first["design_vector"],
                "seeds": seeds,
                "raw_mean_by_task": raw_mean,
                "raw_std_by_task": raw_std,
                "normalized_by_task": normalized,
                "score_by_seed": score_by_seed,
                "equal_weight_score_mean": float(
                    np.mean(score_by_seed)
                ),
                "equal_weight_score_std": float(
                    np.std(score_by_seed, ddof=1)
                ),
                "motor_module_mass_kg": first[
                    "motor_module_mass_kg"
                ],
                "result_paths_by_task": {
                    task: [
                        record["result_path"]
                        for record in task_records[task]
                    ]
                    for task in TASKS
                },
            }
        )
    rows.sort(
        key=lambda row: (
            -row["equal_weight_score_mean"],
            row["equal_weight_score_std"],
            row["motor_module_mass_kg"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    front_ids = {
        row["design_id"]
        for row in pareto_front(rows, "raw_mean_by_task")
    }
    for row in rows:
        row["pareto_nondominated"] = row["design_id"] in front_ids
    return rows


def write_final_report(rows, anchors, search_count):
    lines = [
        "# Four-objective equal-weight co-design",
        "",
        "Tasks: payload, 10-degree uphill stable actual speed, swing "
        "clearance, and jump reachable area. Static slope is excluded.",
        "",
        "The speed task uses real 10-degree terrain, pure-wheel command "
        "(step component 0), the frozen 4.0 m/s-domain policy, and the "
        "450 W/kg motor power envelope. The other task protocols are "
        "unchanged.",
        "",
        "Each normalized task score is zero at the matched nominal design "
        "and one at that task's matched single-objective specialist. The "
        "equal-weight score is their arithmetic mean.",
        "",
        "| rank | design | score mean +/- std | payload kg | speed m/s | "
        "clearance m | jump m^2 | Pareto |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        raw = row["raw_mean_by_task"]
        lines.append(
            f"| {row['rank']} | "
            f"`{[round(value, 6) for value in row['design_vector']]}` | "
            f"{row['equal_weight_score_mean']:.5f} +/- "
            f"{row['equal_weight_score_std']:.5f} | "
            f"{raw['payload']:.5g} | {raw['speed']:.5g} | "
            f"{raw['clearance']:.5g} | {raw['jump']:.6g} | "
            f"{'yes' if row['pareto_nondominated'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Fresh shared search designs: {search_count}.",
            f"Formal seeds: {list(base.FORMAL_SEEDS)}.",
            f"Formal normalization anchors: `{anchors}`.",
        ]
    )
    (OUTPUT / "FINAL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "STARTED").write_text(
        base.now_iso() + "\n", encoding="ascii"
    )
    (OUTPUT / "DONE").unlink(missing_ok=True)
    (OUTPUT / "EXIT_CODE").unlink(missing_ok=True)
    snapshot_sources()
    archive_policies()
    archive_speed_contract()
    base.atomic_json(
        OUTPUT / "RUN_MANIFEST.json",
        {
            "started_at": base.now_iso(),
            "tasks": list(TASKS),
            "slope_excluded": True,
            "weights": WEIGHTS,
            "policies": POLICIES,
            "single_task_specialists": {
                task: base.design_dict(design)
                for task, design in SPECIALISTS.items()
            },
            "parameters": list(base.PARAMETERS),
            "bounds": {
                name: [float(low), float(high)]
                for name, low, high in zip(
                    base.PARAMETERS, base.LOWER, base.UPPER
                )
            },
            "search_seed": base.SEED,
            "formal_seeds": list(base.FORMAL_SEEDS),
            "bo_batches": [BO1_COUNT, BO2_COUNT],
            "speed_policy_source": str(SPEED_ROOT),
            "speed_training_command_max_mps": SPEED_POLICY[
                "training_command_max_mps"
            ],
            "speed_protocol": SPEED_PROTOCOL,
            "speed_anchor_source": {
                "selection": (
                    "best newly proposed candidate from the completed "
                    "v4 six-dimensional BO"
                ),
                "design_vector": SPEED_SINGLE_RESULT["design_vector"],
                "objective_mean": SPEED_SINGLE_RESULT["objective_mean"],
                "objective_std": SPEED_SINGLE_RESULT["objective_std"],
                "seeds": SPEED_SINGLE_RESULT["seeds"],
            },
            "normalization": (
                "per-task matched nominal=0, matched specialist=1"
            ),
        },
    )

    initial_designs = shared_initial_designs()
    initial_raw = evaluate_raw(
        "four_objective_initial_search", initial_designs
    )
    base.save_records(OUTPUT / "INITIAL_RAW_RECORDS.json", initial_raw)
    anchors = search_anchors(initial_raw)
    base.atomic_json(OUTPUT / "SEARCH_NORMALIZATION.json", anchors)
    all_raw = list(initial_raw)
    search_rows = aggregate_search(all_raw, anchors)
    base.atomic_json(
        OUTPUT / "INITIAL_AGGREGATED_RECORDS.json", search_rows
    )
    gp_fits = []

    proposals, fit = base.propose_expected_improvement(
        search_rows, BO1_COUNT, base.SEED + 1700
    )
    gp_fits.append({"phase": "four_objective_bo1", "fit": fit})
    bo1_raw = evaluate_raw("four_objective_bo1_search", proposals)
    all_raw.extend(bo1_raw)
    search_rows = aggregate_search(all_raw, anchors)
    base.save_records(OUTPUT / "SEARCH_AFTER_BO1_RAW.json", all_raw)
    base.atomic_json(
        OUTPUT / "SEARCH_AFTER_BO1_AGGREGATED.json", search_rows
    )

    proposals, fit = base.propose_expected_improvement(
        search_rows, BO2_COUNT, base.SEED + 1800
    )
    gp_fits.append({"phase": "four_objective_bo2", "fit": fit})
    bo2_raw = evaluate_raw("four_objective_bo2_search", proposals)
    all_raw.extend(bo2_raw)
    search_rows = aggregate_search(all_raw, anchors)
    base.save_records(OUTPUT / "ALL_SEARCH_RAW_RECORDS.json", all_raw)
    base.atomic_json(
        OUTPUT / "ALL_SEARCH_AGGREGATED_RECORDS.json", search_rows
    )
    base.atomic_json(OUTPUT / "GP_FITS.json", gp_fits)
    base.atomic_json(
        OUTPUT / "SEARCH_PARETO_FRONT.json",
        pareto_front(search_rows, "raw_by_task"),
    )

    formal_designs = [
        np.asarray(row["design_vector"], dtype=np.float64)
        for row in search_rows[:SEARCH_TOP_COUNT]
    ]
    formal_designs.extend(
        [base.NOMINAL.copy()]
        + [SPECIALISTS[task].copy() for task in TASKS]
    )
    formal_designs = base.unique_designs(formal_designs)
    formal_raw = evaluate_raw(
        "four_objective_formal_repeats",
        formal_designs,
        fidelity="formal",
        seeds=base.FORMAL_SEEDS,
    )
    base.save_records(OUTPUT / "FORMAL_RAW_RECORDS.json", formal_raw)
    anchors = formal_anchors(formal_raw)
    base.atomic_json(OUTPUT / "FORMAL_NORMALIZATION.json", anchors)
    rows = aggregate_formal(formal_raw, anchors)
    base.atomic_json(
        OUTPUT / "FINAL_RESULTS.json",
        {
            "winner": rows[0],
            "finalists": rows,
            "pareto_front": [
                row for row in rows if row["pareto_nondominated"]
            ],
            "formal_normalization": anchors,
            "weights": WEIGHTS,
        },
    )
    with (OUTPUT / "FINAL_RESULTS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "rank",
            *base.PARAMETERS,
            "equal_weight_score_mean",
            "equal_weight_score_std",
            "payload",
            "speed",
            "clearance",
            "jump",
            "pareto_nondominated",
            "motor_module_mass_kg",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = {
                "rank": row["rank"],
                **row["design"],
                "equal_weight_score_mean": row[
                    "equal_weight_score_mean"
                ],
                "equal_weight_score_std": row[
                    "equal_weight_score_std"
                ],
                **row["raw_mean_by_task"],
                "pareto_nondominated": row["pareto_nondominated"],
                "motor_module_mass_kg": row["motor_module_mass_kg"],
            }
            writer.writerow(output_row)
    write_final_report(rows, anchors, len(search_rows))

    time.sleep(2.0)
    gpu_state = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    ).stdout.strip()
    base.atomic_json(
        OUTPUT / "LIFECYCLE.json",
        {
            "completed_at": base.now_iso(),
            "gpu_state_after_children": gpu_state,
            "search_design_count": len(search_rows),
            "formal_design_count": len(rows),
            "formal_result_count": len(formal_raw),
        },
    )
    base.update_status(
        stage="complete",
        completed_jobs=1,
        total_jobs=1,
        failed_jobs=[],
        eta_seconds=0.0,
    )
    (OUTPUT / "EXIT_CODE").write_text("0\n", encoding="ascii")
    (OUTPUT / "DONE").write_text(
        base.now_iso() + "\n", encoding="ascii"
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
