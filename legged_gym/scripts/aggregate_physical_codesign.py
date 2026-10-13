#!/usr/bin/env python3
"""Aggregate five physical co-design metrics from fixed result JSON files."""

import argparse
import csv
import json
import math
from pathlib import Path


LOCOMOTION_TASKS = ("clearance", "slope", "payload", "speed")
METRIC_FIELDS = (
    ("clearance", "H_clear90_m"),
    ("slope", "theta90_deg"),
    ("payload", "M90_kg"),
    ("speed", "V90_mps"),
    ("jump", "A_jump_m2"),
)
NORMALIZATION_FIELDS = {
    "clearance": "H_clear90_m",
    "slope": "theta90_midpoint_deg",
    "payload": "M90_kg",
    "speed": "V90_mps",
    "jump": "A_jump_m2",
}
SLOPE_NORMALIZATION_SPAN_DEG = 12.0
JUMP_NORMALIZATION_SPAN_M2 = 0.16 * 0.14
EXPECTED_UPDATES = {
    "clearance": 100,
    "slope": 200,
    "payload": 100,
    "speed": 400,
    "jump": 400,
}
EXPECTED_SLOPE_LEVELS = (0.0, 0.0, 0.0, 1.5, 3.0, 6.0, 9.0, 12.0)
EXPECTED_TRAINING_LEVELS = {
    "clearance": ("clearance_levels_m", (0.22,) * 8),
    "payload": (
        "payload_levels_kg",
        (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0),
    ),
    "speed": (
        "speed_levels_mps",
        (0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6),
    ),
}
REFINED_EVALUATION_LEVELS = {
    "slope": tuple(float(value) for value in range(13)),
    "payload": tuple(float(value) for value in range(0, 46, 5)),
    "speed": (0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--nominal-label", default="nominal")
    parser.add_argument("--expected-seed", type=int, default=20260724)
    parser.add_argument(
        "--eval-override-root",
        type=Path,
        help=(
            "optional root containing label/{slope,payload,speed}/result.json "
            "from the high-replication evaluation protocol"
        ),
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_candidates(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = [row["label"] for row in rows]
    if len(labels) != len(set(labels)):
        raise ValueError("candidate labels must be unique")
    for row in rows:
        row["design"] = (
            float(row["thigh_scale"]),
            float(row["shank_scale"]),
            float(row["motor_scale"]),
        )
    return rows


def assert_design(record, expected, path):
    design = record.get("design", {})
    actual = (
        float(design.get("thigh", math.nan)),
        float(design.get("shank", math.nan)),
        float(design.get("motor", math.nan)),
    )
    if any(
        not math.isfinite(value)
        or abs(value - target) > 5.0e-4
        for value, target in zip(actual, expected)
    ):
        raise ValueError(
            f"{path}: design {actual} does not match candidate {expected}"
        )


def assert_close(actual, expected, path, field, tolerance=1.0e-9):
    if not math.isfinite(float(actual)) or abs(float(actual) - expected) > tolerance:
        raise ValueError(f"{path}: {field} {actual} != {expected}")


def assert_training_contract(task, training, path):
    expected_updates = EXPECTED_UPDATES[task]
    if int(training.get("updates", -1)) != expected_updates:
        raise ValueError(
            f"{path}: updates {training.get('updates')} != {expected_updates}"
        )
    if int(training.get("num_envs", -1)) != 4096:
        raise ValueError(f"{path}: num_envs must be 4096")

    if task == "jump":
        if training.get("base_checkpoint") != 4000:
            raise ValueError(f"{path}: jump base checkpoint must be 4000")
        assert_close(
            training.get("learning_rate", math.nan),
            1.0e-3,
            path,
            "learning_rate",
        )
        assert_close(
            training.get("entropy_coef", math.nan),
            0.005,
            path,
            "entropy_coef",
        )
        assert_close(
            training.get("est_learning_rate", math.nan),
            1.0e-3,
            path,
            "est_learning_rate",
        )
        if training.get("schedule") != "adaptive":
            raise ValueError(f"{path}: jump schedule must be adaptive")
        if tuple(training.get("height_command_range_m", [])) != (0.1, 0.16):
            raise ValueError(f"{path}: jump height command range mismatch")
        if tuple(training.get("distance_command_range_m", [])) != (0.05, 0.14):
            raise ValueError(f"{path}: jump distance command range mismatch")
        assert_close(
            training.get("resume_noise_std", math.nan),
            0.5,
            path,
            "resume_noise_std",
        )
        return

    if training.get("physical_task") != task:
        raise ValueError(
            f"{path}: physical_task {training.get('physical_task')} != {task}"
        )
    if training.get("base_checkpoint") != 27000:
        raise ValueError(f"{path}: locomotion base checkpoint must be 27000")
    assert_close(
        training.get("learning_rate", math.nan),
        1.0e-4,
        path,
        "learning_rate",
    )
    assert_close(
        training.get("entropy_coef", math.nan),
        0.001,
        path,
        "entropy_coef",
    )
    assert_close(
        training.get("est_learning_rate", math.nan),
        0.0,
        path,
        "est_learning_rate",
    )
    if training.get("schedule") != "fixed":
        raise ValueError(f"{path}: locomotion schedule must be fixed")
    if task == "slope":
        levels = tuple(
            float(value) for value in training.get("slope_levels_deg", [])
        )
        if levels != EXPECTED_SLOPE_LEVELS:
            raise ValueError(f"{path}: slope curriculum mismatch: {levels}")
        assert_close(
            training.get("total_vel_tracking_sigma", math.nan),
            0.2,
            path,
            "total_vel_tracking_sigma",
        )
        assert_close(
            training.get("total_vel_error_reward", math.nan),
            -4.0,
            path,
            "total_vel_error_reward",
        )
    else:
        field, expected_levels = EXPECTED_TRAINING_LEVELS[task]
        levels = tuple(float(value) for value in training.get(field, []))
        if levels != expected_levels:
            raise ValueError(
                f"{path}: {task} curriculum mismatch: {levels}"
            )


def assert_refined_evaluation_contract(task, evaluation, path):
    if task not in REFINED_EVALUATION_LEVELS:
        return
    levels = tuple(float(value) for value in evaluation.get("levels", []))
    if levels != REFINED_EVALUATION_LEVELS[task]:
        raise ValueError(f"{path}: refined {task} levels mismatch: {levels}")
    if int(evaluation.get("replicas_per_level", -1)) != 128:
        raise ValueError(f"{path}: refined evaluation must use 128 replicas")
    assert_close(evaluation.get("settle_s", math.nan), 2.0, path, "settle_s")
    assert_close(
        evaluation.get("evaluation_s", math.nan),
        8.0,
        path,
        "evaluation_s",
    )
    assert_close(evaluation.get("friction", math.nan), 0.8, path, "friction")
    if task == "slope":
        discretization = evaluation.get("terrain_discretization", {})
        assert_close(
            discretization.get("horizontal_scale_m", math.nan),
            0.1,
            path,
            "terrain horizontal scale",
        )
        assert_close(
            discretization.get("vertical_scale_m", math.nan),
            0.001,
            path,
            "terrain vertical scale",
        )


def load_metric(
    root,
    label,
    task,
    design,
    expected_seed,
    eval_override_root=None,
):
    refined = eval_override_root is not None and task in REFINED_EVALUATION_LEVELS
    if refined:
        eval_job_dir = eval_override_root / label / task
        eval_dir = eval_job_dir
        nested_eval_dir = eval_job_dir / task
        if not (eval_dir / "result.json").is_file() and (
            nested_eval_dir / "result.json"
        ).is_file():
            eval_dir = nested_eval_dir
        eval_done_path = eval_job_dir / "DONE"
    else:
        eval_dir = root / label / task / "eval"
        extended_dir = root / label / task / "eval_extended"
        if extended_dir.exists():
            extended_result = extended_dir / "result.json"
            extended_done = extended_dir / "DONE"
            if not extended_result.is_file() or not extended_done.is_file():
                raise FileNotFoundError(
                    f"incomplete extended evaluation: {extended_dir}"
                )
            eval_dir = extended_dir
        eval_done_path = eval_dir / "DONE"
    eval_path = eval_dir / "result.json"
    train_dir = root / label / task / "train"
    train_path = train_dir / "result.json"
    if not eval_path.is_file():
        raise FileNotFoundError(eval_path)
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    for done_path in (eval_done_path, train_dir / "DONE"):
        if not done_path.is_file():
            raise FileNotFoundError(done_path)
    evaluation = load_json(eval_path)
    training = load_json(train_path)
    assert_design(evaluation, design, eval_path)
    assert_design(training, design, train_path)
    if refined:
        assert_refined_evaluation_contract(task, evaluation, eval_path)
    for record, path in ((evaluation, eval_path), (training, train_path)):
        if int(record.get("seed", -1)) != expected_seed:
            raise ValueError(
                f"{path}: seed {record.get('seed')} != {expected_seed}"
            )
    assert_training_contract(task, training, train_path)
    censor = "none"
    derived = {}
    if task == "jump":
        reachable = evaluation.get("reachable_area", {})
        value = reachable.get("metric_value_m2")
        reliable_count = int(reachable.get("reliable_grid_count", -1))
        total_count = int(reachable.get("total_grid_count", -1))
        if reliable_count == 0 and total_count > 0:
            censor = "left"
        elif reliable_count == total_count and total_count > 0:
            censor = "right"
    else:
        summary = evaluation.get("summary", {})
        value = summary.get("metric_value")
        continuous_reliable = list(
            summary.get("continuous_reliable_mask", [])
        )
        if value is None and summary.get("lowest_level_passes") is False:
            value = 0.0
            censor = "left"
        elif continuous_reliable and all(continuous_reliable):
            censor = "right"
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"{eval_path}: metric is missing or non-finite")
    value = float(value)

    if task == "slope":
        levels = [float(level) for level in evaluation.get("levels", [])]
        reliable = continuous_reliable
        if not levels or len(levels) != len(reliable):
            raise ValueError(f"{eval_path}: invalid slope reliability data")
        if censor == "left":
            midpoint = 0.0
        else:
            passed = [index for index, flag in enumerate(reliable) if flag]
            if not passed:
                raise ValueError(f"{eval_path}: slope pass index is missing")
            last = passed[-1]
            midpoint = (
                0.5 * (levels[last] + levels[last + 1])
                if last + 1 < len(levels)
                else levels[last]
            )
        derived["theta90_midpoint_deg"] = midpoint

    return value, evaluation, training, censor, derived


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    candidates = load_candidates(args.candidates)
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    audit = {}

    for candidate in candidates:
        label = candidate["label"]
        design = candidate["design"]
        row = {
            "label": label,
            "thigh_scale": design[0],
            "shank_scale": design[1],
            "motor_scale": design[2],
            "role": candidate.get("role", ""),
            "status": "complete",
            "error": "",
        }
        audit[label] = {}
        try:
            for task, field in METRIC_FIELDS:
                value, evaluation, training, censor, derived = load_metric(
                    args.root,
                    label,
                    task,
                    design,
                    args.expected_seed,
                    args.eval_override_root,
                )
                row[field] = value
                row[f"{field}_censor"] = censor
                row.update(derived)
                audit[label][task] = {
                    "evaluation": evaluation,
                    "training": training,
                    "censor": censor,
                    "derived": derived,
                }
        except (FileNotFoundError, ValueError, KeyError) as exc:
            row["status"] = "incomplete"
            row["error"] = str(exc)
            if not args.allow_missing:
                raise
        records.append(row)

    nominal = next(
        (row for row in records if row["label"] == args.nominal_label),
        None,
    )
    if nominal is None or nominal["status"] != "complete":
        if not args.allow_missing:
            raise ValueError("complete nominal result is required")
        nominal = None

    normalized_fields = []
    for _, field in METRIC_FIELDS:
        normalized_fields.append(f"{field}_normalized")
    for row in records:
        if row["status"] != "complete" or nominal is None:
            row["equal_weight_score"] = None
            continue
        normalized = []
        for task, field in METRIC_FIELDS:
            source_field = NORMALIZATION_FIELDS[task]
            reference = float(nominal[source_field])
            if task not in ("slope", "jump") and reference <= 0.0:
                raise ValueError(
                    f"nominal reference {source_field} must be positive, "
                    f"got {reference}"
                )
            if task == "slope":
                value = (
                    1.0
                    + (float(row[source_field]) - reference)
                    / SLOPE_NORMALIZATION_SPAN_DEG
                )
            elif task == "jump":
                value = (
                    1.0
                    + (float(row[source_field]) - reference)
                    / JUMP_NORMALIZATION_SPAN_M2
                )
            else:
                value = float(row[source_field]) / reference
            row[f"{field}_normalized"] = value
            normalized.append(value)
        row["equal_weight_score"] = sum(normalized) / len(normalized)

    ranked = sorted(
        records,
        key=lambda row: (
            row["status"] == "complete",
            (
                row.get("equal_weight_score", -math.inf)
                if isinstance(row.get("equal_weight_score"), (int, float))
                and math.isfinite(row["equal_weight_score"])
                else -math.inf
            ),
        ),
        reverse=True,
    )
    for rank, row in enumerate(
        (row for row in ranked if row["status"] == "complete"),
        start=1,
    ):
        row["rank_equal_weight"] = rank

    metric_columns = [field for _, field in METRIC_FIELDS]
    censor_columns = [f"{field}_censor" for _, field in METRIC_FIELDS]
    table_fields = [
        "rank_equal_weight",
        "label",
        "thigh_scale",
        "shank_scale",
        "motor_scale",
        "role",
        *metric_columns,
        "theta90_midpoint_deg",
        *censor_columns,
        *normalized_fields,
        "equal_weight_score",
        "status",
        "error",
    ]
    write_csv(args.out / "five_metrics.csv", table_fields, ranked)

    sensitivity = []
    all_metric_sensitivity = []
    if nominal is not None:
        for row in records:
            if row["status"] != "complete":
                continue
            non_jump = [
                row[f"{field}_normalized"]
                for _, field in METRIC_FIELDS[:-1]
            ]
            jump = row["A_jump_m2_normalized"]
            sensitivity_row = {
                "label": row["label"],
                "thigh_scale": row["thigh_scale"],
                "shank_scale": row["shank_scale"],
                "motor_scale": row["motor_scale"],
            }
            for jump_weight in (0.10, 0.20, 0.30):
                score = (
                    (1.0 - jump_weight) * sum(non_jump) / len(non_jump)
                    + jump_weight * jump
                )
                sensitivity_row[f"score_jump_{int(100 * jump_weight)}pct"] = score
            sensitivity.append(sensitivity_row)
        for jump_weight in (10, 20, 30):
            key = f"score_jump_{jump_weight}pct"
            order = sorted(
                sensitivity, key=lambda row: row[key], reverse=True
            )
            for rank, row in enumerate(order, start=1):
                row[f"rank_jump_{jump_weight}pct"] = rank
        sensitivity_fields = [
            "label",
            "thigh_scale",
            "shank_scale",
            "motor_scale",
            "score_jump_10pct",
            "rank_jump_10pct",
            "score_jump_20pct",
            "rank_jump_20pct",
            "score_jump_30pct",
            "rank_jump_30pct",
        ]
        write_csv(
            args.out / "weight_sensitivity.csv",
            sensitivity_fields,
            sensitivity,
        )

        normalized_by_task = {
            task: f"{field}_normalized"
            for task, field in METRIC_FIELDS
        }
        for row in records:
            if row["status"] != "complete":
                continue
            sensitivity_row = {
                "label": row["label"],
                "thigh_scale": row["thigh_scale"],
                "shank_scale": row["shank_scale"],
                "motor_scale": row["motor_scale"],
            }
            for focus_task, focus_field in normalized_by_task.items():
                other_fields = [
                    field
                    for task, field in normalized_by_task.items()
                    if task != focus_task
                ]
                for focus_weight in (0.10, 0.20, 0.30):
                    score = (
                        focus_weight * row[focus_field]
                        + (1.0 - focus_weight)
                        * sum(row[field] for field in other_fields)
                        / len(other_fields)
                    )
                    key = (
                        f"score_{focus_task}_"
                        f"{int(100 * focus_weight)}pct"
                    )
                    sensitivity_row[key] = score
            all_metric_sensitivity.append(sensitivity_row)

        score_fields = []
        rank_fields = []
        for focus_task, _ in METRIC_FIELDS:
            for focus_weight in (10, 20, 30):
                score_key = f"score_{focus_task}_{focus_weight}pct"
                rank_key = f"rank_{focus_task}_{focus_weight}pct"
                ordered = sorted(
                    all_metric_sensitivity,
                    key=lambda row: row[score_key],
                    reverse=True,
                )
                for rank, row in enumerate(ordered, start=1):
                    row[rank_key] = rank
                score_fields.append(score_key)
                rank_fields.append(rank_key)
        all_sensitivity_fields = [
            "label",
            "thigh_scale",
            "shank_scale",
            "motor_scale",
        ]
        for score_field, rank_field in zip(score_fields, rank_fields):
            all_sensitivity_fields.extend((score_field, rank_field))
        write_csv(
            args.out / "all_metric_weight_sensitivity.csv",
            all_sensitivity_fields,
            all_metric_sensitivity,
        )

    summary = {
        "root": str(args.root.resolve()),
        "candidates": str(args.candidates.resolve()),
        "expected_seed": args.expected_seed,
        "eval_override_root": (
            str(args.eval_override_root.resolve())
            if args.eval_override_root is not None
            else None
        ),
        "nominal_label": args.nominal_label,
        "metric_weights": {field: 0.2 for _, field in METRIC_FIELDS},
        "normalization_fields": NORMALIZATION_FIELDS,
        "normalization_modes": {
            "clearance": "ratio_to_within_seed_nominal",
            "slope": (
                "1 + (theta90_midpoint_deg - within_seed_nominal) / "
                f"{SLOPE_NORMALIZATION_SPAN_DEG:g} deg"
            ),
            "payload": "ratio_to_within_seed_nominal",
            "speed": "ratio_to_within_seed_nominal",
            "jump": (
                "1 + (A_jump_m2 - within_seed_nominal) / "
                f"{JUMP_NORMALIZATION_SPAN_M2:g} m^2"
            ),
        },
        "slope_normalization_note": (
            "theta90_midpoint_deg is the midpoint of the final passing "
            "and first failing tested angle; raw theta90_deg remains the "
            "conservative discrete lower bound. A fixed 12 deg span is used "
            "for additive normalization because division by a near-zero "
            "nominal slope threshold is ill-conditioned."
        ),
        "jump_normalization_note": (
            "A fixed 0.16 m * 0.14 m = 0.0224 m^2 span is used for "
            "additive normalization. This preserves a finite score when "
            "the within-seed nominal policy has no grid cell meeting the "
            "90% reliability threshold; the zero area remains reported as "
            "left-censored."
        ),
        "complete_count": sum(
            row["status"] == "complete" for row in records
        ),
        "candidate_count": len(records),
        "ranking": ranked,
        "all_metric_weight_sensitivity": all_metric_sensitivity,
        "audit": audit,
    }
    with (args.out / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    print(
        f"[aggregate] complete={summary['complete_count']}/"
        f"{summary['candidate_count']} out={args.out}"
    )


if __name__ == "__main__":
    main()
