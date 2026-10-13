#!/usr/bin/env python3
"""Run a frozen five-seed paired tiebreak for the top two co-designs."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy import stats

import run_four_objective_codesign as combined


OUTPUT = combined.OUTPUT
NEW_SEEDS = tuple(
    int(value)
    for value in os.environ.get(
        "FOUR_OBJ_TIEBREAK_SEEDS",
        "20260732,20260733,20260734,20260735,20260736",
    ).split(",")
    if value.strip()
)


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="ascii")
    os.replace(temporary, path)


def main():
    started_at = combined.base.now_iso()
    atomic_text(OUTPUT / "TIEBREAK_STARTED", started_at + "\n")
    (OUTPUT / "TIEBREAK_DONE").unlink(missing_ok=True)
    (OUTPUT / "TIEBREAK_EXIT_CODE").unlink(missing_ok=True)

    source_target = (
        OUTPUT
        / "tiebreak_code_snapshot"
        / "run_four_objective_tiebreak.py"
    )
    source_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), source_target)

    final = combined.base.read_json(OUTPUT / "FINAL_RESULTS.json")
    original_rows = final["finalists"]
    if len(original_rows) < 2:
        raise RuntimeError("formal result does not contain two finalists")
    original_ids = [
        original_rows[0]["design_id"],
        original_rows[1]["design_id"],
    ]
    designs = [
        np.asarray(row["design_vector"], dtype=np.float64)
        for row in original_rows[:2]
    ]
    combined.base.atomic_json(
        OUTPUT / "TIEBREAK_MANIFEST.json",
        {
            "started_at": started_at,
            "purpose": (
                "confirmatory paired-seed ranking of the original "
                "five-seed formal top two"
            ),
            "design_ids": original_ids,
            "design_vectors": [
                [float(value) for value in design] for design in designs
            ],
            "new_seeds": list(NEW_SEEDS),
            "fidelity": "formal",
            "weights": combined.WEIGHTS,
            "normalization": (
                "frozen from the original five-seed formal run"
            ),
            "protocols_unchanged": True,
        },
    )

    new_records = combined.evaluate_raw(
        "four_objective_tiebreak_repeats",
        designs,
        fidelity="formal",
        seeds=NEW_SEEDS,
    )
    combined.base.save_records(
        OUTPUT / "TIEBREAK_NEW_RAW_RECORDS.json", new_records
    )

    old_records = combined.base.read_json(
        OUTPUT / "FORMAL_RAW_RECORDS.json"
    )
    selected_old = [
        record
        for record in old_records
        if record["design_id"] in original_ids
    ]
    all_records = selected_old + new_records
    combined.base.save_records(
        OUTPUT / "TIEBREAK_ALL_TEN_SEED_RAW_RECORDS.json",
        all_records,
    )
    anchors = final["formal_normalization"]
    rows = combined.aggregate_formal(all_records, anchors)
    rows_by_id = {row["design_id"]: row for row in rows}

    first = rows_by_id[original_ids[0]]
    second = rows_by_id[original_ids[1]]
    first_scores = np.asarray(first["score_by_seed"], dtype=np.float64)
    second_scores = np.asarray(second["score_by_seed"], dtype=np.float64)
    if first["seeds"] != second["seeds"]:
        raise RuntimeError("tiebreak seed order mismatch")
    difference = first_scores - second_scores
    mean_difference = float(np.mean(difference))
    std_difference = float(np.std(difference, ddof=1))
    sem = std_difference / np.sqrt(len(difference))
    critical = float(stats.t.ppf(0.975, len(difference) - 1))
    interval = [
        mean_difference - critical * sem,
        mean_difference + critical * sem,
    ]
    paired_test = stats.ttest_rel(first_scores, second_scores)
    winner = max(
        rows,
        key=lambda row: (
            row["equal_weight_score_mean"],
            -row["equal_weight_score_std"],
            -row["motor_module_mass_kg"],
        ),
    )

    result = {
        "winner": winner,
        "rows": rows,
        "original_rank_order": original_ids,
        "paired_first_minus_second": {
            "seeds": first["seeds"],
            "differences": [float(value) for value in difference],
            "mean": mean_difference,
            "std": std_difference,
            "confidence_interval_95_t": interval,
            "paired_t_p_descriptive": float(paired_test.pvalue),
            "first_win_count": int(np.sum(difference > 0.0)),
            "second_win_count": int(np.sum(difference < 0.0)),
            "tie_count": int(np.sum(difference == 0.0)),
        },
        "normalization": anchors,
    }
    combined.base.atomic_json(
        OUTPUT / "TIEBREAK_TEN_SEED_RESULTS.json", result
    )

    lines = [
        "# Ten-seed paired tiebreak",
        "",
        "Original formal normalization anchors and 25% weights are frozen.",
        "",
        "| rank | design | score mean +/- std | payload | speed | "
        "clearance | jump |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(
        sorted(
            rows,
            key=lambda item: -item["equal_weight_score_mean"],
        ),
        start=1,
    ):
        raw = row["raw_mean_by_task"]
        lines.append(
            f"| {rank} | `{[round(value, 6) for value in row['design_vector']]}` "
            f"| {row['equal_weight_score_mean']:.6f} +/- "
            f"{row['equal_weight_score_std']:.6f} | "
            f"{raw['payload']:.6g} | {raw['speed']:.6g} | "
            f"{raw['clearance']:.6g} | {raw['jump']:.7g} |"
        )
    lines.extend(
        [
            "",
            "Paired original-rank-1 minus original-rank-2 score:",
            "",
            f"- mean: `{mean_difference:+.6f}`;",
            f"- standard deviation: `{std_difference:.6f}`;",
            (
                "- paired 95% t interval: "
                f"`[{interval[0]:+.6f}, {interval[1]:+.6f}]`;"
            ),
            (
                "- descriptive paired t-test p: "
                f"`{float(paired_test.pvalue):.6g}`;"
            ),
            (
                f"- seed wins: original rank 1 `{int(np.sum(difference > 0))}`, "
                f"original rank 2 `{int(np.sum(difference < 0))}`."
            ),
        ]
    )
    (OUTPUT / "TIEBREAK_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

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
    combined.base.atomic_json(
        OUTPUT / "TIEBREAK_LIFECYCLE.json",
        {
            "completed_at": combined.base.now_iso(),
            "new_job_count": len(new_records),
            "combined_record_count": len(all_records),
            "gpu_state_after_children": gpu_state,
        },
    )
    combined.base.update_status(
        stage="tiebreak_complete",
        completed_jobs=1,
        total_jobs=1,
        failed_jobs=[],
        eta_seconds=0.0,
    )
    atomic_text(OUTPUT / "TIEBREAK_EXIT_CODE", "0\n")
    atomic_text(
        OUTPUT / "TIEBREAK_DONE",
        combined.base.now_iso() + "\n",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        atomic_text(OUTPUT / "TIEBREAK_EXIT_CODE", "1\n")
        raise
