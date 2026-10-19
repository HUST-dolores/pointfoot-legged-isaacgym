"""Freeze the validation-selected models and explicitly enumerate formal batches."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

from .geometry import sha256, write_json
from .robot import ROOT


def main(a):
    data = ROOT / "logs/payload_retention_20261018"
    reports = data / "reports"
    reports.mkdir(exist_ok=True)
    selected = json.loads((data / "validation_residual/selected_checkpoints.json").read_text())
    assert set(selected) == {"seed_303", "seed_404"}
    for seed in (303, 404):
        completed = json.loads((data / f"train_residual_gpu1_s{seed}/result.json").read_text())
        assert completed["iterations"] == 2000 and completed["cleanup_completed"]
        assert completed["base_actor_unchanged"] and completed["encoder_unchanged"]
    formal = ["formal_baseline_seed23456", "baseline_formal/formal_baseline_seed34567", "baseline_formal/formal_baseline_seed45678"]
    remote_root = Path("/home/ps/mydrive/xu/payload_slide_lab_20261018/repo")
    for record in selected.values():
        seed, iteration = record["training_seed"], record["iteration"]
        checkpoint = ROOT / Path(record["checkpoint"]).relative_to(remote_root)
        assert sha256(checkpoint) == record["checkpoint_sha256"]
        for evaluation_seed in (23456, 34567, 45678):
            name = f"heldout_precomputed_s{seed}_i{iteration}/s{seed}_i{iteration}_seed{evaluation_seed}"
            result = json.loads((data / name / "result.json").read_text())
            assert result["checkpoint_sha256"] == record["checkpoint_sha256"]
            assert result["seed"] == evaluation_seed and result["trials"] == 136 and result["cleanup_completed"]
            formal.append(name)
        alias = data / f"selected_seed{seed}.pt"
        if alias.exists():
            assert sha256(alias) == record["checkpoint_sha256"]
        else:
            shutil.copy2(checkpoint, alias)
    primary = max(selected.values(), key=lambda r: (r["control_success"] == 1., r["in_domain_success"], -r["fall_rate"], -r["iteration"], -r["training_seed"]))
    primary_alias = data / "selected_policy.pt"
    if primary_alias.exists():
        assert sha256(primary_alias) == primary["checkpoint_sha256"]
    else:
        shutil.copy2(data / f"selected_seed{primary['training_seed']}.pt", primary_alias)
    selection = {"primary": primary, "selected_by_training_seed": selected, "formal_directories": formal,
        "selection_uses_validation_seed_only": 12345, "formal_seeds": [23456, 34567, 45678],
        "checkpoint_selection_not_based_on_formal_results": True, "retained_updates_per_training_seed": 2000,
        "effective_sample_budget_per_training_seed": 2000 * 512 * 24,
        "actual_compute_also_includes_discarded_updates_and_diagnostics": True,
        "simulated_payload_ground_truth_feedback": True}
    write_json(data / "final_selection.json", selection)
    timing = {}
    for seed in (303, 404):
        section_names = (["train_residual_pilot_s303"] if seed == 303 else []) + [f"train_residual_s{seed}", f"train_residual_gpu1_s{seed}"]
        sections = []
        for name in section_names:
            folder = data / name
            metadata = json.loads((folder / "metadata.json").read_text())
            result_file = folder / "result.json"
            stop_file = folder / "STOPPED.json"
            if result_file.exists():
                result = json.loads(result_file.read_text())
                elapsed = result["elapsed_s"]
                completed = True
            else:
                result = json.loads(stop_file.read_text())
                elapsed = result["old_elapsed_s_last_report"]
                completed = False
            sections.append({"directory": name, "started": metadata["started"], "elapsed_s": elapsed,
                             "elapsed_from_completed_run": completed, "gpu": metadata["visible_gpu"]})
        timing[str(seed)] = {"sections": sections, "sum_section_wall_s_approx": sum(s["elapsed_s"] for s in sections),
                            "not_exclusive_gpu_compute_seconds": True}
    write_json(reports / "training_time_sections.json", timing)
    if a.mark_complete:
        verification = json.loads((reports / "verification.json").read_text())
        process_audit = json.loads((reports / "process_audit.json").read_text())
        summary = json.loads((reports / "summary.json").read_text())
        assert verification["status"] == "passed" and not process_audit["matching_retention_processes"]
        assert summary["unique_trials"] == 1224
        write_json(data / "task_finished.json", {"finished": datetime.now().isoformat(), **selection,
            "all_retention_processes_exited": True, "formal_trials": summary["unique_trials"],
            "independent_verification_passed": True, "other_training_processes_preserved": process_audit["other_processes_preserved"],
            "main_nominal_urdf_unchanged": True, "first_stage_prototype_not_stationary_position_hold": True})
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mark-complete", action="store_true")
    main(p.parse_args())
