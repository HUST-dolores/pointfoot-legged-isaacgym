"""Independent NumPy/SciPy audit of recorded geometry, impulses and retained trials."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import CORNERS, measure, write_json


def verify_batch(path):
    meta = json.loads((path / "result.json").read_text())
    cases = json.loads((path / "cases.json").read_text())
    scenarios = json.loads((path / "scenarios.json").read_text())
    materials = json.loads((path / "materials.json").read_text())
    settings = meta["settings"]
    with np.load(path / "trace.npz", allow_pickle=False) as f:
        root, box = f["root"].astype(float), f["box"].astype(float)
        shape = root.shape[:2]
        rr = Rotation.from_quat(root[..., 3:7].reshape(-1, 4)).as_matrix().reshape(*shape, 3, 3)
        rb = Rotation.from_quat(box[..., 3:7].reshape(-1, 4)).as_matrix().reshape(*shape, 3, 3)
        plate = root[..., :3] + np.einsum("...ij,j->...i", rr, [0., 0., settings["plate_z"]])
        derived = measure(box[..., :3], rb, plate, rr, f["load_contact"].astype(float))
        active = f["has_load"].astype(bool) & (f["time_s"] > .54)
        margin_error = float(np.abs(derived["footprint_margin_m"] - f["margins"].min(-1))[active].max())
        gap_error = float(np.abs(derived["normal_gap_m"] - f["gap"])[active].max())
        force_error = float(np.abs(derived["normal_force_N"] - f["normal_force"])[active].max())
        assert margin_error < 1e-4 and gap_error < 1e-4 and force_error < 1e-3, (margin_error, gap_error, force_error)
        orientation = np.einsum("...ji,...jk->...ik", rr, rb)
        tilt_deg = np.degrees(np.arccos(np.clip(orientation[..., 2, 2], -1., 1.)))
        local = np.einsum("...ji,...j->...i", rr, box[..., :3] - plate)
        corners = local[..., None, :] + np.einsum("...ij,cj->...ci", orientation, CORNERS)
        lo, hi = corners.min(-2), corners.max(-2)
        overlap = ((hi[..., :2] > -.25) & (lo[..., :2] < .25)).all(-1)
        supported = overlap & (np.abs(derived["normal_gap_m"]) < .015) & (derived["normal_force_N"] > .1 * settings["mass"] * 9.81) & (orientation[..., 2, 2] > .5)
        assert np.array_equal(supported[active], f["supported"][active]), "support classification mismatch"
        departure = ((~overlap) & (derived["normal_gap_m"] < -.015)) | (hi[..., 2] < settings["plate_top"] - .04)
        falling = (-rr[..., 2, 2] > -.5) | (root[..., 2] < .35)
        drop_count = np.zeros(shape[1], dtype=int)
        fall_count = np.zeros(shape[1], dtype=int)
        for k in range(shape[0]):
            drop_active = f["has_load"][k].astype(bool) & (f["time_s"][k] > settings["spawn_time"] + .2)
            drop_count = np.where(drop_active & departure[k], drop_count + 1, 0)
            fall_count = np.where(falling[k], fall_count + 1, 0)
            assert np.array_equal(drop_count >= round(settings["drop_confirm"] / .02), f["drop"][k]), "drop event reconstruction mismatch"
            assert np.array_equal(fall_count >= round(settings["fall_confirm"] / .02), f["fall"][k]), "fall event reconstruction mismatch"
        assert np.all(f["commands_and_payload_context"][..., :3] == 0.), "executed motion command changed"
        strict_slip = []
        # A failed box remains physical; all assigned trials and full post-failure traces are retained.
        for i, case in enumerate(cases):
            t = f["time_s"][:, i]
            force = f["force"][:, i].astype(float)
            nonzero = np.linalg.norm(force, axis=-1) > .01
            expected_samples = 2 if case["nominal_dv"] > 0 else 0
            assert int(nonzero.sum()) == expected_samples
            if nonzero.any():
                np.testing.assert_allclose(t[nonzero], [2.02, 2.04], atol=1e-5)
                impulse = np.linalg.norm(force.sum(0) * .02)
                assert abs(impulse - settings["mass"] * case["nominal_dv"]) < 1e-4
            assert np.all(force[t > 2.05] == 0.)
            assert int(f["spawns"][:, i].max()) == 1
            mask = f["has_load"][:, i].astype(bool)
            assert np.ptp(f["pose_writes"][mask, i]) == 0
            drop = f["drop"][:, i].astype(bool)
            fall = f["fall"][:, i].astype(bool)
            end = t >= settings["duration"] - .5
            success = not (drop | fall).any() and supported[end, i].mean() >= .8
            assert case["success"] == success
            strict = active[:, i] & supported[:, i] & (derived["footprint_margin_m"][:, i] >= 0.) & (tilt_deg[:, i] < 12.)
            strict &= (np.abs(derived["normal_gap_m"][:, i]) < .008) & (derived["normal_force_N"][:, i] > .2 * settings["mass"] * 9.81)
            bottom_delta = derived["bottom_center_plate"][:, i, :2] - np.asarray(scenarios[i]["spawn_xy"])
            strict_slip.append(float(np.linalg.norm(bottom_delta[strict], axis=-1).max()) if strict.any() else 0.)
        for material in materials:
            assert material["wheel_friction_unchanged"]
            assert abs(material["mass"] - settings["mass"]) < 1e-5
            assert np.allclose(material["box_friction"], settings["friction"], atol=1e-6)
            assert np.allclose([material["robot_friction"][i] for i in material["base_shapes"]], settings["friction"], atol=1e-6)
        assert np.isfinite(root).all() and np.isfinite(box).all()
        assert meta["verified_spawns_next_physics_step"] == len(cases)
        assert meta["cleanup_completed"] and abs(meta["actual_final_time_s"] - settings["duration"]) < 1e-4
    return {"directory": str(path), "trials": len(cases), "margin_reconstruction_max_error_m": margin_error,
            "gap_reconstruction_max_error_m": gap_error, "force_reconstruction_max_error_N": force_error,
            "finite_states": True, "finite_duration_completed": True,
            "single_spawn_no_post_spawn_pose_writes": True, "actual_impulses_verified": True,
            "success_recomputed_from_states": True, "executed_motion_commands_zero": True,
            "drop_and_fall_events_rebuilt_from_geometry_and_persistence": True,
            "strict_supported_flat_sliding_over_2cm_trials": sum(x >= .02 for x in strict_slip),
            "strict_supported_flat_max_displacement_m": max(strict_slip),
            "strict_sliding_uses_bottom_center_gap_force_tilt_and_full_footprint": True,
            "mass_and_material_readback_verified": True}


def verify_checkpoint(checkpoint, original):
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reference = torch.load(original, map_location="cpu", weights_only=False)
    assert state["architecture"] == "bounded_residual"
    for key, value in reference["encoder_state_dict"].items():
        assert torch.equal(value, state["encoder_state_dict"][key]), "encoder changed"
    compared = 0
    for key, value in reference["model_state_dict"].items():
        if key.startswith("actor."):
            target = "actor.base." + key[len("actor."):]
            assert torch.equal(value, state["model_state_dict"][target]), "frozen balancing actor changed"
            compared += 1
    return {"checkpoint": str(checkpoint), "encoder_bitwise_equal_to_original": True,
            "base_actor_bitwise_equal_to_original": True, "base_actor_tensors_checked": compared}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, nargs="+", default=[])
    p.add_argument("--original-checkpoint", type=Path)
    a = p.parse_args()
    result = {"batches": [verify_batch(path) for path in a.batches],
              "checkpoints": [verify_checkpoint(path, a.original_checkpoint) for path in a.checkpoints],
              "status": "passed"}
    write_json(a.output, result)
    print(json.dumps(result, indent=2))
