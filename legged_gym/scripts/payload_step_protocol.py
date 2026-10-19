"""Frozen inputs and event definitions for the matched payload-step evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np


POLICIES = {
    "rl_s1": ("RL-only", 1, "Jul05_15-13-14_rlonly_loadest", 16000, 20250705),
    "rl_s2": ("RL-only", 2, "Jul05_15-21-33_rlonly_loadest_s2", 16000, 20250706),
    "mg_s1": ("Model-guided", 1, "Jul05_23-30-43_mg_e1_s1", 11000, 20250705),
    "mg_s2": ("Model-guided", 2, "Jul05_23-33-51_mg_e1_s2", 16000, 20250706),
}
SEEDS = (12345, 20261016, 20261017)
MASS_KG = 5.9268727
DT = 0.02
DROP_S = 0.20
CATCH_DEADLINE_S = 2.0
BOX_BOUND_RADIUS_M = float(np.sqrt(3.0) * 0.05)
PLATE_HALF_WIDTH_M = 0.25
PLATE_TOP_Z_M = 0.0225


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(key.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def make_scene(seed, num_envs=30, steps=2000):
    def rng(stream):
        return np.random.default_rng(np.random.SeedSequence([seed, stream]))
    scene = {
        "body_mass_inertia_scale": rng(1).uniform(0.95, 1.05, (num_envs, 32)),
        "initial_joint_offset": rng(2).uniform(-0.5, 0.5, (num_envs, 8)),
        "initial_root_velocity": rng(3).uniform(-0.5, 0.5, (num_envs, 6)),
        "imu_pitch_roll_rad": np.deg2rad(rng(4).uniform(-1.2, 1.2, (num_envs, 2))),
        "box_offset_body": np.column_stack((rng(5).uniform(-0.17, 0.21, num_envs),
                                             rng(6).uniform(-0.19, 0.19, num_envs),
                                             np.full(num_envs, 0.10))),
        "gait": np.column_stack((rng(7).uniform(1.0, 1.5, num_envs),
                                  np.full(num_envs, 0.5), np.full(num_envs, 0.5),
                                  rng(8).uniform(0.0, 0.1, num_envs))),
        "noise_uniform": rng(9).uniform(-1, 1, (steps + 3, num_envs, 20)),
    }
    return {key: value.astype(np.float32) for key, value in scene.items()}


def trajectory_events(trace, env_id, condition, duration_s=40.0):
    time = trace["time_s"].astype(np.float64)
    failed = trace["first_failure"][..., env_id].astype(bool)
    failure_indices = np.flatnonzero(failed)
    failure_time = float(time[failure_indices[0]]) if failure_indices.size else None
    valid = np.ones(time.size, dtype=bool) if failure_time is None else time < failure_time - 1e-7
    on = trace["on_body"][:, env_id].astype(bool)
    generated = trace["has_load"][:, env_id].astype(bool)
    support_indices = np.flatnonzero(on & valid)
    support_time = float(time[support_indices[0]]) if support_indices.size else None
    row = {
        "env_id": env_id, "first_robot_failure_s": failure_time,
        "support_established_s": support_time, "first_drop_start_s": None,
        "first_drop_confirm_s": None, "catch_failed": False,
        "drop_confirmed": False, "robot_completed": failure_time is None,
        "loaded_completed": None, "eligible_samples": int(valid.sum()),
        "on_body_fraction": float(np.mean(on[valid])) if np.any(valid) else None,
    }
    if condition == "free_box":
        pos = trace["box_relative_position"][:, env_id]
        # A bounding sphere gives a conservative orientation-independent departure test.
        departed = (np.abs(pos[:, 0]) > PLATE_HALF_WIDTH_M + BOX_BOUND_RADIUS_M)
        departed |= np.abs(pos[:, 1]) > PLATE_HALF_WIDTH_M + BOX_BOUND_RADIUS_M
        departed |= pos[:, 2] + BOX_BOUND_RADIUS_M < PLATE_TOP_Z_M
        count, start = 0, None
        required = int(round(DROP_S / DT)) + 1
        for index in range(time.size):
            qualifies = (valid[index] and generated[index] and departed[index] and not on[index]
                         and support_time is not None and time[index] > support_time)
            if qualifies:
                if count == 0:
                    start = float(time[index])
                count += 1
                if count >= required:
                    row.update(first_drop_start_s=start, first_drop_confirm_s=float(time[index]),
                               drop_confirmed=True)
                    break
            else:
                count, start = 0, None
        observed_deadline = duration_s >= CATCH_DEADLINE_S
        row["catch_failed"] = bool(observed_deadline and
                                   (support_time is None or support_time > CATCH_DEADLINE_S))
        row["loaded_completed"] = bool(row["robot_completed"] and not row["catch_failed"]
                                        and not row["drop_confirmed"])
    end = duration_s + DT / 2
    if failure_time is not None:
        end = min(end, failure_time)
    if row["first_drop_start_s"] is not None:
        end = min(end, row["first_drop_start_s"])
    if row["catch_failed"]:
        end = min(end, CATCH_DEADLINE_S)
    row["motion_end_exclusive_s"] = float(end)
    return row


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
