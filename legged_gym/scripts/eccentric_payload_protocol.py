"""Frozen paired static-payload inputs and independently counted physical events."""

import numpy as np

from legged_gym.scripts.payload_step_protocol import (
    DT, array_digest, make_scene as walking_scene, sha256_file, write_json,
    BOX_BOUND_RADIUS_M, PLATE_HALF_WIDTH_M, PLATE_TOP_Z_M,
)


POLICIES = {
    "rl": ("RL-only", "Jun05_03-01-38_wide2-30_rl_only_seed1", 11000),
    "mg": ("Model-guided", "Jun04_23-18-34_wide2-30_model_guided_seed1", 11000),
}
MASSES = (8, 13, 18, 23, 26)
PLACEMENTS = {"centered": 0.0, "eccentric": 0.15}
SEEDS = (12345, 20261018, 20261019)
MAIN_MASS, MAIN_SEED, MAIN_ENV = 13, 12345, 0
DURATION_S = 40.0


def make_scene(seed):
    scene = walking_scene(seed)
    scene["box_offset_body"][:] = [0.0, 0.0, 0.10]
    return scene


def batch_id(policy, mass, placement, seed):
    return f"{policy}_m{mass}_{placement}_eval{seed}"


def angles(root):
    q = root[..., 3:7].astype(np.float64)
    x, y, z, w = [q[..., i] for i in range(4)]
    roll = np.rad2deg(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    pitch = np.rad2deg(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))
    upright_cos = 1-2*(x*x+y*y)
    return roll, pitch, upright_cos


def first_sustained(times, condition, duration):
    start = None
    for i, active in enumerate(condition):
        if not active:
            start = None
        elif start is None:
            start = i
        if start is not None and times[i]-times[start] >= duration-1e-5:
            return float(times[start]), float(times[i])
    return None, None


def events(z, env):
    t = z["time_s"].astype(float)
    root = z["root_state"][:, env]
    _, _, upright_cos = angles(root)
    # A geometric fall is independent of the historical accumulated termination counter.
    fall_start, fall_confirm = first_sustained(t, (upright_cos < .5) | (root[:, 2] < .25), .10)
    on, generated = z["on_body"][:, env].astype(bool), z["has_load"][:, env].astype(bool)
    support = np.flatnonzero(on & generated)
    support_time = float(t[support[0]]) if support.size else None
    p = z["box_relative_position"][:, env]
    outside = (np.abs(p[:, 0]) > PLATE_HALF_WIDTH_M+BOX_BOUND_RADIUS_M)
    outside |= np.abs(p[:, 1]) > PLATE_HALF_WIDTH_M+BOX_BOUND_RADIUS_M
    outside |= p[:, 2]+BOX_BOUND_RADIUS_M < PLATE_TOP_Z_M
    eligible = generated & ~on & outside
    eligible &= t > support_time if support_time is not None else np.zeros_like(eligible)
    drop_start, drop_confirm = first_sustained(t, eligible, .20)
    failed = np.flatnonzero(z["first_failure"][:, env])
    termination = float(t[failed[0]]) if failed.size else None
    catch_failed = support_time is None or support_time > 2.0
    row = {
        "fall": int(fall_start is not None), "fall_start_s": fall_start, "fall_confirm_s": fall_confirm,
        "drop": int(drop_start is not None), "drop_start_s": drop_start, "drop_confirm_s": drop_confirm,
        "support_established_s": support_time, "catch_failed": int(catch_failed),
        "original_termination": int(termination is not None), "original_termination_s": termination,
        "fall_and_drop": int(fall_start is not None and drop_start is not None),
        "drop_before_fall": int(drop_start is not None and (fall_start is None or drop_start < fall_start)),
        "drop_after_or_with_fall": int(drop_start is not None and fall_start is not None and drop_start >= fall_start),
        "intact_40s": int(fall_start is None and drop_start is None and not catch_failed),
    }
    limits = [DURATION_S+DT/2]
    limits += [v for v in (fall_start, termination, drop_start) if v is not None]
    if catch_failed:
        limits.append(2.0)
    row["at_risk_end_exclusive_s"] = min(limits)
    return row
