"""Fixed commands for an existing unified policy, not a mode-transition controller."""

import numpy as np

from legged_gym.scripts.payload_step_protocol import DT, make_scene, array_digest, sha256_file, write_json
from legged_gym.scripts.eccentric_payload_protocol import angles, first_sustained


RUN = "Jul03_15-59-41_"
CHECKPOINT = 13000
SEEDS = (12345, 20261018, 20261019)
COMMANDS = np.array([[.5, 0.], [.3, .2], [.1, .4], [.5, 0.]], dtype=np.float32)
PHASE_NAMES = ("roll_initial", "mixed", "step_dominant", "roll_return")
MAIN_SEED, MAIN_ENV = 12345, 0


def phase_at_step(step, preceding_interval=False):
    value=np.asarray(step, dtype=np.int64)-(1 if preceding_interval else 0)
    return np.clip(value//500, 0, 3)


def command_at_step(step):
    return COMMANDS[phase_at_step(step)]


def first_failure(z, env):
    t=z["time_s"]
    terminated=np.flatnonzero(z["first_failure"][:,env])
    _,_,upright=angles(z["root_state"][:,env])
    fall,confirmed=first_sustained(t,(upright<.5)|(z["root_state"][:,env,2]<.25),.1)
    original=float(t[terminated[0]]) if terminated.size else None
    limits=[v for v in (fall,original) if v is not None]
    return {"fall_start_s":fall,"fall_confirm_s":confirmed,"original_termination_s":original,
            "first_invalid_s":min(limits) if limits else None}
