"""Restore class-style BaseConfig objects from a saved run's JSON files."""

import json
import os


def apply_config_dict(config_object, values):
    """Recursively apply a JSON-compatible mapping to a BaseConfig object."""
    applied = 0
    for key, value in values.items():
        if key.startswith("__"):
            continue
        if isinstance(value, dict):
            current = getattr(config_object, key, None)
            if current is not None and hasattr(current, "__dict__"):
                applied += apply_config_dict(current, value)
            else:
                # Some config leaves are mappings by design, such as
                # init_state.default_joint_angles.
                setattr(config_object, key, dict(value))
                applied += 1
        else:
            setattr(config_object, key, value)
            applied += 1
    return applied


def load_saved_run_configs(env_cfg, train_cfg, log_root, run_name):
    """Load and apply env_cfg.json and train_cfg.json from one run directory."""
    run_dir = os.path.join(log_root, str(run_name))
    env_path = os.path.join(run_dir, "env_cfg.json")
    train_path = os.path.join(run_dir, "train_cfg.json")
    missing = [
        path for path in (env_path, train_path) if not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(
            "saved run config is incomplete: " + ", ".join(missing)
        )
    with open(env_path, encoding="utf-8") as handle:
        env_values = json.load(handle)
    with open(train_path, encoding="utf-8") as handle:
        train_values = json.load(handle)
    env_count = apply_config_dict(env_cfg, env_values)
    train_count = apply_config_dict(train_cfg, train_values)
    print(
        f"[saved-config] restored {env_count} env and {train_count} train "
        f"values from {run_dir}",
        flush=True,
    )
    return run_dir
