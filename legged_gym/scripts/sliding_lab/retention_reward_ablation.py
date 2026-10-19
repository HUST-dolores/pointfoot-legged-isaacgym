"""Explicit, auditable single-term reward overrides for resumed retention jobs."""

from dataclasses import replace
import math


ARMS = {
    "control": {"alive_weight": 2.0, "drop_cost": 12.0},
    "alive4": {"alive_weight": 4.0, "drop_cost": 12.0},
    "drop24": {"alive_weight": 2.0, "drop_cost": 24.0},
}


def override_rewards(settings, alive_weight=None, drop_cost=None):
    updates = {}
    for name, value in (("alive_weight", alive_weight), ("drop_cost", drop_cost)):
        if value is not None:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            updates[name] = float(value)
    return replace(settings, **updates)


def settings_changes(before, after):
    return {key: {"before": value, "after": after.dictionary()[key]}
            for key, value in before.dictionary().items()
            if value != after.dictionary()[key]}
