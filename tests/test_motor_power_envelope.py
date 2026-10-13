import math
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "legged_gym"
    / "utils"
    / "motor_cost.py"
)
SPEC = importlib.util.spec_from_file_location("motor_cost_under_test", MODULE_PATH)
MOTOR_COST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOTOR_COST)
available_motor_torque_nm = MOTOR_COST.available_motor_torque_nm
relative_motor_power_w = MOTOR_COST.relative_motor_power_w


def test_nominal_wheel_power_anchor():
    assert relative_motor_power_w(0.365, 1.0) == pytest.approx(164.25)


def test_low_speed_uses_peak_torque():
    assert available_motor_torque_nm(
        40.0,
        0.365,
        0.6,
        0.0,
    ) == pytest.approx(24.0)


def test_high_speed_uses_power_limit():
    speed = 5.0 / 0.127
    expected = 0.365 * 450.0 / speed
    assert available_motor_torque_nm(
        40.0,
        0.365,
        1.0,
        speed,
    ) == pytest.approx(expected)


def test_linear_power_scaling_preserves_monotonic_design_order():
    speed = 5.0 / 0.127
    limits = [
        available_motor_torque_nm(
            40.0,
            0.365,
            scale,
            speed,
        )
        for scale in (0.6, 1.0, 1.6)
    ]
    assert limits[0] < limits[1] < limits[2]
    assert limits[2] / limits[0] == pytest.approx(1.6 / 0.6)


@pytest.mark.parametrize(
    "bad_kwargs",
    (
        {"nominal_peak_torque_nm": 0.0},
        {"nominal_mass_kg": 0.0},
        {"scale": 0.0},
        {"specific_power_w_per_kg": 0.0},
        {"power_exponent": 0.0},
        {"min_speed_rad_s": 0.0},
    ),
)
def test_invalid_power_model_inputs_raise(bad_kwargs):
    kwargs = {
        "nominal_peak_torque_nm": 40.0,
        "nominal_mass_kg": 0.365,
        "scale": 1.0,
        "output_speed_rad_s": math.pi,
        "specific_power_w_per_kg": 450.0,
        "power_exponent": 1.0,
        "min_speed_rad_s": 1.0e-3,
    }
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError):
        available_motor_torque_nm(**kwargs)
