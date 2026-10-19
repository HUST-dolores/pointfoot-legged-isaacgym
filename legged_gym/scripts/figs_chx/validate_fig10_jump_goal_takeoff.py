#!/usr/bin/env python3
"""Validate Figure 10 calculations against geometry and archived source methods."""

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import fig0_sixd_design_variables as style
import fig03_four_capability_definitions as geometry
import fig10_jump_goal_takeoff as figure


def validate():
    cfg = figure.load_policy(figure.DEFAULT_CONFIG)
    gravity = abs(cfg["sim"]["gravity"][2])
    time, x, z, vx, vz = figure.ballistic_reference(gravity)
    np.testing.assert_allclose([x[0], z[0], x[-1], z[-1]], [0, 0, figure.DISTANCE_M, 0], atol=1e-12)
    np.testing.assert_allclose(z.max(), figure.COM_RISE_M, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.gradient(x, time, edge_order=2), vx, atol=1e-11)
    np.testing.assert_allclose(np.gradient(z, time, edge_order=2), vz - gravity * time, atol=1e-11)

    robot = style.CollisionRobot(figure.DEFAULT_URDF)
    inertials = figure.load_inertials(figure.DEFAULT_URDF)
    push, air = figure.make_scenes(robot, inertials)
    c0 = figure.center_of_mass(push, inertials)
    ca = figure.center_of_mass(air, inertials)
    np.testing.assert_allclose(ca[2] - c0[2], figure.COM_RISE_M, atol=1e-12)
    assert np.linalg.norm(ca - air.frames["base_Link"][:3, 3]) > 0.1
    shift = np.array([0.17, -0.08, 0.23])
    geometry.translate_scene(push, shift)
    np.testing.assert_allclose(figure.center_of_mass(push, inertials), c0 + shift, atol=1e-12)
    geometry.translate_scene(push, -shift)
    for side in "LR":
        assert abs(figure.wheel_bottom(push, side)[2]) < 1e-10
        assert figure.wheel_bottom(air, side)[2] > 0
    clearance = min(figure.wheel_bottom(air, side)[2] for side in "LR")
    assert abs(clearance - figure.COM_RISE_M) > 0.05

    wheel = np.mean([push.centers[f"wheel_{side}_Joint"] for side in "LR"], axis=0)
    vector = (c0 - wheel)[[0, 2]]
    length, angle = np.linalg.norm(vector), np.arctan2(vector[0], vector[1])
    np.testing.assert_allclose(length * np.array([np.sin(angle), np.cos(angle)]), vector, atol=1e-12)
    dl, da, step = 0.7, 0.4, 1e-6
    coordinates = lambda t: (length + dl * t) * np.array([np.sin(angle + da * t), np.cos(angle + da * t)])
    fd = (coordinates(step) - coordinates(-step)) / (2 * step)
    derivative = dl * np.array([np.sin(angle), np.cos(angle)]) + length * da * np.array([np.cos(angle), -np.sin(angle)])
    np.testing.assert_allclose(fd, derivative, atol=1e-10)
    assert np.all(derivative > 0)

    lo_h, hi_h = cfg["commands"]["jump_height_cmd_range"]
    lo_d, hi_d = cfg["commands"]["jump_dist_cmd_range"]
    heights, distances = np.meshgrid(np.linspace(lo_h, hi_h, 3), np.linspace(lo_d, hi_d, 3), indexing="ij")
    heights, distances = heights.ravel(), distances.ravel()
    n = len(heights)
    target_vz, target_vx, _ = figure.policy_targets(cfg, heights, distances)
    methods = {"_reward_jump_takeoff_vz", "_reward_jump_fwd_vel", "_compute_torques"}
    tree = ast.parse(figure.DEFAULT_ENV.read_text(encoding="utf-8"))
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in methods]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<archived Figure 10 methods>", "exec"), namespace)
    dtype = torch.float64
    tensor = lambda data: torch.as_tensor(data, dtype=dtype)
    zeros = lambda *shape: torch.zeros(*shape, dtype=dtype)
    ones = lambda *shape: torch.ones(*shape, dtype=dtype)
    env = SimpleNamespace(
        cfg=SimpleNamespace(env=SimpleNamespace(**cfg["env"]), control=SimpleNamespace(**cfg["control"]),
                            rewards=SimpleNamespace(**cfg["rewards"]), commands=SimpleNamespace(**cfg["commands"])),
        num_envs=n, jump_height_cmd=tensor(heights[:, None]), jump_dist_cmd=tensor(distances[:, None]),
        jump_takeoff_lat_ratio=zeros(n), jump_takeoff_roll=zeros(n), jump_takeoff_fire=torch.ones(n, dtype=torch.bool),
        jump_takeoff_base_vz=tensor(target_vz), jump_takeoff_quality=ones(n),
        jump_takeoff_dir=tensor(np.tile([1.0, 0.0], (n, 1))), jumped_flag=torch.zeros(n, dtype=torch.bool),
        jump_state=torch.full((n,), 2, dtype=torch.long), foot_velocities=zeros(n, 2, 3),
        _jump_dist_long_gain=lambda: ones(n),
        dof_pos=zeros(n, 8), dof_vel=zeros(n, 8), default_dof_pos=zeros(n, 8),
        p_gains=ones(n, 8), d_gains=zeros(n, 8), torques_scale=ones(n, 8),
        _motor_available_torque_limits=lambda: ones(n, 8) * 1000,
        _jump_phase_elapsed=lambda: ones(n) * 0.20,
    )
    env.foot_velocities[:, :, 0] = tensor(target_vx)[:, None]
    torch.testing.assert_close(namespace["_reward_jump_takeoff_vz"](env), ones(n))
    torch.testing.assert_close(namespace["_reward_jump_fwd_vel"](env), ones(n))

    def forbidden_reference(*args):
        raise AssertionError("The archived trajectory-disabled policy must not call the explicit reference generator")

    env._apply_jump_explicit_trajectory = forbidden_reference
    env.jump_state[:] = 1
    torque = namespace["_compute_torques"](env, zeros(n, 8))
    h_signal = (heights - 0.5 * (lo_h + hi_h)) / (0.5 * (hi_h - lo_h))
    d_signal = (distances - 0.5 * (lo_d + hi_d)) / (0.5 * (hi_d - lo_d))
    expected = zeros(n, 8)
    expected[:, 1] = tensor(cfg["control"]["jump_hip_distance_residual"] * d_signal)
    expected[:, 5] = -expected[:, 1]
    expected[:, 2] = tensor(-cfg["control"]["jump_knee_height_residual"] * h_signal)
    expected[:, 6] = -expected[:, 2]
    torch.testing.assert_close(torque, expected)
    env._jump_phase_elapsed = lambda: zeros(n)
    torch.testing.assert_close(namespace["_compute_torques"](env, zeros(n, 8)), zeros(n, 8))
    env.jump_state[:] = 2
    env._jump_phase_elapsed = lambda: ones(n) * 0.20
    torch.testing.assert_close(namespace["_compute_torques"](env, zeros(n, 8)), zeros(n, 8))
    return {
        "scope": "Analytical/geometry/source-method consistency, NOT simulator performance or trajectory feasibility",
        "ballistic_endpoint_and_apex": "pass",
        "ballistic_velocity_finite_difference": "pass",
        "centroid_translation_and_distinction_from_body_origin": "pass",
        "com_rise_distinct_from_wheel_clearance": "pass",
        "virtual_extension_angle_velocity_finite_difference": "pass",
        "archived_reward_target_cases": n,
        "archived_command_action_correction_cases": n,
        "action_test_controller": "Identity P gain, zero D gain and loose limits to isolate the source action mapping",
        "explicit_trajectory_not_invoked": "pass",
        "prepush_and_flight_command_correction_gates": "pass",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = validate()
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
