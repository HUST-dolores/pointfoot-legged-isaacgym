"""Isolated Isaac Gym payload-on-deck sliding diagnostic.

This script intentionally does not load the robot or policy.  It creates one
fixed deck box and one dynamic payload box, then applies either a velocity kick
or a deck slope.  Use it to separate contact/material behavior from the robot's
closed-loop platform motion.
"""

import argparse
import math
import os
import sys

from isaacgym import gymapi, gymtorch
import torch

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _make_sim(gym, args):
    sim_params = gymapi.SimParams()
    sim_params.dt = args.dt
    sim_params.substeps = args.substeps
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = args.pipeline == "gpu"
    sim_params.physx.use_gpu = args.pipeline == "gpu"
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = args.num_position_iterations
    sim_params.physx.num_velocity_iterations = args.num_velocity_iterations
    sim_params.physx.contact_offset = args.contact_offset
    sim_params.physx.rest_offset = args.rest_offset
    sim_params.physx.friction_offset_threshold = args.friction_offset_threshold
    sim_params.physx.friction_correlation_distance = args.friction_correlation_distance
    compute_device = 0 if args.pipeline == "gpu" else -1
    graphics_device = -1 if args.headless else 0
    return gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)


def _set_asset_friction(gym, asset, friction):
    props = gym.get_asset_rigid_shape_properties(asset)
    for p in props:
        p.friction = float(friction)
        if friction <= 1e-9:
            p.rolling_friction = 0.0
            p.torsion_friction = 0.0
    gym.set_asset_rigid_shape_properties(asset, props)


def _quat_y(theta):
    return gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 1.0, 0.0), theta)


def _rot_y(theta, v):
    c, s = math.cos(theta), math.sin(theta)
    x, y, z = v
    return torch.tensor(
        [c * x + s * z, y, -s * x + c * z],
        dtype=torch.float32,
    )


def _vec3(v):
    return gymapi.Vec3(float(v[0]), float(v[1]), float(v[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument(
        "--real_time",
        action="store_true",
        default=False,
        help="When using the viewer, sync playback to wall-clock time.",
    )
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--deck_size_x", type=float, default=0.50)
    parser.add_argument("--deck_size_y", type=float, default=0.50)
    parser.add_argument("--deck_size_z", type=float, default=0.02)
    parser.add_argument("--deck_z", type=float, default=0.50)
    parser.add_argument("--load_size", type=float, default=0.10)
    parser.add_argument("--load_mass", type=float, default=20.0)
    parser.add_argument("--load_x", type=float, default=0.18)
    parser.add_argument("--load_y", type=float, default=0.00)
    parser.add_argument("--load_friction", type=float, default=0.0)
    parser.add_argument("--deck_friction", type=float, default=0.0)
    parser.add_argument(
        "--slope_deg",
        type=float,
        default=0.0,
        help="Positive value tilts the deck so +x is downhill.",
    )
    parser.add_argument("--kick_vx", type=float, default=0.0)
    parser.add_argument("--kick_vy", type=float, default=0.0)
    parser.add_argument("--kick_time", type=float, default=0.5)
    parser.add_argument("--num_position_iterations", type=int, default=4)
    parser.add_argument("--num_velocity_iterations", type=int, default=0)
    parser.add_argument("--contact_offset", type=float, default=0.01)
    parser.add_argument("--rest_offset", type=float, default=0.0)
    parser.add_argument("--friction_offset_threshold", type=float, default=0.04)
    parser.add_argument("--friction_correlation_distance", type=float, default=0.025)
    args = parser.parse_args()

    gym = gymapi.acquire_gym()
    sim = _make_sim(gym, args)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym sim")

    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create Isaac Gym viewer")

    env = gym.create_env(
        sim,
        gymapi.Vec3(-1.0, -1.0, 0.0),
        gymapi.Vec3(1.0, 1.0, 1.0),
        1,
    )

    deck_opts = gymapi.AssetOptions()
    deck_opts.fix_base_link = True
    deck_asset = gym.create_box(
        sim, args.deck_size_x, args.deck_size_y, args.deck_size_z, deck_opts
    )
    _set_asset_friction(gym, deck_asset, args.deck_friction)

    load_opts = gymapi.AssetOptions()
    load_opts.fix_base_link = False
    load_opts.density = args.load_mass / (args.load_size ** 3)
    load_asset = gym.create_box(
        sim, args.load_size, args.load_size, args.load_size, load_opts
    )
    _set_asset_friction(gym, load_asset, args.load_friction)

    theta = math.radians(args.slope_deg)
    deck_center = torch.tensor([0.0, 0.0, args.deck_z], dtype=torch.float32)
    deck_pose = gymapi.Transform()
    deck_pose.p = _vec3(deck_center)
    deck_pose.r = _quat_y(theta)
    deck_handle = gym.create_actor(env, deck_asset, deck_pose, "fixed_deck", 0, 0, 0)

    local_load = torch.tensor(
        [
            args.load_x,
            args.load_y,
            0.5 * args.deck_size_z + 0.5 * args.load_size + 0.001,
        ],
        dtype=torch.float32,
    )
    load_pose = gymapi.Transform()
    load_pose.p = _vec3(deck_center + _rot_y(theta, local_load))
    load_pose.r = _quat_y(theta)
    load_handle = gym.create_actor(env, load_asset, load_pose, "payload", 0, 0, 0)

    if viewer is not None:
        gym.viewer_camera_look_at(
            viewer,
            env,
            gymapi.Vec3(0.9, -0.8, 0.85),
            gymapi.Vec3(0.0, 0.0, args.deck_z),
        )

    deck_idx = gym.get_actor_index(env, deck_handle, gymapi.DOMAIN_SIM)
    load_idx = gym.get_actor_index(env, load_handle, gymapi.DOMAIN_SIM)

    gym.prepare_sim(sim)
    root_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_tensor)
    gym.refresh_actor_root_state_tensor(sim)

    total_steps = int(round(args.duration / args.dt))
    kick_step = int(round(args.kick_time / args.dt))
    kicked = False
    start_pos = root_states[load_idx, 0:3].clone()

    print(
        "[slide-scene] "
        f"pipeline={args.pipeline} deck_friction={args.deck_friction:g} "
        f"load_friction={args.load_friction:g} slope_deg={args.slope_deg:g} "
        f"kick=({args.kick_vx:g},{args.kick_vy:g}) load_mass={args.load_mass:g}"
    )
    print(
        "[slide-scene] "
        f"deck_idx={deck_idx} load_idx={load_idx} start_pos={start_pos.tolist()}"
    )

    for step in range(total_steps + 1):
        t = step * args.dt
        if not kicked and step >= kick_step and (
            abs(args.kick_vx) > 0.0 or abs(args.kick_vy) > 0.0
        ):
            root_states[load_idx, 7] = args.kick_vx
            root_states[load_idx, 8] = args.kick_vy
            idx = torch.tensor([load_idx], dtype=torch.int32, device=root_states.device)
            gym.set_actor_root_state_tensor_indexed(
                sim, root_tensor, gymtorch.unwrap_tensor(idx), 1
            )
            kicked = True
            print(
                f"[slide-scene] kick step={step} t={t:.3f}s "
                f"vel={root_states[load_idx, 7:10].tolist()}"
            )

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)

        if viewer is not None:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            if args.real_time:
                gym.sync_frame_time(sim)
            if gym.query_viewer_has_closed(viewer):
                break

        if step % args.print_every == 0 or step == total_steps:
            pos = root_states[load_idx, 0:3].detach().cpu()
            vel = root_states[load_idx, 7:10].detach().cpu()
            delta = pos - start_pos.cpu()
            on_deck = (
                abs(float(pos[0])) <= 0.5 * args.deck_size_x + 0.5 * args.load_size
                and abs(float(pos[1])) <= 0.5 * args.deck_size_y + 0.5 * args.load_size
                and float(pos[2]) > args.deck_z - 0.2
            )
            print(
                f"[slide-scene] step={step:04d} t={t:.3f}s "
                f"pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) "
                f"d=({delta[0]:+.4f},{delta[1]:+.4f},{delta[2]:+.4f}) "
                f"vel=({vel[0]:+.4f},{vel[1]:+.4f},{vel[2]:+.4f}) "
                f"on_deck={on_deck}"
            )

    final_pos = root_states[load_idx, 0:3].detach().cpu()
    final_delta = final_pos - start_pos.cpu()
    print(
        "[slide-scene] summary "
        f"final_delta=({final_delta[0]:+.4f},{final_delta[1]:+.4f},{final_delta[2]:+.4f}) "
        f"final_pos=({final_pos[0]:+.4f},{final_pos[1]:+.4f},{final_pos[2]:+.4f})"
    )

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
