"""Paired retention evaluation and viewer; never respawn within a trial."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import isaacgym  # noqa: F401
from isaacgym import gymapi
import numpy as np
import torch

from .geometry import write_json, sha256
from .retention_env import build_environment
from .retention_protocol import RetentionSettings, eval_cases
from .retention_train import networks, exact_seed
from .robot import ROOT
from legged_gym.utils.helpers import set_seed


def main(a):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    exact_seed(a.seed)
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    settings = RetentionSettings()
    if a.checkpoint:
        checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        settings = RetentionSettings(**checkpoint["settings"])
    if a.duration is not None:
        settings.duration = a.duration
    if a.viewer or a.capture:
        angle = {"forward": 0., "left": np.pi / 2, "backward": np.pi, "right": -np.pi / 2}[a.direction]
        cases = [{"nominal_dv": a.dv, "angle": angle, "direction": a.direction, "repeat": 0}]
    else:
        cases = eval_cases(a.repeats)
    env, cfg = build_environment(settings, len(cases), a.seed, train=False,
                                viewer=a.viewer, capture=a.capture, evaluation_cases=cases)
    algorithm, _, preservation_error, _ = networks(resume=a.checkpoint)
    algorithm.actor_critic.eval()
    records = {}
    camera = None
    frames = []
    start = time.monotonic()

    def add(key, value):
        records.setdefault(key, []).append(value.detach().cpu().numpy().copy())

    try:
        env.reset()
        env.evaluation_mode = True
        env.episode_stats.clear()
        scenarios = [{**c, "spawn_xy": env.spawn_xy[i].cpu().tolist()} for i, c in enumerate(cases)]
        if a.capture:
            prop = gymapi.CameraProperties()
            prop.width, prop.height = 960, 720
            prop.horizontal_fov = 50
            prop.use_collision_geometry = True
            camera = env.gym.create_camera_sensor(env.envs[0], prop)
            if camera < 0:
                raise RuntimeError("camera failed")
            (a.output / "frames").mkdir()
        palette = {"base": (.35, .58, .78), "abad": (.78, .48, .56), "hip": (.88, .69, .30),
                   "knee": (.29, .68, .56), "wheel": (.35, .34, .41)}
        if a.viewer or a.capture:
            for handle, actor, box in zip(env.envs, env.actor_handles, env.load_handles):
                env.gym.set_rigid_body_color(handle, box, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(.95, .43, .12))
                for i, name in enumerate(env.body_names):
                    color = next((c for prefix, c in palette.items() if prefix in name), (.5, .6, .7))
                    env.gym.set_rigid_body_color(handle, actor, i, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*color))
        with torch.inference_mode():
            remaining_steps = round(settings.duration / env.dt) - int(env.episode_length_buf[0])
            for index in range(remaining_steps):
                obs, history, commands, critic = env.get_observations()
                latent = algorithm.encode_for_policy(history, obs)
                action = algorithm.actor_critic.act_inference(torch.cat((latent, obs, commands), -1))
                if not bool(torch.isfinite(action).all()):
                    raise FloatingPointError("actor output")
                env.step(action)
                for key, value in env.last_transition.items():
                    add(key, value)
                geom = env.geometry()
                for key, value in {"time_s": env.episode_length_buf.float() * env.dt,
                    "drop_latched": env.drop_latched, "fall_latched": env.fall_latched,
                    "actions": action, "observations": obs, "commands_and_payload_context": commands,
                    "load_contact": env.contact_forces[:, env.num_bodies],
                    "wheel_contact": env.contact_forces[:, env.feet_indices],
                    "gravity": env.projected_gravity, "has_load": env.has_load,
                    "box_relative_velocity": geom["relative_velocity"]}.items():
                    add(key, value)
                if a.capture or a.viewer:
                    target = env.root_states[env.actor_indices[0], :3].cpu().numpy() + np.array([0., 0., -.25])
                    camera_pos = target + np.array([1.1, -1.5, 1.05]) * a.camera_scale
                    if a.viewer:
                        env.gym.viewer_camera_look_at(env.viewer, env.envs[0], gymapi.Vec3(*camera_pos), gymapi.Vec3(*target))
                    if camera is not None and index % 5 == 0:
                        from PIL import Image
                        env.gym.set_camera_location(camera, env.envs[0], gymapi.Vec3(*camera_pos), gymapi.Vec3(*target))
                        env.gym.fetch_results(env.sim, True)
                        env.gym.step_graphics(env.sim)
                        env.gym.render_all_camera_sensors(env.sim)
                        rgb = np.asarray(env.gym.get_camera_image(env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR)).reshape(720, 960, 4)[..., :3]
                        if rgb.std() < 3:
                            raise RuntimeError("blank image")
                        name = f"frame_{len(frames):04d}.png"
                        Image.fromarray(rgb).save(a.output / "frames" / name)
                        frames.append({"file": name, "time_s": float(env.episode_length_buf[0]) * env.dt})
                    if a.viewer:
                        if env.gym.query_viewer_has_closed(env.viewer):
                            break
                        delay = (index + 1) * env.dt - (time.monotonic() - start)
                        if delay > 0:
                            time.sleep(delay)
        arrays = {k: np.stack(v) for k, v in records.items()}
        np.savez_compressed(a.output / "trace.npz", **arrays)
        metrics = []
        for i, case in enumerate(cases):
            t = arrays["time_s"][:, i]
            drop = arrays["drop"][:, i].astype(bool)
            fall = arrays["fall"][:, i].astype(bool)
            failed = drop | fall
            first = int(np.flatnonzero(failed)[0]) if failed.any() else len(t)
            valid = (t >= settings.spawn_time + .2) & (np.arange(len(t)) < first)
            supported = arrays["supported"][:, i].astype(bool)
            slip = np.linalg.norm(arrays["local"][:, i, :2] - np.asarray(scenarios[i]["spawn_xy"]), axis=-1)
            max_slip = float(slip[valid & supported].max()) if (valid & supported).any() else 0.
            late = supported[t >= settings.duration - .5]
            late_support = float(late.mean()) if len(late) else 0.
            end_success = bool(t[-1] >= settings.duration - env.dt / 2) and not failed.any() and late_support >= .8
            root = arrays["root"][:, i]
            dxy = np.linalg.norm(root[:, :2] - root[0, :2], axis=-1)
            metrics.append({"env": i, "seed": a.seed, **case, "success": bool(end_success),
                "drop": bool(drop.any()), "fall": bool(fall.any()), "last_half_s_support_fraction": late_support,
                "first_failure_s": float(t[first]) if first < len(t) else None,
                "max_supported_slip_m": max_slip, "sliding_observed": max_slip >= .02,
                "min_margin_before_failure_m": float(arrays["margins"][:, i].min(-1)[valid].min()) if valid.any() else None,
                "max_robot_displacement_before_failure_m": float(dxy[:max(1, first)].max()),
                "spawn_count": int(arrays["spawns"][:, i].max()),
                "pose_writes_after_first_spawn": int(np.ptp(arrays["pose_writes"][arrays["has_load"][:, i], i]))})
        write_json(a.output / "cases.json", metrics)
        write_json(a.output / "scenarios.json", scenarios)
        write_json(a.output / "materials.json", env.material_audit)
        disturbed = [m for m in metrics if m["nominal_dv"] > 0]
        summary = {"seed": a.seed, "trials": len(metrics), "disturbed_trials": len(disturbed),
            "disturbed_success_rate": float(np.mean([m["success"] for m in disturbed])),
            "disturbed_drop_rate": float(np.mean([m["drop"] for m in disturbed])),
            "disturbed_fall_rate": float(np.mean([m["fall"] for m in disturbed])),
            "disturbed_sliding_rate": float(np.mean([m["sliding_observed"] for m in disturbed])),
            "all_trials_single_spawn": all(m["spawn_count"] == 1 for m in metrics),
            "no_pose_writes_after_spawn": all(m["pose_writes_after_first_spawn"] == 0 for m in metrics),
            "checkpoint": str(a.checkpoint) if a.checkpoint else "original_frozen_balancing_policy",
            "checkpoint_sha256": sha256(a.checkpoint) if a.checkpoint else None,
            "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("retention*.py")},
            "settings": settings.dictionary(), "elapsed_s": time.monotonic() - start,
            "expanded_actor_preservation_max_error": preservation_error, "camera_frames": frames,
            "camera_scale": a.camera_scale,
            "verified_spawns_next_physics_step": env.verified_spawns,
            "root_state_submissions_batched": True, "actual_final_time_s": float(arrays["time_s"][-1, 0])}
        assert summary["all_trials_single_spawn"] and summary["no_pose_writes_after_spawn"]
        assert env.verified_spawns == len(cases)
    finally:
        if camera is not None:
            env.gym.destroy_camera_sensor(env.sim, env.envs[0], camera)
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)
    summary["cleanup_completed"] = True
    write_json(a.output / "result.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("source_sha256", "camera_frames", "settings")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--duration", type=float)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--capture", action="store_true")
    p.add_argument("--direction", choices=("forward", "left", "backward", "right"), default="forward")
    p.add_argument("--dv", type=float, default=.65)
    p.add_argument("--camera-scale", type=float, default=1.2)
    p.add_argument("--clean-exit", action="store_true")
    a = p.parse_args()
    if a.output is None:
        a.output = ROOT / "logs/payload_retention_play" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    main(a)
    if a.clean_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
