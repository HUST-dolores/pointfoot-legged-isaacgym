#!/usr/bin/env python3
"""Train the payload encoder while keeping a validated jump controller frozen."""

import argparse
import copy
import csv
import json
import os
import sys
import time
from datetime import datetime

import isaacgym  # noqa: F401
import torch
import torch.nn.functional as F

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.algorithm.mlp_encoder import MLP_Encoder
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.scripts.train_jump_load_estimation import _restore_source_cfg
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict, get_load_path


def _save_json(path, value):
    with open(path, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--encoder_updates", type=int, default=1200)
    parser.add_argument("--save_interval", type=int, default=300)
    parser.add_argument("--encoder_lr", type=float, default=1.0e-3)
    parser.add_argument("--history_length", type=int, default=50)
    parser.add_argument("--include_torques", action="store_true")
    parser.add_argument("--crouch_mass_weight", type=float, default=20.0)
    parser.add_argument("--flight_mass_weight", type=float, default=20.0)
    parser.add_argument("--landed_mass_weight", type=float, default=10.0)
    parser.add_argument("--output_run_name", required=True)
    extra, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _restore_source_cfg(env_cfg, train_cfg, args)
    control_encoder_cfg = class_to_dict(train_cfg.MLP_Encoder)
    source_history_length = int(control_encoder_cfg["obs_history_length"])
    source_input_dim = int(control_encoder_cfg["num_input_dim"])
    source_frame_size = source_input_dim // source_history_length
    estimator_frame_size = source_frame_size + (
        int(env_cfg.env.num_actions) if extra.include_torques else 0
    )
    env_cfg.env.encoder_include_torques = extra.include_torques
    env_cfg.env.obs_history_length = extra.history_length
    train_cfg.MLP_Encoder.obs_history_length = extra.history_length
    train_cfg.MLP_Encoder.num_input_dim = (
        estimator_frame_size * extra.history_length
    )
    env_cfg.env.num_envs = int(args.num_envs or 4096)
    dr = env_cfg.domain_rand
    dr.add_random_load = False
    dr.per_env_load_mass = False
    dr.load_drop_terminal = False
    dr.attached_payload = True
    dr.attached_payload_mass_range = [1.0, 6.0]
    dr.attached_payload_offset_range_x = [-0.02, 0.02]
    dr.attached_payload_offset_range_y = [-0.02, 0.02]
    dr.attached_payload_offset_range_z = [0.10, 0.10]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    source_path = get_load_path(
        os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            args.task,
            train_cfg.runner.experiment_name,
        ),
        load_run=args.load_run,
        checkpoint=args.checkpoint,
    )
    source = torch.load(source_path, map_location=env.device)
    runner.alg.actor_critic.load_state_dict(source["model_state_dict"])
    estimator_state = copy.deepcopy(source["encoder_state_dict"])
    if extra.include_torques:
        current_state = runner.alg.encoder.state_dict()
        input_weight_key = next(
            key
            for key in estimator_state
            if key.endswith("gru.weight_ih_l0")
        )
        expanded_weight = current_state[input_weight_key].clone()
        expanded_weight[:, :source_frame_size] = estimator_state[input_weight_key]
        estimator_state[input_weight_key] = expanded_weight
    runner.alg.encoder.load_state_dict(estimator_state)
    control_encoder = MLP_Encoder(**control_encoder_cfg).to(env.device)
    control_encoder.load_state_dict(source["encoder_state_dict"])
    control_encoder.eval()
    for parameter in control_encoder.parameters():
        parameter.requires_grad_(False)

    actor = runner.alg.actor_critic.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    estimator = runner.alg.encoder.train()
    optimizer = torch.optim.Adam(estimator.parameters(), lr=extra.encoder_lr)

    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", args.task, train_cfg.runner.experiment_name
    )
    run_name = (
        datetime.now().strftime("%b%d_%H-%M-%S")
        + "_"
        + extra.output_run_name
    )
    run_dir = os.path.join(log_root, run_name)
    os.makedirs(run_dir, exist_ok=False)
    _save_json(os.path.join(run_dir, "env_cfg.json"), class_to_dict(env_cfg))
    train_cfg_dict = class_to_dict(train_cfg)
    train_cfg_dict["encoder_only"] = {
        "control_checkpoint": source_path,
        "control_encoder_frozen": True,
        "control_encoder_cfg": control_encoder_cfg,
        "estimator_history_length": extra.history_length,
        "estimator_frame_size": estimator_frame_size,
        "estimator_include_torques": extra.include_torques,
        "attached_payload_mass_range_kg": [1.0, 6.0],
        "phase_mass_weights": {
            "idle": 1.0,
            "crouch": extra.crouch_mass_weight,
            "flight": extra.flight_mass_weight,
            "landed": extra.landed_mass_weight,
        },
    }
    _save_json(os.path.join(run_dir, "train_cfg.json"), train_cfg_dict)

    obs, obs_history, commands, critic_obs = env.get_observations()
    mass_scale = float(env.cfg.normalization.obs_scales.mass_scale)
    rows = []
    start = time.time()

    def save_checkpoint(update):
        checkpoint = dict(source)
        checkpoint["encoder_state_dict"] = estimator.state_dict()
        checkpoint["control_encoder_state_dict"] = source["encoder_state_dict"]
        checkpoint["iter"] = int(source.get("iter", args.checkpoint)) + update
        checkpoint["encoder_only_metadata"] = train_cfg_dict["encoder_only"]
        checkpoint["encoder_optimizer_state_dict"] = optimizer.state_dict()
        path = os.path.join(run_dir, f"model_{checkpoint['iter']}.pt")
        torch.save(checkpoint, path)
        print(f"[jump-load-encoder] saved {path}")

    for update in range(1, extra.encoder_updates + 1):
        prediction = estimator(obs_history)
        vel_target = critic_obs[:, 0:3]
        mass_target = critic_obs[:, 3:4]
        com_target = critic_obs[:, 4:7]
        phase = env.jump_state
        sample_weight = torch.ones_like(mass_target)
        sample_weight[phase == 1] = extra.crouch_mass_weight
        sample_weight[phase == 2] = extra.flight_mass_weight
        sample_weight[phase == 3] = extra.landed_mass_weight
        vel_loss = F.mse_loss(prediction[:, 0:3], vel_target)
        mass_loss = (
            (prediction[:, 3:4] - mass_target).square() * sample_weight
        ).sum() / sample_weight.sum()
        com_loss = (
            (prediction[:, 4:7] - com_target).square()
            * sample_weight
        ).sum() / (3.0 * sample_weight.sum())
        loss = 0.2 * vel_loss + 5.0 * mass_loss + com_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(estimator.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            estimator_frames = obs_history.reshape(
                env.num_envs, extra.history_length, estimator_frame_size
            )
            control_history = estimator_frames[
                :, -source_history_length:, :source_frame_size
            ].reshape(env.num_envs, source_input_dim)
            control_latent = control_encoder.encode(
                control_history
            )
            actions = actor.act_inference(
                torch.cat((control_latent, obs, commands), dim=-1)
            )
            mass_error_kg = (
                prediction[:, 3:4] - mass_target
            ) / mass_scale
            mass_rmse_kg = torch.sqrt(torch.mean(mass_error_kg.square())).item()
            mass_bias_kg = torch.mean(mass_error_kg).item()
            flight = phase == 2
            flight_rmse_kg = (
                torch.sqrt(torch.mean(mass_error_kg[flight].square())).item()
                if flight.any()
                else float("nan")
            )

        obs, rews, dones, infos, obs_history, commands, critic_obs = env.step(actions)
        elapsed = time.time() - start
        row = {
            "update": update,
            "loss": float(loss.item()),
            "vel_loss": float(vel_loss.item()),
            "mass_loss": float(mass_loss.item()),
            "com_loss": float(com_loss.item()),
            "mass_rmse_kg": mass_rmse_kg,
            "mass_bias_kg": mass_bias_kg,
            "flight_rmse_kg": flight_rmse_kg,
            "flight_fraction": float(flight.float().mean().item()),
            "done_fraction": float(dones.float().mean().item()),
            "elapsed_s": elapsed,
        }
        rows.append(row)
        if update == 1 or update % 20 == 0:
            eta = elapsed / update * (extra.encoder_updates - update)
            print(
                "[jump-load-encoder] "
                f"{update}/{extra.encoder_updates} loss={row['loss']:.5f} "
                f"mass_rmse={mass_rmse_kg:.3f}kg "
                f"flight_rmse={flight_rmse_kg:.3f}kg "
                f"flight={100.0 * row['flight_fraction']:.1f}% "
                f"ETA={eta / 60.0:.1f}min"
            )
        if update % extra.save_interval == 0 or update == extra.encoder_updates:
            save_checkpoint(update)

    csv_path = os.path.join(run_dir, "encoder_training.csv")
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[jump-load-encoder] DONE run={run_name} elapsed={time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
