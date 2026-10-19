"""PPO adaptation for retention, with the original state estimator frozen."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import time

import isaacgym  # noqa: F401
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from legged_gym.algorithm.actor_critic import ActorCritic
from legged_gym.algorithm.mlp_encoder import MLP_Encoder
from legged_gym.algorithm.ppo import PPO
from .geometry import sha256, write_json
from .retention_env import build_environment, source_directory
from legged_gym.utils.helpers import class_to_dict, set_seed
from .retention_protocol import CONTEXT_DIM, RetentionSettings
from .robot import ROOT


def exact_seed(seed):
    # The shared helper interprets seed=1 as random. This experiment needs literal seeds.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def networks(device="cuda:0", resume=None, learning_rate=1e-4, action_std=0.20):
    source = source_directory()
    train = json.loads((source / "train_cfg.json").read_text())
    saved = torch.load(source / "model_11000.pt", map_location=device, weights_only=False)
    encoder = MLP_Encoder(**train["MLP_Encoder"]).to(device).eval()
    encoder.load_state_dict(saved["encoder_state_dict"], strict=True)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    actor = ActorCritic(58 + CONTEXT_DIM, 65 + CONTEXT_DIM, 8, **train["policy"]).to(device)
    weights = actor.state_dict()
    old = saved["model_state_dict"]
    for name in weights:
        if name == "actor.0.weight":
            weights[name].zero_()
            weights[name][:, :58] = old[name]
        elif name == "critic.0.weight":
            weights[name].zero_()
            weights[name][:, :58] = old[name][:, :58]
            weights[name][:, -7:] = old[name][:, 58:]
        else:
            weights[name] = old[name]
    actor.load_state_dict(weights, strict=True)
    # Verify adaptation starts from exactly the original deterministic controller.
    reference = ActorCritic(58, 65, 8, **train["policy"]).to(device)
    reference.load_state_dict(old, strict=True)
    with torch.no_grad():
        original = torch.randn(32, 58, device=device)
        expanded = torch.cat((original, torch.randn(32, CONTEXT_DIM, device=device)), -1)
        error = (actor.act_inference(expanded) - reference.act_inference(original)).abs().max().item()
        assert error < 1e-5, error
        # Old value estimates belong to a different reward. Relearn the value head.
        actor.critic[-1].weight.zero_()
        actor.critic[-1].bias.zero_()
        actor.logstd.fill_(np.log(action_std))
    del reference
    state = None
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        actor.load_state_dict(state["model_state_dict"], strict=True)
        encoder.load_state_dict(state["encoder_state_dict"], strict=True)
    algorithm_cfg = dict(train["algorithm"])
    algorithm_cfg.update(learning_rate=learning_rate, schedule="fixed", gamma=0.995,
                         num_learning_epochs=4, num_mini_batches=4, entropy_coef=0.002,
                         early_stop=True, desired_kl=0.015)
    algorithm = PPO(1, encoder, actor, device=device, **algorithm_cfg)
    algorithm.extra_optimizer = None
    if state:
        algorithm.optimizer.load_state_dict(state["optimizer_state_dict"])
    return algorithm, train, error, state


def save_checkpoint(path, algorithm, iteration, settings, seed):
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(".tmp")
    torch.save({"model_state_dict": algorithm.actor_critic.state_dict(),
                "encoder_state_dict": algorithm.encoder.state_dict(),
                "optimizer_state_dict": algorithm.optimizer.state_dict(),
                "iteration": iteration, "settings": settings.dictionary(), "seed": seed,
                "context_dim": CONTEXT_DIM, "sim_payload_state_feedback": True}, temp)
    os.replace(temp, path)


def main(a):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    exact_seed(a.seed)
    if (a.output / "result.json").exists():
        raise FileExistsError("Completed training is immutable; choose another directory")
    if a.output.exists() and not a.resume:
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True, exist_ok=True)
    settings = RetentionSettings(max_dv=a.max_dv, curriculum_iterations=a.curriculum_iterations,
                                 center_weight=a.center_weight)
    if a.resume:
        previous = torch.load(a.resume, map_location="cpu", weights_only=False)
        settings = RetentionSettings(**previous["settings"])
    env, cfg = build_environment(settings, a.num_envs, a.seed)
    writer = SummaryWriter(str(a.output / "tensorboard"))
    start = time.monotonic()
    started = datetime.now().isoformat()
    algorithm, train, preservation_error, state = networks(resume=a.resume, learning_rate=a.lr, action_std=a.action_std)
    algorithm.num_group = a.num_envs
    algorithm.init_storage(a.num_envs, 24, [48], [65 + CONTEXT_DIM], [880], [3 + CONTEXT_DIM], [8])
    before_encoder = {k: v.clone() for k, v in algorithm.encoder.state_dict().items()}
    iteration_start = state["iteration"] if state else 0
    source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("retention*.py")}
    metadata = {"started": started, "settings": settings.dictionary(), "seed": a.seed,
        "num_envs": a.num_envs, "horizon": 24, "target_iterations": a.iterations,
        "source_hashes": source_hashes, "base_policy": str(source_directory() / "model_11000.pt"),
        "base_policy_sha256": sha256(source_directory() / "model_11000.pt"),
        "nominal_urdf_sha256": sha256(ROOT / "resources/robots/WF_TRON1A/urdf/robot.urdf"),
        "expanded_actor_preservation_max_error": preservation_error,
        "encoder_frozen": True, "payload_state_feedback": "simulation ground truth in plate coordinates",
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ppo_core": "repository PPO, unchanged", "argv": vars(a).copy()}
    metadata["argv"] = {k: str(v) if isinstance(v, Path) else v for k, v in metadata["argv"].items()}
    write_json(a.output / "metadata.json", metadata)
    write_json(a.output / "env_cfg.json", class_to_dict(cfg))
    write_json(a.output / "materials.json", env.material_audit)
    metrics_file = (a.output / "progress.jsonl").open("a", buffering=1)
    episodes_file = (a.output / "episodes.jsonl").open("a", buffering=1)
    no_update_pose_writes = True
    try:
        env.reset()
        obs, history, commands, critic = env.get_observations()
        if not state:
            save_checkpoint(a.output / "model_0.pt", algorithm, 0, settings, a.seed)
        for iteration in range(iteration_start, a.iterations):
            env.set_learning_iteration(iteration)
            env.episode_stats.clear()
            tick = time.monotonic()
            with torch.inference_mode():
                for _ in range(24):
                    action = algorithm.act(obs, history, commands, critic)
                    # Archived reset_idx mutates history in place before storage.add_transitions.
                    algorithm.transition.observations = obs.clone()
                    algorithm.transition.observation_history = history.clone()
                    algorithm.transition.commands = commands.clone()
                    obs, rewards, dones, infos, history, commands, critic = env.step(action)
                    if not bool(torch.isfinite(obs).all() & torch.isfinite(rewards).all()):
                        raise FloatingPointError("Nonfinite simulator state")
                    algorithm.process_env_step(rewards, dones, infos, obs)
                latent = algorithm.encode_for_policy(history, obs)
                algorithm.compute_returns(torch.cat((critic, commands, latent), -1))
            # Updates are not episode boundaries and must never move/remove the box.
            physical_before = env.root_states[env.load_indices].clone()
            pose_writes_before = env.pose_writes.clone()
            if iteration == iteration_start or (iteration + 1) % 10 == 0:
                with torch.no_grad():
                    st = algorithm.storage
                    flat_obs = st.observations.flatten(0, 1)[:2048]
                    latent_check = algorithm.encode_for_policy(st.observation_history.flatten(0, 1)[:2048], flat_obs)
                    reconstructed = algorithm.actor_critic.act_inference(torch.cat((latent_check, flat_obs, st.commands.flatten(0, 1)[:2048]), -1))
                    rollout_error = (reconstructed - st.mu.flatten(0, 1)[:2048]).abs().max().item()
                    print(json.dumps({"pre_update_rollout_action_reconstruction_error": rollout_error}), flush=True)
                    assert rollout_error < 1e-4, "Stored rollout observations do not reproduce the collection policy"
            losses = algorithm.update()
            assert torch.equal(physical_before, env.root_states[env.load_indices])
            assert torch.equal(pose_writes_before, env.pose_writes)
            assert all(torch.equal(v, before_encoder[k]) for k, v in algorithm.encoder.state_dict().items())
            if not all(np.isfinite(float(v)) for v in losses[:4]):
                raise FloatingPointError("Nonfinite PPO loss")
            for ep in env.episode_stats:
                ep["iteration"] = iteration + 1
                episodes_file.write(json.dumps(ep) + "\n")
            if (iteration + 1) % a.save_every == 0 or iteration + 1 == a.iterations:
                save_checkpoint(a.output / f"model_{iteration + 1}.pt", algorithm, iteration + 1, settings, a.seed)
            if (iteration + 1) % 10 == 0 or iteration == iteration_start:
                wall = time.monotonic() - start
                seconds_per_iteration = wall / (iteration + 1 - iteration_start)
                row = {"iteration": iteration + 1, "elapsed_s": wall, "last_iteration_s": time.monotonic() - tick,
                    "eta_s": seconds_per_iteration * (a.iterations - iteration - 1),
                    "pulse_max_dv": env.pulse_max_dv, "value_loss": losses[0], "surrogate_loss": losses[2],
                    "kl": losses[3], "action_std": float(algorithm.actor_critic.logstd.exp().mean()),
                    "episode_count_last_iteration": len(env.episode_stats),
                    "timeout_rate_last_iteration": float(np.mean([ep["timeout"] for ep in env.episode_stats])) if env.episode_stats else None,
                    "drop_rate_last_iteration": float(np.mean([ep["drop"] for ep in env.episode_stats])) if env.episode_stats else None,
                    "reward_terms_last_step": env.term_means, "payload_state_unchanged_by_optimizer": no_update_pose_writes}
                metrics_file.write(json.dumps(row) + "\n")
                write_json(a.output / "status.json", row)
                print(json.dumps(row), flush=True)
                for key in ("value_loss", "surrogate_loss", "kl", "action_std", "pulse_max_dv"):
                    writer.add_scalar(key, row[key], iteration + 1)
                for key, value in env.term_means.items():
                    writer.add_scalar("reward/" + key, value, iteration + 1)
    finally:
        metrics_file.close()
        episodes_file.close()
        writer.close()
        env.gym.destroy_sim(env.sim)
    write_json(a.output / "result.json", {"completed": datetime.now().isoformat(), "iterations": a.iterations,
        "elapsed_s": time.monotonic() - start, "cleanup_completed": True,
        "payload_state_unchanged_by_optimizer": no_update_pose_writes, "encoder_unchanged": True})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--action-std", type=float, default=.20)
    p.add_argument("--max-dv", type=float, default=.75)
    p.add_argument("--curriculum-iterations", type=int, default=1000)
    p.add_argument("--center-weight", type=float, default=.25)
    p.add_argument("--resume", type=Path)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
