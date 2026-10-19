"""Check physical respawns with a frozen policy before trusting repeated-episode training."""
import argparse
import json
from pathlib import Path

import isaacgym  # noqa: F401
import torch
import numpy as np
from .retention_protocol import RetentionSettings
from .retention_env import build_environment
from .retention_train import exact_seed, networks
from .geometry import write_json


def main(a):
    torch.set_num_threads(1)
    exact_seed(101)
    settings = RetentionSettings()
    env, _ = build_environment(settings, a.num_envs, 101)
    alg, _, _, _ = networks(resume=a.checkpoint, action_std=.08)
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    episodes = []
    traces = []
    try:
        env.reset()
        env.episode_stats.clear()
        with torch.inference_mode():
            for index in range(1250):
                obs, history, commands, critic = env.get_observations()
                latent = alg.encode_for_policy(history, obs)
                inputs = torch.cat((latent, obs, commands), -1)
                action = alg.actor_critic.act(inputs) if a.sample else alg.actor_critic.act_inference(inputs)
                env.step(action)
                if env.episode_length_buf[0] in (25, 30, 35, 40):
                    g = env.geometry()
                    rows.append({"step": index, "episode_time": float(env.episode_length_buf[0]) * env.dt,
                        "root": env.root_states[env.actor_indices[0]].cpu().tolist(),
                        "box": env.root_states[env.load_indices[0]].cpu().tolist(),
                        "local": g["local"][0].cpu().tolist(), "gap": float(g["gap"][0]),
                        "support": bool(g["supported"][0]), "force": env.contact_forces[0, env.num_bodies].cpu().tolist()})
                for row in env.episode_stats:
                    episodes.append(row)
                env.episode_stats.clear()
                traces.append(torch.cat((env.root_states[env.actor_indices], env.root_states[env.load_indices],
                    env.episode_length_buf[:, None], env.has_load[:, None]), -1).cpu().numpy().copy())
        np.savez_compressed(a.output / "trace.npz", states=np.stack(traces))
        write_json(a.output / "rows.json", rows)
        write_json(a.output / "episodes.json", episodes)
        print(json.dumps({"episodes": len(episodes), "early_drops": sum(e["length_s"] < 1.9 for e in episodes),
                         "length_percentiles": np.percentile([e["length_s"] for e in episodes], [0, 25, 50, 75, 100]).tolist(),
                         "first_rows": rows[:12]}), flush=True)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--sample", action="store_true")
    main(p.parse_args())
