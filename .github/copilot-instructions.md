# Copilot Instructions for pointfoot-legged-gym

## Project shape
- This repo is an Isaac Gym legged-robot RL stack built around `legged_gym`.
- Each task is a pair of files: `<task>.py` and `<task>_config.py` under `legged_gym/envs/<task>/`.
- The env class derives from `BaseTask`; the config class derives from `BaseConfig`.
- Task registration happens in `legged_gym/envs/__init__.py` via `task_registry.register(...)`.

## Runtime flow
- `legged_gym/scripts/train.py` calls `task_registry.make_env(...)`, then `make_alg_runner(...)`, then `learn(...)`.
- `legged_gym/scripts/play.py` loads the registered task, resumes a checkpoint, and runs policy inference.
- The runner wires `MLP_Encoder` + `ActorCritic` + `PPO`; rollout length is `runner.num_steps_per_env` per iteration.
- `OnPolicyRunner` logs TensorBoard scalars such as `Loss/encoder`, `Train/mean_episode_length`, and env-specific load stats.

## Important code patterns
- Reward functions are discovered by name from non-zero entries in `cfg.rewards.scales`; keep function names aligned with config keys.
- Observations are split into policy obs and critic obs; if you change one, update `num_observations` / `num_critic_observations` in the config.
- The encoder auxiliary loss is tied to the critic target slice; keep encoder output size, critic target slice, and `extra_loss` shape consistent.
- `BaseTask` caches root state, base velocity, foot state, and load state; reset code must resynchronize those buffers after `reset_idx(...)`.

## Load / domain-rand specifics
- Random load spawning is controlled in `BaseTask` with `add_random_load`, `load_start_time_s`, `load_duration_s`, and `load_interval_s`.
- Load actors have their own indices (`load_indices`) and should not be mixed with robot termination indices.
- Load mass / COM randomization is tracked through `base_mass`, `base_com`, `base_mass0`, and `base_com0`.

## Project conventions
- This codebase prefers config-driven behavior over hard-coded constants.
- Keep edits localized to the relevant task/config/runner files; avoid refactoring unrelated tasks.
- Many behaviors are verified by TensorBoard and short manual runs rather than a formal test suite.
- Use the provided logging and viewer controls: `v` toggles rendering sync during training.

## Environment and dependencies
- Follow the README: Python 3.8 is recommended, with PyTorch 2.2.2 + CUDA 12.1 and Isaac Gym Preview 3.
- Typical install flow is `pip install -e .` after Isaac Gym is installed.

## Good edit targets
- Add new task logic in `legged_gym/envs/<task>/<task>.py`.
- Tune reward weights, obs sizes, and load settings in the matching `<task>_config.py`.
- Update training / logging behavior in `legged_gym/algorithm/on_policy_runner.py` and `legged_gym/algorithm/ppo.py`.

## When changing behavior
- If you modify reward or observation shapes, update config sizes, runner wiring, and any ONNX / JIT export paths.
- If you add a new task, register it and make sure `train.py` / `play.py` can resolve it through `task_registry`.
- Prefer concrete, reproducible changes over speculative cleanup.
