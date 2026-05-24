# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a legged robot reinforcement learning (RL) training framework built on NVIDIA's Isaac Gym. It trains multiple robot morphologies (Pointfoot, Solefoot, Wheelfoot) using PPO with advanced domain randomization and load carrying capabilities. The codebase includes a dual-head auxiliary encoder for estimating robot load mass and center-of-mass, which is jointly trained with the policy.

Key characteristics:
- Multi-environment simulation (up to 8000+ parallel environments on GPU)
- Configurable task and training parameters via hierarchical config classes
- Task registry pattern for environment registration and instantiation
- Advanced load management with dynamic spawning and removal
- Observation filtering via Butterworth low-pass filters
- Terrain and command curriculum learning

## Quick Start Commands

### Training
```bash
export ROBOT_TYPE=WF_TRON1A  # Choose: PF_TRON1A, WF_TRON1A, SF_TRON1A, etc.
python legged_gym/scripts/train.py --task=wheelfoot_flat --headless
```

Key training flags:
- `--headless` - Disable visualization (faster training)
- `--num_envs N` - Override number of parallel environments
- `--seed S` - Set random seed
- `--max_iterations N` - Override max training iterations
- `--resume` - Resume from last checkpoint
- `--load_run <run_name>` - Resume from specific run
- `--checkpoint N` - Resume from specific iteration

### Playing / Testing
```bash
python legged_gym/scripts/play.py --task=wheelfoot_flat --load_run <run_name> --checkpoint <iter>
```

During playback, press `v` to toggle rendering sync. The play script spawns a single environment for interactive testing.

### Exporting Policies
```bash
python legged_gym/scripts/export_policy_as_onnx.py
```

## Architecture Overview

### Core Execution Flow

```
train.py
  → task_registry.make_env(task_name)
      ├─ Creates task class (e.g., BipedWF)
      └─ Loads env config (BipedCfgWF)
  
  → task_registry.make_alg_runner(env)
      ├─ Creates MLP_Encoder (auxiliary state estimator)
      ├─ Creates ActorCritic policy
      ├─ Creates PPO algorithm
      └─ Returns OnPolicyRunner
  
  → OnPolicyRunner.learn(num_iterations)
      ├─ For each iteration:
      │   ├─ Rollout: Collect num_steps_per_env transitions per environment
      │   ├─ Compute: Rewards, returns, advantages
      │   ├─ Update: Encoder, actor, critic networks
      │   └─ Log: TensorBoard scalars and checkpoint
      └─ Save final policy
```

### Key Classes and Responsibilities

**Environment Layer (legged_gym/envs/)**
- `VecEnv` - Abstract interface for vectorized environments
- `BaseTask` - Core RL task with Isaac Gym integration
  - Manages simulation state (root states, DOF states, forces)
  - Computes observations and rewards
  - Handles domain randomization and load management
  - Provides `step()` and `reset_idx()` for training loop
- `BipedWF`, `BipedPF`, `BipedSF` - Robot-specific task implementations
  - Extend `BaseTask` with task-specific observation/reward logic
  - Implement `_init_buffers()`, `_compute_observations()`, reward functions

**Configuration Layer (legged_gym/envs/*/`_config.py`)**
- `BaseConfig` - Base class with recursive member initialization
- `BipedCfgWF`, etc. - Nested config classes
  - `env` - Environment parameters (num_envs, obs/action sizes, episode length)
  - `terrain` - Terrain curriculum settings
  - `commands` - Command ranges and curriculum
  - `domain_rand` - Randomization parameters (friction, mass, load, etc.)
  - `rewards.scales` - Reward function weights (keys auto-map to methods)
  - `normalization` - Observation/action scaling factors
  - `control` - Joint stiffness/damping, action scaling
  - `asset` - URDF path, foot properties, load estimation params

**Training Layer (legged_gym/algorithm/)**
- `OnPolicyRunner` - Main training orchestrator
  - Collects rollouts of `num_steps_per_env` transitions per iteration
  - Updates encoder, actor, critic via PPO
  - Logs metrics and saves checkpoints
- `PPO` - Proximal Policy Optimization algorithm
  - Computes policy/value losses, entropy regularization
  - Handles auxiliary encoder losses (load mass/COM estimation)
  - Implements gradient clipping and learning rate scheduling
- `MLP_Encoder` - Auxiliary encoder for state estimation
  - Processes observation history to estimate load mass and COM
  - Outputs latent state used by actor/critic
  - Can use dual-head architecture (separate vel/mass/com branches)
- `ActorCritic` - Policy and value networks
  - Actor outputs action distribution (mean + log_std)
  - Critic outputs value estimate
  - Both take encoder latent + commands as input

**Utility Layer (legged_gym/utils/)**
- `task_registry` - Registry pattern for task registration and instantiation
- `helpers` - Config loading, seed setting, path management
- `Logger` - TensorBoard and W&B logging
- `Terrain` - Procedural terrain generation with curriculum

## Important Design Patterns

### 1. Reward Function Discovery
Reward functions are auto-discovered from non-zero entries in `cfg.rewards.scales`. If a key exists (e.g., `tracking_lin_vel`), the framework calls `_reward_<key>()` method.
- **Pattern**: Keep function names aligned with config keys
- **Example**: `scales.tracking_lin_vel = 4.0` → calls `_reward_tracking_lin_vel()`

### 2. Observation Construction
Observations are split into:
- **Policy obs** (`num_observations`) - Used by actor/critic
- **Critic obs** (`num_critic_observations`) - Used by value network (often includes height map samples)
- **Encoder input** (`encoder_obs_history`) - Stacked observation history for auxiliary encoder

**Important**: If you modify observation components, update config sizes in:
- `env.num_observations` / `env.num_critic_observations`
- `MLP_Encoder.num_input_dim` (= stacked observation history size)
- Reward function signatures (if they depend on new obs)

### 3. Auxiliary Encoder Losses
The encoder is trained with separate MSE/Huber losses for:
- **Velocity estimation** - Predicts base velocity from observations
- **Load mass estimation** - Predicts estimated load mass
- **Load COM estimation** - Predicts load center-of-mass offset

Loss weights and regression type configured in:
- `algorithm.extra_loss_*_w` - Weight multipliers
- `algorithm.extra_loss_regression` - "mse", "huber", or "smooth_l1"
- `algorithm.extra_loss_load_boost` - Extra weight when load is detected on body

### 4. Load Management
Loads are dynamically spawned/removed during episodes:
- Controlled by `domain_rand.add_random_load`, `load_start_time_s`, `load_duration_s`, `load_interval_s`
- Load actors have separate indices (`self.load_indices`) - never mix with robot actor indices
- Methods: `add_load()`, `remove_load()`, `_maybe_spawn_loads()`, `get_load_stats()`
- Load mass and COM randomization persist across spawns via `base_mass`, `base_com` buffers

### 5. Config-Driven Behavior
Avoid hard-coded constants. Use config classes to control:
- Simulation parameters (dt, gravity, physics engine)
- Observation/action dimensions and normalization
- Reward scales and termination thresholds
- Domain randomization ranges
- Algorithm hyperparameters (learning rate, clip_param, etc.)

## Key Files to Understand

### Task-Specific
- `legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py` - Core task implementation
  - `_init_buffers()` - Buffer allocation
  - `_parse_cfg()` - Config parsing and derived parameters
  - `compute_observations()` - Observation construction
  - `reset_idx()` - Episode reset logic
  - Reward methods (e.g., `compute_reward_tracking_lin_vel()`)
- `legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py` - Configs for WF_TRON1A

### Framework Core
- `legged_gym/envs/base/base_task.py` - Common task logic
  - Simulation creation and stepping
  - State buffer management
  - Load spawning and removal
  - Generic reward computation loop
- `legged_gym/utils/task_registry.py` - Task registration and factory methods

### Training
- `legged_gym/scripts/train.py` - Entry point
- `legged_gym/algorithm/on_policy_runner.py` - Rollout and update loop
- `legged_gym/algorithm/ppo.py` - Loss computation and optimization

## Common Edits and Patterns

### Adding a New Reward Function
1. Create method `compute_reward_<name>(self)` in task class
2. Return a tensor of shape `(num_envs,)` with reward values
3. Add `scales.<name> = <weight>` to config rewards
4. Update `num_observations` in config if you added new obs components used by the reward

### Modifying Observations
1. Edit observation construction in `compute_observations()` or `_compute_observations()`
2. Update `num_observations` and `num_critic_observations` in config
3. Update encoder input size if using history stacking
4. Verify reward functions still index correct obs dimensions

### Changing Domain Randomization
1. Edit `domain_rand.*_range` and enable flags in config
2. Implement randomization logic in `_randomize_*()` methods in task class
<!-- 3. Consider load statistics impact (e.g., `load_enable_iter` controls when load spawning starts) -->

### Tuning Load Estimation
- Adjust `extra_loss_*_w` for loss weighting (mass, COM, velocity branches)
- Use `extra_loss_load_boost` to up-weight loss when load detected on body
- Change `extra_loss_regression` from "mse" to "huber" for robustness to outliers
- Monitor TensorBoard: `Loss/encoder`, `Metric/mass_mse`, `Metric/com_mse`

## Testing and Debugging

### Single Environment Debugging
```bash
python legged_gym/scripts/play.py --task=wheelfoot_flat --load_run <run> --checkpoint <iter>
```
- Environment spawns with 1 parallel env for interactive inspection
- Press `v` to toggle rendering; `ESC` to quit

### Viewing Training Logs
```bash
tensorboard --logdir logs/
```
Key scalars:
- `Loss/encoder` - Auxiliary encoder loss (velocity, mass, COM)
- `Loss/surrogate` - Policy gradient loss
- `Loss/value` - Value network loss
- `Metric/*_mse` - Unweighted MSE for load estimation targets
- `Train/mean_episode_reward` - Average return
- `Train/mean_episode_length` - Average episode length

### Config Inspection
Saved configs are logged to `logs/<experiment>/<robot_type>/<datetime>/`:
- `env_cfg.json` - Resolved environment config
- `train_cfg.json` - Resolved training config
- `*.py` - Task and config source files

## Robot Types and Files

Each robot type is defined by:
1. URDF in `resources/robots/<ROBOT_TYPE>/urdf/robot.urdf`
2. Environment class (e.g., `BipedWF` for wheelfoot)
3. Config class (e.g., `BipedCfgWF`)
4. Registration in `legged_gym/envs/__init__.py`

Supported types:
- `PF_TRON1A`, `PF_P441A`, `PF_P441B`, `PF_P441C`, `PF_P441C2` - Pointfoot variants
- `WF_TRON1A` - Wheelfoot with wheels
- `SF_TRON1A` - Solefoot variant

## Integration with Copilot Instructions

This CLAUDE.md complements `.github/copilot-instructions.md`, which contains:
- Reward function naming conventions
- Observation/action dimension consistency patterns
- Encoder auxiliary loss wiring details
- Load management specifics (indices, buffers)
- TensorBoard logging conventions

Refer to copilot-instructions.md for detailed implementation patterns.
