"""CPU/GPU contact, actor-observation and small PPO smoke experiment.

Example: python legged_gym/scripts/ship_motion_probe.py --mode probe --device cpu
This is a wiring/contact check, not evidence of a trained balancing controller.
"""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("ROBOT_TYPE", "WF_TRON1A")
from isaacgym import gymapi, gymutil
import torch
from legged_gym.envs.wheelfoot_ship.wheelfoot_ship import BipedWFShip
from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_config import BipedCfgWFShip, BipedCfgPPOWFShip
from legged_gym.algorithm.on_policy_runner import OnPolicyRunner
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.saved_config import apply_config_dict
from legged_gym.utils.ship_warm_start import migrate_flat_checkpoint
from legged_gym.utils.ship_motion import surface_pose, heading_yaw_rate


def survival_checkpoint(failure_counts,steps,dt):
    """Read-only diagnostics; never passed to policy observations or rewards."""
    survivors=int((failure_counts==0).sum())
    return {'steps':steps,'simulated_seconds':steps*dt,'survivors':survivors,
            'survival_fraction':survivors/len(failure_counts),
            'failure_resets':int(failure_counts.sum())}


def record_first_failure(first_steps,failure_mask,step):
    first_steps[(first_steps<0) & failure_mask]=step


def yaw_from_quaternion(q):
    x,y,z,w=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "train", "eval"), default="probe")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--warm-start", type=Path, help="Evaluate a migrated flat controller without PPO updates")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--motion-scale", type=float, default=1.0)
    parser.add_argument("--roll-deg",type=float)
    parser.add_argument("--pitch-deg",type=float)
    parser.add_argument("--heave-m",type=float)
    parser.add_argument("--fixed-amplitude",action="store_true",
                        help="Every environment experiences the stated maximum amplitude")
    parser.add_argument("--fixed-frequency-hz",type=float,nargs=3,metavar=("ROLL","PITCH","HEAVE"))
    imu_override=parser.add_mutually_exclusive_group()
    imu_override.add_argument("--imu-off", action="store_true")
    imu_override.add_argument("--imu-on", action="store_true")
    parser.add_argument('--locomotion-mode',choices=['balance','roll','step'])
    parser.add_argument("--reference-mode",choices=['body','gravity_projection'])
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1 or args.iterations < 1 or args.motion_scale < 0:
        parser.error("counts must be positive and motion-scale nonnegative")
    if args.mode == "eval" and (args.checkpoint is None) == (args.warm_start is None):
        parser.error("--mode eval requires exactly one of --checkpoint or --warm-start")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    cfg, train_cfg = BipedCfgWFShip(), BipedCfgPPOWFShip()
    env_type=BipedWFShip
    saved_values=None
    saved_mode='balance'
    if args.mode == "eval" and args.checkpoint is not None:
        saved_env = args.checkpoint.parent/"env_cfg.json"
        saved_values=json.loads(saved_env.read_text())
        saved_mode=saved_values.get('ship_mobility',{}).get('mode','balance')
        if args.locomotion_mode is not None and args.locomotion_mode!=saved_mode:
            parser.error('locomotion mode does not match the checkpoint config')
    locomotion_mode=args.locomotion_mode or saved_mode
    if saved_values is not None and saved_values.get('ship_model_prior',{}).get('enabled',False):
        if locomotion_mode!='balance':
            parser.error('geometric model-prior checkpoint is standing balance only')
        from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_prior import BipedWFShipPrior
        from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_prior_config import BipedCfgWFShipPrior,BipedCfgPPOWFShipPrior
        cfg,train_cfg=BipedCfgWFShipPrior(),BipedCfgPPOWFShipPrior()
        env_type=BipedWFShipPrior
    if locomotion_mode!='balance':
        if args.warm_start is not None:
            parser.error('legacy flat warm-start evaluation is balance-only')
        from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_mobility import BipedWFShipMobility
        from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_mobility_config import (
            BipedCfgWFShipRoll,BipedCfgWFShipStep,BipedCfgPPOWFShipMobility,
        )
        cfg=(BipedCfgWFShipRoll if locomotion_mode=='roll' else BipedCfgWFShipStep)()
        train_cfg=BipedCfgPPOWFShipMobility()
        env_type=BipedWFShipMobility
    if saved_values is not None:
        apply_config_dict(cfg,saved_values)
    if locomotion_mode!='balance':
        cfg.ship_mobility.speed_curriculum=False
    cfg.seed = args.seed
    cfg.env.num_envs = args.num_envs
    # An omitted override must honor a separately trained no-IMU checkpoint.
    if args.imu_off:cfg.ship.imu_enabled=False
    elif args.imu_on:cfg.ship.imu_enabled=True
    if args.reference_mode is not None:cfg.ship.reference_mode=args.reference_mode
    cfg.ship.curriculum = False
    cfg.ship.initial_difficulty = 1.0
    cfg.ship.training_schedule = False  # evaluation severity is independent of training iteration
    cfg.ship.corner_fraction = 0.0  # held-out tests never inherit training anchor assignments
    # Longer verification must not quietly restart the sea-state every 20 s
    # through the training timeout; actual falls remain failures and resets.
    cfg.env.episode_length_s = max(cfg.env.episode_length_s,
                                  args.steps*cfg.sim.dt*cfg.control.decimation + .02)
    bounds=[args.roll_deg,args.pitch_deg,args.heave_m]
    if any(x is not None for x in bounds):
        if not all(x is not None and x>=0 for x in bounds):
            parser.error('roll-deg, pitch-deg and heave-m must be supplied together and nonnegative')
        if args.motion_scale>0 and not any(x>0 for x in bounds):
            parser.error('nonstatic evaluation requires at least one nonzero motion axis')
        cfg.ship.amplitude_max=[args.roll_deg*3.141592653589793/180,
                                args.pitch_deg*3.141592653589793/180,args.heave_m]
        cfg.ship.amplitude_min=[.4*x for x in cfg.ship.amplitude_max]
    cfg.ship.amplitude_min = [v*args.motion_scale for v in cfg.ship.amplitude_min]
    cfg.ship.amplitude_max = [v*args.motion_scale for v in cfg.ship.amplitude_max]
    if args.fixed_amplitude:
        cfg.ship.amplitude_min = list(cfg.ship.amplitude_max)
    if args.fixed_frequency_hz is not None:
        if any(v <= 0 for v in args.fixed_frequency_hz):
            parser.error('fixed frequencies must be positive')
        cfg.ship.frequency_min = list(args.fixed_frequency_hz)
        cfg.ship.frequency_max = list(args.fixed_frequency_hz)
    params = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), params)
    params.use_gpu_pipeline = args.device.startswith("cuda")
    params.physx.use_gpu = args.device.startswith("cuda")
    params.physx.num_threads = 2
    params.physx.num_velocity_iterations = 2
    params.physx.num_position_iterations = 8
    args.output.mkdir(parents=True, exist_ok=False)
    env = None
    result = {"mode": args.mode, "device": args.device, "seed": args.seed,
              "imu_enabled": cfg.ship.imu_enabled, "motion_scale": args.motion_scale,
              "reference_mode":getattr(cfg.ship,'reference_mode','body'),
              "fixed_amplitude":args.fixed_amplitude,
              "amplitude_min":cfg.ship.amplitude_min,"amplitude_max":cfg.ship.amplitude_max,
              "frequency_min":cfg.ship.frequency_min,"frequency_max":cfg.ship.frequency_max,
              "episode_length_s":cfg.env.episode_length_s,
              "balance_claim": "not_established_by_this_smoke_test"}
    result['locomotion_mode']=locomotion_mode
    started = time.monotonic()
    try:
        env = env_type(cfg, params, gymapi.SIM_PHYSX, args.device, not args.viewer)
        assert env.gym.get_sim_actor_count(env.sim) == 2*env.num_envs
        assert not torch.isin(env.actor_indices, env.deck_indices).any()
        mass = env.gym.get_actor_rigid_body_properties(env.envs[0], env.load_handles[0])[0].mass
        assert abs(mass-cfg.ship.mass_kg) < 1.0
        result["deck_mass_kg"] = mass
        if args.mode == "train":
            train_cfg.MLP_Encoder.hidden_dims = [64]
            train_cfg.MLP_Encoder.vel_head_hidden_dims = [32]
            train_cfg.MLP_Encoder.mass_head_hidden_dims = [32]
            train_cfg.policy.actor_hidden_dims = [128, 64]
            train_cfg.policy.critic_hidden_dims = [128, 64]
            train_cfg.algorithm.num_learning_epochs = 1
            train_cfg.algorithm.num_mini_batches = 2
            train_cfg.runner.save_interval = 1
            runner = OnPolicyRunner(env, class_to_dict(train_cfg), str(args.output), args.device)
            with (args.output/"train.log").open("w") as stream, redirect_stdout(stream):
                runner.learn(args.iterations, init_at_random_ep_len=False)
            result.update(passed=True, iterations=args.iterations,
                          checkpoint=str(args.output/f"model_{args.iterations}.pt"))
        else:
            runner = None
            if args.mode == "eval":
                if args.checkpoint is not None:
                    apply_config_dict(train_cfg, json.loads((args.checkpoint.parent/"train_cfg.json").read_text()))
                runner = OnPolicyRunner(env, class_to_dict(train_cfg), str(args.output), args.device)
                if args.warm_start is not None:
                    source = torch.load(args.warm_start, map_location="cpu")
                    model, encoder = migrate_flat_checkpoint(source, runner.alg.actor_critic.state_dict(), runner.alg.encoder.state_dict())
                    runner.alg.actor_critic.load_state_dict(model)
                    runner.alg.encoder.load_state_dict(encoder)
                else:
                    runner.load(str(args.checkpoint))
                runner.alg.actor_critic.eval()
                runner.alg.encoder.eval()
            with torch.inference_mode():
                env.reset()
                physical_velocity=env.dof_state.view(env.num_envs,env.num_dof,2)[:,:,1]
                if cfg.env.dof_vel_use_pos_diff:
                    assert env.dof_vel.data_ptr()!=physical_velocity.data_ptr()
                    physical_velocity.fill_(3.)
                    env.dof_vel.fill_(2.)
                    env.reset_idx(torch.arange(env.num_envs,device=env.device))
                    assert torch.count_nonzero(physical_velocity)==0, 'physical joint reset retained old velocity'
                    assert torch.count_nonzero(env.dof_vel)==0
                result['physical_dof_velocity_reset_checked']=True
                # Regression for fall-reset histories: no previous fallen IMU
                # state may survive after the robot root has been reset upright.
                env.base_quat[:]=torch.tensor([0.,0.70710678,0.,0.70710678],device=env.device)
                env.base_ang_vel[:]=torch.tensor([0.,1.,0.],device=env.device)
                env.projected_gravity[:]=torch.tensor([1.,0.,0.],device=env.device)
                env.reset_idx(torch.arange(env.num_envs,device=env.device))
                frame_width=cfg.env.num_observations*2-cfg.env.num_actions
                reset_frames=env.encoder_obs_history.reshape(env.num_envs,cfg.env.obs_history_length,frame_width)
                expected_gravity=env.gravity_vec[:,None,:].expand(-1,cfg.env.obs_history_length,-1)
                torch.testing.assert_close(reset_frames[:,:,3:6],expected_gravity)
                filtered_offset=cfg.env.num_observations
                torch.testing.assert_close(reset_frames[:,:,filtered_offset+3:filtered_offset+6],expected_gravity)
                assert torch.count_nonzero(reset_frames[:,:,:3])==0
                assert torch.count_nonzero(reset_frames[:,:,filtered_offset:filtered_offset+3])==0
                result['reset_history_checked']=True
                # A stale edge flag must not terminate an upright, centered
                # robot again on the tick immediately following an edge reset.
                env.contact_forces.zero_()
                env.edge_reset_buf.fill_(True)
                env._update_ship_relative_state()
                env.check_termination()
                assert not env.reset_buf.any(), 'stale edge reset leaked into the next episode'
                result['edge_reset_checked']=True
                env.episode_length_buf.fill_(env.max_episode_length+1)
                zero = torch.zeros(env.num_envs,env.num_actions,device=env.device)
                forced = env.step(zero)
                assert forced[2].all() and forced[3]['time_outs'].all()
                timed_out_snapshot = forced[3]['time_outs'].clone()
                following = env.step(zero)
                torch.testing.assert_close(following[3]['time_outs'],env.time_out_buf)
                assert not following[3]['time_outs'].any(), 'stale timeout caused repeated PPO bootstrap'
                assert timed_out_snapshot.all()
                result['timeout_refresh_checked']=True
                env.reset_idx(torch.arange(env.num_envs,device=env.device))
                contact_max = sensor_delta_max = action_delta = 0.0
                roll_max = pitch_max = heave_max = 0.0
                posture_error_sum = 0.0
                wheel_deck_pair_count = 0
                angle_max = 0.0
                resets = 0
                peak_dof_velocity = 0.0
                termination_counts = dict(contact=0, inversion=0, edge=0, low_height=0)
                failures_per_env = torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
                first_failure_steps = torch.full_like(failures_per_env,-1)
                survival_timeline = []
                position_error_sum = velocity_error_sum = 0.0
                episode_steps = torch.zeros(env.num_envs, device=env.device)
                completed_episode_s = []
                mobility_command_sum=mobility_error_sum=mobility_single_support=0.
                mobility_contact_sum=torch.zeros(2,device=env.device)
                initial_sensor = env.deck_imu.clone()
                previous_yaw=yaw_from_quaternion(env.root_states[env.actor_indices,3:7]).clone()
                cumulative_yaw=torch.zeros(env.num_envs,device=env.device)
                peak_yaw=torch.zeros_like(cumulative_yaw)
                yaw_rate_squared_sum=torch.zeros((),device=env.device)
                yaw_metric_count=torch.zeros((),device=env.device)
                for step_index in range(args.steps):
                    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
                    if runner is not None:
                        current_obs, current_history, current_commands, _ = env.get_observations()
                        latent = runner.alg.encode_for_policy(current_history, current_obs)
                        actions = runner.alg.actor_critic.act_inference(
                            torch.cat((latent, current_obs, current_commands), dim=-1))
                    obs, reward, dones, info, history, commands, critic = env.step(
                        actions)
                    assert torch.isfinite(obs).all() and torch.isfinite(history).all()
                    assert torch.isfinite(reward).all() and torch.isfinite(critic).all()
                    assert obs.shape == (env.num_envs, cfg.env.num_observations)
                    # Robot proprioception gets actor noise after critic construction.
                    # Deck IMU already models its own noise, shared by both consumers.
                    torch.testing.assert_close(critic[:, 27:36], obs[:, env.imu_slice])
                    if not cfg.ship.imu_enabled:
                        assert torch.count_nonzero(obs[:, env.imu_slice]) == 0
                        frames = history.view(env.num_envs, cfg.env.obs_history_length, -1)
                        assert torch.count_nonzero(frames[:, :, env.imu_slice]) == 0
                        assert torch.count_nonzero(frames[:, :, cfg.env.num_observations+20:cfg.env.num_observations+29]) == 0
                    if locomotion_mode!='balance':
                        mobility_command_sum+=float(env.commands[:,0].abs().mean())
                        mobility_error_sum+=float((env.base_lin_vel[:,0]-env.commands[:,0]).square().mean())
                        mobility_contact_sum+=env.mobility_contacts.float().mean(0)
                        mobility_single_support+=float((env.mobility_contacts.sum(-1)==1).float().mean())
                    contact_max = max(contact_max, float(env.contact_forces[:, env.feet_indices].norm(dim=-1).max()))
                    if args.device == "cpu":
                        wheel_ids = set(env.feet_indices.cpu().tolist())
                        for contact in env.gym.get_env_rigid_contacts(env.envs[0]):
                            b0, b1 = int(contact["body0"]), int(contact["body1"])
                            assert b0 >= 0 and b1 >= 0, "unexpected static-ground contact"
                            wheel_deck_pair_count += int(
                                (b0 == env.num_bodies and b1 in wheel_ids)
                                or (b1 == env.num_bodies and b0 in wheel_ids))
                    sensor_delta_max = max(sensor_delta_max, float((env.deck_imu-initial_sensor).abs().max()))
                    root = env.root_states[env.deck_indices]
                    angle_max = max(angle_max, float(root[:, 3:5].abs().max())*2)
                    x,y,z,w = root[:,3:7].unbind(-1)
                    roll = torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
                    pitch = torch.asin((2*(w*y-z*x)).clamp(-1,1))
                    roll_max = max(roll_max,float(roll.abs().max()))
                    pitch_max = max(pitch_max,float(pitch.abs().max()))
                    top,_ = surface_pose(root,cfg.ship.thickness)
                    heave_max = max(heave_max,float((top[:,2]-env.ship_pivot[:,2]).abs().max()))
                    posture_error_sum += float(env.projected_gravity[:,:2].square().sum(-1).mean())
                    resets += int(dones.sum())
                    peak_dof_velocity = max(peak_dof_velocity, float(env.dof_vel.abs().max()))
                    for key, mask in env.ship_termination_causes.items():
                        termination_counts[key] += int((mask & dones.bool()).sum())
                    timed_out = info.get('time_outs',torch.zeros_like(dones)).bool()
                    failure_mask=dones.bool() & ~timed_out
                    robot_state=env.root_states[env.actor_indices]
                    current_yaw=yaw_from_quaternion(robot_state[:,3:7])
                    increment=torch.atan2(torch.sin(current_yaw-previous_yaw),torch.cos(current_yaw-previous_yaw))
                    # Exclude terminal/post-reset states and never count a reset
                    # jump as physical turning. Keep the first-failure prefix.
                    yaw_valid=(failures_per_env==0) & ~dones.bool()
                    cumulative_yaw+=torch.where(yaw_valid,increment,torch.zeros_like(increment))
                    previous_yaw=current_yaw.clone()
                    if (step_index+1)*env.dt>=2.:
                        peak_yaw=torch.maximum(peak_yaw,torch.where(yaw_valid,cumulative_yaw.abs(),torch.zeros_like(cumulative_yaw)))
                        rate=heading_yaw_rate(robot_state[:,3:7],robot_state[:,10:13])
                        yaw_rate_squared_sum+=(rate.square()*yaw_valid).sum()
                        yaw_metric_count+=yaw_valid.sum()
                    record_first_failure(first_failure_steps,failure_mask,step_index+1)
                    failures_per_env += failure_mask.long()
                    if (step_index+1)%3000==0 or step_index+1==args.steps:
                        survival_timeline.append(survival_checkpoint(failures_per_env,step_index+1,env.dt))
                    episode_steps += 1
                    finished = dones.bool()
                    completed_episode_s.extend((episode_steps[finished]*env.dt).cpu().tolist())
                    episode_steps[finished] = 0
                    position_error_sum += float(env.ship_local_position[:, :2].norm(dim=-1).mean())
                    velocity_error_sum += float(env.ship_relative_velocity[:, :2].norm(dim=-1).mean())
                # Verify the actor sees these channels directly, independently of latent.
                if runner is None:
                    runner = OnPolicyRunner(env, class_to_dict(train_cfg), str(args.output), args.device)
                obs, history, commands, critic = env.get_observations()
                latent = runner.alg.encode_for_policy(history, obs)
                actor_input = torch.cat((latent, obs, commands), dim=-1)
                perturbed = actor_input.clone()
                perturbed[:, latent.shape[1]+20:latent.shape[1]+29] += 0.25
                action_delta = float((runner.alg.actor_critic.act_inference(perturbed)
                                      - runner.alg.actor_critic.act_inference(actor_input)).abs().max())
                # Zero IMU sensitivity is valid for migrated controllers (new
                # weights start at zero) and ablated policies. Report it, but
                # do not confuse a learned zero weight with a disconnected input.
                sensitivity_ok = args.mode == "eval" or action_delta > 1e-7
                result.update(passed=(contact_max > 1 and sensitivity_ok
                                      and float(env.ship_pose_error_max.max()) < 0.005
                                      and (args.device != "cpu" or wheel_deck_pair_count > 0)
                                      and (args.motion_scale == 0 or angle_max > 0.001)),
                              steps=args.steps, actor_obs_dim=env.num_obs,
                              encoder_history_dim=env.encoder_obs_history.shape[1],
                              peak_foot_contact_n=contact_max, max_deck_tilt_rad=angle_max,
                              max_actual_roll_deg=roll_max*180/3.141592653589793,
                              max_actual_pitch_deg=pitch_max*180/3.141592653589793,
                              max_actual_heave_m=heave_max,
                              mean_upright_error=posture_error_sum/args.steps,
                              wheel_deck_contact_pairs=(wheel_deck_pair_count if args.device == "cpu" else None),
                              max_sensor_change=sensor_delta_max, direct_actor_imu_sensitivity=action_delta,
                              max_deck_position_error_m=float(env.ship_pose_error_max.max()),
                              robot_resets=resets, policy="checkpoint" if args.mode == "eval" else "zero_action_PD",
                              peak_dof_velocity_rad_s=peak_dof_velocity,
                              termination_counts=termination_counts,
                              failure_resets=int(failures_per_env.sum()),
                              survival_fraction=float((failures_per_env==0).float().mean()),
                              survival_timeline=survival_timeline,
                              first_failure_times_s=[None if value<0 else value*env.dt
                                                     for value in first_failure_steps.cpu().tolist()],
                              mean_completed_episode_s=(sum(completed_episode_s)/len(completed_episode_s) if completed_episode_s else None),
                              mean_deck_relative_position_error_m=position_error_sum/args.steps,
                              mean_deck_relative_speed_m_s=velocity_error_sum/args.steps)
                result['yaw_metrics']={
                    'scope':'world heading change, unwrapped; >=2s and before each first failure; no reset jumps',
                    'heading_rate_rms_rad_s':None if not yaw_metric_count else float((yaw_rate_squared_sum/yaw_metric_count).sqrt()),
                    'peak_heading_change_median_deg':float(peak_yaw.median()*180/3.141592653589793),
                    'peak_heading_change_p95_deg':float(torch.quantile(peak_yaw,.95)*180/3.141592653589793),
                    'per_environment_peak_heading_change_deg':(peak_yaw*180/3.141592653589793).cpu().tolist(),
                    'full_survivor_count':int((failures_per_env==0).sum()),
                }
                if locomotion_mode!='balance':
                    result['mobility_metrics']={
                        'mean_abs_command_m_s':mobility_command_sum/args.steps,
                        'forward_velocity_rmse_m_s':(mobility_error_sum/args.steps)**.5,
                        'wheel_contact_fraction':(mobility_contact_sum/args.steps).cpu().tolist(),
                        'single_support_fraction':mobility_single_support/args.steps,
                        'command_profile':'position-independent smooth forward/back shuttle',
                        'mobility_success_claim':False,
                    }
                if not result["passed"]:
                    raise RuntimeError("contact/observation probe failed")
        (args.output/"env_cfg.json").write_text(json.dumps(class_to_dict(cfg), indent=2))
        (args.output/"train_cfg.json").write_text(json.dumps(class_to_dict(train_cfg), indent=2))
    except Exception as error:
        result.update(passed=False, error=repr(error))
        raise
    finally:
        result["elapsed_s"] = time.monotonic()-started
        (args.output/"result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        if env is not None:
            if env.viewer is not None:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
