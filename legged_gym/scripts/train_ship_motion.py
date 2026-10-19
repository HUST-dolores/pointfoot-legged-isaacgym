"""Isolated staged ship training with explicit legacy-policy migration.

Each run owns its source snapshot, configs, checkpoints and atomic progress
file. Start on static deck, then ramp roll, pitch and heave. No future motion
phase or curriculum scale is added to the actor input.
"""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('ROBOT_TYPE','WF_TRON1A')
import isaacgym
from isaacgym import gymapi,gymutil
import torch
from legged_gym.envs.wheelfoot_ship.wheelfoot_ship import BipedWFShip
from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_config import BipedCfgWFShip,BipedCfgPPOWFShip
from legged_gym.algorithm.on_policy_runner import OnPolicyRunner
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.ship_warm_start import migrate_flat_checkpoint, migrate_ship_mobility_checkpoint, ship_finetune_state_dicts, ship_prior_state_dicts
from legged_gym.utils.saved_config import apply_config_dict


def write_json(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


class ShipProgressRunner(OnPolicyRunner):
    def log(self,locs,**kwargs):
        super().log(locs,**kwargs)
        elapsed=locs['collection_time']+locs['learn_time']
        reward=sum(locs['rewbuffer'])/len(locs['rewbuffer']) if locs['rewbuffer'] else None
        length=sum(locs['lenbuffer'])/len(locs['lenbuffer']) if locs['lenbuffer'] else None
        ship=self.env.cfg.ship
        cap=getattr(self,'action_noise_cap',None)
        if cap is not None:
            with torch.no_grad():self.alg.actor_critic.logstd.clamp_(max=math.log(cap))
        if (ship.motion_gate_enabled and ship.motion_start_iteration is None
            and locs['it']+1 >= ship.warmup_iterations and length is not None
            and length*self.env.dt >= ship.motion_gate_episode_s):
            ship.motion_start_iteration=locs['it']+1
            write_json(Path(self.log_dir)/'env_cfg.json',class_to_dict(self.env.cfg))
            print(f'Motion gate opened at iteration {ship.motion_start_iteration}',flush=True)
        values={
            'state':'running','iteration':locs['it']+1,
            'updated_utc':datetime.now(timezone.utc).isoformat(),
            'pid':os.getpid(),'mean_reward':reward,
            'mean_episode_length_s':None if length is None else length*self.env.dt,
            'value_loss':float(locs['mean_value_loss']),
            'surrogate_loss':float(locs['mean_surrogate_loss']),
            'encoder_loss':float(locs['mean_extra_loss']),
            'policy_kl':float(locs['mean_kl']),
            'policy_updates':locs.get('extra_metrics',{}).get('policy_updates'),
            'current_learning_rate':float(self.alg.learning_rate),
            'max_action_noise_std':float(self.alg.actor_critic.logstd.exp().max()),
            'encoder_training_enabled':self.alg.extra_optimizer is not None,
            'fps':self.env.num_envs*self.num_steps_per_env/max(elapsed,1e-9),
            'motion_scale':self.env.training_motion_scale().cpu().tolist(),
            'motion_gate_open':ship.motion_start_iteration is not None,
            'motion_start_iteration':ship.motion_start_iteration,
            'max_deck_position_error_m':float(self.env.ship_pose_error_max.max()),
            'total_environment_steps':self.tot_timesteps,
        }
        if not all(math.isfinite(values[k]) for k in ['value_loss','surrogate_loss','encoder_loss']):
            raise RuntimeError('non-finite PPO loss')
        values.update({key:value for key,value in locs.get('extra_metrics',{}).items()
                       if key.startswith('encoder_')})
        write_json(Path(self.log_dir)/'progress.json',values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--label',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--num-envs',type=int,default=4096)
    p.add_argument('--iterations',type=int,default=12000)
    p.add_argument('--episode-length-s',type=float,help='Override the training episode horizon; default follows env config')
    p.add_argument('--yaw-tracking-scale',type=float,help='Optional existing yaw-rate tracking reward; default leaves rewards unchanged')
    p.add_argument('--heading-yaw-tracking-scale',type=float,help='Optional actual world-heading-rate tracking reward, not body gyro z')
    p.add_argument('--encoder-kl-limit',type=float,help='Optional sampled mean KL cap on the separate supervised encoder update')
    p.add_argument('--warmup-iterations',type=int,default=1000)
    p.add_argument('--motion-ramp-iterations',type=int,default=4000)
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=2026101401)
    p.add_argument('--imu-off',action='store_true')
    p.add_argument('--locomotion-mode',choices=['balance','roll','step'],default='balance')
    p.add_argument('--ship-warm-start',type=Path,help='Explicit 37->40 ship-to-mobility migration; new optimizer')
    p.add_argument('--ship-finetune',type=Path,help='Same-layout 37-observation ship weights/configs; fresh optimizer and run')
    p.add_argument('--model-prior',action='store_true',help='Standing ship only: add sensor-only geometric prior24 as ordinary observations; ship-finetune migrates obs37 -> obs61')
    p.add_argument('--prior-horizon-s',type=float,help='Optional positive short-horizon prediction time; requires --model-prior')
    p.add_argument('--warm-start',type=Path)
    p.add_argument('--noise-std',type=float,default=.25)
    p.add_argument('--save-interval',type=int,default=250)
    p.add_argument('--roll-deg',type=float,default=20)
    p.add_argument('--pitch-deg',type=float,default=12)
    p.add_argument('--heave-m',type=float,default=.15)
    p.add_argument('--physics-cpu',action='store_true')
    p.add_argument('--reference-mode',choices=['body','gravity_projection'],default=None)
    p.add_argument('--schedule',choices=['fixed','adaptive'],default='fixed')
    p.add_argument('--kl-early-stop',action='store_true')
    p.add_argument('--freeze-encoder',action='store_true')
    p.add_argument('--entropy-coef',type=float,default=.01)
    p.add_argument('--max-noise-std',type=float)
    p.add_argument('--corner-fraction',type=float,default=0.)
    args=p.parse_args()
    if args.model_prior and (args.locomotion_mode!='balance' or args.warm_start is not None or args.ship_warm_start is not None):
        p.error('model-prior is standing balance only; use --ship-finetune for existing ship weights')
    if args.prior_horizon_s is not None and (not args.model_prior or not math.isfinite(args.prior_horizon_s) or args.prior_horizon_s<=0):
        p.error('prior-horizon-s must be finite and positive and requires --model-prior')
    if args.ship_finetune is not None and (args.warm_start is not None or args.ship_warm_start is not None or args.locomotion_mode!='balance'):
        p.error('ship-finetune is balance-only and exclusive with other initializations')
    if args.heading_yaw_tracking_scale is not None and (not math.isfinite(args.heading_yaw_tracking_scale) or args.heading_yaw_tracking_scale<0):
        p.error('heading yaw tracking scale must be finite and nonnegative')
    if args.heading_yaw_tracking_scale is not None and args.yaw_tracking_scale is not None:
        p.error('choose either heading-rate or legacy body-yaw reward, not both')
    if args.ship_warm_start is not None and (args.warm_start is not None or args.locomotion_mode=='balance'):
        p.error('ship-warm-start is for mobility only and cannot be combined with flat warm-start')
    if args.warm_start is not None and args.locomotion_mode!='balance':
        p.error('mobility uses --ship-warm-start; flat migration is for balance only')
    if args.encoder_kl_limit is not None and (not math.isfinite(args.encoder_kl_limit) or args.encoder_kl_limit<=0):
        p.error('encoder KL limit must be finite and positive')
    if args.yaw_tracking_scale is not None and (not math.isfinite(args.yaw_tracking_scale) or args.yaw_tracking_scale<0):
        p.error('yaw tracking scale must be finite and nonnegative')
    if args.episode_length_s is not None and (not math.isfinite(args.episode_length_s) or args.episode_length_s<=0):
        p.error('episode length must be finite and positive')
    if not 0<=args.corner_fraction<=1 or args.entropy_coef<0:
        p.error('corner-fraction must be [0,1] and entropy coefficient nonnegative')
    if args.max_noise_std is not None and args.max_noise_std<args.noise_std:
        p.error('noise cap must be at least the initial noise standard deviation')
    if min(args.iterations,args.num_envs,args.save_interval,args.motion_ramp_iterations,args.learning_rate,args.noise_std)<=0:
        p.error('counts, learning rate and action noise must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'progress.json').exists():
        p.error('refusing to overwrite an existing run')
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    cfg,train=BipedCfgWFShip(),BipedCfgPPOWFShip()
    if args.model_prior:
        from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_prior_config import BipedCfgWFShipPrior,BipedCfgPPOWFShipPrior
        cfg,train=BipedCfgWFShipPrior(),BipedCfgPPOWFShipPrior()
    if args.ship_finetune is not None:
        source_env=json.loads((args.ship_finetune.parent/'env_cfg.json').read_text())
        source_train=json.loads((args.ship_finetune.parent/'train_cfg.json').read_text())
        source_obs=source_env['env']['num_observations']
        allowed_obs=(37,61) if args.model_prior else (37,)
        source_has_prior=bool(source_env.get('ship_model_prior',{}).get('enabled',False))
        if source_has_prior!=(source_obs==61):
            raise ValueError('source model-prior flag and observation dimensions disagree')
        if (source_obs not in allowed_obs or source_env['env']['num_critic_observations']!=source_obs+7
            or source_env['env']['num_actions']!=8 or source_env['env']['use_gait_phase']
            or source_env['env']['use_gait_stepping'] or 'ship' not in source_env):
            raise ValueError('finetune config is not the standing ship observation contract')
        apply_config_dict(cfg,source_env)
        apply_config_dict(train,source_train)
    env_type=BipedWFShip
    if args.locomotion_mode!='balance':
        from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_mobility import BipedWFShipMobility
        from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_mobility_config import (
            BipedCfgWFShipRoll,BipedCfgWFShipStep,BipedCfgPPOWFShipMobility,
        )
        cfg=(BipedCfgWFShipRoll if args.locomotion_mode=='roll' else BipedCfgWFShipStep)()
        train=BipedCfgPPOWFShipMobility()
        env_type=BipedWFShipMobility
    if args.model_prior:
        from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_prior import BipedWFShipPrior
        env_type=BipedWFShipPrior
        cfg.env.num_observations=61
        cfg.env.num_critic_observations=68
        cfg.ship_model_prior.enabled=True
        if args.prior_horizon_s is not None:
            cfg.ship_model_prior.prediction_horizon_s=args.prior_horizon_s
        train.MLP_Encoder.num_input_dim=(61+53)*cfg.env.obs_history_length
        train.runner.experiment_name='WF_TRON1A_ship_prior'
    cfg.seed=args.seed
    cfg.env.num_envs=args.num_envs
    if args.yaw_tracking_scale is not None:
        cfg.rewards.scales.tracking_ang_vel=args.yaw_tracking_scale
    if args.heading_yaw_tracking_scale is not None:
        cfg.rewards.scales.ship_yaw_rate=args.heading_yaw_tracking_scale
    if args.episode_length_s is not None:
        cfg.env.episode_length_s=args.episode_length_s
    cfg.ship.imu_enabled=not args.imu_off
    if args.reference_mode is not None:
        cfg.ship.reference_mode=args.reference_mode
    cfg.ship.corner_fraction=args.corner_fraction
    cfg.ship.curriculum=False
    cfg.ship.initial_difficulty=1.
    cfg.ship.training_schedule=True
    cfg.ship.warmup_iterations=args.warmup_iterations
    cfg.ship.motion_ramp_iterations=args.motion_ramp_iterations
    cfg.ship.motion_gate_enabled=True
    cfg.ship.motion_start_iteration=None
    cfg.ship.motion_gate_episode_s=16.0
    cfg.ship.amplitude_max=[math.radians(args.roll_deg),math.radians(args.pitch_deg),args.heave_m]
    cfg.ship.amplitude_min=[.4*v for v in cfg.ship.amplitude_max]
    train.seed=args.seed
    if args.encoder_kl_limit is not None:
        train.algorithm.encoder_kl_limit=args.encoder_kl_limit
    train.algorithm.learning_rate=args.learning_rate
    train.algorithm.schedule=args.schedule
    train.algorithm.early_stop=args.kl_early_stop
    train.algorithm.entropy_coef=args.entropy_coef
    train.algorithm.est_learning_rate=1e-4
    if args.freeze_encoder:train.algorithm.est_learning_rate=0.
    train.runner.ship_freeze_encoder=args.freeze_encoder
    train.runner.ship_action_noise_cap=args.max_noise_std
    train.runner.save_interval=args.save_interval
    train.runner.max_iterations=args.iterations
    params=gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
    use_gpu=args.device.startswith('cuda') and not args.physics_cpu
    params.use_gpu_pipeline=use_gpu
    params.physx.use_gpu=use_gpu
    params.physx.num_threads=4
    params.physx.num_position_iterations=8
    params.physx.num_velocity_iterations=2
    params.physx.max_gpu_contact_pairs=2**20
    started=time.monotonic()
    manifest={**vars(args),'output':str(args.output),'warm_start':None if args.warm_start is None else str(args.warm_start),
              'ship_warm_start':None if args.ship_warm_start is None else str(args.ship_warm_start),
              'ship_finetune':None if args.ship_finetune is None else str(args.ship_finetune),
              'initial_checkpoint_sha256':None if args.ship_finetune is None else hashlib.sha256(args.ship_finetune.read_bytes()).hexdigest(),
              'source_root':str(Path(__file__).resolve().parents[2]),'pid':os.getpid(),
              'started_utc':datetime.now(timezone.utc).isoformat(),'physics_gpu':use_gpu,
              'imu_channels':[20,29]}
    write_json(args.output/'manifest.json',manifest)
    write_json(args.output/'env_cfg.json',class_to_dict(cfg))
    write_json(args.output/'train_cfg.json',class_to_dict(train))
    env=None
    try:
        env=env_type(cfg,params,gymapi.SIM_PHYSX,'cpu' if args.physics_cpu else args.device,True)
        runner=ShipProgressRunner(env,class_to_dict(train),str(args.output),args.device)
        runner.action_noise_cap=args.max_noise_std
        if args.warm_start:
            checkpoint=torch.load(args.warm_start,map_location='cpu')
            model,encoder=migrate_flat_checkpoint(checkpoint,runner.alg.actor_critic.state_dict(),runner.alg.encoder.state_dict())
            runner.alg.actor_critic.load_state_dict(model)
            runner.alg.encoder.load_state_dict(encoder)
            print('Migrated flat policy: actor 38->47, critic 45->54, GRU frame 48->66',flush=True)
        if args.ship_warm_start:
            checkpoint=torch.load(args.ship_warm_start,map_location='cpu')
            model,encoder=migrate_ship_mobility_checkpoint(checkpoint,runner.alg.actor_critic.state_dict(),runner.alg.encoder.state_dict())
            runner.alg.actor_critic.load_state_dict(model)
            runner.alg.encoder.load_state_dict(encoder)
            print('Migrated ship policy: actor 47->50, critic 54->57, GRU frame 66->72; new optimizer',flush=True)
        if args.ship_finetune:
            checkpoint=torch.load(args.ship_finetune,map_location='cpu')
            loader=ship_prior_state_dicts if args.model_prior else ship_finetune_state_dicts
            model,encoder=loader(checkpoint,runner.alg.actor_critic.state_dict(),runner.alg.encoder.state_dict())
            runner.alg.actor_critic.load_state_dict(model)
            runner.alg.encoder.load_state_dict(encoder)
            print('Loaded ship weights only; '+('prior obs61/GRU114, added columns zero' if args.model_prior else 'same obs37/GRU66')+'; fresh optimizer and iteration0',flush=True)
        with torch.no_grad():
            runner.alg.actor_critic.logstd.fill_(math.log(args.noise_std))
        if args.freeze_encoder:
            runner.alg.extra_optimizer=None
            runner.alg.encoder.requires_grad_(False)
        runner.learn(args.iterations,init_at_random_ep_len=False)
        progress=json.loads((args.output/'progress.json').read_text())
        progress.update(state='complete',elapsed_s=time.monotonic()-started,
                        checkpoint=str(args.output/f'model_{args.iterations}.pt'))
        write_json(args.output/'progress.json',progress)
        write_json(args.output/'result.json',progress)
    except Exception as error:
        write_json(args.output/'progress.json',{'state':'failed','label':args.label,'error':repr(error),
                   'updated_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid()})
        raise
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__=='__main__':
    main()
