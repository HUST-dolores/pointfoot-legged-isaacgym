"""Actual simulator reset must preserve a frozen PPO transition likelihood."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('ROBOT_TYPE','WF_TRON1A')
from isaacgym import gymapi,gymutil
import torch
from legged_gym.envs.wheelfoot_ship.wheelfoot_ship import BipedWFShip
from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_config import BipedCfgWFShip,BipedCfgPPOWFShip
from legged_gym.algorithm.on_policy_runner import OnPolicyRunner
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.saved_config import apply_config_dict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--num-envs',type=int,default=4)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(2026101505);torch.set_num_threads(1)
    cfg,train=BipedCfgWFShip(),BipedCfgPPOWFShip()
    apply_config_dict(cfg,json.loads((a.checkpoint.parent/'env_cfg.json').read_text()))
    apply_config_dict(train,json.loads((a.checkpoint.parent/'train_cfg.json').read_text()))
    cfg.env.num_envs=a.num_envs;cfg.ship.curriculum=False;cfg.ship.training_schedule=False
    cfg.ship.corner_fraction=0.;cfg.ship.amplitude_min=cfg.ship.amplitude_max=[0.,0.,0.]
    params=gymapi.SimParams();gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
    params.use_gpu_pipeline=a.device.startswith('cuda');params.physx.use_gpu=params.use_gpu_pipeline
    params.physx.num_threads=1;params.physx.num_position_iterations=8;params.physx.num_velocity_iterations=2
    env=BipedWFShip(cfg,params,gymapi.SIM_PHYSX,a.device,True)
    try:
        runner=OnPolicyRunner(env,class_to_dict(train),None,a.device)
        state=torch.load(a.checkpoint,map_location=a.device)
        runner.alg.actor_critic.load_state_dict(state['model_state_dict'])
        runner.alg.encoder.load_state_dict(state['encoder_state_dict'])
        runner.alg.extra_optimizer=None;runner.alg.encoder.requires_grad_(False)
        obs,history,commands,critic=env.get_observations();before=history.clone()
        env.episode_length_buf.fill_(env.max_episode_length+1)
        with torch.inference_mode():
            actions=runner.alg.act(obs,history,commands,critic)
            data=env.step(actions)
            drift=float((runner.alg.transition.observation_history-before).abs().max())
            runner.alg.process_env_step(data[1],data[2],data[3],data[0])
            s=runner.alg.storage
            latent=runner.alg.encode_for_policy(s.observation_history[0],s.observations[0])
            actor_input=torch.cat((latent,s.observations[0],s.commands[0]),dim=-1)
            mean=runner.alg.actor_critic.act_inference(actor_input)
            error=float((mean-s.mu[0]).abs().max())
            kl=float(((mean-s.mu[0]).square()/(2*s.sigma[0].square())).sum(-1).mean())
        result={'passed':bool(data[2].all()) and drift==0 and error<1e-6 and kl<1e-10,
                'device':a.device,'num_envs':a.num_envs,'forced_timeout_all':bool(data[2].all()),
                'history_changed_after_env_step':drift,'unchanged_policy_mean_error':error,
                'unchanged_policy_kl':kl,'encoder_frozen':True,'checkpoint':str(a.checkpoint)}
        (a.output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
        if not result['passed']:raise RuntimeError('rollout likelihood did not roundtrip through reset')
    finally:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
