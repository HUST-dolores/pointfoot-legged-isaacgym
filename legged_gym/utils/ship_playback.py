"""Single ship play/evaluation entry: preserve saved sensors, use the probe loop.

No simulator is imported here. Legacy non-ship play behavior is unchanged.
Explicit input/output paths and a recorded protocol avoid ambiguous playback.
"""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

SHIP_TASK_MODES={'wheelfoot_ship':'balance','wheelfoot_ship_prior':'balance','wheelfoot_ship_roll':'roll','wheelfoot_ship_step':'step'}


def ship_play_request(args, root):
    """Resolve one immutable checkpoint and explicit evaluation overrides."""
    task=getattr(args,'task',None)
    if task not in SHIP_TASK_MODES:
        raise ValueError('not a supported ship playback task')
    run=getattr(args,'load_run',None)
    number=getattr(args,'checkpoint',None)
    if not run or number is None or number<0:
        raise ValueError('ship play requires explicit --load_run and --checkpoint >= 0')
    run=Path(run)
    if not run.is_absolute():
        default_experiment=('WF_TRON1A_ship_prior' if task=='wheelfoot_ship_prior' else 'WF_TRON1A_ship' if task=='wheelfoot_ship' else 'WF_TRON1A_ship_mobility')
        experiment=getattr(args,'experiment_name',None) or default_experiment
        run=Path(root)/'logs'/task/experiment/run
    checkpoint=run/f'model_{number}.pt'
    for file in [checkpoint,run/'env_cfg.json',run/'train_cfg.json']:
        if not file.is_file():raise FileNotFoundError(file)
    saved=json.loads((run/'env_cfg.json').read_text())
    if 'ship' not in saved:raise ValueError('checkpoint config is not a ship task')
    if bool(saved.get('ship_model_prior',{}).get('enabled',False))!=(task=='wheelfoot_ship_prior'):
        raise ValueError('model-prior checkpoints require wheelfoot_ship_prior task; legacy checkpoints require legacy task')
    mode=saved.get('ship_mobility',{}).get('mode','balance')
    if mode!=SHIP_TASK_MODES[task]:raise ValueError('task and saved locomotion mode differ')
    # These flags previously changed ship conditions through the legacy path.
    # Do not silently ignore an explicitly requested incompatible experiment.
    if getattr(args,'no_noise',False):raise ValueError('unified ship play preserves saved noise; --no_noise is not allowed')
    if any(abs(float(getattr(args,k,0.) or 0.))>1e-12 for k in ['cmd_vx','cmd_vy','cmd_yaw','cmd_step','slope_deg']):
        raise ValueError('unified ship play uses saved commands and fixed world gravity; command/slope overrides are not allowed')
    if getattr(args,'load_hold',False) or getattr(args,'no_load',False):
        raise ValueError('legacy payload overrides do not apply to ship play')
    history=getattr(args,'history_length',None)
    if history is not None and history>0 and history!=saved['env']['obs_history_length']:
        raise ValueError('unified ship play preserves saved history length')
    scale=float(getattr(args,'ship_motion_scale',1.))
    steps=getattr(args,'ship_eval_steps',3000)
    count=getattr(args,'num_envs',None)
    count=64 if count is None else count
    seed=getattr(args,'seed',None)
    seed=2026101499 if seed is None else seed
    if not math.isfinite(scale) or scale<0 or type(steps) is not int or steps<1 or type(count) is not int or count<1:
        raise ValueError('positive integer env/step counts and finite nonnegative motion scale required')
    if type(seed) is not int:raise ValueError('seed must be an integer')
    output=getattr(args,'ship_eval_output',None)
    if output is None:
        stamp=datetime.now().strftime('%Y%m%dT%H%M%S%f')
        output=Path(root)/'logs'/'wheelfoot_ship'/f'unified_play_{stamp}_{seed}_model{number}'
    output=Path(output)
    if output.exists():raise FileExistsError('refusing to overwrite ship play output: '+str(output))
    amplitude=saved['ship']['amplitude_max']
    frequency=saved['ship']['frequency_max']
    if len(amplitude)!=3 or len(frequency)!=3 or any(not math.isfinite(v) or v<0 for v in amplitude) or any(not math.isfinite(v) or v<=0 for v in frequency):
        raise ValueError('invalid saved ship amplitude/frequency')
    device=getattr(args,'rl_device','cuda:0')
    sim_device=getattr(args,'sim_device',device)
    if sim_device!=device:raise ValueError('ship play currently requires matching sim_device and rl_device')
    argv=['--mode','eval','--checkpoint',str(checkpoint),'--device',device,
        '--num-envs',str(count),'--steps',str(steps),'--seed',str(seed),'--motion-scale',str(scale),
        '--roll-deg',str(math.degrees(amplitude[0])),'--pitch-deg',str(math.degrees(amplitude[1])),
        '--heave-m',str(amplitude[2]),'--fixed-amplitude','--fixed-frequency-hz',*[str(v) for v in frequency],
        '--output',str(output)]
    # Always honor the checkpoint sensor contract; never enable IMU by default
    # when loading the separately trained no-IMU policy.
    if not saved['ship']['imu_enabled']:argv.append('--imu-off')
    if not getattr(args,'headless',False):argv.append('--viewer')
    metadata={'entrypoint':'play.py -> unified ship_motion_probe.main','task':task,'locomotion_mode':mode,
        'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'seed':seed,'num_envs':count,'steps':steps,'motion_scale':scale,
        'fixed_amplitude_rad_m':[v*scale for v in amplitude],'fixed_frequency_hz':frequency,
        'saved_imu_enabled':saved['ship']['imu_enabled'],'saved_noise':saved['noise'],
        'saved_reference_mode':saved['ship']['reference_mode'],
        'policy_mode':'mean action, no action sampling; saved observation noise remains enabled',
        'legacy_play_overrides_applied':False,'probe_argv':argv}
    return argv,output,saved,metadata


def verify_ship_play_contract(saved, actual):
    """Permit only declared test-severity/horizon/population overrides."""
    ship_keys=['imu_enabled','imu_delay_steps','gyro_noise_std','accel_noise_std',
        'attitude_noise_std','sensor_offset','gyro_scale','accel_scale','reference_mode',
        'length','width','thickness','mass_kg','friction','edge_margin','ramp_s','surface_z']
    for key in ship_keys:
        if actual['ship'].get(key)!=saved['ship'].get(key):
            raise RuntimeError('saved ship sensor/physics setting changed: '+key)
    if saved.get('ship_model_prior',{})!=actual.get('ship_model_prior',{}):
        raise RuntimeError('saved model-prior sensor/geometry/prediction contract changed')
    for section in ['noise','control','normalization','init_state','domain_rand','sim','commands']:
        if actual[section]!=saved[section]:
            # Gym resolves seconds-based integer intervals in domain_rand.
            if section=='domain_rand':
                a=dict(actual[section]);b=dict(saved[section]);a.pop('push_interval',None);b.pop('push_interval',None)
                if a==b:continue
            raise RuntimeError('saved playback contract changed: '+section)
    for key in ['num_observations','num_critic_observations','num_actions','obs_history_length']:
        if actual['env'][key]!=saved['env'][key]:raise RuntimeError('saved observation contract changed: '+key)


def run_unified_ship_play(args):
    from legged_gym import LEGGED_GYM_ROOT_DIR
    from legged_gym.scripts import ship_motion_probe
    argv,output,saved,metadata=ship_play_request(args,LEGGED_GYM_ROOT_DIR)
    print('[SHIP PLAY] full saved config; saved IMU/noise/frames; fixed-amplitude evaluation; legacy overrides bypassed',flush=True)
    previous=sys.argv
    try:
        sys.argv=['ship_motion_probe.py',*argv]
        ship_motion_probe.main()
        actual=json.loads((output/'env_cfg.json').read_text())
        verify_ship_play_contract(saved,actual)
        metadata['contract_verified']=True
        (output/'play_protocol.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
        print('[SHIP PLAY] SAVED_CONTRACT_VERIFIED '+str(output),flush=True)
    finally:
        sys.argv=previous
