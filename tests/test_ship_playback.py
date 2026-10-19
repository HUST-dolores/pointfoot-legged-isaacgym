"""Unified ship play config/input contracts without loading Isaac Gym."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as S
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('unified_ship_playback',ROOT/'legged_gym/utils/ship_playback.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(tmp_path,imu=True):
    run=tmp_path/'saved_run';run.mkdir()
    saved={'env':{'num_observations':37,'num_critic_observations':44,'num_actions':8,'obs_history_length':10},
        'ship':{'imu_enabled':imu,'amplitude_max':[.3490658503988659,.20943951023931956,.15],
                'frequency_max':[.35,.3,.4],'reference_mode':'gravity_projection',
                'imu_delay_steps':1,'gyro_noise_std':.005,'accel_noise_std':.03,'attitude_noise_std':.002,
                'sensor_offset':[0,0,.1],'gyro_scale':1.,'accel_scale':1/9.81,'length':6.,'width':4.,
                'thickness':.2,'mass_kg':10000.,'friction':.9,'edge_margin':.4,'ramp_s':2.,'surface_z':1.5},
        'noise':{'add_noise':True,'noise_level':1.5},'control':{'decimation':4},
        'sim':{'dt':.005,'gravity':[0,0,-9.81]},'normalization':{'obs_scales':{'ang_vel':.25}},
        'init_state':{'pos':[0,0,.9664]},'domain_rand':{'push_interval':3000,'push_robots':False},
        'commands':{'ranges':{'ang_vel_yaw':[0,0]}}}
    (run/'env_cfg.json').write_text(json.dumps(saved));(run/'train_cfg.json').write_text('{}')
    (run/'model_12000.pt').write_bytes(b'fixture weights, not loaded by pure config tests')
    args=S(task='wheelfoot_ship',load_run=str(run),checkpoint=12000,rl_device='cuda:0',sim_device='cuda:0',
           headless=True,seed=2026101499,num_envs=64,ship_eval_steps=3000,ship_motion_scale=1.,
           ship_eval_output=str(tmp_path/'new_output'))
    return args,saved

@pytest.mark.parametrize('imu',[False,True])
def test_saved_imu_noise_frames_and_fixed_severity_are_preserved(tmp_path,imu):
    args,saved=fixture(tmp_path,imu)
    argv,output,_,metadata=m.ship_play_request(args,tmp_path)
    assert ('--imu-off' in argv)==(not imu)
    assert '--viewer' not in argv and '--fixed-amplitude' in argv
    assert metadata['saved_imu_enabled']==imu
    assert metadata['saved_noise']==saved['noise']
    assert metadata['saved_reference_mode']=='gravity_projection'
    assert metadata['fixed_frequency_hz']==[.35,.3,.4]
    assert metadata['legacy_play_overrides_applied'] is False
    assert not output.exists()

@pytest.mark.parametrize('scale',[0.,.25,1.])
def test_explicit_static_mild_and_full_protocols(tmp_path,scale):
    args,saved=fixture(tmp_path);args.ship_motion_scale=scale
    argv,_,_,meta=m.ship_play_request(args,tmp_path)
    assert argv[argv.index('--motion-scale')+1]==str(scale)
    assert meta['fixed_amplitude_rad_m']==[v*scale for v in saved['ship']['amplitude_max']]

@pytest.mark.parametrize('field,value',[
    ('no_noise',True),('cmd_yaw',.2),('cmd_vx',.3),('slope_deg',10.),('load_hold',True),
    ('ship_motion_scale',float('nan')),('ship_eval_steps',0),('num_envs',True),
    ('history_length',5),('sim_device','cpu'),('checkpoint',-1),
])
def test_undeclared_or_invalid_overrides_rejected(tmp_path,field,value):
    args,_=fixture(tmp_path);setattr(args,field,value)
    with pytest.raises(ValueError):m.ship_play_request(args,tmp_path)
    assert not (tmp_path/'new_output').exists()

def test_output_is_never_overwritten(tmp_path):
    args,_=fixture(tmp_path);Path(args.ship_eval_output).mkdir()
    with pytest.raises(FileExistsError):m.ship_play_request(args,tmp_path)

def test_task_mismatch_rejected(tmp_path):
    args,_=fixture(tmp_path);args.task='wheelfoot_ship_step'
    with pytest.raises(ValueError):m.ship_play_request(args,tmp_path)

def test_contract_allows_only_declared_population_and_motion_changes(tmp_path):
    _,saved=fixture(tmp_path);actual=copy.deepcopy(saved)
    actual['env']['num_envs']=64;actual['env']['episode_length_s']=60.02
    actual['ship']['amplitude_min']=actual['ship']['amplitude_max']
    actual['ship']['training_schedule']=False;actual['ship']['corner_fraction']=0.
    actual['domain_rand']['push_interval']=150
    m.verify_ship_play_contract(saved,actual)

@pytest.mark.parametrize('section,key,value',[
    ('ship','imu_enabled',False),('ship','imu_delay_steps',0),('ship','gyro_scale',.5),
    ('ship','reference_mode','body'),('noise','noise_level',.5),('sim','gravity',[0,1,-9.8]),
    ('env','num_observations',40),('control','decimation',2),
])
def test_actual_output_contract_catches_sensor_frame_noise_or_physics_drift(tmp_path,section,key,value):
    _,saved=fixture(tmp_path);actual=copy.deepcopy(saved);actual[section][key]=value
    with pytest.raises(RuntimeError):m.verify_ship_play_contract(saved,actual)

def test_generic_play_ship_dispatch_precedes_all_legacy_overrides():
    tree=ast.parse((ROOT/'legged_gym/scripts/play.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='play')
    first=function.body[0]
    assert isinstance(first,ast.If)
    assert set(ast.literal_eval(first.test.comparators[0]))==set(m.SHIP_TASK_MODES)
    assert any(isinstance(n,ast.Return) for n in first.body)
    assert any(isinstance(n,ast.ImportFrom) and n.module=='legged_gym.utils.ship_playback' for n in first.body)

@pytest.mark.parametrize('saved_imu',[False,True])
def test_probe_without_explicit_override_honors_saved_imu(saved_imu):
    tree=ast.parse((ROOT/'legged_gym/scripts/ship_motion_probe.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Attribute)
              and isinstance(n.test.value,ast.Name) and n.test.value.id=='args' and n.test.attr=='imu_off')
    cfg=S(ship=S(imu_enabled=saved_imu));args=S(imu_off=False,imu_on=False)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual probe sensor override>','exec'),dict(args=args,cfg=cfg))
    assert cfg.ship.imu_enabled==saved_imu


def test_prior_play_requires_matching_task_and_preserves_prior_settings(tmp_path):
    args,saved=fixture(tmp_path)
    saved['env'].update(num_observations=61,num_critic_observations=68)
    saved['ship_model_prior']={'enabled':True,'prediction_horizon_s':.1,'initial_relative_yaw_rad':0.}
    (Path(args.load_run)/'env_cfg.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError):m.ship_play_request(args,tmp_path)
    args.task='wheelfoot_ship_prior'
    _,_,_,metadata=m.ship_play_request(args,tmp_path)
    assert metadata['task']=='wheelfoot_ship_prior'
    actual=copy.deepcopy(saved)
    m.verify_ship_play_contract(saved,actual)
    actual['ship_model_prior']['prediction_horizon_s']=.2
    with pytest.raises(RuntimeError):m.verify_ship_play_contract(saved,actual)
