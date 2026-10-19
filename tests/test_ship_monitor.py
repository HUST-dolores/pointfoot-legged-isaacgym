"""Fleet report regressions; no network, training or systemd actions."""
import importlib.util
import json
import shlex
import pytest
from pathlib import Path
from types import SimpleNamespace


def load_monitor():
    path = Path(__file__).resolve().parents[1]/"legged_gym/scripts/monitor_ship_training.py"
    spec = importlib.util.spec_from_file_location("ship_monitor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_probe_preserves_written_metrics(monkeypatch):
    module = load_monitor()
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return SimpleNamespace(returncode=1, stderr="probe check failed", stdout="")
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({"passed": False, "failure_resets": 100}))
    monkeypatch.setattr(module.subprocess, "run", run)
    job = {"output": "/isolated/run", "python": "/env/python", "source": "/isolated/source",
           "latest_checkpoint": "/isolated/run/model_250.pt", "gpu": 1, "host": "lab",
           "roll_deg": 20, "pitch_deg": 12, "heave_m": .15, "imu_enabled": False}
    value = module.evaluate(job, "test", .25)
    assert value["failure_resets"] == 100
    assert value["execution_returncode"] == 1
    assert "--imu-off" in calls[0][-1]
    assert "--motion-scale 0.25" in calls[0][-1]


def test_terminal_checkpoint_is_evaluated_on_all_three_scales(monkeypatch, tmp_path):
    module = load_monitor()
    manifest = tmp_path/"manifest.json"
    job = {"label": "run", "host": "lab", "gpu": 0,"output":"/isolated/run"}

    manifest.write_text(json.dumps({"jobs": [job]}))
    collected = {**job, "progress": {"state": "exited_without_completion"},
                 "latest_checkpoint": "/isolated/model_250.pt", "exit_code": 139,
                 "physx_buffer_errors": 20}
    monkeypatch.setattr(module, "collect", lambda host, jobs: {"jobs": [collected], "gpu": ""})
    scales = []
    def evaluate(job, stamp, scale):
        scales.append(scale)
        return {"failure_resets": 3}
    monkeypatch.setattr(module, "evaluate", evaluate)
    monkeypatch.setattr(module.sys, "argv", ["monitor", "--manifest", str(manifest), "--evaluate"])
    module.main()
    assert scales == [0., .25, 1.]
    report = json.loads((tmp_path/"status.json").read_text())
    assert set(report["jobs"][0]["evaluation"]) == {"0", "0.25", "1"}
    assert "139" in (tmp_path/"status.md").read_text()
    assert "仿真无效警告" in (tmp_path/"status.md").read_text()


def test_quick_poll_preserves_evaluation_and_its_original_model(monkeypatch, tmp_path):
    module=load_monitor()
    job={"label":"run","host":"lab","gpu":0,"output":"/isolated/run"}
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"jobs":[job]}))
    old={**job,"latest_checkpoint":"/isolated/model_250.pt",
         "evaluation":{"1":{"survival_fraction":.5,"failure_resets":4}},
         "evaluation_updated_utc":"2026-10-14T15:00:00+00:00"}
    (tmp_path/"status.json").write_text(json.dumps({"jobs":[old]}))
    current={**job,"progress":{"state":"running"},"latest_checkpoint":"/isolated/model_500.pt"}
    monkeypatch.setattr(module,"collect",lambda host,jobs:{"jobs":[current],"gpu":""})
    monkeypatch.setattr(module.sys,"argv",["monitor","--manifest",str(manifest)])
    module.main()
    report=json.loads((tmp_path/"status.json").read_text())["jobs"][0]
    assert report["latest_checkpoint"].endswith("model_500.pt")
    assert report["evaluation_checkpoint"].endswith("model_250.pt")
    assert report["evaluation"]["1"]["survival_fraction"]==.5
    assert "model_250.pt" in (tmp_path/"status.md").read_text()

def evaluation_job(**options):
    return {"label":"duration","output":"/isolated/run","python":"/env/python",
            "source":"/isolated/source","latest_checkpoint":"/isolated/run/model_250.pt",
            "gpu":1,"host":"lab","roll_deg":20,"pitch_deg":12,"heave_m":.15,
            "imu_enabled":True,**options}

@pytest.mark.parametrize('override', [False, True])
def test_evaluation_default_and_long_steps(monkeypatch, override):
    module=load_monitor()
    expected=3000 if override else 1000
    job=evaluation_job(**({'evaluation_steps':expected} if override else {}))
    calls=[]
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0,stderr='',stdout='' if len(calls)==1 else json.dumps({'passed':True,'steps':expected}))
    monkeypatch.setattr(module.subprocess,'run',run)
    result=module.evaluate(job,'duration',1.)
    args=shlex.split(calls[0][-1])
    assert args[args.index('--steps')+1]==str(expected)
    assert result['steps']==expected

@pytest.mark.parametrize('override',[False,True])
def test_evaluation_default_and_explicit_seed(monkeypatch,override):
    module=load_monitor();expected=2026101504 if override else 2026101499
    job=evaluation_job(**({'evaluation_seed':expected} if override else {}))
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0,stderr='',stdout='' if len(calls)==1 else json.dumps({'seed':expected}))
    monkeypatch.setattr(module.subprocess,'run',run)
    result=module.evaluate(job,'seed-test',1.)
    args=shlex.split(calls[0][-1])
    assert args[args.index('--seed')+1]==str(expected)
    assert result['seed']==expected

@pytest.mark.parametrize('seed',[None,True,1.5,'5'])
def test_invalid_evaluation_seed_never_launch(monkeypatch,seed):
    module=load_monitor()
    def unexpected(*args,**kwargs):raise AssertionError('invalid seed must not launch SSH')
    monkeypatch.setattr(module.subprocess,'run',unexpected)
    with pytest.raises(ValueError,match='evaluation_seed'):
        module.evaluate(evaluation_job(evaluation_seed=seed),'bad-seed',1.)

@pytest.mark.parametrize('steps', [0, -1, 1.5, True])
def test_invalid_evaluation_steps_never_launch(monkeypatch, steps):
    module=load_monitor()
    def unexpected(*args, **kwargs):
        raise AssertionError('must not launch SSH for invalid steps')
    monkeypatch.setattr(module.subprocess,'run',unexpected)
    with pytest.raises(ValueError,match='positive integer'):
        module.evaluate(evaluation_job(evaluation_steps=steps),'invalid',1.)

@pytest.mark.parametrize('actual_steps', [1000, 3000])
def test_report_uses_actual_duration_not_planned(monkeypatch, tmp_path, actual_steps):
    module=load_monitor()
    job=evaluation_job(evaluation_steps=3000)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'jobs':[job]}))
    collected={**job,'progress':{'state':'running'}}
    monkeypatch.setattr(module,'collect',lambda host,jobs:{'jobs':[collected],'gpu':''})
    monkeypatch.setattr(module,'evaluate',lambda job,stamp,scale:{'steps':actual_steps,'survival_fraction':1.,'execution_returncode':0})
    monkeypatch.setattr(module.sys,'argv',['monitor','--manifest',str(manifest),'--evaluate','--eval-scales','1'])
    module.main()
    report=(tmp_path/'status.md').read_text()
    assert f'| {actual_steps*.02:.1f} |' in report
    assert '每环境 20 秒' not in report
