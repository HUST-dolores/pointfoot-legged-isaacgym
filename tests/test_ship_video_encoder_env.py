"""Actual encoder environment helper must not alter native simulator environment."""
import ast
import os
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('library_path',[None,'/tmp/fake-conda/lib:/tmp/fake-torch/lib'])
def test_encoder_library_isolation_does_not_mutate_parent(monkeypatch,library_path):
    if library_path is None:monkeypatch.delenv('LD_LIBRARY_PATH',raising=False)
    else:monkeypatch.setenv('LD_LIBRARY_PATH',library_path)
    monkeypatch.setenv('SHIP_ENCODER_TEST','preserved')
    before=dict(os.environ)
    tree=ast.parse((ROOT/'legged_gym/scripts/render_ship_deck_video.py').read_text())
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='ffmpeg_environment')
    ns={'os':os}
    exec(compile(ast.Module(body=[helper],type_ignores=[]),'<actual encoder environment>','exec'),ns)
    child=ns['ffmpeg_environment']()
    assert 'LD_LIBRARY_PATH' not in child
    assert child=={k:v for k,v in before.items() if k!='LD_LIBRARY_PATH'}
    assert dict(os.environ)==before
    assert child['SHIP_ENCODER_TEST']=='preserved'

@pytest.mark.parametrize('script',['render_ship_deck_video.py','render_ship_policy_video.py'])
def test_both_real_video_encoders_use_isolated_child_env(script):
    tree=ast.parse((ROOT/'legged_gym/scripts'/script).read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen']
    assert len(calls)==1
    keyword=next(k for k in calls[0].keywords if k.arg=='env')
    assert isinstance(keyword.value,ast.Call) and keyword.value.func.id=='ffmpeg_environment'
