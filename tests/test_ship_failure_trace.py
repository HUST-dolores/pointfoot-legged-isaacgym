"""Read-only terminal capture contract, using the actual callback AST."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

def make_trace_env():
    spec = importlib.util.spec_from_file_location('trace_math', ROOT/'legged_gym/utils/ship_motion.py')
    math_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math_module)
    class Parent:
        def reset_idx(self, env_ids):
            self.parent_input = self.root_states.clone()
            self.root_states[self.actor_indices[env_ids], :3] = 99
            self.base_height[env_ids] = 99
            self.parent_called = True
            return 'base_reset_done'
    tree = ast.parse((ROOT/'legged_gym/scripts/ship_failure_trace.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FailureTraceShip')
    namespace = {'probe':SimpleNamespace(BipedWFShip=Parent),'TRACES':[],
                 'surface_pose':math_module.surface_pose,'inverse_rotate':math_module.inverse_rotate}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual failure capture>', 'exec'), namespace)
    env = namespace['FailureTraceShip']()
    env.num_envs, env.num_dof, env.dt = 2, 8, .02
    env.cfg = SimpleNamespace(ship=SimpleNamespace(thickness=.2,length=6.,width=4.))
    env.actor_indices = torch.tensor([0,2])
    env.deck_indices = torch.tensor([1,3])
    env.root_states = torch.zeros(4,13)
    env.root_states[:,6] = 1
    env.root_states[env.actor_indices,2] = 1.8
    env.root_states[env.deck_indices,2] = 1.4
    env.reset_buf = torch.tensor([True,True])
    env.time_out_buf = torch.tensor([False,False])
    env.ship_time = torch.tensor([4.,4.])
    env.episode_length_buf = torch.tensor([200,200])
    env.base_height = torch.tensor([.3,.3])
    env.gravity_vec = torch.tensor([[0.,0.,-1.],[0.,0.,-1.]])
    env.projected_gravity = env.gravity_vec.clone()
    env.dof_state = torch.zeros(16,2)
    env.dof_state[:,1] = 3
    env.dof_vel = torch.full((2,8),2.)  # separate estimate, not physical state
    env.actions = torch.full((2,8),.15)
    env.feet_indices = torch.tensor([0,1])
    env.foot_positions = torch.tensor([[[0.,.2,1.6],[0.,-.2,1.6]]]*2)
    env.contact_forces = torch.tensor([[[0.,0.,10.],[0.,0.,0.]]]*2)
    env.ship_termination_causes = {'low_height':torch.tensor([True,True])}
    return env, namespace['TRACES']

def test_physical_capture_is_before_parent_reset_and_read_only():
    env, records = make_trace_env()
    original = env.root_states.clone()
    result = env.reset_idx(torch.tensor([0]))
    assert result == 'base_reset_done'
    torch.testing.assert_close(env.parent_input, original)
    assert len(records) == 1 and records[0]['env_id'] == 0
    record = records[0]
    assert record['robot_root'][2] == pytest.approx(1.8)
    assert record['physical_height_m'] == pytest.approx(.3,abs=1e-6)
    assert record['height_cache_error_m'] < 1e-6
    assert record['physical_dof_velocity'] == [3.]*8
    assert record['actions'] == pytest.approx([.15]*8)
    assert env.root_states[0,2] == 99
    assert not record['any_foot_center_outside_xy']
    assert record['foot_contact_n'] == [10.,0.]

def test_foot_exit_is_recorded_separately_from_root_failure_label():
    env, records = make_trace_env()
    env.foot_positions[0,0,1] = 2.2
    env.reset_idx(torch.tensor([0]))
    assert records[0]['any_foot_center_outside_xy']
    assert records[0]['feet_deck_xyz_m'][0][1] == pytest.approx(2.2)

@pytest.mark.parametrize('excluded', ['timeout','explicit_reset','startup'])
def test_nonfall_and_startup_events_excluded_without_skipping_reset(excluded):
    env, records = make_trace_env()
    if excluded == 'timeout': env.time_out_buf[0] = True
    if excluded == 'explicit_reset': env.reset_buf[0] = False
    if excluded == 'startup': env.ship_time[0] = 1.9
    env.reset_idx(torch.tensor([0]))
    assert records == [] and env.parent_called

def test_mismatched_caches_are_reported_not_used_as_physical_truth():
    env, records = make_trace_env()
    env.base_height[0] = .7
    env.projected_gravity[0,0] = .1
    env.reset_idx(torch.tensor([0]))
    assert records[0]['physical_height_m'] == pytest.approx(.3,abs=1e-6)
    assert records[0]['height_cache_error_m'] == pytest.approx(.4,abs=1e-6)
    assert records[0]['gravity_cache_error'] == pytest.approx(.1,abs=1e-6)
