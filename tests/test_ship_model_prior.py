"""Analytical sensor-only prior, actual URDF FK/IK, and compatibility tests."""
import argparse
import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace as S

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


m = module('ship_prior_math', 'legged_gym/utils/ship_model_prior.py')
warm = module('ship_prior_migration', 'legged_gym/utils/ship_warm_start.py')
NAMES = [['abad_L_Joint', 'hip_L_Joint', 'knee_L_Joint'],
         ['abad_R_Joint', 'hip_R_Joint', 'knee_R_Joint']]


def kin():
    return m.WheelLegKinematics(ROOT/'resources/robots/WF_TRON1A/urdf/robot.urdf',
        NAMES, ['wheel_L_Joint', 'wheel_R_Joint'], dtype=torch.float64)


def sensors(n=2):
    imu = torch.zeros(n, 9, dtype=torch.float64); imu[:, 5] = 9.81; imu[:, 8] = -1
    body_gyro = torch.zeros(n, 3, dtype=torch.float64)
    gravity = body_gyro.clone(); gravity[:, 2] = -1
    q = torch.zeros(n, 2, 3, dtype=torch.float64)
    force = torch.full((n, 2), 30., dtype=torch.float64)
    return imu, body_gyro, gravity, q, force


def test_zero_fk_matches_all_urdf_joint_offsets():
    k = kin()
    expected = torch.tensor([[[-.02144, .1485, -.77982], [-.02144, -.1485, -.77982]]], dtype=torch.float64)
    torch.testing.assert_close(k.forward(torch.zeros(1, 2, 3, dtype=torch.float64)), expected)


def test_exact_spatial_jacobian_matches_finite_differences():
    k = kin(); q = torch.tensor([[[.1, .2, -.2], [-.1, -.2, .2]]], dtype=torch.float64)
    _, jacobian = k.forward(q, jacobian=True)
    eps = 1e-6
    for j in range(3):
        plus, minus = q.clone(), q.clone(); plus[:, :, j] += eps; minus[:, :, j] -= eps
        torch.testing.assert_close((k.forward(plus)-k.forward(minus))/(2*eps), jacobian[..., j], atol=1e-9, rtol=1e-6)


def test_so3_zero_gradients_and_opposite_vector_alignment_are_finite():
    v = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    r = m.so3_exp(v)
    assert torch.isfinite(torch.autograd.grad(r.sum(), v)[0]).all()
    a = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
    result = m.apply_rotation(m.so3_exp(m.alignment_vector(a, -a)), a)
    torch.testing.assert_close(result, -a, atol=1e-12, rtol=0)


def test_static_flat_prior_is_neutral_reference_not_action():
    prior = m.ShipModelPrior(kin(), 2, .02, .127)
    output = prior.update(*sensors())
    assert output.shape == (2, 24)
    torch.testing.assert_close(output[:, :3], torch.tensor([[0., 0., 1.]], dtype=torch.float64).repeat(2, 1))
    assert output[:, 6:20].abs().max() < 1e-12
    assert (output[:, 20:24] == 1).all()


def test_update_does_not_mutate_any_sensor_or_joint_input():
    data=sensors(); data[0][:,0]=.5
    original=[x.clone() for x in data]
    m.ShipModelPrior(kin(),2,.02,.127).update(*data)
    for before,after in zip(original,data):torch.testing.assert_close(before,after)


def test_valid_joint_references_reach_the_predicted_support_plane():
    k=kin();prior=m.ShipModelPrior(k,2,.02,.127)
    data=sensors();data[0][:,0]=.5
    output=prior.update(*data)
    wheels=k.forward(data[3]);normal=output[:,:3];future=output[:,3:6]
    contact=wheels-.127*normal[:,None,:]
    displacement=output[:,6:12].reshape(2,2,3)/prior.contact_displacement_scale
    target=contact+displacement+.127*future[:,None,:]
    q_ref=data[3]+output[:,14:20].reshape(2,2,3)/prior.joint_angle_scale
    assert (output[:,22:24]==1).all()
    assert ((k.forward(q_ref)-target).norm(dim=-1)<prior.ik_tolerance_m).all()


def test_positive_roll_predicts_left_rise_and_opposite_leg_compensation():
    prior = m.ShipModelPrior(kin(), 2, .02, .127)
    data = sensors(); data[0][:, 0] = .5
    output = prior.update(*data)
    expected = torch.tensor([0., -math.sin(.05), math.cos(.05)], dtype=torch.float64)
    torch.testing.assert_close(output[0, 3:6], expected)
    displacement = output[:, 6:12].reshape(2, 2, 3)
    assert (displacement[:, 0, 2] > 0).all() and (displacement[:, 1, 2] < 0).all()
    assert (output[:, 12] < 0).all() and (output[:, 13] > 0).all()
    assert (output[:, 22:24] == 1).all()


def test_relative_yaw_calibration_changes_rotation_axis_not_magnitude():
    prior = m.ShipModelPrior(kin(), 2, .02, .127, {'initial_relative_yaw_rad': math.pi/2})
    data = sensors(); data[0][:, 0] = .5
    output = prior.update(*data)
    assert output[0, 3] > 0 and abs(float(output[0, 4])) < 1e-12
    torch.testing.assert_close(output[0, 5], torch.tensor(math.cos(.05), dtype=torch.float64))


def test_common_body_and_deck_rotation_preserves_relative_attitude():
    prior = m.ShipModelPrior(kin(), 2, .02, .127, {'gravity_correction_gain': 0.})
    data = sensors(); data[0][:, 2] = .2; data[1][:, 2] = .2
    for _ in range(30):
        prior.update(*data)
    torch.testing.assert_close(prior.relative_rotation, torch.eye(3, dtype=torch.float64).repeat(2, 1, 1), atol=1e-12, rtol=0)


def test_gravity_alignment_uses_body_tilt_without_true_quaternions():
    prior = m.ShipModelPrior(kin(), 2, .02, .127)
    data = sensors(); data[2][:, 1] = -math.sin(.1); data[2][:, 2] = -math.cos(.1)
    output = prior.update(*data)
    torch.testing.assert_close(output[:, :3], -data[2], atol=1e-12, rtol=0)
    assert output[:, 14:20].abs().max() > .01


def test_airborne_leg_reference_neutral_and_contact_hysteresis():
    prior = m.ShipModelPrior(kin(), 2, .02, .127)
    data = sensors(); data[0][:, 0] = .5; data[4][:, 1] = 0
    output = prior.update(*data)
    assert not output[:, 17:20].any() and not output[:, 23].any()
    data[4][:, 0] = 3; prior.update(*data)
    assert prior.contacts[:, 0].all()
    data[4][:, 0] = 1; prior.update(*data)
    assert not prior.contacts.any()
    assert not prior.features[:, 6:24].any()
    data[4][:, 0] = 3; prior.update(*data)
    assert not prior.contacts.any()


def test_no_imu_removes_all_derived_channels_and_observer_state():
    prior = m.ShipModelPrior(kin(), 2, .02, .127)
    data = sensors(); data[0][:, 0] = 1
    prior.update(*data)
    output = prior.update(*data, imu_enabled=False)
    assert not output.any() and not prior.initialized.any() and not prior.contacts.any()


def test_reset_is_per_environment_and_clears_references():
    prior = m.ShipModelPrior(kin(), 2, .02, .127)
    data = sensors(); data[0][:, 0] = 1; prior.update(*data)
    preserved = prior.features[1].clone()
    prior.reset(torch.tensor([0]))
    assert not prior.features[0].any() and not prior.initialized[0]
    torch.testing.assert_close(prior.features[1], preserved)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_bad_sensor_invalidates_only_its_own_environment(bad):
    prior = m.ShipModelPrior(kin(), 2, .02, .127)
    data = sensors(); data[0][0, 0] = bad
    output = prior.update(*data)
    assert torch.isfinite(output).all() and not output[0].any()
    assert output[1, 22:24].all() and not prior.initialized[0]


def test_unreachable_ik_is_marked_invalid_and_neutral():
    prior = m.ShipModelPrior(kin(), 2, .02, .127, {'max_joint_delta_rad': 1e-7, 'ik_tolerance_m': 1e-9})
    data = sensors(); data[0][:, 0] = 2
    output = prior.update(*data)
    assert not output[:, 22:24].any() and not output[:, 12:20].any()


def test_joint_angle_and_velocity_bounds_are_respected():
    k = kin(); prior = m.ShipModelPrior(k, 2, .02, .127, {'prediction_horizon_s': .01})
    data = sensors(); data[0][:, 0] = 5
    output = prior.update(*data)
    delta = output[:, 14:20].reshape(2, 2, 3)
    assert (delta.abs() <= k.velocity_limits*.01+1e-9).all()
    assert (delta.abs() <= .25+1e-9).all()
    assert ((data[3]+delta) >= k.lower).all() and ((data[3]+delta) <= k.upper).all()


@pytest.mark.parametrize('settings', [
    {'prediction_horizon_s': 0.}, {'ik_iterations': 0}, {'ik_iterations': True},
    {'gravity_correction_gain': 1.1}, {'ik_damping': float('nan')},
    {'contact_off_threshold_n': 5.}, {'initial_relative_yaw_rad': float('inf')},
])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError): m.ShipModelPrior(kin(), 2, .02, .127, settings)


def states():
    source_model = {'actor.0.weight': torch.randn(16, 47), 'critic.0.weight': torch.randn(16, 54), 'logstd': torch.randn(8)}
    source_encoder = {'encoder.gru.weight_ih_l0': torch.randn(24, 66), 'encoder.gru.bias_ih_l0': torch.randn(24)}
    target_model = {**source_model, 'actor.0.weight': torch.zeros(16, 71), 'critic.0.weight': torch.zeros(16, 78)}
    target_encoder = {**source_encoder, 'encoder.gru.weight_ih_l0': torch.zeros(24, 114)}
    return {'model_state_dict': source_model, 'encoder_state_dict': source_encoder}, target_model, target_encoder


def test_zero_column_migration_preserves_all_three_input_paths():
    source, model_template, encoder_template = states()
    model, encoder = warm.ship_prior_state_dicts(source, model_template, encoder_template)
    for key in ['actor.0.weight', 'critic.0.weight']:
        old = source['model_state_dict'][key]; x = torch.randn(5, old.shape[1])
        expanded = torch.randn(5, old.shape[1]+24)
        expanded[:, :36] = x[:, :36]; expanded[:, 60:] = x[:, 36:]
        torch.testing.assert_close(x@old.T, expanded@model[key].T)
        assert not model[key][:, 36:60].any()
    x = torch.randn(5, 66); expanded = torch.randn(5, 114)
    expanded[:, :29] = x[:, :29]; expanded[:, 53:90] = x[:, 29:]
    key = 'encoder.gru.weight_ih_l0'
    torch.testing.assert_close(x@source['encoder_state_dict'][key].T, expanded@encoder[key].T)
    assert not encoder[key][:, 29:53].any() and not encoder[key][:, 90:114].any()


def test_prior_continuation_is_exact_without_aliasing():
    source, model, encoder = states()
    source['model_state_dict'] = model; source['encoder_state_dict'] = encoder
    copied = warm.ship_prior_state_dicts(source, model, encoder)
    for origin, result in zip([model, encoder], copied):
        for key in origin:
            torch.testing.assert_close(origin[key], result[key]); assert origin[key].data_ptr() != result[key].data_ptr()


@pytest.mark.parametrize('problem', ['wrong_actor', 'wrong_gru', 'nan', 'keys'])
def test_migration_rejects_incompatible_or_nonfinite_weights(problem):
    source, model, encoder = states()
    if problem == 'wrong_actor': source['model_state_dict']['actor.0.weight'] = torch.zeros(16, 50)
    if problem == 'wrong_gru': source['encoder_state_dict']['encoder.gru.weight_ih_l0'] = torch.zeros(24, 72)
    if problem == 'nan': source['model_state_dict']['logstd'][0] = float('nan')
    if problem == 'keys': source['encoder_state_dict']['extra'] = torch.ones(1)
    with pytest.raises(ValueError): warm.ship_prior_state_dicts(source, model, encoder)


def test_actor_features_do_not_read_privileged_world_or_future_state():
    tree = ast.parse((ROOT/'legged_gym/envs/wheelfoot_ship/wheelfoot_ship_prior.py').read_text())
    names = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    for forbidden in ['root_states', 'base_quat', 'foot_positions', 'ship_phase', 'ship_frequency', 'ship_time', '_deck_command']:
        assert forbidden not in names


def test_prior_reset_precedes_parent_history_construction():
    tree = ast.parse((ROOT/'legged_gym/envs/wheelfoot_ship/wheelfoot_ship_prior.py').read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'reset_idx')
    events = []
    class Base:
        def reset_idx(self, ids): events.append('parent_history')
    ns = {'Base': Base, 'torch': torch}
    cls = ast.ClassDef(name='PriorTest', bases=[ast.Name(id='Base', ctx=ast.Load())], keywords=[], body=[fn], decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), '<actual reset>', 'exec'), ns)
    env = ns['PriorTest'](); env.ship_model_prior = S(reset=lambda ids: events.append('clear_prior'))
    env.reset_idx(torch.tensor([0]))
    assert events == ['clear_prior', 'parent_history']


def test_prior_cli_rejects_ambiguous_modes_before_output(monkeypatch, tmp_path):
    tree = ast.parse((ROOT/'legged_gym/scripts/train_ship_motion.py').read_text())
    body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main').body
    prefix = []
    for n in body:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'mkdir': break
        prefix.append(n)
    for extra in [['--model-prior', '--locomotion-mode', 'step'], ['--model-prior', '--warm-start', 'flat.pt'],
                  ['--prior-horizon-s', '.1'], ['--model-prior', '--prior-horizon-s', 'nan']]:
        folder = tmp_path/'not_created'
        monkeypatch.setattr(sys, 'argv', ['train', '--output', str(folder), '--label', 'test', *extra])
        with pytest.raises(SystemExit):
            exec(compile(ast.Module(body=prefix, type_ignores=[]), '<actual CLI>', 'exec'),
                 dict(argparse=argparse, Path=Path, math=math, __doc__='test'))
        assert not folder.exists()
