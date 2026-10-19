"""Analytical motion/IMU tests; no simulator or GPU required."""
import importlib.util
from pathlib import Path
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "ship_motion_math", Path(__file__).resolve().parents[1]/"legged_gym/utils/ship_motion.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def sample(t, heave=0.06):
    return m.motion_state(torch.tensor([t], dtype=torch.float64),
                          torch.tensor([[0.23, 0.19, heave]], dtype=torch.float64),
                          torch.tensor([[0.31, 0.27, 0.41]], dtype=torch.float64),
                          torch.tensor([[0.4, 1.1, 2.0]], dtype=torch.float64),
                          torch.tensor([[2., -3., 1.5]], dtype=torch.float64), 0.2, 2.0)


def test_quintic_startup_has_zero_velocity_and_acceleration():
    root, accel, alpha = sample(0)
    torch.testing.assert_close(root[:, 3:7], torch.tensor([[0.,0.,0.,1.]], dtype=torch.float64))
    assert torch.count_nonzero(root[:, 7:]) == 0
    assert torch.count_nonzero(accel) == 0
    assert torch.count_nonzero(alpha) == 0


@pytest.mark.parametrize("time", [0.7, 2.7])
def test_root_velocities_and_acceleration_match_finite_differences(time):
    eps = 1e-5
    root, accel, alpha = sample(time)
    before = sample(time-eps)[0]
    after = sample(time+eps)[0]
    torch.testing.assert_close((after[:, :3]-before[:, :3])/(2*eps), root[:, 7:10], atol=1e-8, rtol=1e-6)
    torch.testing.assert_close((after[:, 7:10]-before[:, 7:10])/(2*eps), accel, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close((after[:, 10:]-before[:, 10:])/(2*eps), alpha, atol=1e-8, rtol=1e-6)
    axis = torch.tensor([[1.,0.,0.]], dtype=torch.float64)
    dr = (m.rotate(after[:, 3:7], axis)-m.rotate(before[:, 3:7], axis))/(2*eps)
    torch.testing.assert_close(dr, torch.cross(root[:, 10:], m.rotate(root[:, 3:7], axis), dim=-1), atol=1e-8, rtol=1e-6)


def test_surface_pivot_and_plane_geometry_are_consistent():
    root, _, _ = sample(2.7, heave=0)
    top, normal = m.surface_pose(root, 0.2)
    torch.testing.assert_close(top, torch.tensor([[2.,-3.,1.5]], dtype=torch.float64))
    torch.testing.assert_close(m.point_velocity(root, top), torch.zeros_like(top), atol=1e-10, rtol=0)
    local = torch.tensor([[0.7,-0.3,0.]], dtype=torch.float64)
    point = top + m.rotate(root[:, 3:7], local)
    torch.testing.assert_close(m.plane_height(point[:, None, :2], top, normal)[:, 0], point[:, 2])


def test_surface_imu_specific_force_includes_rotating_com_offset():
    root, accel, alpha = sample(2.7, heave=0)
    offset = torch.tensor([0.,0.,0.1], dtype=torch.float64)
    imu = m.imu_sample(root, accel, alpha, offset)
    # The top-center sensor is the rotation pivot: world acceleration is zero.
    gravity = torch.tensor([[0.,0.,9.81]], dtype=torch.float64)
    torch.testing.assert_close(imu[:, 3:6], m.inverse_rotate(root[:, 3:7], gravity))
    torch.testing.assert_close(imu[:, 6:], -imu[:, 3:6]/9.81)


def test_heave_is_observable_as_specific_force():
    root, accel, alpha = sample(2.7)
    imu = m.imu_sample(root, accel, alpha, torch.tensor([0.,0.,0.1], dtype=torch.float64))
    world_force = m.rotate(root[:, 3:7], imu[:, 3:6])
    assert abs(float(world_force[0, 2])-9.81) > 0.01
    torch.testing.assert_close(world_force[:, :2], torch.zeros(1,2,dtype=torch.float64), atol=1e-9, rtol=0)


def test_invalid_ramp_is_rejected():
    z = torch.zeros(1,3)
    with pytest.raises(ValueError):
        m.motion_state(torch.zeros(1), z, z, z, z, 0.2, 0)


@pytest.mark.parametrize('roll_deg',[0.,20.])
def test_gravity_projection_keeps_world_upright_standing_reference_zero(roll_deg):
    import math
    r=math.radians(roll_deg);h=.7664
    rate=math.radians(20)*2*math.pi*.35 if roll_deg==0 else 0.
    deck=torch.zeros(1,13,dtype=torch.float64)
    deck[:,3:7]=torch.tensor([[math.sin(r/2),0.,0.,math.cos(r/2)]],dtype=torch.float64)
    pivot=torch.tensor([[0.,0.,1.5]],dtype=torch.float64)
    deck[:,:3]=pivot+m.rotate(deck[:,3:7],torch.tensor([[0.,0.,-.1]],dtype=torch.float64))
    deck[:,10]=rate
    deck[:,7:10]=torch.cross(deck[:,10:13],deck[:,:3]-pivot,dim=-1)
    robot=torch.zeros_like(deck);robot[:,6]=1.;robot[:,2]=1.5+h/math.cos(r)
    robot[:,9]=h/math.cos(r)*math.tan(r)*rate
    position,velocity,relative=m.gravity_projected_support_state(deck,robot,.2)
    torch.testing.assert_close(position,torch.zeros_like(position),atol=1e-12,rtol=0)
    torch.testing.assert_close(velocity,torch.zeros_like(velocity),atol=1e-12,rtol=0)
    torch.testing.assert_close(relative[:,:2],torch.zeros(1,2,dtype=torch.float64),atol=1e-12,rtol=0)
    surface,normal=m.surface_pose(deck,.2)
    torch.testing.assert_close(((robot[:,:3]-surface)*normal).sum(-1),torch.tensor([h],dtype=torch.float64))
    if roll_deg==0:
        legacy=robot[:,7:10]-m.point_velocity(deck,robot[:,:3])
        assert float(legacy[:,:2].norm())>.58
    else:
        legacy_position=m.inverse_rotate(deck[:,3:7],robot[:,:3]-surface)
        assert float(legacy_position[0,1])>.27


def test_gravity_projection_matches_existing_xy_reference_on_flat_translation():
    deck=torch.zeros(1,13,dtype=torch.float64);deck[:,6]=1.
    deck[:,:3]=torch.tensor([[2.,-1.,1.4]],dtype=torch.float64)
    deck[:,7:10]=torch.tensor([[.2,-.1,.03]],dtype=torch.float64)
    robot=torch.zeros_like(deck);robot[:,6]=1.
    robot[:,:3]=torch.tensor([[2.4,-1.3,2.5]],dtype=torch.float64)
    robot[:,7:10]=torch.tensor([[.5,.7,-.1]],dtype=torch.float64)
    position,velocity,relative=m.gravity_projected_support_state(deck,robot,.2)
    surface,_=m.surface_pose(deck,.2)
    legacy_position=m.inverse_rotate(deck[:,3:7],robot[:,:3]-surface)
    legacy=robot[:,7:10]-m.point_velocity(deck,robot[:,:3])
    torch.testing.assert_close(position[:,:2],legacy_position[:,:2])
    torch.testing.assert_close(velocity[:,:2],legacy[:,:2])
    torch.testing.assert_close(relative,legacy)


def test_gravity_projection_separates_tangent_motion_from_vertical_height_adjustment():
    deck,_,_=sample(2.7)
    surface,_=m.surface_pose(deck,.2)
    local=torch.tensor([[.7,-.3,0.]],dtype=torch.float64)
    support=surface+m.rotate(deck[:,3:7],local)
    command=torch.tensor([[.3,-.2,0.]],dtype=torch.float64)
    tangent=m.rotate(deck[:,3:7],command)
    vertical=torch.tensor([[0.,0.,.31]],dtype=torch.float64)
    robot=torch.zeros_like(deck);robot[:,6]=1.
    robot[:,:3]=support+torch.tensor([[0.,0.,.9]],dtype=torch.float64)
    robot[:,7:10]=m.point_velocity(deck,support)+tangent+vertical
    position,velocity,relative=m.gravity_projected_support_state(deck,robot,.2)
    torch.testing.assert_close(position,local,atol=1e-12,rtol=0)
    torch.testing.assert_close(velocity,command,atol=1e-12,rtol=0)
    torch.testing.assert_close(relative,tangent+vertical,atol=1e-12,rtol=0)


def test_gravity_projection_batched_yaw_decks_preserve_surface_velocity():
    import math
    deck=torch.zeros(2,13,dtype=torch.float64);deck[:,2]=1.4;deck[:,6]=1.
    deck[1,3:7]=torch.tensor([0.,0.,math.sin(.35),math.cos(.35)],dtype=torch.float64)
    deck[:,7:10]=torch.tensor([.1,-.2,.03],dtype=torch.float64);deck[:,12]=.4
    surface,_=m.surface_pose(deck,.2)
    local=torch.tensor([[.2,-.3,0.],[-.4,.1,0.]],dtype=torch.float64)
    support=surface+m.rotate(deck[:,3:7],local)
    command=torch.tensor([[.2,.1,0.],[-.3,.2,0.]],dtype=torch.float64)
    robot=torch.zeros_like(deck);robot[:,6]=1.
    robot[:,:3]=support+torch.tensor([0.,0.,.9],dtype=torch.float64)
    robot[:,7:10]=m.point_velocity(deck,support)+m.rotate(deck[:,3:7],command)+torch.tensor([0.,0.,.5],dtype=torch.float64)
    position,velocity,_=m.gravity_projected_support_state(deck,robot,.2)
    torch.testing.assert_close(position,local,atol=1e-12,rtol=0)
    torch.testing.assert_close(velocity,command,atol=1e-12,rtol=0)
