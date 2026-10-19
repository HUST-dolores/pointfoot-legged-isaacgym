"""Batched prescribed deck motion and IMU geometry. Quaternion order: xyzw.

The pivot is the center of the upper deck surface, not the box COM. Gravity
stays fixed in the inertial world. No simulator or project imports are needed.
"""
import math
import torch


def rotate(q, v):
    uv = torch.cross(q[..., :3], v, dim=-1)
    return v + 2 * (q[..., 3:] * uv + torch.cross(q[..., :3], uv, dim=-1))


def inverse_rotate(q, v):
    return rotate(torch.cat((-q[..., :3], q[..., 3:]), dim=-1), v)


def heading_yaw_rate(quaternion, world_angular_velocity):
    """Derivative of atan2(world_forward_y, world_forward_x), in rad/s.

    Unlike body gyro z this is the actual heading rate when the body tilts.
    Near a vertical forward axis yaw is undefined; denominator regularization
    keeps the reward finite while the existing inversion termination applies.
    Physical state is used for reward/diagnostics only, never added to actor obs.
    """
    axis = torch.zeros_like(world_angular_velocity)
    axis[..., 0] = 1.
    forward = rotate(quaternion, axis)
    derivative = torch.cross(world_angular_velocity, forward, dim=-1)
    return ((forward[..., 0]*derivative[..., 1]-forward[..., 1]*derivative[..., 0])
            / forward[..., :2].square().sum(-1).clamp(min=1e-6))


def motion_state(time, amplitude, frequency, phase, pivot, thickness, ramp_s):
    """Return box root state, COM acceleration, and world angular acceleration.

    amplitude columns: roll [rad], pitch [rad], heave [m]. R = Ry(pitch) Rx(roll).
    Quintic startup has zero initial velocity and acceleration even with random
    phases. Velocities include the rotating COM offset below the surface pivot.
    """
    if ramp_s <= 0 or thickness <= 0:
        raise ValueError("ramp_s and thickness must be positive")
    t = time.unsqueeze(-1)
    u = (t / ramp_s).clamp(0, 1)
    s = 10*u**3 - 15*u**4 + 6*u**5
    ds = (30*u**2 - 60*u**3 + 30*u**4) / ramp_s
    dds = (60*u - 180*u**2 + 120*u**3) / ramp_s**2
    w = 2 * math.pi * frequency
    angle = w * t + phase
    sn, cs = torch.sin(angle), torch.cos(angle)
    value = amplitude * s * sn
    rate = amplitude * (ds * sn + s * w * cs)
    accel = amplitude * (dds * sn + 2*ds*w*cs - s*w*w*sn)
    r, p, h = value.unbind(-1)
    rd, pd, hd = rate.unbind(-1)
    rdd, pdd, hdd = accel.unbind(-1)
    sr, cr = torch.sin(r/2), torch.cos(r/2)
    sp, cp = torch.sin(p/2), torch.cos(p/2)
    quat = torch.stack((sr*cp, cr*sp, -sr*sp, cr*cp), dim=-1)
    omega = torch.stack((rd*torch.cos(p), pd, -rd*torch.sin(p)), dim=-1)
    alpha = torch.stack((rdd*torch.cos(p)-rd*pd*torch.sin(p), pdd,
                         -rdd*torch.sin(p)-rd*pd*torch.cos(p)), dim=-1)
    offset = torch.zeros_like(pivot)
    offset[..., 2] = -thickness / 2
    arm = rotate(quat, offset)
    position = pivot + arm
    position[..., 2] += h
    velocity = torch.cross(omega, arm, dim=-1)
    velocity[..., 2] += hd
    linear_accel = (torch.cross(alpha, arm, dim=-1)
                    + torch.cross(omega, torch.cross(omega, arm, dim=-1), dim=-1))
    linear_accel[..., 2] += hdd
    return torch.cat((position, quat, velocity, omega), dim=-1), linear_accel, alpha


def surface_pose(root, thickness):
    offset = torch.zeros_like(root[..., :3])
    offset[..., 2] = thickness / 2
    normal = torch.zeros_like(offset)
    normal[..., 2] = 1
    return root[..., :3] + rotate(root[..., 3:7], offset), rotate(root[..., 3:7], normal)


def plane_height(points_xy, surface, normal):
    """World z of the infinite support plane; bounds are checked separately."""
    delta = points_xy - surface[:, None, :2]
    return surface[:, None, 2] - (delta * normal[:, None, :2]).sum(-1) / normal[:, None, 2]


def point_velocity(root, point):
    return root[..., 7:10] + torch.cross(root[..., 10:13], point-root[..., :3], dim=-1)


def gravity_projected_support_state(deck, robot, thickness):
    """Alternative reference for world-upright standing on an upward deck.

    Project the robot root vertically (fixed world gravity) onto the plane.
    Return that point's deck coordinates, its tangential relative velocity,
    and the root's world velocity relative to the physical support point.
    Vertical body-height adjustments must not become tangential locomotion.
    This is distinct from holding the elevated body point fixed in deck axes.
    """
    surface, normal = surface_pose(deck, thickness)
    z = plane_height(robot[:, None, :2], surface, normal)[:, 0:1]
    support = torch.cat((robot[:, :2], z), dim=-1)
    relative_world = robot[:, 7:10] - point_velocity(deck, support)
    horizontal = relative_world[:, :2]
    tangent_z = -(horizontal*normal[:, :2]).sum(-1,keepdim=True)/normal[:, 2:3]
    tangent_world = torch.cat((horizontal,tangent_z),dim=-1)
    return (inverse_rotate(deck[:, 3:7],support-surface),
            inverse_rotate(deck[:, 3:7],tangent_world),relative_world)


def imu_sample(root, linear_accel, angular_accel, sensor_offset):
    """Deck-frame gyro, specific force, and gravity-direction attitude estimate.

    Offset is measured from box COM in deck coordinates. Accelerometer reads
    R^T(a_sensor - g), so a stationary level sensor reads +9.81 on its z axis.
    The last 3 channels model an attitude estimator, not a raw accelerometer.
    """
    q = root[..., 3:7]
    arm = rotate(q, sensor_offset.expand_as(linear_accel))
    omega = root[..., 10:13]
    a_sensor = linear_accel + torch.cross(angular_accel, arm, dim=-1)
    a_sensor += torch.cross(omega, torch.cross(omega, arm, dim=-1), dim=-1)
    gravity = torch.zeros_like(linear_accel)
    gravity[..., 2] = -9.81
    return torch.cat((inverse_rotate(q, omega), inverse_rotate(q, a_sensor-gravity),
                      inverse_rotate(q, gravity/9.81)), dim=-1)


def shuttle_velocity(time, max_speed, period_s, ramp_s):
    """Position-independent forward/back command for a finite deck.

    This uses elapsed episode time, never robot/deck pose or a future sea state.
    After startup each complete period has zero commanded displacement.
    """
    if not all(math.isfinite(v) and v > 0 for v in (max_speed, period_s, ramp_s)):
        raise ValueError('shuttle speed, period and ramp must be finite and positive')
    u = (time/ramp_s).clamp(0, 1)
    ramp = 10*u**3-15*u**4+6*u**5
    return max_speed*ramp*torch.sin(2*math.pi*time/period_s)


def mobility_clock(time, step_mode, frequency_hz, stance_duty):
    """Ordinary requested-mode/clock features and alternating contact targets.

    The clock is a controller gait clock, NOT the ship-wave phase. Rolling
    features are all zero and both wheels are requested to remain supported.
    """
    if not math.isfinite(frequency_hz) or frequency_hz <= 0 or not .5 <= stance_duty < 1:
        raise ValueError('positive gait frequency and stance duty in [0.5,1) required')
    mode = torch.as_tensor(step_mode, device=time.device, dtype=time.dtype).expand_as(time)
    phase = torch.remainder(time*frequency_hz, 1.)
    features = torch.stack((mode, mode*torch.sin(2*math.pi*phase),
                            mode*torch.cos(2*math.pi*phase)), dim=-1)
    phases = torch.stack((phase, torch.remainder(phase+.5, 1.)), dim=-1)
    contacts = torch.where(mode[:, None].bool(), (phases < stance_duty).to(time.dtype),
                           torch.ones_like(phases))
    return features, contacts


def mobility_observations(flat_obs, deck_imu, gait_features):
    """40 ordinary features: state20, IMU9, requested gait3, previous actions8."""
    if flat_obs.shape[-1] != 28 or deck_imu.shape != (*flat_obs.shape[:-1], 9) or gait_features.shape != (*flat_obs.shape[:-1], 3):
        raise ValueError('mobility observation contract requires flat28 + IMU9 + gait3')
    return torch.cat((flat_obs[..., :20], deck_imu, gait_features, flat_obs[..., 20:]), dim=-1)


def foot_deck_state(deck, feet, thickness, radius):
    """Normal wheel clearance and tangential CONTACT-point slip, not center speed.

    feet contains physical world position/quaternion/linear/angular velocity.
    Rolling without slip is allowed: spinning wheels cancel their translating
    centers at the contact point. All quantities are relative to the deck.
    """
    top, normal = surface_pose(deck, thickness)
    arm = -radius*normal[:, None, :].expand_as(feet[..., :3])
    contact_point = feet[..., :3]+arm
    support = point_velocity(deck[:, None, :].expand(-1, feet.shape[1], -1), contact_point)
    contact_velocity = feet[..., 7:10]+torch.cross(feet[..., 10:13], arm, dim=-1)
    relative = contact_velocity-support
    tangential = relative-(relative*normal[:, None, :]).sum(-1, keepdim=True)*normal[:, None, :]
    clearance = ((feet[..., :3]-top[:, None, :])*normal[:, None, :]).sum(-1)-radius
    return clearance.clamp(min=0), tangential
