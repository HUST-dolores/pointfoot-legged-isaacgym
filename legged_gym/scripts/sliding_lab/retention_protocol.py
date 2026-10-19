"""Task constants and tensor-only geometry; no Isaac Gym dependency."""

from dataclasses import asdict, dataclass
import math


CONTEXT_DIM = 15


@dataclass
class RetentionSettings:
    mass: float = 5.93
    friction: float = 0.05
    plate_z: float = 0.06
    plate_half: float = 0.25
    plate_top: float = 0.0025
    box_half: float = 0.05
    duration: float = 10.0
    spawn_time: float = 0.5
    pulse_time: float = 2.0
    pulse_duration: float = 0.04
    drop_confirm: float = 0.16
    fall_confirm: float = 0.10
    start_dv: float = 0.25
    max_dv: float = 0.75
    curriculum_iterations: int = 1000
    initial_offset: float = 0.05
    alive_weight: float = 2.0
    center_weight: float = 0.25
    edge_weight: float = 4.0
    outward_weight: float = 2.0
    drop_cost: float = 12.0
    fall_cost: float = 12.0

    def dictionary(self):
        return asdict(self)


def quat_rotate(q, v):
    import torch
    uv = torch.cross(q[..., :3], v, dim=-1)
    return v + 2.0 * (q[..., 3:4] * uv + torch.cross(q[..., :3], uv, dim=-1))


def quat_inverse_rotate(q, v):
    import torch
    return quat_rotate(torch.cat((-q[..., :3], q[..., 3:4]), -1), v)


def quat_multiply(a, b):
    import torch
    xyz = a[..., 3:4] * b[..., :3] + b[..., 3:4] * a[..., :3] + torch.cross(a[..., :3], b[..., :3], dim=-1)
    w = a[..., 3:4] * b[..., 3:4] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True)
    return torch.cat((xyz, w), -1)


def payload_geometry(root, box, force, settings):
    import torch
    q = root[:, 3:7]
    delta = box[:, :3] - root[:, :3]
    local = quat_inverse_rotate(q, delta)
    local = local - local.new_tensor([0.0, 0.0, settings.plate_z])
    velocity = quat_inverse_rotate(q, box[:, 7:10] - root[:, 7:10] - torch.cross(root[:, 10:13], delta, dim=-1))
    inverse = torch.cat((-q[:, :3], q[:, 3:4]), -1)
    relative_q = quat_multiply(inverse, box[:, 3:7])
    corners = local.new_tensor([[x, y, z] for x in (-settings.box_half, settings.box_half)
                               for y in (-settings.box_half, settings.box_half)
                               for z in (-settings.box_half, settings.box_half)])
    corners = quat_rotate(relative_q[:, None, :].expand(-1, 8, -1), corners[None].expand(len(root), -1, -1)) + local[:, None, :]
    lo, hi = corners.amin(1), corners.amax(1)
    margins = torch.stack((lo[:, 0] + settings.plate_half, settings.plate_half - hi[:, 0],
                           lo[:, 1] + settings.plate_half, settings.plate_half - hi[:, 1]), -1)
    overlap = ((hi[:, :2] > -settings.plate_half) & (lo[:, :2] < settings.plate_half)).all(-1)
    gap = lo[:, 2] - settings.plate_top
    normal_force = quat_inverse_rotate(q, force)[:, 2]
    up = quat_rotate(relative_q, local.new_tensor([0.0, 0.0, 1.0]).expand(len(root), -1))
    supported = overlap & (gap.abs() < 0.015) & (normal_force > 0.1 * settings.mass * 9.81) & (up[:, 2] > 0.5)
    # Partial overhang and short airborne gaps are not terminal events.
    departure = ((~overlap) & (gap < -0.015)) | (hi[:, 2] < settings.plate_top - 0.04)
    context = torch.cat((local * local.new_tensor([4., 4., 10.]) - local.new_tensor([0., 0., 0.525]),
                         velocity, up, margins * 5., supported[:, None].float(),
                         torch.ones((len(root), 1), device=root.device)), -1).clamp(-5., 5.)
    assert context.shape[-1] == CONTEXT_DIM
    return {"local": local, "relative_velocity": velocity, "margins": margins, "gap": gap,
            "normal_force": normal_force, "supported": supported, "departure": departure,
            "overlap": overlap, "up": up, "context": context}


def retained_reward(geometry, active, gravity, velocity, ang_velocity, height, actions, previous_actions,
                    torques, drop, fall, settings, dt):
    import torch
    xy = geometry["local"][:, :2]
    m = geometry["margins"]
    outward = torch.stack((-geometry["relative_velocity"][:, 0], geometry["relative_velocity"][:, 0],
                            -geometry["relative_velocity"][:, 1], geometry["relative_velocity"][:, 1]), -1).clamp_min(0.)
    edge = ((0.06 - m) / 0.12).clamp(0., 2.)
    risk_gate = ((0.10 - m) / 0.10).clamp(0., 2.)
    enabled = active.float()
    terms = {
        "retained": settings.alive_weight * geometry["supported"].float() * enabled,
        "center": -settings.center_weight * (xy / 0.2).square().sum(-1).clamp_max(4.) * enabled,
        "edge": -settings.edge_weight * edge.square().mean(-1) * enabled,
        "outward": -settings.outward_weight * ((outward / 0.4).square() * risk_gate).mean(-1).clamp_max(5.) * enabled,
        "upright": 0.5 * torch.exp(-3. * gravity[:, :2].square().sum(-1)),
        "height": -0.5 * ((height - 0.7664) / 0.2).square().clamp_max(4.),
        "velocity": -0.12 * velocity[:, :2].square().sum(-1).clamp_max(9.),
        "angular": -0.03 * ang_velocity[:, :2].square().sum(-1).clamp_max(25.),
        "smooth": -0.01 * (actions - previous_actions).square().mean(-1),
        "effort": -0.00002 * torques.square().sum(-1),
    }
    terms = {key: value * dt for key, value in terms.items()}
    terms["drop"] = -settings.drop_cost * drop.float()
    terms["fall"] = -settings.fall_cost * fall.float()
    return sum(terms.values()), terms


def eval_cases(repeats=8):
    cases = []
    for magnitude in (0.25, 0.45, 0.65, 0.85):
        for direction, angle in (("forward", 0.), ("left", math.pi / 2), ("backward", math.pi), ("right", -math.pi / 2)):
            for repeat in range(repeats):
                cases.append({"nominal_dv": magnitude, "angle": angle, "direction": direction, "repeat": repeat})
    for repeat in range(repeats):
        cases.append({"nominal_dv": 0., "angle": 0., "direction": "none", "repeat": repeat})
    return cases
