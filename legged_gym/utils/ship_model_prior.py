"""Sensor-only, local geometric ship support references (not a controller).

No simulator imports, world root poses, future wave phase, or true deck motion.
Relative yaw needs an initial calibration; gravity cannot observe its drift.
The plane pivots at the current support centroid: common heave/translation is
not observable from the inputs here and is deliberately not invented.
"""
import math
import xml.etree.ElementTree as ET

import torch

FEATURE_DIM = 24
FEATURE_SLICES = {
    'deck_normal_body': slice(0, 3),
    'predicted_deck_normal_body': slice(3, 6),
    'contact_compensation': slice(6, 12),
    'leg_length_delta': slice(12, 14),
    'joint_angle_delta': slice(14, 20),
    'contacts': slice(20, 22),
    'ik_valid': slice(22, 24),
}


def skew(v):
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)


def so3_exp(v):
    """Rodrigues exponential, stable at zero and differentiable."""
    theta = v.norm(dim=-1, keepdim=True)
    k = skew(v)
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(k)
    a = torch.sinc(theta/math.pi)[..., None]
    b = (.5*torch.sinc(theta/(2*math.pi)).square())[..., None]
    return eye+a*k+b*(k@k)


def unit(v):
    return v/v.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def alignment_vector(a, b):
    """Axis-angle taking unit a to unit b, including the antiparallel case."""
    a, b = unit(a), unit(b)
    cross = torch.cross(a, b, dim=-1)
    sine = cross.norm(dim=-1, keepdim=True)
    cosine = (a*b).sum(-1, keepdim=True).clamp(-1, 1)
    angle = torch.atan2(sine, cosine)
    result = cross/sine.clamp(min=1e-8)*angle
    basis = torch.nn.functional.one_hot(a.abs().argmin(-1), 3).to(a.dtype)
    opposite = unit(torch.cross(a, basis, dim=-1))*math.pi
    return torch.where((sine < 1e-7) & (cosine < 0), opposite, result)


def apply_rotation(rotation, vector):
    return (rotation@vector.unsqueeze(-1)).squeeze(-1)


def _rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
            [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]


class WheelLegKinematics:
    """Exact nominal URDF wheel-center FK/Jacobian; excludes wheel spin DOF.

    Supports the two configured 3-revolute-joint chains with fixed joint-origin
    rotations and a wheel joint as endpoint. Per-env morphology is not assumed.
    """
    def __init__(self, urdf_path, joint_names, wheel_joint_names, device='cpu', dtype=torch.float32):
        if len(joint_names) != 2 or any(len(leg) != 3 for leg in joint_names) or len(wheel_joint_names) != 2:
            raise ValueError('prior requires two three-joint wheel legs')
        joints = {node.attrib['name']: node for node in ET.parse(urdf_path).getroot().findall('joint')}
        offsets, rotations, axes, lows, highs, velocities, tips = [], [], [], [], [], [], []
        bases = []
        for names, wheel_name in zip(joint_names, wheel_joint_names):
            chain = [joints[name] for name in names]
            wheel = joints[wheel_name]
            bases.append(chain[0].find('parent').attrib['link'])
            for left, right in zip(chain, chain[1:]+[wheel]):
                if left.find('child').attrib['link'] != right.find('parent').attrib['link']:
                    raise ValueError('URDF joints must be a consecutive wheel-leg chain')
            xyzs, rpys, directions, lower, upper, speed = [], [], [], [], [], []
            for node in chain:
                if node.attrib['type'] != 'revolute':
                    raise ValueError('leg prior only supports limited revolute joints')
                origin = node.find('origin')
                xyzs.append([float(v) for v in origin.attrib.get('xyz', '0 0 0').split()])
                rpys.append(_rpy_matrix([float(v) for v in origin.attrib.get('rpy', '0 0 0').split()]))
                directions.append([float(v) for v in node.find('axis').attrib['xyz'].split()])
                limit = node.find('limit').attrib
                lower.append(float(limit['lower'])); upper.append(float(limit['upper'])); speed.append(float(limit['velocity']))
            offsets.append(xyzs); rotations.append(rpys); axes.append(directions)
            lows.append(lower); highs.append(upper); velocities.append(speed)
            tips.append([float(v) for v in wheel.find('origin').attrib.get('xyz', '0 0 0').split()])
        if bases[0] != bases[1]:
            raise ValueError('wheel legs must share one body frame')
        self.joint_names = tuple(tuple(names) for names in joint_names)
        for name, value in [('offsets', offsets), ('rotations', rotations), ('axes', axes),
                            ('lower', lows), ('upper', highs), ('velocity_limits', velocities), ('tips', tips)]:
            tensor = torch.tensor(value, device=device, dtype=dtype)
            if not torch.isfinite(tensor).all():
                raise ValueError('nonfinite URDF geometry')
            setattr(self, name, tensor)
        if (self.axes.norm(dim=-1) < 1e-8).any() or (self.lower >= self.upper).any() or (self.velocity_limits <= 0).any():
            raise ValueError('invalid URDF axes or joint limits')
        self.axes = unit(self.axes)
        self.hip_origins = self.offsets[:, 0]

    def forward(self, q, jacobian=False):
        if q.ndim != 3 or q.shape[1:] != (2, 3):
            raise ValueError('joint positions must have shape [batch,2,3]')
        rotation = torch.eye(3, device=q.device, dtype=q.dtype).expand(q.shape[0], 2, 3, 3)
        position = torch.zeros_like(q)
        origins, axes = [], []
        for j in range(3):
            position = position+apply_rotation(rotation, self.offsets[:, j])
            rotation = rotation@self.rotations[:, j]
            origins.append(position)
            axes.append(apply_rotation(rotation, self.axes[:, j]))
            rotation = rotation@so3_exp(self.axes[:, j]*q[:, :, j:j+1])
        end = position+apply_rotation(rotation, self.tips)
        if not jacobian:
            return end
        columns = [torch.cross(axis, end-origin, dim=-1) for axis, origin in zip(axes, origins)]
        return end, torch.stack(columns, dim=-1)


class ShipModelPrior:
    """Stateful causal dual-IMU attitude observer + contact-constrained IK.

    update() is called ONCE per policy tick; observations only read its cache.
    All references are ordinary inputs, never actuator targets or rewards.
    """
    DEFAULTS = {
        'prediction_horizon_s': .10, 'initial_relative_yaw_rad': 0.,
        'gravity_correction_gain': .04, 'upright_gain': .5, 'max_upright_correction_rad': .10,
        'contact_on_threshold_n': 5., 'contact_off_threshold_n': 2.,
        'max_contact_displacement_m': .06, 'max_joint_delta_rad': .25,
        'ik_damping': .02, 'ik_iterations': 5, 'ik_tolerance_m': .005,
        'joint_limit_margin_rad': .01, 'minimum_normal_cosine': .35,
        'contact_displacement_scale': 10., 'leg_length_scale': 10., 'joint_angle_scale': 1.,
    }

    def __init__(self, kinematics, batch_size, dt, wheel_radius, settings=None):
        self.kinematics = kinematics
        for key, default in self.DEFAULTS.items():
            value = settings.get(key, default) if isinstance(settings, dict) else getattr(settings, key, default)
            setattr(self, key, value)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('nonfinite prior setting: '+key)
        positive = ['prediction_horizon_s', 'max_upright_correction_rad', 'contact_on_threshold_n',
                    'max_contact_displacement_m', 'max_joint_delta_rad', 'ik_damping', 'ik_tolerance_m',
                    'contact_displacement_scale', 'leg_length_scale', 'joint_angle_scale']
        if any(getattr(self, key) <= 0 for key in positive) or not 0 < dt or not math.isfinite(dt) or not 0 < wheel_radius or not math.isfinite(wheel_radius):
            raise ValueError('prior time, radius, limits and scales must be finite and positive')
        if (not 0 <= self.gravity_correction_gain <= 1 or not 0 <= self.upright_gain <= 1
            or not 0 < self.minimum_normal_cosine <= 1
            or not 0 <= self.contact_off_threshold_n < self.contact_on_threshold_n
            or not 0 <= self.joint_limit_margin_rad
            or type(self.ik_iterations) is not int or not 1 <= self.ik_iterations <= 20):
            raise ValueError('invalid observer/contact/IK settings')
        if ((kinematics.upper-kinematics.lower) <= 2*self.joint_limit_margin_rad).any():
            raise ValueError('joint-limit margin consumes the joint range')
        self.dt, self.wheel_radius = dt, wheel_radius
        self.features = kinematics.offsets.new_zeros(batch_size, FEATURE_DIM)
        self.relative_rotation = torch.eye(3, device=self.features.device, dtype=self.features.dtype).repeat(batch_size, 1, 1)
        yaw = self.features.new_tensor([0., 0., self.initial_relative_yaw_rad])
        self.calibration_rotation = so3_exp(yaw)
        self.contacts = torch.zeros(batch_size, 2, device=self.features.device, dtype=torch.bool)
        self.initialized = torch.zeros(batch_size, device=self.features.device, dtype=torch.bool)
        self.reset(torch.arange(batch_size, device=self.features.device))

    def reset(self, ids):
        self.relative_rotation[ids] = self.calibration_rotation
        self.contacts[ids] = False
        self.initialized[ids] = False
        self.features[ids] = 0

    def _attitude(self, deck_imu, body_gyro, body_gravity):
        # R_BD maps deck-frame vectors into the robot body frame. Its yaw
        # derives from calibration + gyro integration, NOT simulator headings.
        rotation = so3_exp(-body_gyro*self.dt)@self.relative_rotation@so3_exp(deck_imu[:, :3]*self.dt)
        rotation = torch.where(self.initialized[:, None, None], rotation, self.relative_rotation)
        observed = apply_rotation(rotation, unit(deck_imu[:, 6:9]))
        correction = alignment_vector(observed, body_gravity)
        gain = torch.where(self.initialized, self.features.new_full(self.initialized.shape, self.gravity_correction_gain), self.features.new_ones(self.initialized.shape))
        rotation = so3_exp(correction*gain[:, None])@rotation
        self.relative_rotation[:] = rotation
        self.initialized[:] = True
        normal = rotation[:, :, 2]
        # Predict deck rotation in the CURRENT body frame. The desired body
        # tilt correction is accounted for separately, rather than assuming
        # the robot will blindly continue its current angular motion.
        predicted = (rotation@so3_exp(deck_imu[:, :3]*self.prediction_horizon_s))[:, :, 2]
        return unit(normal), unit(predicted)

    def update(self, deck_imu, body_gyro, body_gravity, q, contact_force_norms, imu_enabled=True):
        n = self.features.shape[0]
        if (deck_imu.shape != (n, 9) or body_gyro.shape != (n, 3) or body_gravity.shape != (n, 3)
            or q.shape != (n, 2, 3) or contact_force_norms.shape != (n, 2)):
            raise ValueError('prior sensor batch shape mismatch')
        if not imu_enabled:
            self.reset(torch.arange(n, device=self.features.device))
            return self.features
        sensor_valid = (torch.isfinite(deck_imu).all(-1) & torch.isfinite(body_gyro).all(-1)
                        & torch.isfinite(body_gravity).all(-1) & torch.isfinite(q).all(dim=-1).all(-1)
                        & torch.isfinite(contact_force_norms).all(-1)
                        & (deck_imu[:, 6:9].norm(dim=-1) > .5) & (body_gravity.norm(dim=-1) > .5))
        sensor_valid &= (torch.isfinite(body_gyro.norm(dim=-1))
                         & torch.isfinite(deck_imu.norm(dim=-1))
                         & torch.isfinite(q.norm(dim=-1)).all(-1))
        # Bad measurements invalidate only their own environments; do not let
        # a NaN reach the batched linear solver or persist in the observer.
        bad = (~sensor_valid).nonzero(as_tuple=False).flatten()
        self.reset(bad)
        deck_imu = torch.where(sensor_valid[:, None], deck_imu, torch.zeros_like(deck_imu))
        body_gyro = torch.where(sensor_valid[:, None], body_gyro, torch.zeros_like(body_gyro))
        body_gravity = torch.where(sensor_valid[:, None], body_gravity, torch.zeros_like(body_gravity))
        q = torch.where(sensor_valid[:, None, None], q, torch.zeros_like(q))
        contact_force_norms = torch.where(sensor_valid[:, None], contact_force_norms, torch.zeros_like(contact_force_norms))
        normal, future_normal = self._attitude(deck_imu, body_gyro, body_gravity)
        self.contacts[:] = torch.where(self.contacts,
            contact_force_norms > self.contact_off_threshold_n,
            contact_force_norms > self.contact_on_threshold_n) & sensor_valid[:, None]
        wheels = self.kinematics.forward(q)
        contact_points = wheels-self.wheel_radius*normal[:, None, :]
        support = self.contacts.to(q.dtype)
        centroid = (contact_points*support[:, :, None]).sum(1)/support.sum(1, keepdim=True).clamp(min=1)
        cosine = (normal*future_normal).sum(-1)
        geometry_valid = sensor_valid & (cosine >= self.minimum_normal_cosine) & (support.sum(-1) > 0)
        distance = ((centroid[:, None, :]-contact_points)*future_normal[:, None, :]).sum(-1)/cosine.clamp(min=self.minimum_normal_cosine)[:, None]
        distance = distance.clamp(-self.max_contact_displacement_m, self.max_contact_displacement_m)
        compensation = distance[:, :, None]*normal[:, None, :]
        compensated_wheels = contact_points+compensation+self.wheel_radius*future_normal[:, None, :]
        up = -unit(body_gravity)
        desired_up = torch.zeros_like(up); desired_up[:, 2] = 1
        tilt = alignment_vector(up, desired_up)*self.upright_gain
        tilt *= (self.max_upright_correction_rad/tilt.norm(dim=-1, keepdim=True).clamp(min=1e-8)).clamp(max=1)
        target = apply_rotation(so3_exp(tilt)[:, None], compensated_wheels)
        active = self.contacts & geometry_valid[:, None]
        compensation = torch.where(active[:, :, None], compensation, torch.zeros_like(compensation))
        target = torch.where(active[:, :, None], target, wheels)
        k = self.kinematics
        delta_limit = torch.minimum(torch.full_like(k.lower, self.max_joint_delta_rad), k.velocity_limits*self.prediction_horizon_s)
        lower = torch.maximum(k.lower+self.joint_limit_margin_rad, q-delta_limit)
        upper = torch.minimum(k.upper-self.joint_limit_margin_rad, q+delta_limit)
        # An already out-of-range measured joint invalidates its leg; do not
        # manufacture a reference outside the original position/rate bounds.
        active &= (lower <= upper).all(-1)
        reference = q.clone()
        eye = torch.eye(3, device=q.device, dtype=q.dtype)
        for _ in range(self.ik_iterations):
            current, jacobian = k.forward(reference, jacobian=True)
            error = target-current
            solve = torch.linalg.solve(jacobian@jacobian.transpose(-1, -2)+self.ik_damping**2*eye, error.unsqueeze(-1))
            step = (jacobian.transpose(-1, -2)@solve).squeeze(-1).clamp(-.10, .10)
            candidate = torch.minimum(torch.maximum(reference+step, lower), upper)
            reference = torch.where(active[:, :, None], candidate, q)
        residual = (k.forward(reference)-target).norm(dim=-1)
        valid = active & (residual <= self.ik_tolerance_m)
        # Failed IK yields a neutral reference and explicit invalid flag,
        # rather than presenting a physically infeasible target as achievable.
        reference = torch.where(valid[:, :, None], reference, q)
        leg_delta = ((k.forward(reference)-k.hip_origins).norm(dim=-1)
                     -(wheels-k.hip_origins).norm(dim=-1))
        output = torch.cat((normal, future_normal,
            compensation.reshape(n, 6)*self.contact_displacement_scale,
            leg_delta*self.leg_length_scale,
            (reference-q).reshape(n, 6)*self.joint_angle_scale,
            support, valid.to(q.dtype)), -1)
        self.features[:] = torch.where(sensor_valid[:, None], output, torch.zeros_like(output))
        self.reset(bad)
        return self.features
