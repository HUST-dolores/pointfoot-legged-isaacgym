#!/usr/bin/env python3
"""Measure the native cylinder-to-capsule replacement extent on a plane."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def main(output):
    import isaacgym  # noqa: F401
    from isaacgym import gymapi, gymtorch
    import torch
    import numpy as np
    from legged_gym.scripts.payload_step_protocol import write_json

    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    robot = ET.Element("robot", name="wheel_collision_probe")
    link = ET.SubElement(robot, "link", name="probe")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value="1")
    ET.SubElement(inertial, "inertia", ixx="0.005", ixy="0", ixz="0", iyy="0.005", iyz="0", izz="0.005")
    collision = ET.SubElement(link, "collision")
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "cylinder", radius="0.127", length="0.05")
    ET.ElementTree(robot).write(output / "probe.urdf", encoding="utf-8", xml_declaration=True)
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = .005
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.use_gpu_pipeline = params.physx.use_gpu = True
    params.physx.num_position_iterations = 8
    params.physx.num_threads = 2
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    try:
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        plane.restitution = 0
        gym.add_ground(sim, plane)
        options = gymapi.AssetOptions()
        options.replace_cylinder_with_capsule = True
        asset = gym.load_asset(sim, str(output), "probe.urdf", options)
        env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
        for index, vertical in enumerate((False, True)):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(index, 0, .5)
            if not vertical:
                pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.pi/2)
            actor = gym.create_actor(env, asset, pose, str(index), index, 0)
            props = gym.get_actor_rigid_shape_properties(env, actor)
            for p in props:
                p.restitution = 0
            gym.set_actor_rigid_shape_properties(env, actor, props)
        gym.prepare_sim(sim)
        roots = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        for _ in range(600):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        heights = roots[:, 2].cpu().numpy().tolist()
        result = {"horizontal_axis_rest_height_m": heights[0], "vertical_axis_rest_height_m": heights[1],
                  "nominal_radius_m": .127, "nominal_cylinder_length_m": .05,
                  "inferred_capsule_half_segment_m": heights[1]-heights[0],
                  "expected_capsule_heights_m": [.127, .152]}
        write_json(output / "result.json", result)
        print(result, flush=True)
    finally:
        gym.destroy_sim(sim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output.resolve())
