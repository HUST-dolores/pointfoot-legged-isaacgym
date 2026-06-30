"""Phase-1 workstation validation: generate WF_TRON1A leg morphologies and load each in Isaac Gym
to verify the hard constraint (identical DOF/body count across morphologies) and sane mass/geometry.

This does NOT test dynamic balance (a wheeled biped is an inverted pendulum and cannot stand under
passive PD; balance is validated later with the trained spatial-DR policy). It validates that the
generated URDFs are well-formed, load, keep topology constant, and that scale=1.0 reproduces the
base robot's mass.

Run on the workstation:
    conda activate cdesign
    cd /home/ps/mydrive/xu/pointfoot-legged-gym
    python legged_gym/scripts/test_morphology_load.py
"""

import os
import argparse
import xml.etree.ElementTree as ET

from isaacgym import gymapi

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403  # initialize package in train.py's order (avoids circular import)
from legged_gym.utils.morphology_urdf import generate_morphology_set


def _make_sim(gym):
    sp = gymapi.SimParams()
    sp.dt = 1.0 / 200.0
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.use_gpu_pipeline = True
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 4
    sp.physx.num_velocity_iterations = 0
    sp.physx.use_gpu = True
    # compute device 0, graphics device -1 (headless; we only need physics asset stats, no rendering)
    return gym.create_sim(0, -1, gymapi.SIM_PHYSX, sp)


def _asset_options():
    o = gymapi.AssetOptions()
    o.fix_base_link = False
    o.collapse_fixed_joints = True
    o.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    o.flip_visual_attachments = False
    o.replace_cylinder_with_capsule = False
    return o


def _urdf_total_mass(path):
    """Sum of all <mass value=...> in a URDF (mass is conserved regardless of fixed-joint collapse)."""
    root = ET.parse(path).getroot()
    return float(sum(float(m.get("value")) for m in root.iter("mass")))


def _load_stats(gym, sim, urdf_dir, fname, opts):
    asset = gym.load_asset(sim, urdf_dir, fname, opts)
    ndof = gym.get_asset_dof_count(asset)
    nbody = gym.get_asset_rigid_body_count(asset)
    total_mass = _urdf_total_mass(os.path.join(urdf_dir, fname))
    return ndof, nbody, total_mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--urdf_dir",
        default=os.path.join(LEGGED_GYM_ROOT_DIR, "resources/robots/WF_TRON1A/urdf"),
    )
    ap.add_argument("--base_urdf", default="robot.urdf")
    ap.add_argument("--scales", default="0.8,1.0,1.2",
                    help="uniform thigh=shank scales to generate")
    ap.add_argument("--prefix", default="morph")
    args = ap.parse_args()

    base_urdf_path = os.path.join(args.urdf_dir, args.base_urdf)
    scales = [(float(s), float(s)) for s in args.scales.split(",")]

    print(f"=== generating {len(scales)} morphologies into {args.urdf_dir} ===")
    manifest = generate_morphology_set(base_urdf_path, args.urdf_dir, scales, args.prefix)
    for m in manifest:
        print(f"  [{m['index']:03d}] {m['file']}  t={m['thigh_scale']} s={m['shank_scale']}  "
              f"nominal_base_h={m['nominal_base_height']:.4f} m")

    gym = gymapi.acquire_gym()
    sim = _make_sim(gym)
    opts = _asset_options()

    print("=== base robot.urdf reference ===")
    b_dof, b_body, b_mass = _load_stats(gym, sim, args.urdf_dir, args.base_urdf, opts)
    print(f"  base: dof={b_dof} bodies={b_body} total_mass={b_mass:.4f} kg")

    print("=== morphologies ===")
    dof_set, body_set = set(), set()
    scale1_mass = None
    for m in manifest:
        ndof, nbody, mass = _load_stats(gym, sim, args.urdf_dir, m["file"], opts)
        dof_set.add(ndof)
        body_set.add(nbody)
        if abs(m["thigh_scale"] - 1.0) < 1e-9 and abs(m["shank_scale"] - 1.0) < 1e-9:
            scale1_mass = mass
        print(f"  [{m['index']:03d}] t={m['thigh_scale']} s={m['shank_scale']}  "
              f"dof={ndof} bodies={nbody} total_mass={mass:.4f} kg")

    print("=== checks ===")
    print(f"  DOF set across morphologies: {dof_set}")
    print(f"  body-count set across morphologies: {body_set}")
    ok = True
    if len(dof_set) != 1:
        print("  FAIL: DOF count varies across morphologies"); ok = False
    if len(body_set) != 1:
        print("  FAIL: rigid-body count varies across morphologies"); ok = False
    if dof_set and next(iter(dof_set)) != b_dof:
        print(f"  WARN: morph DOF {dof_set} != base DOF {b_dof}")
    if scale1_mass is not None:
        dm = abs(scale1_mass - b_mass)
        print(f"  scale=1.0 mass {scale1_mass:.4f} vs base {b_mass:.4f} (|d|={dm:.5f} kg)")
        if dm > 1e-3:
            print("  FAIL: scale=1.0 morphology mass differs from base"); ok = False
    print("  RESULT:", "PASS" if ok else "FAIL")

    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
