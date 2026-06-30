"""Leg-morphology URDF generator for WF_TRON1A spatial domain randomization (co-design Phase 1).

Emulates Chen et al. "Pretraining-finetuning Framework for Efficient Co-design" (arXiv:2407.06770):
each parallel env can load a robot with a different leg geometry. This module produces N URDF
variants by scaling the thigh and shank *segment lengths* of the base WF_TRON1A URDF.

Mapping (verified against resources/robots/WF_TRON1A/urdf/robot.urdf):
  * Thigh segment  = link ``hip_{L,R}_Link``  + its downstream joint ``knee_{L,R}_Joint`` origin
                     (knee origin ~ (-0.15, +-0.0205, -0.25981); sagittal length ~0.30 m).
  * Shank segment  = link ``knee_{L,R}_Link`` + its downstream joint ``wheel_{L,R}_Joint`` origin
                     (wheel origin ~ (0.15, +-0.0435, -0.25981)).
  * Wheel link (radius 0.127 m, mass 1.08 kg) is NOT scaled; only its mounting point moves with the shank.
  * Left/right kept symmetric (same scale). Only the x,z components of origins are scaled
    (the segments lie in the sagittal x-z plane); y (hip width / wheel offset) is preserved.

Scaling rules per segment with factor s (length scale), following the paper's URDF-edit table:
  * downstream joint origin x,z  *= s   (moves the child joint along the lengthened segment)
  * segment link inertial origin x,z *= s
  * segment link mass *= s              (mass proportional to length, fixed cross-section)
  * segment link inertia *= s**3        (slender-rod approx: transverse term dominates; v1, see WORKLOG)
  * segment link collision cylinder length *= s, its origin x,z *= s

Pure-Python (xml.etree); no Isaac Gym dependency. Safe to run offline.
"""

import os
import json
import copy
import xml.etree.ElementTree as ET

# --- geometry constants read from the base URDF (meters) ---
ABAD_Z = 0.2602            # |base_Link -> abad joint| vertical drop (NOT scaled)
SEG_Z = 0.25981            # nominal thigh and shank vertical component (each), at scale 1.0
WHEEL_RADIUS = 0.127       # constant

# name -> segment. Only links/joints listed here are modified; everything else is copied verbatim.
THIGH_LINKS = ("hip_L_Link", "hip_R_Link")
THIGH_JOINTS = ("knee_L_Joint", "knee_R_Joint")     # downstream of the thigh
SHANK_LINKS = ("knee_L_Link", "knee_R_Link")
SHANK_JOINTS = ("wheel_L_Joint", "wheel_R_Joint")   # downstream of the shank

_INERTIA_KEYS = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")


def _fmt(v):
    """Format a float compactly without spurious precision noise."""
    return f"{v:.9g}"


def _scale_origin_xz(origin_elem, s):
    """Scale x and z of an <origin xyz="x y z"/> by s; keep y."""
    if origin_elem is None or origin_elem.get("xyz") is None:
        return
    x, y, z = (float(v) for v in origin_elem.get("xyz").split())
    origin_elem.set("xyz", f"{_fmt(x * s)} {_fmt(y)} {_fmt(z * s)}")


def _scale_segment_link(link, s):
    """Scale a segment link's inertial (origin xz, mass*s, inertia*s^3) and collision (origin xz, cyl length*s)."""
    inertial = link.find("inertial")
    if inertial is not None:
        _scale_origin_xz(inertial.find("origin"), s)
        mass = inertial.find("mass")
        if mass is not None and mass.get("value") is not None:
            mass.set("value", _fmt(float(mass.get("value")) * s))
        inertia = inertial.find("inertia")
        if inertia is not None:
            for k in _INERTIA_KEYS:
                if inertia.get(k) is not None:
                    inertia.set(k, _fmt(float(inertia.get(k)) * s ** 3))
    for coll in link.findall("collision"):
        _scale_origin_xz(coll.find("origin"), s)
        cyl = coll.find("geometry/cylinder")
        if cyl is not None and cyl.get("length") is not None:
            cyl.set("length", _fmt(float(cyl.get("length")) * s))


def _kinematic_joint(root, name):
    """Return the real kinematic <joint name=...> (the one with an <origin>), skipping
    gazebo/transmission <joint name=...> references that share the same name but have no origin."""
    for j in root.findall("joint"):
        if j.get("name") == name and j.find("origin") is not None:
            return j
    return None


def scale_robot(base_root, thigh_scale, shank_scale):
    """Return a deep-copied URDF <robot> element with thigh/shank scaled."""
    root = copy.deepcopy(base_root)
    for lname in THIGH_LINKS:
        link = root.find(f"link[@name='{lname}']")
        if link is not None:
            _scale_segment_link(link, thigh_scale)
    for jname in THIGH_JOINTS:
        j = _kinematic_joint(root, jname)
        if j is not None:
            _scale_origin_xz(j.find("origin"), thigh_scale)
    for lname in SHANK_LINKS:
        link = root.find(f"link[@name='{lname}']")
        if link is not None:
            _scale_segment_link(link, shank_scale)
    for jname in SHANK_JOINTS:
        j = _kinematic_joint(root, jname)
        if j is not None:
            _scale_origin_xz(j.find("origin"), shank_scale)
    return root


def nominal_base_height(thigh_scale, shank_scale):
    """Zero-pose base height above ground (m): vertical leg reach + wheel radius.
    Used to spawn each morphology at a sane height (avoid ground penetration / drop shock)."""
    return ABAD_Z + SEG_Z * thigh_scale + SEG_Z * shank_scale + WHEEL_RADIUS


def generate_morphology_set(base_urdf, out_dir, scales_list, prefix="morph"):
    """Write one URDF per (thigh_scale, shank_scale) in scales_list, plus <prefix>_manifest.json.

    Args:
        base_urdf: path to the base WF_TRON1A robot.urdf.
        out_dir:   directory to write morph_*.urdf into (use the same urdf/ dir so ../meshes/ stays valid).
        scales_list: iterable of (thigh_scale, shank_scale) tuples.
        prefix:    filename prefix.
    Returns:
        list of manifest dicts (index, file, thigh_scale, shank_scale, nominal_base_height).
    """
    base_root = ET.parse(base_urdf).getroot()
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i, (st, ss) in enumerate(scales_list):
        root = scale_robot(base_root, float(st), float(ss))
        fname = f"{prefix}_{i:03d}.urdf"
        ET.ElementTree(root).write(
            os.path.join(out_dir, fname), encoding="utf-8", xml_declaration=True
        )
        manifest.append(
            {
                "index": i,
                "file": fname,
                "thigh_scale": float(st),
                "shank_scale": float(ss),
                "nominal_base_height": nominal_base_height(float(st), float(ss)),
            }
        )
    manifest_path = os.path.join(out_dir, f"{prefix}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(
            {"base_urdf": os.path.abspath(base_urdf), "morphologies": manifest},
            f,
            indent=2,
        )
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate WF_TRON1A leg-morphology URDF variants.")
    ap.add_argument("--base_urdf", required=True, help="path to base WF_TRON1A robot.urdf")
    ap.add_argument("--out_dir", required=True, help="output dir (use the same urdf/ dir)")
    ap.add_argument("--prefix", default="morph")
    ap.add_argument(
        "--scales",
        default="0.8,1.0,1.2",
        help="comma list of uniform thigh=shank scales (e.g. '0.8,1.0,1.2')",
    )
    args = ap.parse_args()
    scales = [(float(s), float(s)) for s in args.scales.split(",")]
    man = generate_morphology_set(args.base_urdf, args.out_dir, scales, args.prefix)
    for m in man:
        print(
            f"[{m['index']:03d}] {m['file']}  t={m['thigh_scale']} s={m['shank_scale']}  "
            f"base_h={m['nominal_base_height']:.4f} m"
        )
