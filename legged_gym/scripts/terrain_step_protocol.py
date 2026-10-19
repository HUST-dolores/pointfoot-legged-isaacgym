"""Frozen terrain meshes and measurement helpers for the July stepping policies."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from legged_gym.scripts.payload_step_protocol import make_scene, sha256_file, write_json, array_digest


POLICIES = {
    "base": ("Base", "Jul02_12-12-17_", 10000),
    "adapted": ("Terrain-adapted", "Jul02_16-11-09_", 14500),
}
TERRAINS = ("flat", "slope5", "rough5cm")
SEEDS = (12345, 20261016, 20261017)
DT = .02
DX = .1
X0, X1 = -5., 15.
Y0, Y1 = -4., 4.
NX, NY = 201, 81
FRICTION = .4
RESTITUTION = .8
FRAME_TIMES = (2., 2.24, 2.48, 2.72)


def anchors(num_envs=30):
    ids = np.arange(num_envs)
    return np.column_stack(((ids // 5) * 24., (ids % 5) * 12.)).astype(np.float32)


def make_terrain_scene(seed):
    scene = make_scene(seed, num_envs=30, steps=300)
    x = np.linspace(X0, X1, NX)
    y = np.linspace(Y0, Y1, NY)
    scene["height_flat"] = np.zeros((30, NX, NY), dtype=np.float32)
    scene["height_slope5"] = np.broadcast_to(
        (x * np.tan(np.deg2rad(5.)))[None, :, None], (30, NX, NY)).astype(np.float32).copy()
    cx, cy = np.linspace(X0, X1, 51), np.linspace(Y0, Y1, 21)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    coarse = np.random.default_rng(np.random.SeedSequence([seed, 210])).uniform(-.05, .05, (30, 51, 21))
    scene["rough_coarse_height"] = coarse.astype(np.float32)
    scene["height_rough5cm"] = np.stack([
        RegularGridInterpolator((cx, cy), field, method="linear")(points).reshape(NX, NY)
        for field in coarse]).astype(np.float32)
    scene["mesh_anchors_xy"] = anchors()
    return scene


def triangle_mesh(heights, anchor_xy):
    count, nx, ny = heights.shape
    xx, yy = np.meshgrid(np.linspace(X0, X1, nx), np.linspace(Y0, Y1, ny), indexing="ij")
    vertices = np.empty((count, nx, ny, 3), dtype=np.float32)
    vertices[..., 0] = xx[None] + anchor_xy[:, None, None, 0]
    vertices[..., 1] = yy[None] + anchor_xy[:, None, None, 1]
    vertices[..., 2] = heights
    a = np.arange(nx * ny).reshape(nx, ny)[:-1, :-1]
    lower = np.stack((a, a + ny, a + 1), axis=-1).reshape(-1, 3)
    upper = np.stack((a + ny + 1, a + 1, a + ny), axis=-1).reshape(-1, 3)
    one = np.concatenate((lower, upper), axis=0)
    triangles = np.concatenate([one + env * nx * ny for env in range(count)], axis=0).astype(np.uint32)
    return vertices.reshape(-1, 3), triangles


def sample_height_numpy(heights, env_ids, local_xy):
    p = np.asarray(local_xy)
    u = np.clip((p[..., 0] - X0) / DX, 0, heights.shape[1] - 1)
    v = np.clip((p[..., 1] - Y0) / DX, 0, heights.shape[2] - 1)
    i = np.minimum(np.floor(u).astype(int), heights.shape[1] - 2)
    j = np.minimum(np.floor(v).astype(int), heights.shape[2] - 2)
    u, v = u - i, v - j
    z00, z10 = heights[env_ids, i, j], heights[env_ids, i + 1, j]
    z01, z11 = heights[env_ids, i, j + 1], heights[env_ids, i + 1, j + 1]
    return np.where(u + v <= 1, z00 + (z10-z00)*u + (z01-z00)*v,
                    z11 + (z01-z11)*(1-u) + (z10-z11)*(1-v))


def wheel_geometry(urdf):
    root = ET.parse(urdf).getroot()
    offsets, radii, widths = [], [], []
    for side in ("L", "R"):
        collision = root.find(f"link[@name='wheel_{side}_Link']/collision")
        cylinder = collision.find("geometry/cylinder")
        offsets.append([float(x) for x in collision.find("origin").get("xyz").split()])
        rpy = [float(x) for x in collision.find("origin").get("rpy").split()]
        if not np.allclose(np.abs(rpy), [np.pi/2, 0, 0], atol=1e-6):
            raise ValueError("wheel collision axis changed: measurement needs an explicit axis update")
        radii.append(float(cylinder.get("radius")))
        widths.append(float(cylinder.get("length")))
    return np.asarray(offsets, np.float32), np.asarray(radii, np.float32), np.asarray(widths, np.float32)


def cylinder_vertical_extent(axis_z, radius, width):
    return radius * np.sqrt(np.maximum(0, 1 - np.asarray(axis_z) ** 2)) + .5 * width * np.abs(axis_z)


def capsule_vertical_extent(axis_z, radius, segment_length):
    return radius + .5 * segment_length * np.abs(axis_z)


def prepare(output):
    output = Path(output) / "scenarios"
    output.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for seed in SEEDS:
        scene = make_terrain_scene(seed)
        path = output / f"seed_{seed}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as previous:
                if array_digest(dict(previous)) != array_digest(scene):
                    raise ValueError(f"existing frozen scene differs: {path}")
        else:
            np.savez_compressed(path, **scene)
        manifests[str(seed)] = {"file_sha256": sha256_file(path), "array_sha256": array_digest(scene),
            "rough_min_m": float(scene["height_rough5cm"].min()),
            "rough_max_m": float(scene["height_rough5cm"].max()),
            "height_sha256": {t: array_digest({t: scene["height_"+t]}) for t in TERRAINS}}
    write_json(output / "manifest.json", {"seeds": manifests, "num_envs": 30,
        "grid_spacing_m": DX, "rough_control_grid_m": .4, "rough_sampling_support_m": [-.05, .05],
        "patch_bounds_m": [X0, X1, Y0, Y1], "friction": FRICTION, "restitution": RESTITUTION,
        "mesh_diagonal": "v00-v10-v01 and v11-v01-v10", "uphill_direction": "+world X"})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
