#!/usr/bin/env python3
"""Four task scenes with measurement definitions, using collision geometry."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Polygon, Rectangle
import numpy as np

import fig0_sixd_design_variables as style


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARCHIVED_URDF = HERE / "nominal_source.urdf"
DEFAULT_URDF = (ARCHIVED_URDF if ARCHIVED_URDF.is_file() else
                REPO / "resources/robots/WF_TRON1A/urdf/robot.urdf")
DEFAULT_OUTPUT = HERE / "out/four_capability_definitions"
FONT_SIZE = 14
FIGURE_SIZE = (12.8, 9.0)
MEASURE_COLOR = "#238B7B"
BOUNDARY_COLOR = "#487FB7"
LOAD_COLOR = "#BE9153"
style.COLLISION_COLORS = {**style.COLLISION_COLORS, "load": "#CFAC72"}
REACHABLE_POINTS = np.array([[0.22, 0.95], [0.54, 0.68], [0.92, 0.36]])
TEXT = {
    "zh": {
        "payload_title": "(a)  承载：可靠载荷边界",
        "uphill_title": "(b)  上坡：沿坡实际速度",
        "clearance_title": "(c)  迈步：轮胎离地净空",
        "jump_title": "(d)  跳跃：实际高度与距离",
        "payload": "附加载荷",
        "load_axis": "载荷",
        "success_axis": "成功率",
        "ground": "地面",
        "speed": "实测沿坡速度",
        "angle": "坡角 10°",
        "lowest_point": "轮胎最低点",
        "clearance": "净空",
        "takeoff": "起飞",
        "landing": "首次着陆",
        "actual_height": "实际高度",
        "actual_distance": "实际距离",
        "reachable": "可靠可达区域",
        "height_axis": "高度",
        "distance_axis": "距离",
    },
    "en": {
        "payload_title": "(a)  Payload: reliable mass limit",
        "uphill_title": "(b)  Uphill: actual along-slope speed",
        "clearance_title": "(c)  Stepping: wheel-ground clearance",
        "jump_title": "(d)  Jumping: actual height and distance",
        "payload": "Added payload",
        "load_axis": "Payload",
        "success_axis": "Success rate",
        "ground": "Ground",
        "speed": "Actual uphill speed",
        "angle": "Slope 10°",
        "lowest_point": "Wheel lowest point",
        "clearance": "Clearance",
        "takeoff": "Takeoff",
        "landing": "First landing",
        "actual_height": "Actual height",
        "actual_distance": "Actual distance",
        "reachable": "Reliable region",
        "height_axis": "Height",
        "distance_axis": "Distance",
    },
}


def axis_rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def make_scene(robot, angles=None, offset=(0, 0, 0), slope=0.0, payload=False):
    angles = angles or {}
    world = np.eye(4)
    world[:3, :3] = style.rotation_rpy(np.array([0.0, -slope, 0.0]))
    world[:3, 3] = offset
    frames = {"base_Link": world @ robot.frames["base_Link"]}
    remaining = list(robot.joints.values())
    while remaining:
        resolved = []
        for joint in remaining:
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            if parent not in frames:
                continue
            rotation = np.eye(4)
            angle = float(angles.get(joint.get("name"), 0.0))
            if angle:
                rotation[:3, :3] = axis_rotation(style.values(joint.find("axis"), "xyz"), angle)
            frames[child] = frames[parent] @ style.origin_transform(joint.find("origin")) @ rotation
            resolved.append(joint)
        if not resolved:
            raise ValueError("Cannot resolve the scene kinematic tree")
        remaining = [joint for joint in remaining if joint not in resolved]
    centers = {name: frames[joint.find("child").get("link")][:3, 3].copy()
               for name, joint in robot.joints.items()}
    collisions = []
    for collision in robot.collisions:
        local = np.linalg.inv(robot.frames[collision.link]) @ collision.transform
        collisions.append(style.Collision(collision.link, collision.kind,
                                          frames[collision.link] @ local,
                                          collision.dimensions.copy()))
    if payload:
        local = np.eye(4)
        local[:3, 3] = [0.03, 0, 0.10]
        load_frame = frames["base_Link"] @ local
        collisions.append(style.Collision("load_Link", "box", load_frame,
                                          np.array([0.20, 0.18, 0.15])))
        frames["load_Link"] = load_frame
    return SimpleNamespace(frames=frames, centers=centers, collisions=collisions)


def draw_robot(ax, scene, projection, side=None, alpha=1.0):
    style.draw_collisions(ax, scene, projection, side=side)
    ax.collections[-1].set_alpha(alpha)
    sides = side if side else "LR"
    for leg in sides:
        abad = scene.centers[f"abad_{leg}_Joint"]
        local = np.array([0.05556, 0.105 if leg == "L" else -0.105, -0.167, 1])
        body_bottom = (scene.frames["base_Link"] @ local)[:3]
        chain = projection.xy([body_bottom, abad, scene.centers[f"hip_{leg}_Joint"],
                               scene.centers[f"knee_{leg}_Joint"], scene.centers[f"wheel_{leg}_Joint"]])
        ax.plot(chain[:, 0], chain[:, 1], color="#81929E", linewidth=0.85,
                linestyle=(0, (3, 2)), alpha=alpha, zorder=3)


def ground_plane(ax, projection, slope=0.0):
    if np.isclose(projection.view[0], 0):
        x = np.array([-1.2, 1.2])
        z = x * np.tan(slope)
        ax.fill_between(x, z, z - 0.028, color="#E4E9EC", zorder=0)
        ax.plot(x, z, color="#8E9CA5", linewidth=1.1, zorder=1)
    else:
        corners = np.array([[-0.55, -0.35, 0], [0.55, -0.35, 0],
                            [0.55, 0.35, 0], [-0.55, 0.35, 0]])
        corners[:, 2] = corners[:, 0] * np.tan(slope)
        ax.add_patch(Polygon(projection.xy(corners), facecolor="#F1F4F6",
                             edgecolor="#B1BDC5", linewidth=0.65, zorder=0))


def label(ax, position, value, color=style.INK, align="left", **kwargs):
    return ax.text(*position, value, fontsize=FONT_SIZE, color=color,
                   ha=align, va="center", **kwargs)


def leader(ax, target, position, value, color=style.MUTED, align="left"):
    ax.annotate(value, xy=target, xytext=position, ha=align, va="center",
                fontsize=FONT_SIZE, color=color,
                arrowprops={"arrowstyle": "-", "lw": 0.9, "color": color,
                            "shrinkA": 5, "shrinkB": 5}, zorder=8)


def double_arrow(ax, start, end, color=MEASURE_COLOR):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<->", mutation_scale=11,
                                 color=color, linewidth=1.4, shrinkA=0, shrinkB=0, zorder=6))


def prepare_scene_axis(ax, bounds):
    ax.set(xlim=bounds[0], ylim=bounds[1], aspect="equal")
    ax.axis("off")


def wheel_radius(robot):
    return float(next(c.dimensions[0] for c in robot.collisions if c.link == "wheel_L_Link"))


def draw_payload_scene(ax, robot, words):
    projection = style.Projection([4.0, -3.0, 1.4])
    scene = make_scene(robot, payload=True)
    ground_plane(ax, projection)
    draw_robot(ax, scene, projection)
    load_center = projection.xy(scene.frames["load_Link"][:3, 3])
    leader(ax, load_center, (0.16, 1.13), words["payload"] + r" $m$", LOAD_COLOR)
    prepare_scene_axis(ax, ((-0.42, 0.57), (-0.13, 1.27)))


def draw_payload_boundary(ax, words):
    # Shape illustrates the boundary definition; no measured capacity is encoded.
    mass = np.array([0, 0.25, 0.45, 0.58, 0.64, 0.71, 0.79, 0.90, 1.0])
    probability = np.array([1, 1, 0.99, 0.97, 0.93, 0.83, 0.65, 0.30, 0.05])
    boundary = float(np.interp(0.90, probability[::-1], mass[::-1]))
    ax.plot(mass, probability, color=BOUNDARY_COLOR, linewidth=1.9)
    ax.axhline(0.90, color="#85939C", linewidth=0.85, linestyle=(0, (3, 3)))
    ax.plot([boundary, boundary], [0, 0.90], color=MEASURE_COLOR,
            linewidth=1.0, linestyle=(0, (3, 3)))
    ax.scatter([boundary], [0.90], s=25, color=MEASURE_COLOR, zorder=5)
    ax.set(xlim=(0, 1.02), ylim=(0, 1.09))
    ax.set_xticks([boundary], [r"$M_{90}$"])
    ax.set_yticks([0.90], ["90%"])
    ax.set_xlabel(words["load_axis"] + r" $m$", fontsize=FONT_SIZE, labelpad=5)
    ax.text(0, 1.13, words["success_axis"], transform=ax.transAxes, fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE, colors=style.MUTED, length=3)
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color("#92A0A9")
        ax.spines[name].set_linewidth(0.8)
    return boundary


def draw_uphill_scene(ax, robot, words):
    projection = style.Projection([0, -1, 0])
    slope = np.deg2rad(10.0)
    scene = make_scene(robot, slope=slope)
    ground_plane(ax, projection, slope)
    draw_robot(ax, scene, projection, side="L")
    start = np.array([0.07, 0.36])
    end = start + np.array([0.61 * np.cos(slope), 0.61 * np.sin(slope)])
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                 color=MEASURE_COLOR, linewidth=1.8, zorder=6))
    label(ax, (0.10, 0.52), words["speed"] + r" $v_{\parallel}$", MEASURE_COLOR)
    anchor = np.array([0.30, 0.30 * np.tan(slope)])
    ax.plot([anchor[0], 0.95], [anchor[1], anchor[1]], color="#9AA6AE",
            linestyle=(0, (3, 3)), linewidth=0.85)
    ax.add_patch(Arc(anchor, 0.85, 0.85, theta1=0, theta2=10,
                     color=style.INK, linewidth=1.1))
    label(ax, (0.58, -0.018), words["angle"], align="center")
    prepare_scene_axis(ax, ((-0.55, 1.01), (-0.15, 1.22)))


def draw_clearance_scene(ax, robot, words):
    projection = style.Projection([4.0, -3.0, 1.4])
    scene = make_scene(robot, {"hip_L_Joint": -0.65, "knee_L_Joint": 0.20})
    ground_plane(ax, projection)
    draw_robot(ax, scene, projection)
    center = scene.centers["wheel_L_Joint"]
    bottom = center - np.array([0, 0, wheel_radius(robot)])
    ground = bottom.copy()
    ground[2] = 0
    bottom_xy, ground_xy = projection.xy([bottom, ground])
    ax.scatter([bottom_xy[0]], [bottom_xy[1]], s=25, color=style.INK, zorder=7)
    shift = np.array([0.12, 0])
    for endpoint in (bottom_xy, ground_xy):
        ax.plot([endpoint[0], endpoint[0] + shift[0]], [endpoint[1], endpoint[1]],
                color=MEASURE_COLOR, linewidth=0.9, zorder=5)
    double_arrow(ax, ground_xy + shift, bottom_xy + shift)
    label(ax, (bottom_xy[0] + 0.17, (bottom_xy[1] + ground_xy[1]) / 2),
          words["clearance"] + r" $h_{\mathrm{clear}}$", MEASURE_COLOR)
    leader(ax, bottom_xy, (0.27, 0.49), words["lowest_point"])
    leader(ax, ground_xy, (-0.11, -0.12), words["ground"])
    prepare_scene_axis(ax, ((-0.36, 0.93), (-0.19, 1.16)))
    return float(bottom[2])


def wheel_mid(scene):
    return np.mean([scene.centers[f"wheel_{leg}_Joint"] for leg in "LR"], axis=0)


def move_wheel_mid_x(scene, desired):
    shift = desired - wheel_mid(scene)[0]
    translate_scene(scene, np.array([shift, 0, 0]))


def translate_scene(scene, translation):
    for frame in scene.frames.values():
        frame[:3, 3] += translation
    for center in scene.centers.values():
        center[:] += translation
    for collision in scene.collisions:
        collision.transform[:3, 3] += translation


def draw_jump_scene(ax, robot, words):
    projection = style.Projection([0, -1, 0])
    start, finish = -0.44, 0.49
    takeoff = make_scene(robot)
    landing = make_scene(robot, {
        "hip_L_Joint": -0.65, "hip_R_Joint": 0.65,
        "knee_L_Joint": 0.20, "knee_R_Joint": -0.20,
    })
    landing_clearance = min(landing.centers[f"wheel_{leg}_Joint"][2] - wheel_radius(robot)
                            for leg in "LR")
    translate_scene(landing, np.array([0, 0, -landing_clearance]))
    flight = make_scene(robot, {
        "hip_L_Joint": -0.60, "hip_R_Joint": 0.60,
        "knee_L_Joint": 0.30, "knee_R_Joint": -0.30,
    }, offset=(0, 0, 0.18))
    for scene, x in ((takeoff, start), (landing, finish), (flight, (start + finish) / 2)):
        move_wheel_mid_x(scene, x)
    ground_plane(ax, projection)
    draw_robot(ax, takeoff, projection, side="L", alpha=0.25)
    draw_robot(ax, landing, projection, side="L", alpha=0.25)
    draw_robot(ax, flight, projection, side="L")
    radius = wheel_radius(robot)
    height = min(flight.centers[f"wheel_{leg}_Joint"][2] - radius for leg in "LR")
    bottom = projection.xy(wheel_mid(flight) - np.array([0, 0, radius]))
    curve_x = np.linspace(start, finish, 100)
    curve_y = height * 4 * ((curve_x - start) / (finish - start)) * (
        1 - (curve_x - start) / (finish - start)
    )
    ax.plot(curve_x, curve_y, color="#91AAA6", linewidth=0.9,
            linestyle=(0, (3, 3)), zorder=1)
    ax.scatter([bottom[0]], [bottom[1]], s=22, color=style.INK, zorder=7)
    height_x = 0.17
    ax.plot([bottom[0], height_x], [bottom[1], bottom[1]], color=MEASURE_COLOR,
            linewidth=0.85, zorder=5)
    double_arrow(ax, (height_x, 0), (height_x, height))
    label(ax, (0.19, 1.22), words["actual_height"] + r" $H$", MEASURE_COLOR)
    for x in (start, finish):
        ax.plot([x, x], [0.0, -0.11], color="#93A2AC", linewidth=0.85,
                linestyle=(0, (3, 3)))
    double_arrow(ax, (start, -0.10), (finish, -0.10))
    label(ax, ((start + finish) / 2, -0.19), words["actual_distance"] + r" $D$",
          MEASURE_COLOR, align="center")
    label(ax, (start, -0.32), words["takeoff"], style.MUTED, align="center")
    label(ax, (finish, -0.32), words["landing"], style.MUTED, align="center")
    prepare_scene_axis(ax, ((-0.78, 0.98), (-0.40, 1.29)))
    return {"height_illustration_m": float(height), "distance_illustration_m": finish - start}


def coverage_polygon(points):
    # Downward closure of reliable (distance, height) points, matching the evaluator.
    points = sorted((float(distance), float(height)) for distance, height in points)
    if any(points[i][1] < points[i + 1][1] for i in range(len(points) - 1)):
        raise ValueError("Coverage illustration expects nondominated decreasing heights")
    boundary = [(0, points[0][1])]
    for index, (distance, height) in enumerate(points):
        boundary.append((distance, height))
        boundary.append((distance, points[index + 1][1] if index + 1 < len(points) else 0))
    return np.asarray([(0, 0), *boundary, (0, 0)]), np.asarray(boundary)


def covered_area(points):
    previous_height, area = 0.0, 0.0
    for height in sorted(set(float(point[1]) for point in points)):
        envelope = max(float(point[0]) for point in points if point[1] >= height)
        area += (height - previous_height) * envelope
        previous_height = height
    return area


def draw_reachability(ax, words):
    points = REACHABLE_POINTS
    polygon, boundary = coverage_polygon(points)
    ax.add_patch(Polygon(polygon, facecolor="#DCEEE6", edgecolor="none", zorder=1))
    ax.plot(boundary[:, 0], boundary[:, 1], color=MEASURE_COLOR, linewidth=1.6, zorder=3)
    ax.scatter(points[:, 0], points[:, 1], s=23, color=BOUNDARY_COLOR, zorder=4)
    ax.add_patch(Rectangle((0, 0), points[:, 0].max(), points[:, 1].max(),
                           fill=False, edgecolor="#9DAAB2", linewidth=0.8,
                           linestyle=(0, (3, 3)), zorder=2))
    ax.text(0.5, 1.13, words["reachable"], transform=ax.transAxes, ha="center",
            fontsize=FONT_SIZE, color=style.INK)
    ax.text(0.24, 0.19, r"$A_{\mathrm{jump}}$", fontsize=FONT_SIZE, color=MEASURE_COLOR)
    ax.set(xlim=(0, 1.08), ylim=(0, 1.08))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(words["distance_axis"] + r" $D$", fontsize=FONT_SIZE, labelpad=6)
    ax.set_ylabel(words["height_axis"] + r" $H$", fontsize=FONT_SIZE, labelpad=6)
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color("#8A99A3")
        ax.spines[name].set_linewidth(0.8)


def create_figure(robot, language):
    words = TEXT[language]
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=140)
    for position, key in (((0.04, 0.950), "payload_title"), ((0.53, 0.950), "uphill_title"),
                          ((0.04, 0.472), "clearance_title"), ((0.53, 0.472), "jump_title")):
        fig.text(*position, words[key], fontsize=FONT_SIZE, color=style.INK)
    payload = fig.add_axes([0.04, 0.555, 0.255, 0.355])
    payload_limit = fig.add_axes([0.338, 0.625, 0.133, 0.195])
    slope = fig.add_axes([0.53, 0.555, 0.43, 0.355])
    step = fig.add_axes([0.04, 0.085, 0.43, 0.355])
    jump = fig.add_axes([0.515, 0.105, 0.29, 0.33])
    reachable = fig.add_axes([0.835, 0.205, 0.124, 0.17])
    draw_payload_scene(payload, robot, words)
    draw_payload_boundary(payload_limit, words)
    draw_uphill_scene(slope, robot, words)
    draw_clearance_scene(step, robot, words)
    draw_jump_scene(jump, robot, words)
    draw_reachability(reachable, words)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    corpus = "".join(value for language in TEXT.values() for value in language.values())
    style.configure_fonts(args.output_dir, extra_text=corpus)
    robot = style.CollisionRobot(args.urdf)
    style.validate_robot(robot)
    languages = ("zh", "en") if args.language == "both" else (args.language,)
    for language in languages:
        fig = create_figure(robot, language)
        for extension in ("png", "pdf", "svg"):
            target = args.output_dir / f"four_capability_definitions_{language}.{extension}"
            fig.savefig(target, facecolor="white", dpi=args.dpi)
            print(target)
        plt.close(fig)
    area = covered_area(REACHABLE_POINTS)
    rectangle_area = float(REACHABLE_POINTS[:, 0].max() * REACHABLE_POINTS[:, 1].max())
    if not area < rectangle_area:
        raise ValueError("Reachability illustration fails to distinguish coverage from independent extrema")
    manifest = {
        "source_urdf": str(args.urdf.resolve()),
        "source_urdf_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        "figure_type": "Metric-definition schematic, not a measured performance result or policy rollout",
        "robot_visuals": "Colored collision primitives with nominal geometry; static illustrative kinematic poses",
        "motor_housing_dimensions_encode_design_scale": False,
        "payload_curve": "Qualitative definition-only curve; the only numeric success marker is the 90 percent criterion",
        "payload_measure": "Continuous successful mass boundary from low mass, at 90 percent trial success, with neighboring-level interpolation",
        "uphill_measure": "Actual world velocity projected onto ascending terrain tangent; 10th-percentile trial scores, reset/downhill failures zero",
        "uphill_angle_deg": 10,
        "clearance_measure": "Wheel-center height minus fixed wheel radius and ground height, using swing-cycle peaks",
        "clearance_statistic": "10th percentile across environment/leg median valid swing-cycle peaks",
        "jump_height_measure": "Peak over flight of the lower of the two wheel-ground clearances; not COM apex rise",
        "jump_distance_measure": "Wheel-midpoint displacement from actual takeoff to first landing, projected onto takeoff forward direction",
        "jump_reliability": "Per-command physical resolved success rate >=90 percent; successful actual H,D represented by marginal 10th percentiles",
        "jump_area_definition": "Area of the downward union of reliable actual (D,H) pairs, not the convex hull or independent-maxima product",
        "reachability_diagram_points": REACHABLE_POINTS.tolist(),
        "reachability_diagram_coordinates": "Normalized conceptual positions; not experiment samples or physical values",
        "conceptual_coverage_area": area,
        "conceptual_extrema_rectangle_area": rectangle_area,
        "font_size_pt": FONT_SIZE,
        "explanatory_footers": "Removed in V2; success criteria and statistical definitions belong in the accompanying Markdown notes",
        "raster_size_px": [round(size * args.dpi) for size in FIGURE_SIZE],
    }
    (args.output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
