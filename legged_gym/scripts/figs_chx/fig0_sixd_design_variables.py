#!/usr/bin/env python3
"""Export the six-dimensional design definition using URDF collision geometry."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import string
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import PolyCollection
from matplotlib import font_manager
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as DrawingPath
import numpy as np


REPO = Path(__file__).resolve().parents[3]
DEFAULT_URDF = REPO / "resources/robots/WF_TRON1A/urdf/robot.urdf"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "out/sixd_design_variables_v2"
GROUPS = ("abad", "hip", "knee", "wheel")
GROUP_COLORS = {
    "abad": "#367BB5",
    "hip": "#199A84",
    "knee": "#C35C46",
    "wheel": "#8263A6",
}
THIGH_COLOR = "#197B83"
SHANK_COLOR = "#6E60A0"
COLLISION_COLORS = {
    "base": "#6894C5",
    "abad": "#DAA1AD",
    "hip": "#E2B45F",
    "knee": "#6BB6A4",
    "wheel": "#77759F",
}
INK = "#26343E"
MUTED = "#687780"
LIGHT = "#D5DDE2"
TEXT = {
    "zh": {
        "title": "六维结构与驱动设计变量",
        "subtitle": "关节轴线与碰撞体积模型",
        "side_title": "(a)  侧视图：两类长度尺度",
        "drive_title": "(b)  斜视图：四类驱动尺度",
        "thigh": "大腿",
        "shank": "小腿",
        "hip_center": "髋轴中心",
        "knee_center": "膝轴中心",
        "wheel_center": "轮轴中心",
        "forward": "前进方向",
        "left": "左",
        "right": "右",
        "shared": "左右同类驱动共享参数",
        "abad": "侧摆驱动",
        "hip": "髋部驱动",
        "knee": "膝部驱动",
        "wheel": "轮部驱动",
    },
    "en": {
        "title": "Six-dimensional morphology and actuation design",
        "subtitle": "Joint axes and collision-volume model",
        "side_title": "(a)  Side view: two length scales",
        "drive_title": "(b)  Oblique view: four actuator scales",
        "thigh": "Thigh",
        "shank": "Shank",
        "hip_center": "Hip axis",
        "knee_center": "Knee axis",
        "wheel_center": "Wheel axis",
        "forward": "Forward",
        "left": "L",
        "right": "R",
        "shared": "Left/right actuators of each type share one scale",
        "abad": "Abduction",
        "hip": "Hip",
        "knee": "Knee",
        "wheel": "Wheel",
    },
}


def values(element, key, default="0 0 0"):
    return np.asarray([float(x) for x in element.get(key, default).split()])


def rotation_rpy(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (
        np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    )
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def origin_transform(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, :3] = rotation_rpy(values(element, "rpy"))
        transform[:3, 3] = values(element, "xyz")
    return transform


@dataclass
class Collision:
    link: str
    kind: str
    transform: np.ndarray
    dimensions: np.ndarray


class CollisionRobot:
    def __init__(self, urdf):
        root = ET.parse(urdf).getroot()
        self.joints = {j.get("name"): j for j in root.findall("joint")}
        self.frames = {"base_Link": np.eye(4)}
        remaining = list(self.joints.values())
        while remaining:
            resolved = []
            for joint in remaining:
                parent = joint.find("parent").get("link")
                child = joint.find("child").get("link")
                if parent in self.frames:
                    self.frames[child] = self.frames[parent] @ origin_transform(
                        joint.find("origin")
                    )
                    resolved.append(joint)
            if not resolved:
                raise ValueError("URDF tree is disconnected from base_Link")
            remaining = [j for j in remaining if j not in resolved]

        wheel = self.frames["wheel_L_Link"][:3, 3]
        wheel_radius = float(root.find("link[@name='wheel_L_Link']/collision/geometry/cylinder").get("radius"))
        self.ground_shift = wheel_radius - wheel[2]
        for frame in self.frames.values():
            frame[2, 3] += self.ground_shift

        self.centers = {}
        for name, joint in self.joints.items():
            child = joint.find("child").get("link")
            self.centers[name] = self.frames[child][:3, 3].copy()

        # The load deck and IMU are fixed accessories outside the six variables.
        selected = {"base_Link"} | {
            f"{group}_{side}_Link" for group in GROUPS for side in "LR"
        }
        self.collisions = []
        for link in root.findall("link"):
            name = link.get("name")
            if name not in selected:
                continue
            for collision in link.findall("collision"):
                geometry = collision.find("geometry")
                box, cylinder = geometry.find("box"), geometry.find("cylinder")
                transform = self.frames[name] @ origin_transform(collision.find("origin"))
                if box is not None:
                    self.collisions.append(Collision(name, "box", transform, values(box, "size")))
                elif cylinder is not None:
                    self.collisions.append(Collision(name, "cylinder", transform, np.array([
                        float(cylinder.get("radius")), float(cylinder.get("length"))
                    ])))
                else:
                    raise ValueError(f"Unsupported collision primitive on {name}")

    def nominal_length(self, segment, side="L"):
        name = "knee" if segment == "thigh" else "wheel"
        xyz = values(self.joints[f"{name}_{side}_Joint"].find("origin"), "xyz")
        return float(np.linalg.norm(xyz[[0, 2]]))

    def effort(self, group, side="L"):
        return float(self.joints[f"{group}_{side}_Joint"].find("limit").get("effort"))

    def axis(self, group, side):
        joint = self.joints[f"{group}_{side}_Joint"]
        child = joint.find("child").get("link")
        axis = self.frames[child][:3, :3] @ values(joint.find("axis"), "xyz")
        return axis / np.linalg.norm(axis)


class Projection:
    def __init__(self, camera):
        self.view = np.asarray(camera, dtype=float)
        self.view /= np.linalg.norm(self.view)
        self.right = np.cross([0.0, 0.0, 1.0], self.view)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.view, self.right)

    def xy(self, points):
        return np.asarray(points) @ np.stack([self.right, self.up], axis=1)


def box_faces(size):
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ]) * size / 2
    ids = ([0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
           [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7])
    return [vertices[list(face)] for face in ids]


def cylinder_faces(radius, length, samples=64):
    theta = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    ring = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    low = np.column_stack([ring, np.full(samples, -length / 2)])
    high = np.column_stack([ring, np.full(samples, length / 2)])
    faces = [low[::-1], high]
    for i in range(samples):
        j = (i + 1) % samples
        faces.append(np.array([low[i], low[j], high[j], high[i]]))
    return faces


def collision_faces(collision):
    local = (box_faces(collision.dimensions) if collision.kind == "box"
             else cylinder_faces(*collision.dimensions))
    return [face @ collision.transform[:3, :3].T + collision.transform[:3, 3]
            for face in local]


def draw_collisions(ax, robot, projection, side=None):
    faces = []
    light_direction = np.array([-0.4, -0.8, 1.8])
    light_direction /= np.linalg.norm(light_direction)
    for collision in robot.collisions:
        if side and collision.link.endswith(f"_{'R' if side == 'L' else 'L'}_Link"):
            continue
        group = collision.link.split("_")[0]
        base_color = np.array(colors.to_rgb(COLLISION_COLORS[group]))
        for face in collision_faces(collision):
            normal = np.cross(face[1] - face[0], face[2] - face[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            if normal @ projection.view < 1e-5:
                continue
            shade = 0.87 + 0.12 * max(0.0, float(normal @ light_direction))
            depth = float(face.mean(axis=0) @ projection.view)
            faces.append((depth, projection.xy(face), np.clip(base_color * shade, 0, 1)))
    faces.sort(key=lambda face: face[0])
    collection = PolyCollection([face[1] for face in faces],
                                facecolors=[face[2] for face in faces],
                                edgecolors=[face[2] for face in faces],
                                linewidths=0.12, zorder=2)
    ax.add_collection(collection)


def draw_kinematics(ax, robot, projection, sides):
    for side in sides:
        points = [robot.centers[f"{group}_{side}_Joint"] for group in GROUPS]
        body_bottom = np.array([points[0][0], points[0][1],
                                robot.frames["base_Link"][2, 3] - 0.167])
        line = projection.xy(np.vstack([body_bottom, *points]))
        ax.plot(line[:, 0], line[:, 1], color="#8D9BA5", linewidth=1.2,
                linestyle=(0, (3.0, 2.4)), zorder=3)


def text(ax, x, y, value, size=13, color=INK, **kwargs):
    return ax.text(x, y, value, fontsize=size, color=color,
                   va="center", **kwargs)


def leader(ax, target, label, value, color=MUTED, size=11, align="left"):
    ax.annotate(value, xy=target, xytext=label, fontsize=size, color=color,
                ha=align, va="center", annotation_clip=False,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9,
                            "shrinkA": 6, "shrinkB": 6}, zorder=9)


def dimension(ax, start, end, offset, symbol, color, caption, label_position):
    start, end = np.asarray(start), np.asarray(end)
    direction = end - start
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]]) * np.sign(offset)
    a, b = start + normal * abs(offset), end + normal * abs(offset)
    for center, endpoint in ((start, a), (end, b)):
        ext = endpoint + normal * 0.016
        ax.plot([center[0], ext[0]], [center[1], ext[1]], color=color,
                linewidth=0.85, alpha=0.65, zorder=4)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<->", mutation_scale=11,
                                 linewidth=1.4, color=color, shrinkA=0,
                                 shrinkB=0, zorder=5))
    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    if angle < -90:
        angle += 180
    if angle > 90:
        angle -= 180
    midpoint = (a + b) / 2 + normal * 0.033
    text(ax, *midpoint, symbol, size=23, color=color, ha="center",
         rotation=angle, rotation_mode="anchor", zorder=7)
    text(ax, *label_position, caption, size=13, color=color, ha="center",
         fontweight="bold", zorder=7)


def draw_side(ax, robot, words):
    projection = Projection([0, -1, 0])
    draw_collisions(ax, robot, projection, side="L")
    draw_kinematics(ax, robot, projection, "L")
    hip, knee, wheel = [projection.xy(robot.centers[f"{group}_L_Joint"])
                        for group in ("hip", "knee", "wheel")]
    for a, b, color in ((hip, knee, THIGH_COLOR), (knee, wheel, SHANK_COLOR)):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.6, zorder=4)
    ax.scatter(*np.stack([hip, knee, wheel]).T, s=39, c="white",
               edgecolors=INK, linewidths=1.25, zorder=8)

    dimension(ax, hip, knee, -0.128, r"$s_t$", THIGH_COLOR,
              words["thigh"], (-0.405, 0.67))
    dimension(ax, knee, wheel, 0.147, r"$s_s$", SHANK_COLOR,
              words["shank"], (0.22, 0.405))
    leader(ax, hip, (0.135, 0.672), words["hip_center"])
    leader(ax, knee, (-0.355, 0.338), words["knee_center"], align="right")
    leader(ax, wheel, (0.172, 0.127), words["wheel_center"])

    ax.plot([-0.37, 0.32], [0, 0], color=LIGHT, linewidth=1.0, zorder=1)
    ax.annotate("", xy=(0.32, 0.043), xytext=(0.17, 0.043),
                arrowprops={"arrowstyle": "->", "lw": 1.1, "color": MUTED})
    text(ax, 0.246, 0.078, words["forward"], size=10, color=MUTED, ha="center")
    ax.set(xlim=(-0.51, 0.46), ylim=(-0.035, 1.0), aspect="equal")
    ax.axis("off")
    return projection


def draw_rotation_arrow(ax, robot, projection, group, side, radius=0.032):
    axis = robot.axis(group, side)
    up = np.array([0.0, 0.0, 1.0])
    up -= axis * np.dot(up, axis)
    up /= np.linalg.norm(up)
    transverse = np.cross(axis, up)
    start_angle = -np.pi / 2
    horizontal = float(projection.xy(transverse)[0])
    if group == "abad":
        start_angle = 0.0 if horizontal > 0 else np.pi
    elif group == "hip":
        start_angle = np.pi if horizontal > 0 else 0.0
    theta = np.linspace(start_angle, start_angle + np.pi, 96)
    center = robot.centers[f"{group}_{side}_Joint"]
    world = center + radius * (
        np.cos(theta)[:, None] * up + np.sin(theta)[:, None] * transverse
    )
    arrow = FancyArrowPatch(
        path=DrawingPath(projection.xy(world)), arrowstyle="-|>",
        mutation_scale=12, linewidth=1.8, color=GROUP_COLORS[group], zorder=7,
    )
    arrow.set_path_effects([
        patheffects.Stroke(linewidth=3.4, foreground="white"),
        patheffects.Normal(),
    ])
    ax.add_patch(arrow)


def draw_drives(ax, robot, words):
    projection = Projection([4.0, -3.0, 1.4])
    draw_collisions(ax, robot, projection)
    draw_kinematics(ax, robot, projection, "LR")
    for group in GROUPS:
        points = projection.xy([robot.centers[f"{group}_{side}_Joint"] for side in "LR"])
        for side in "LR":
            draw_rotation_arrow(ax, robot, projection, group, side)
        ax.scatter(*points.T, s=17, c=GROUP_COLORS[group], edgecolors="white",
                   linewidths=0.7, zorder=8)

    rows = {"abad": 0.75, "hip": 0.60, "knee": 0.385, "wheel": 0.155}
    for group, y in rows.items():
        target = projection.xy(robot.centers[f"{group}_L_Joint"])
        elbow_x = 0.265
        color = GROUP_COLORS[group]
        ax.plot([target[0] + 0.014, elbow_x, 0.31], [target[1], y, y],
                color=color, linewidth=1.05, zorder=5)
        text(ax, 0.33, y + 0.026, rf"$k_{{\mathrm{{{group}}}}}$",
             size=21, color=color, zorder=9)
        text(ax, 0.33, y - 0.028, words[group], size=12, color=MUTED, zorder=9)

    for side, key in (("L", "left"), ("R", "right")):
        center = projection.xy(robot.centers[f"wheel_{side}_Joint"])
        text(ax, center[0], -0.075, words[key], size=11, color=MUTED, ha="center")
    ax.set(xlim=(-0.27, 0.78), ylim=(-0.105, 1.0), aspect="equal")
    ax.axis("off")
    return projection


def configure_fonts(output_dir, extra_text=""):
    from fontTools import subset
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont

    font_dir = output_dir / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    corpus = string.printable + "".join(
        value for language in TEXT.values() for value in language.values()
    ) + extra_text
    family = "Codesign Noto Sans SC"
    # Convert the small CJK subset to TrueType to avoid CFF-in-CIDFontType2 PDFs.
    for style, weight in (("Regular", 400), ("Bold", 700)):
        source = Path(f"/usr/share/fonts/opentype/noto/NotoSansCJK-{style}.ttc")
        font = TTFont(str(source), fontNumber=2)
        subsetter = subset.Subsetter()
        subsetter.populate(text=corpus)
        subsetter.subset(font)
        glyph_set = font.getGlyphSet()
        glyphs = {}
        for name in font.getGlyphOrder():
            pen = TTGlyphPen(None)
            glyph_set[name].draw(Cu2QuPen(pen, max_err=1.0, reverse_direction=True))
            glyphs[name] = pen.glyph()
        builder = FontBuilder(font["head"].unitsPerEm, isTTF=True)
        builder.setupGlyphOrder(font.getGlyphOrder())
        builder.setupCharacterMap(font.getBestCmap())
        builder.setupGlyf(glyphs)
        builder.setupHorizontalMetrics(font["hmtx"].metrics)
        builder.setupHorizontalHeader(ascent=font["hhea"].ascent,
                                      descent=font["hhea"].descent)
        builder.setupNameTable({
            "familyName": family, "styleName": style,
            "uniqueFontIdentifier": family + style,
            "fullName": family + " " + style,
            "psName": "CodesignNotoSansSC-" + style,
        })
        builder.setupOS2(sTypoAscender=font["OS/2"].sTypoAscender,
                         sTypoDescender=font["OS/2"].sTypoDescender,
                         usWinAscent=font["OS/2"].usWinAscent,
                         usWinDescent=font["OS/2"].usWinDescent,
                         usWeightClass=weight)
        builder.setupPost()
        target = font_dir / f"CodesignNotoSansSC-{style}.ttf"
        builder.save(str(target))
        font_manager.fontManager.addfont(str(target))
        font.close()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
        "mathtext.fontset": "stix", "font.size": 13,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "axes.unicode_minus": False, "figure.facecolor": "white",
    })


def create_figure(robot, language):
    words = TEXT[language]
    fig = plt.figure(figsize=(14.4, 7.0), dpi=140)
    fig.text(0.05, 0.925, words["side_title"], fontsize=16, color=INK, fontweight="bold")
    fig.text(0.54, 0.925, words["drive_title"], fontsize=16, color=INK, fontweight="bold")
    left = fig.add_axes([0.025, 0.090, 0.475, 0.795])
    right = fig.add_axes([0.515, 0.090, 0.465, 0.795])
    draw_side(left, robot, words)
    draw_drives(right, robot, words)

    fig.text(0.7475, 0.043, words["shared"], fontsize=13, ha="center", color=INK)
    return fig


def validate_robot(robot):
    for segment in ("thigh", "shank"):
        if not np.isclose(robot.nominal_length(segment, "L"), robot.nominal_length(segment, "R")):
            raise ValueError(f"Left/right {segment} lengths are different")
    for group in GROUPS:
        if robot.effort(group, "L") != robot.effort(group, "R"):
            raise ValueError(f"Left/right {group} limits are different")
    if len(robot.collisions) != 9:
        raise ValueError("Expected one torso and eight leg/wheel collision primitives")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    configure_fonts(args.output_dir)
    robot = CollisionRobot(args.urdf)
    validate_robot(robot)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    languages = ("zh", "en") if args.language == "both" else (args.language,)
    for language in languages:
        fig = create_figure(robot, language)
        for suffix in ("png", "pdf", "svg"):
            target = args.output_dir / f"sixd_design_variables_{language}.{suffix}"
            fig.savefig(target, dpi=args.dpi, facecolor="white")
            print(target)
        plt.close(fig)
    manifest = {
        "source_urdf": str(args.urdf.resolve()),
        "source_urdf_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        "source_script": str(Path(__file__).resolve()),
        "matplotlib_version": matplotlib.__version__,
        "design_order": ["s_t", "s_s", "k_abad", "k_hip", "k_knee", "k_wheel"],
        "nominal_length_m": {s: robot.nominal_length(s) for s in ("thigh", "shank")},
        "nominal_peak_torque_nm": {g: robot.effort(g) for g in GROUPS},
        "length_convention": "Sagittal x-z joint-axis spacing; lateral y offsets are fixed",
        "joint_center_world_m": {k: v.tolist() for k, v in robot.centers.items()
                                 if k.endswith(("_L_Joint", "_R_Joint"))},
        "included_collision_links": [c.link for c in robot.collisions],
        "omitted_fixed_accessories": ["plate_link", "limx_imu"],
        "mass_model": "m_i(k_i) = m_i0 * k_i**alpha; formal alpha = 1",
        "collision_colors": COLLISION_COLORS,
        "actuator_markers": "Half-circle arrows in each joint's rotation plane; identical 0.032 m schematic radius",
        "rotation_direction": "Increasing angle around the positive URDF joint axis, following the right-hand rule; not a motion command",
        "visible_equations": False,
        "pose": "Nominal URDF zero pose; no inferred motion or optimized hardware",
        "languages": list(languages),
        "raster_size_px": [round(14.4 * args.dpi), round(7.0 * args.dpi)],
        "svg_text": "Glyph outlines for portable rendering; regenerate to edit labels",
        "pdf_fonts": "Embedded TrueType subsets; Simplified Chinese face converted from Noto CJK SC",
    }
    (args.output_dir / "geometry_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
