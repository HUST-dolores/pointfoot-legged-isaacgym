#!/usr/bin/env python3
"""Audit archived active rewards and draw their intended jump-shaping roles."""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

import fig0_sixd_design_variables as style
import fig03_four_capability_definitions as geometry
import fig10_jump_goal_takeoff as jump


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "out/jump_reward_purposes"
FIGURE_SIZE = (14.4, 9.2)
FONT_SIZE = 14
GROUPS = {
    "stand": "待命站稳",
    "crouch": "下蹲准备",
    "push": "蹬伸离地",
    "tuck": "上升收腿",
    "preland": "下降伸腿",
    "land": "触地缓冲",
    "outcome": "目标结果与密集反馈",
    "stability": "横向稳定与左右协调",
    "cost": "安全、存活与动作代价",
}

# One primary group per active term; phase descriptions retain cross-phase use.
CATALOGUE = [
    ("jump_stand", "stand", "待命且至少一轮触地", "机体速度、轮速、站高与直立度", "回到静止站姿", "乘积奖励，任一因素较差都会降低奖励"),
    ("jump_posture", "stand", "待命且至少一轮触地", "形态相关站高与直立度", "建立不塌陷的站立基础", "不直接约束速度"),
    ("jump_idle_low_height", "stand", "待命且至少一轮触地", "站高不足量", "惩罚低矮塌缩待命", "允许站高余量后才惩罚"),
    ("jump_stance", "stand", "待命且至少一轮触地", "机体系双轮前后差、高度差、中点偏移和轮距", "保持合理对称站姿", "几何奖励，不等于实测稳定裕度"),
    ("jump_ground_roll", "stand", "仅待命", "机体水平速度平方", "避免起跳前滚动漂移", "不惩罚蹬地阶段的必要前移"),
    ("jump_crouch_low", "crouch", "准备相前0.35秒", "相对触发时的机体下降量", "鼓励短时下蹲", "目标下降量0.08米；不是固定位置轨迹"),
    ("jump_crouch_extend", "crouch", "准备相、双轮触地且在推蹬时间窗", "机体下降、有效腿长增长和增长速度", "鼓励有下蹲基础的蹬伸", "同时保留当前下蹲量门控；有效腿长不是关节中心间实际杆长"),
    ("jump_crouch_overtime", "crouch", "准备相仍触地且超过起跳期限", "相位逾期时间", "避免长期蹲着不跳", "0.75秒后渐增，1.05秒为超时尺度"),
    ("jump_no_takeoff", "crouch", "准备相1.05秒仍未起飞的超时事件", "起跳超时标志", "惩罚未完成起跳", "事件惩罚，不是每步固定惩罚"),
    ("jump_push_up", "push", "准备相、双轮触地、0.10至0.80秒", "有效腿长增速和机体世界竖直速度", "让蹬伸带来向上运动", "含下蹲、伸展、居中门控；不能仅靠抬轮替代机体上升"),
    ("jump_push_vz_cmd", "push", "准备相、双轮触地、推蹬时间窗", "机体世界竖直速度与高度命令代理目标", "按高度命令形成竖直速度", "目标为0.8乘根号(2gh)；不是全身质心速度"),
    ("jump_push_fwd_cmd", "push", "准备相、双轮触地、推蹬时间窗", "机体系前向速度与距离命令代理目标", "在离地前建立前向速度", "按轮端命令借用弹道公式；不是完整动力学规划"),
    ("jump_takeoff_leg_extend", "push", "首次离地事件", "起跳质量Q", "要求真实蹬伸与机体抬升", "Q取伸展量、机体抬升、伸展速度、准备时间的最小质量，再乘横向质量"),
    ("jump_takeoff_vz", "push", "首次离地事件", "机体世界竖直速度", "校准离地竖直速度", "另乘侧倾和横向偏移门控，不直接乘Q"),
    ("jump_takeoff_angle", "push", "首次离地事件", "双轮中心平均速度相对竖直方向的出射角", "协调水平与竖直出射分量", "不是机体俯仰角；目标含经验增益0.75"),
    ("jump_takeoff_sym", "push", "整个准备相且双轮触地", "双轮竖直力差、左右腿长及轮端几何差", "鼓励两侧共同推蹬", "是准备相密集奖励，不是离地事件奖励"),
    ("jump_bad_takeoff", "push", "离地且Q低于0.20", "起跳质量不足", "抑制缺少蹬伸或横向失稳的起飞", "状态机触发后按1-Q惩罚"),
    ("jump_air_tuck", "tuck", "腾空且机体竖直速度非负", "相对起飞有效腿长的缩短量", "鼓励上升时收腿", "乘Q；缩腿会增加轮端净空，但不等于增加质心上升"),
    ("jump_upright_air", "tuck", "整个腾空相", "横滚方向的重力投影", "控制空中侧倾", "前跳分支允许矢状面俯仰，不要求完全竖直"),
    ("jump_wheel_spin", "tuck", "整个腾空相", "左右轮关节角速度平方", "减少空中车轮空转", "不惩罚蹬地接触时的轮转"),
    ("jump_preland_extend", "preland", "腾空且机体竖直速度为负", "相对飞行最短有效腿长的增长量", "下降时伸腿准备触地", "乘Q；不是保证脚端着地点的规划器"),
    ("jump_land_leg_sync", "preland", "下降腾空或已着陆", "双轮净空差和竖直接触力不平衡", "引导两轮接近同步着陆", "无接触时主要约束轮端高度差"),
    ("foot_landing_vel", "preland", "轮端近地、无接触且下降", "轮端向下速度平方", "降低近地轮端下降速度", "近地阈值0.08米；不是冲击力奖励"),
    ("jump_landing_compress", "land", "首次着陆事件", "下降相最大有效腿长减触地有效腿长", "引导触地时出现几何压缩", "乘Q；不是完整触地后压缩过程或冲量积分"),
    ("jump_land_brake", "land", "已着陆相", "机体速度和双轮转速", "落地后制动停稳", "不直接要求恢复站高；后续待命站姿奖励负责站高"),
    ("jump_relaunch", "land", "已着陆过且持续至少3步无接触", "再次腾空标志", "抑制落地后再次起跳", "过滤不足3步的短暂落地弹跳；是惩罚而非严格单跳保证"),
    ("jump_apex", "outcome", "首次着陆结算一次", "整次飞行的双轮最低净空峰值", "跟踪实际轮端跳高", "虽名apex，正式版本在首次着陆结算全飞行峰值；乘Q"),
    ("jump_height_error", "outcome", "与apex相同的首次结算事件", "峰值净空与高度命令的归一化平方误差", "惩罚跳高命令误差", "有上限且乘Q，不是质心高度误差"),
    ("jump_land_dist", "outcome", "首次着陆结算一次", "从离地到首次着陆的前向轮心中点位移", "跟踪实际飞行跳距", "按离地朝向投影，不计起跳前滚动；乘Q和长距离增益"),
    ("jump_dist_error", "outcome", "首次着陆事件", "飞行跳距与距离命令的归一化平方误差", "惩罚跳距命令误差", "有上限且乘Q"),
    ("jump_est_height", "outcome", "准备或腾空、限制轮上升且峰值标志未锁定", "较低轮净空加其向上速度的弹道峰值预测", "提供跳高的密集学习信号", "轮端会受收腿运动影响，预测不是真实质心弹道；未乘Q"),
    ("jump_est_dist", "outcome", "腾空且首次峰值标志未锁定", "轮心中点位置速度预测的着陆位移", "提供跳距的密集学习信号", "乘Q；峰值锁定后停止，防止重复获利"),
    ("jump_fwd_vel", "outcome", "腾空且首次峰值标志未锁定", "双轮中心平均速度沿离地朝向的分量", "避免只向上跳而缺少前进", "乘Q；目标按高度距离命令的代理滞空时间形成"),
    ("jump_lateral_stability", "stability", "待命与整个跳跃过程", "机体横向偏移比例、横滚和横滚偏航角速度", "减少侧倒和横向失稳", "横向偏移用机体原点B，不是全身质心投影"),
    ("jump_centered", "stability", "待命或准备相且有轮接触", "机体原点相对轮心中点的偏移和横滚", "鼓励起跳前支撑居中", "不是严格的质心支持域稳定性约束"),
    ("jump_abad_still", "stability", "所有相位", "侧摆关节角和角速度", "抑制过大侧摆及快速外摆", "URDF零位下的角度惩罚，不是任意姿态下竖直腿几何条件"),
    ("jump_leg_sync", "stability", "所有相位", "左右腿长、轮端几何、镜像关节角速度误差", "减少两侧不同步", "URDF镜像角满足左右相加为零，而非左右数值相同"),
    ("jump_mirror_pose", "stability", "所有相位", "七类左右镜像几何质量", "奖励左右协调姿态", "软奖励；不能当作强制对称投影"),
    ("jump_lateral_velocity", "stability", "整个跳跃过程，待命权重减半", "机体系侧向速度", "抑制向左或向右跳偏", "平方误差限幅"),
    ("jump_yaw_stability", "stability", "所有相位", "偏航角速度和双轮转速差", "抑制扭头和轮速不一致", "不是绝对航向角跟踪"),
    ("jump_lateral_drift", "stability", "首次着陆事件", "相对离地朝向的横向落点位移", "抑制跳跃落点侧偏", "不惩罚所需的前向飞行距离"),
    ("orientation", "stability", "所有相位，跳跃激活时改用横滚分量", "重力投影姿态误差", "站立时保持直立，跳跃时控制横滚", "激活跳跃后不惩罚矢状面俯仰"),
    ("ang_vel_xy", "stability", "所有相位，跳跃激活时仅横滚角速度", "机体横滚或横滚俯仰角速度", "抑制不必要的姿态晃动", "跳跃激活时放开俯仰角速度"),
    ("action_rate", "cost", "所有相位", "相邻动作差平方", "减少动作突变", "跳跃激活时正则权重放松至0.15"),
    ("action_smooth", "cost", "所有相位", "动作二阶差分平方", "减少动作抖动", "跳跃激活时正则权重放松至0.15"),
    ("torques", "cost", "所有相位", "输出关节力矩平方", "限制不必要的驱动代价", "不是能量积分；跳跃激活时权重放松至0.15"),
    ("dof_acc", "cost", "所有相位", "关节加速度平方", "抑制过激关节变化", "跳跃激活时正则权重放松至0.15"),
    ("dof_pos_limits", "cost", "关节超过软限位时", "超出软关节位置界限的幅值", "避免过度关节运动", "不是碰撞检测，也不是硬限位替代"),
    ("collision", "cost", "被惩罚连杆发生接触时", "指定连杆接触力超过阈值的数量", "避免非预期身体接触", "取决于配置指定的连杆，不惩罚所有接触"),
    ("keep_balance", "cost", "每个计算奖励的仿真步", "恒为1的存活项", "鼓励维持episode", "函数只返回1；名字不代表显式平衡误差"),
]

TEXT = {
    "zh": {
        "a": "(a)  相位相关的动作引导",
        "phases": ["待命站稳", "下蹲准备", "蹬伸离地", "上升收腿", "下降伸腿", "触地缓冲"],
        "roles": [
            ["恢复站高与站姿", "保持机体与双轮静止"],
            ["短时下蹲后蹬伸", "避免久蹲或不跳"],
            ["形成前向与竖直速度", "有效蹬伸、双侧推蹬"],
            ["缩短有效腿长", "减少侧倾与轮空转"],
            ["伸腿准备触地", "双轮同步、降低落速"],
            ["触地压缩、制动停稳", "抑制落地后再次起跳"],
        ],
        "b": "(b)  目标结果与密集反馈",
        "outcomes": [
            ["峰值净空", "跟踪高度命令", "惩罚实际高度误差"],
            ["飞行跳距", "跟踪距离命令", "惩罚实际距离误差"],
            ["预测与速度", "预测峰值和落点", "引导持续前向运动"],
        ],
        "c": "(c)  横向稳定与动作代价",
        "constraints": [
            ["左右协调", "镜像姿态与有效腿长", "支撑居中、约束侧摆"],
            ["侧向与航向", "减小侧移、横滚和偏航", "减小横向落点偏差"],
            ["安全与平滑", "力矩、加速度与动作变化", "关节限位、碰撞与存活"],
        ],
        "footer": "正式策略的奖励设计目的；姿态为示意，非固定参考轨迹或奖励效果的实测证明",
    },
    "en": {
        "a": "(a)  Phase-dependent motion shaping",
        "phases": ["Idle stance", "Crouch", "Push-off", "Ascent / tuck", "Descent / extend", "Touchdown"],
        "roles": [
            ["Restore stance", "Body / wheels at rest"],
            ["Crouch, then extend", "Avoid delay / no jump"],
            ["Build x / z speed", "Extend; push both legs"],
            ["Shorten effective leg", "Limit roll / wheel spin"],
            ["Extend before contact", "Align wheels; slow fall"],
            ["Compress; brake", "Avoid a second jump"],
        ],
        "b": "(b)  Outcomes and dense feedback",
        "outcomes": [
            ["Peak wheel clearance", "Track the height command", "Penalize achieved-height error"],
            ["Airborne travel", "Track the distance command", "Penalize achieved-distance error"],
            ["Prediction and speed", "Predict peak and landing position", "Encourage forward motion"],
        ],
        "c": "(c)  Lateral stability and action costs",
        "constraints": [
            ["Left-right coordination", "Mirror pose and effective leg length", "Center support; limit abduction"],
            ["Lateral / yaw stability", "Limit lateral speed, roll and yaw", "Reduce lateral landing drift"],
            ["Safety and smoothness", "Torque, acceleration and action changes", "Joint limits, collisions and survival"],
        ],
        "footer": "Intended roles of archived rewards; schematic poses, not fixed trajectories or measured reward effects",
    },
}
COLORS = ["#487FB7", "#B38A31", "#238B7B", "#85719C", "#238B7B", "#B34959"]
POSES = [(0.20, 0.10, 0.00), (0.425, 0.85, 0.00), (0.45, -0.20, 0.00),
         (0.425, 1.15, 0.18), (0.35, -0.10, 0.10), (0.425, 0.45, 0.00)]


def audited_records(cfg, source):
    documented_settings = {"jump_crouch_low_reward_s": 0.35, "jump_crouch_height_drop_target": 0.08,
                           "jump_takeoff_deadline_s": 0.75, "jump_crouch_timeout_s": 1.05,
                           "jump_push_start_s": 0.10, "jump_push_end_s": 0.80,
                           "jump_takeoff_vz_command_gain": 0.80, "jump_takeoff_quality_min": 0.20,
                           "jump_takeoff_angle_gain": 0.75, "jump_regularizer_relax": 0.15,
                           "about_landing_threshold": 0.08}
    for name, expected in documented_settings.items():
        if not np.isclose(cfg["rewards"][name], expected):
            raise ValueError(f"Update the archived-catalogue descriptions before using a new {name}")
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BipedWF")
    functions = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    active = {name: weight for name, weight in cfg["rewards"]["scales"].items() if weight != 0}
    names = [row[0] for row in CATALOGUE]
    if len(names) != len(set(names)) or set(names) != set(active):
        raise ValueError(f"Catalogue/config mismatch: {set(names) ^ set(active)}")
    records = []
    for name, group, phase, quantity, purpose, caveat in CATALOGUE:
        node = functions["_reward_" + name]
        records.append({"reward": name, "raw_config_scale": active[name],
                        "sign": "encourage" if active[name] > 0 else "penalize",
                        "primary_group": group, "primary_group_zh": GROUPS[group],
                        "phase_zh": phase, "quantity_zh": quantity, "purpose_zh": purpose,
                        "caveat_zh": caveat, "source_method": node.name,
                        "source_line": node.lineno, "source_end_line": node.end_lineno})
    inactive = sorted(name[len("_reward_"):] for name in functions
                      if name.startswith("_reward_jump_") and name[len("_reward_"):] not in active)
    return records, inactive, functions


def make_phase_scenes(robot):
    scenes = []
    lengths = []
    for hip, knee, clearance in POSES:
        angles = {"hip_L_Joint": hip, "knee_L_Joint": knee,
                  "hip_R_Joint": -hip, "knee_R_Joint": -knee}
        scene = geometry.make_scene(robot, angles)
        wheel = np.mean([scene.centers[f"wheel_{side}_Joint"] for side in "LR"], axis=0)
        bottom = min(jump.wheel_bottom(scene, side)[2] for side in "LR")
        geometry.translate_scene(scene, [-wheel[0], 0, clearance - bottom])
        base = scene.frames["base_Link"][:3, 3]
        lengths.append(float(np.mean([np.linalg.norm((scene.centers[f"wheel_{side}_Joint"] - base)[[0, 2]])
                                      for side in "LR"])))
        scenes.append(scene)
    if not (lengths[1] < lengths[2] and lengths[3] < lengths[2]
            and lengths[4] > lengths[3] and lengths[5] < lengths[4]):
        raise ValueError("Poses must illustrate crouch/push, tuck/extension and compression in that order")
    for scene, (hip, knee, _) in zip(scenes, POSES):
        for name, angle in (("hip_L_Joint", hip), ("knee_L_Joint", knee),
                            ("hip_R_Joint", -hip), ("knee_R_Joint", -knee)):
            limit = robot.joints[name].find("limit")
            if not float(limit.get("lower")) <= angle <= float(limit.get("upper")):
                raise ValueError(f"Illustrative pose outside {name} limits")
    return scenes, lengths


def check_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [obj for obj in fig.findobj(Text) if obj.get_visible() and obj.get_text()]
    boxes = [obj.get_window_extent(renderer) for obj in texts]
    for obj, box in zip(texts, boxes):
        if not np.isclose(obj.get_fontsize(), FONT_SIZE):
            raise ValueError("All labels must be 14 pt")
        if not all(fig.bbox.contains(*corner) for corner in box.get_points()):
            raise ValueError("Text outside page: " + obj.get_text())
    for i, first in enumerate(boxes):
        for j, second in enumerate(boxes[:i]):
            if first.padded(6).overlaps(second):
                raise ValueError("Overlapping labels: " + texts[i].get_text() + " / " + texts[j].get_text())
    return {"text_count": len(texts), "font_size_pt": FONT_SIZE, "text_overlap_pairs": 0,
            "all_text_within_export": True}


def create_figure(scenes, language):
    words = TEXT[language]
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=140)
    fig.text(0.04, 0.955, words["a"], fontsize=FONT_SIZE, color=style.INK)
    centers = np.linspace(0.11, 0.89, 6)
    for i, (center, scene) in enumerate(zip(centers, scenes)):
        fig.text(center, 0.90, words["phases"][i], fontsize=FONT_SIZE,
                 color=COLORS[i], ha="center")
        ax = fig.add_axes([center - 0.071, 0.656, 0.142, 0.215])
        ax.fill_between([-0.50, 0.50], -0.025, 0.0, color="#E6ECEF", zorder=0)
        ax.plot([-0.50, 0.50], [0, 0], color="#93A3AE", linewidth=0.85)
        geometry.draw_robot(ax, scene, style.Projection([0, -1, 0]), side="L")
        ax.set(xlim=(-0.55, 0.55), ylim=(-0.07, 1.30), aspect="equal")
        ax.axis("off")
        if i < 5:
            arrow_ax = fig.add_axes([center + 0.066, 0.74, 0.025, 0.05])
            jump.arrow(arrow_ax, (0.1, 0.5), (0.9, 0.5), style.MUTED)
            arrow_ax.set(xlim=(0, 1), ylim=(0, 1))
            arrow_ax.axis("off")
        for line, text in enumerate(words["roles"][i]):
            fig.text(center, 0.623 - line * 0.038, text, fontsize=FONT_SIZE,
                     color=style.INK, ha="center")
    for y in (0.548, 0.320):
        fig.add_artist(plt.Line2D([0.04, 0.96], [y, y], transform=fig.transFigure,
                                 color="#D1DADF", linewidth=0.8))
    for title_key, rows_key, title_y, row_y in (("b", "outcomes", 0.505, 0.452),
                                               ("c", "constraints", 0.276, 0.222)):
        fig.text(0.04, title_y, words[title_key], fontsize=FONT_SIZE, color=style.INK)
        for i, (center, rows) in enumerate(zip((0.195, 0.50, 0.805), words[rows_key])):
            for line, text in enumerate(rows):
                fig.text(center, row_y - line * 0.041, text, fontsize=FONT_SIZE,
                         color=COLORS[(i * 2) % len(COLORS)] if line == 0 else style.INK,
                         ha="center")
    fig.text(0.5, 0.055, words["footer"], fontsize=FONT_SIZE, color=style.MUTED, ha="center")
    return fig, check_layout(fig)


def validate_reward_gates(cfg, source):
    import torch

    cls = next(node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "BipedWF")
    cls.bases = []
    wanted = {"_reward_jump_air_tuck", "_reward_jump_preland_extend", "_reward_jump_landing_compress",
              "_reward_jump_relaunch", "_reward_jump_apex", "_reward_jump_height_error",
              "_reward_jump_land_dist", "_reward_jump_dist_error", "_reward_jump_upright_air",
              "_reward_jump_land_brake", "_jump_regularizer_weight", "_reward_keep_balance"}
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    dummy = namespace["BipedWF"]()

    def ns(value):
        return SimpleNamespace(**{key: ns(item) if isinstance(item, dict) else item
                                  for key, item in value.items()})

    dummy.cfg = ns(cfg)
    dummy.num_envs, dummy.device = 4, "cpu"
    dummy.actor_indices = torch.arange(4)
    dummy.root_states = torch.zeros(4, 13)
    dummy.root_states[:, 9] = torch.tensor([1.0, -1.0, 1.0, 1.0])
    dummy.jump_state = torch.tensor([2, 2, 3, 2])
    dummy.jump_takeoff_quality = torch.tensor([1.0, 1.0, 1.0, 0.0])
    dummy.jump_takeoff_leg_len = torch.full((4,), 0.70)
    dummy.jump_min_flight_leg_len = torch.full((4,), 0.40)
    dummy._jump_leg_length = lambda: torch.full((4,), 0.55)
    torch.testing.assert_close(dummy._reward_jump_air_tuck(), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(dummy._reward_jump_preland_extend(), torch.tensor([0.0, 1.0, 0.0, 0.0]))
    dummy.jump_land_fire = torch.tensor([True, False, True, True])
    dummy.jump_preland_leg_len = torch.full((4,), 0.70)
    dummy.jump_land_leg_len = torch.full((4,), 0.60)
    torch.testing.assert_close(dummy._reward_jump_landing_compress(), torch.tensor([1.0, 0.0, 1.0, 0.0]))
    dummy.contact_forces = torch.zeros(4, 2, 3)
    dummy.feet_indices = [0, 1]
    dummy.jump_landed_once = torch.tensor([True, True, False, True])
    dummy.jump_air_steps = torch.tensor([2, 3, 4, 4])
    dummy.contact_forces[3, 0, 2] = 2.0
    torch.testing.assert_close(dummy._reward_jump_relaunch(), torch.tensor([0.0, 1.0, 0.0, 0.0]))
    dummy.jump_peak_height = torch.full((4,), 0.12)
    dummy.jump_height_cmd = torch.full((4, 1), 0.12)
    dummy.jump_apex_fire = torch.tensor([True, False, True, True])
    torch.testing.assert_close(dummy._reward_jump_apex(), torch.tensor([1.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(dummy._reward_jump_height_error(), torch.zeros(4))
    dummy.jump_land_disp = torch.full((4,), 0.10)
    dummy.jump_dist_cmd = torch.full((4, 1), 0.10)
    dummy._jump_dist_long_gain = lambda: torch.ones(4)
    torch.testing.assert_close(dummy._reward_jump_land_dist(), torch.tensor([1.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(dummy._reward_jump_dist_error(), torch.zeros(4))
    dummy.projected_gravity = torch.tensor([[0.6, 0.0, -0.8], [0.0, 0.6, -0.8],
                                             [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    dummy.jump_state = torch.tensor([2, 2, 3, 2])
    torch.testing.assert_close(dummy._reward_jump_upright_air(), torch.tensor([1.0, 0.8, 0.0, 1.0]))
    dummy.jump_state = torch.tensor([0, 1, 2, 3])
    dummy.cmd_jump = torch.zeros(4, 1)
    torch.testing.assert_close(dummy._jump_regularizer_weight(), torch.tensor([1.0, 0.15, 0.15, 0.15]))
    torch.testing.assert_close(dummy._reward_keep_balance(), torch.ones(4))
    dummy.base_lin_vel = torch.zeros(4, 3)
    dummy.dof_vel = torch.zeros(4, 8)
    torch.testing.assert_close(dummy._reward_jump_land_brake(), torch.tensor([0.0, 0.0, 0.0, 1.0]))
    return {"scope": "Catalogue/source and isolated reward-gate consistency, not reward ablation or simulator performance",
            "all_active_rewards_mapped_once": True, "active_total": len(CATALOGUE),
            "active_jump_specific": sum(row[0].startswith("jump_") for row in CATALOGUE),
            "source_methods_exercised": sorted(wanted),
            "phase_event_quality_gates": "pass", "relaunch_three_step_filter": "pass",
            "airborne_pitch_not_penalized_as_roll": "pass", "regularizer_relaxation": "pass",
            "keep_balance_is_constant_survival_term": "pass"}


def write_audit(output, records, inactive, sources, lengths, layout, report, dpi):
    with (output / "active_reward_catalogue.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    mapping = {"active_rewards": records, "jump_methods_without_active_scale": inactive,
               "weight_semantics": "Raw configuration scales, not percentages or observed reward contributions"}
    (output / "reward_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 各项奖励与设计目的对照", "", "来源为本轮正式策略配置和归档环境源码。原始权重不代表贡献百分比。", ""]
    for group, label in GROUPS.items():
        lines.extend(["## " + label, "", "| 奖励项 | 原始权重 | 生效阶段 | 量与目的 | 边界说明 |", "|---|---:|---|---|---|"])
        for row in records:
            if row["primary_group"] == group:
                lines.append(f"| `{row['reward']}` | {row['raw_config_scale']:g} | {row['phase_zh']} | {row['quantity_zh']}；{row['purpose_zh']} | {row['caveat_zh']}；源代码第{row['source_line']}行 |")
        lines.append("")
    lines.extend(["## 有函数但未启用的跳跃项", "", ", ".join("`" + name + "`" for name in inactive),
                  "", "不把这些项列为正式策略已经采用的奖励。图上按目的合并，完整覆盖见CSV/JSON。", ""])
    (output / "奖励目的与权重对照.md").write_text("\n".join(lines))
    report["layout"] = layout
    (output / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    manifest = {"type": "Intended reward purposes for archived trajectory-disabled forward-jump policy",
                "measurements_or_ablation_results": False, "original_figures_overwritten": False,
                "source_files": {name: {"path": str(path.resolve()),
                                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                 for name, path in sources.items()},
                "active_reward_count": len(records), "active_jump_reward_count": 40,
                "inactive_jump_methods": inactive, "effective_length_definition": "Mean sagittal base-origin-to-wheel-center distance; not whole-CoM length or literal link length",
                "illustrative_symmetric_joint_poses_rad_and_wheel_clearance_m": POSES,
                "illustrative_effective_lengths_m": lengths, "pose_sequence_is_policy_rollout": False,
                "jump_use_explicit_trajectory": False, "normal_font_size_pt": FONT_SIZE,
                "raster_size_px": [round(value * dpi) for value in FIGURE_SIZE]}
    (output / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=jump.DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=jump.DEFAULT_CONFIG)
    parser.add_argument("--env-source", type=Path, default=jump.DEFAULT_ENV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    cfg = jump.load_policy(args.config)
    records, inactive, _ = audited_records(cfg, args.env_source)
    report = validate_reward_gates(cfg, args.env_source)
    robot = style.CollisionRobot(args.urdf)
    style.validate_robot(robot)
    scenes, lengths = make_phase_scenes(robot)
    corpus = "".join(str(words) for words in TEXT.values())
    style.configure_fonts(args.output_dir, extra_text=corpus)
    plt.rcParams.update({"font.size": FONT_SIZE, "font.weight": "normal"})
    layout = {}
    for language in ("zh", "en") if args.language == "both" else (args.language,):
        fig, layout[language] = create_figure(scenes, language)
        for extension in ("png", "pdf", "svg"):
            target = args.output_dir / f"jump_reward_purposes_{language}.{extension}"
            fig.savefig(target, dpi=args.dpi, facecolor="white")
            print(target)
        plt.close(fig)
    write_audit(args.output_dir, records, inactive,
                {"nominal_urdf": args.urdf, "archived_config": args.config, "archived_environment": args.env_source},
                lengths, layout, report, args.dpi)
    print(f"Audited {len(records)} active terms; originals unchanged; isolated-source tests passed")


if __name__ == "__main__":
    main()
