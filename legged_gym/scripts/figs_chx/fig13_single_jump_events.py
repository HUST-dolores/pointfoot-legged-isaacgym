#!/usr/bin/env python3
"""Single-jump event organization and reward gates from archived source."""

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
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text
import numpy as np
import torch

import fig0_sixd_design_variables as style


HERE = Path(__file__).resolve().parent
EXPERIMENT = Path("/home/xu/Desktop/codesign_worklog/nested_four_objective_sixd_2026-07-30/results/final_nested_bo")
DEFAULT_CONFIG = (HERE / "source_snapshot/jump_winner_env_cfg.json" if (HERE / "source_snapshot").is_dir()
                  else EXPERIMENT / "winner_policies/jump/seed_20260811/env_cfg.json")
DEFAULT_SOURCE = (HERE / "source_snapshot/wheelfoot_flat.py" if (HERE / "source_snapshot").is_dir()
                  else EXPERIMENT / "code_snapshot/legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py")
DEFAULT_OUTPUT = HERE / "out/single_jump_events"
FIGURE_SIZE = (14.4, 8.1)
FONT_SIZE = 14
LATCH_COLOR = "#487FB7"
AUX_COLOR = "#238B7B"
RESULT_COLOR = "#B38A31"
PENALTY_COLOR = "#B34959"
PROTECTION_COLOR = "#85719C"
TEXT = {
    "zh": {
        "a": "(a)  单次命令的事件与奖励门控",
        "b": "(b)  首次触地后的反弹判别：连续无接触控制步",
        "events": ["命令锁存", "蓄力", "离地", "机体顶点", "首次触地", "恢复／解锁"],
        "states": ["蓄力", "腾空", "着陆恢复"],
        "latch": "命令锁存位",
        "height_aux": "高度预测辅助",
        "distance_aux": "跳距／速度辅助",
        "outcome": "高度／距离结算",
        "close": "关闭预测／速度辅助",
        "once": "仅一次",
        "short": "短暂反弹",
        "long": "持续再次腾空",
        "counts": ["第 1 步", "第 2 步", "第 3 步起"],
        "not_penalized": "1–2 步：不触发再次起跳惩罚",
        "penalized": "≥3 步：再次起跳惩罚",
        "no_repeat": "已结算项不再发放；预测／速度辅助保持关闭",
        "footer": "事件顺序示意，非实测时长；任务组织针对完整单跳，不表示滚动—跳跃模式过渡",
    },
    "en": {
        "a": "(a)  Events and reward gates for one command",
        "b": "(b)  Post-contact bounce check: consecutive no-contact control steps",
        "events": ["Latch command", "Prepare", "Liftoff", "Body apex", "First contact", "Recover / release"],
        "states": ["CROUCH", "FLIGHT", "LANDED"],
        "latch": "Command latch",
        "height_aux": "Height-prediction aid",
        "distance_aux": "Distance / speed aids",
        "outcome": "Height / distance score",
        "close": "Prediction / speed aids off",
        "once": "Once only",
        "short": "Brief bounce",
        "long": "Sustained reflight",
        "counts": ["Step 1", "Step 2", "Step 3+"],
        "not_penalized": "1–2 steps: no relaunch penalty",
        "penalized": "3+ steps: relaunch penalty",
        "no_repeat": "No repeat event score; prediction / speed aids stay off",
        "footer": "Event-order schematic, not measured timing; complete single-jump task, not rolling-to-jumping transitions",
    },
}


def namespace(value):
    return SimpleNamespace(**{key: namespace(item) if isinstance(item, dict) else item
                              for key, item in value.items()})


class SourceFixture:
    """Inject contacts/states into actual archived functions, not a simulator."""

    def __init__(self, cfg, source):
        tree = ast.parse(source.read_text())
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BipedWF")
        methods = {node.name: node for node in original.body if isinstance(node, ast.FunctionDef)}
        names = {"_update_jump_command", "_jump_takeoff_quality", "_jump_phase_elapsed",
                 "_jump_dist_long_gain", "_reward_jump_apex", "_reward_jump_height_error",
                 "_reward_jump_land_dist", "_reward_jump_dist_error", "_reward_jump_est_height",
                 "_reward_jump_est_dist", "_reward_jump_fwd_vel", "_reward_jump_relaunch",
                 "_reward_jump_air_tuck", "_reward_jump_preland_extend"}
        self.method_lines = {name: methods[name].lineno for name in sorted(names)}

        def quat_apply(q, v):
            xyz = q[:, :3]
            return v + 2 * q[:, 3:4] * torch.cross(xyz, v, dim=-1) + 2 * torch.cross(xyz, torch.cross(xyz, v, dim=-1), dim=-1)

        scope = {"torch": torch, "quat_apply": quat_apply}
        cls = ast.ClassDef(name="ArchivedJumpTest", bases=[], keywords=[],
                           body=[methods[name] for name in sorted(names)], decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
        exec(compile(module, str(source), "exec"), scope)
        self.env = scope["ArchivedJumpTest"]()
        env = self.env
        env.num_envs, env.device, env.cfg = 1, "cpu", namespace(cfg)
        env.dt = cfg["sim"]["dt"] * cfg["control"]["decimation"]
        env.feet_indices = [0, 1]
        allocations = [node for node in ast.walk(methods["_init_buffers"])
                       if isinstance(node, ast.Assign) and len(node.targets) == 1
                       and isinstance(node.targets[0], ast.Attribute)
                       and isinstance(node.targets[0].value, ast.Name)
                       and node.targets[0].value.id == "self"
                       and (node.targets[0].attr.startswith("jump_")
                            or node.targets[0].attr in ("cmd_jump", "jumped_flag"))]
        for allocation in sorted(allocations, key=lambda node: node.lineno):
            exec(compile(ast.Module(body=[allocation], type_ignores=[]), str(source), "exec"), {"self": env, "torch": torch})
        env.episode_length_buf = torch.zeros(1, dtype=torch.long)
        env.actor_indices = torch.tensor([0])
        env.root_states = torch.zeros(1, 13)
        env.base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        env.forward_vec = torch.tensor([[1.0, 0.0, 0.0]])
        env.base_lin_vel = torch.zeros(1, 3)
        env.base_ang_vel = torch.zeros(1, 3)
        env.dof_pos = torch.zeros(1, 8)
        env.contact_forces = torch.zeros(1, 2, 3)
        env.foot_positions = torch.tensor([[[0.0, 0.165, 0.15], [0.0, -0.165, 0.15]]])
        env.foot_velocities = torch.zeros(1, 2, 3)
        env.foot_heights = torch.zeros(1, 2)
        env.base_height = torch.full((1,), cfg["rewards"]["base_height_target"])
        env._base_height_target_tensor = lambda: torch.full((1,), cfg["rewards"]["base_height_target"])
        env._test_leg_length = 0.65
        env._jump_leg_length = lambda: torch.full((1,), env._test_leg_length)
        env._jump_leg_lengths = lambda: torch.full((1, 2), env._test_leg_length)
        env._jump_lateral_metrics = lambda: (torch.zeros(1), torch.zeros(1), torch.zeros(1))
        env._jump_symmetry_errors = lambda: tuple(torch.zeros(1) for _ in range(7))
        env.jump_trigger_time[:] = env.dt
        env.jump_command_override = torch.tensor([[0.13, 0.10]])
        self.rows = []

    def step(self, scenario, event, contact=True, vz=0.0, clearance=0.0,
             height_delta=0.0, leg=0.65, speed=0.4, x=0.0):
        env = self.env
        env.episode_length_buf += 1
        env.contact_forces[:] = 0
        if contact:
            env.contact_forces[:, :, 2] = 20
        env.root_states[:, 9] = vz
        env.base_lin_vel[:] = 0
        env.base_lin_vel[:, 0] = speed
        env.base_height[:] = env.cfg.rewards.base_height_target + height_delta
        env._test_leg_length = leg
        env.foot_positions[:, :, 0] = x
        env.foot_positions[:, :, 2] = 0.15 + clearance
        env.foot_heights[:] = clearance
        env.foot_velocities[:] = 0
        env.foot_velocities[:, :, 0] = 0.35
        env.foot_velocities[:, :, 2] = vz
        env._update_jump_command()
        row = {"scenario": scenario, "injected_event": event,
               "test_control_step": int(env.episode_length_buf.item()),
               "status": "injected_source_unit_test_not_physical_rollout",
               "state": int(env.jump_state.item()), "command_latched": bool(env.cmd_jump.item()),
               "height_command": float(env.jump_height_cmd.item()), "distance_command": float(env.jump_dist_cmd.item()),
               "apex_flag_locked": bool(env.jumped_flag.item()),
               "height_settlement_pulse": bool(env.jump_apex_fire.item()),
               "distance_settlement_pulse": bool(env.jump_land_fire.item()),
               "landed_once": bool(env.jump_landed_once.item()), "air_steps": int(env.jump_air_steps.item()),
               "recorded_peak_wheel_clearance": float(env.jump_peak_height.item())}
        for short, method in (("height_score", "_reward_jump_apex"), ("distance_score", "_reward_jump_land_dist"),
                              ("height_aid", "_reward_jump_est_height"), ("distance_aid", "_reward_jump_est_dist"),
                              ("speed_aid", "_reward_jump_fwd_vel"), ("relaunch_indicator", "_reward_jump_relaunch"),
                              ("tuck_reward", "_reward_jump_air_tuck"), ("preland_reward", "_reward_jump_preland_extend")):
            row[short] = float(getattr(env, method)().item())
        self.rows.append(row)
        return row

    def fly_to_first_contact(self, scenario, cross_apex=True):
        self.step(scenario, "latch_command")
        self.env.jump_command_override[:] = torch.tensor([[0.16, 0.14]])
        self.step(scenario, "prepare", height_delta=-0.08, leg=0.55, vz=0.1)
        self.step(scenario, "prepare_hold", height_delta=-0.05, leg=0.58, vz=0.2)
        self.step(scenario, "liftoff", contact=False, height_delta=0.02, leg=0.68, vz=1.0, clearance=0.02)
        self.step(scenario, "ascent", contact=False, height_delta=0.10, leg=0.62, vz=0.6, clearance=0.10, x=0.03)
        if cross_apex:
            self.step(scenario, "body_apex", contact=False, height_delta=0.12, leg=0.58, vz=0.0, clearance=0.12, x=0.06)
            self.step(scenario, "wheel_peak_after_body_apex", contact=False, height_delta=0.10,
                      leg=0.54, vz=-0.2, clearance=0.16, x=0.08)
            self.step(scenario, "descent_extension", contact=False, height_delta=0.06,
                      leg=0.66, vz=-0.4, clearance=0.08, x=0.09)
        self.step(scenario, "first_touchdown", contact=True, vz=-0.5, clearance=0.0, x=0.10)


def validate_events(cfg, source):
    fixtures = []
    nominal = SourceFixture(cfg, source)
    nominal.fly_to_first_contact("normal_recovery")
    contact = nominal.rows[-1]
    assert contact["height_settlement_pulse"] and contact["distance_settlement_pulse"]
    assert contact["height_score"] > 0 and contact["distance_score"] > 0
    assert np.isclose(contact["recorded_peak_wheel_clearance"], 0.16)
    for row in nominal.rows:
        assert np.isclose(row["height_command"], 0.13) and np.isclose(row["distance_command"], 0.10)
    before = next(row for row in nominal.rows if row["injected_event"] == "ascent")
    apex = next(row for row in nominal.rows if row["injected_event"] == "body_apex")
    takeoff = next(row for row in nominal.rows if row["injected_event"] == "liftoff")
    descent = next(row for row in nominal.rows if row["injected_event"] == "descent_extension")
    assert all(before[key] > 0 for key in ("height_aid", "distance_aid", "speed_aid"))
    assert not apex["height_settlement_pulse"]
    assert takeoff["state"] == 2 and takeoff["air_steps"] == 1
    assert descent["preland_reward"] > 0
    assert all(apex[key] == 0 for key in ("height_aid", "distance_aid", "speed_aid"))
    nominal.step("normal_recovery", "landed_hold")
    recovery = nominal.step("normal_recovery", "recover_release", speed=0.0)
    assert recovery["state"] == 0 and not recovery["command_latched"] and recovery["apex_flag_locked"]
    nominal.env.jump_trigger_time[:] = (nominal.env.episode_length_buf.item() + 1) * nominal.env.dt
    fresh = nominal.step("normal_recovery", "new_command_rearms", vz=0.1)
    assert fresh["state"] == 1 and fresh["command_latched"] and not fresh["apex_flag_locked"]
    assert np.isclose(fresh["height_command"], 0.16)
    fixtures.append(nominal)

    for count in (1, 2, 3, 4):
        fixture = SourceFixture(cfg, source)
        name = f"postcontact_no_contact_{count}_steps"
        fixture.fly_to_first_contact(name)
        for i in range(count):
            row = fixture.step(name, f"reflight_{i + 1}", contact=False, vz=0.5, clearance=0.02, x=0.11)
            assert row["air_steps"] == i + 1
            assert bool(row["relaunch_indicator"]) == (i + 1 >= 3)
            assert all(row[key] == 0 for key in ("height_aid", "distance_aid", "speed_aid", "height_score", "distance_score"))
            assert row["state"] == (3 if i < 2 else 2)
        returned = fixture.step(name, "contact_again", speed=0.4, x=0.12)
        assert returned["air_steps"] == 0 and returned["relaunch_indicator"] == 0
        fixture.step(name, "hold_after_bounce", speed=0.4, x=0.12)
        assert sum(row["height_settlement_pulse"] for row in fixture.rows) == 1
        assert sum(row["distance_settlement_pulse"] for row in fixture.rows) == 1
        fixtures.append(fixture)

    fallback = SourceFixture(cfg, source)
    fallback.fly_to_first_contact("no_apex_crossing_before_contact", cross_apex=False)
    assert fallback.rows[-1]["height_settlement_pulse"] and fallback.rows[-1]["apex_flag_locked"]
    fixtures.append(fallback)

    timeout = SourceFixture(cfg, source)
    timeout.fly_to_first_contact("landed_timeout_release")
    for _ in range(int(np.ceil((cfg["rewards"]["jump_landed_timeout_s"] + 0.04) / timeout.env.dt))):
        row = timeout.step("landed_timeout_release", "hold_not_settled", speed=0.8, height_delta=-0.2)
        if row["state"] == 0:
            break
    assert row["state"] == 0 and not row["command_latched"]
    fixtures.append(timeout)

    for fixture in fixtures:
        assert sum(row["height_settlement_pulse"] for row in fixture.rows) == 1
        assert sum(row["distance_settlement_pulse"] for row in fixture.rows) == 1
    report = {"scope": "Actual archived event/reward functions with injected tensors; not simulator/policy performance",
              "scenario_count": len(fixtures), "single_height_and_distance_settlement_per_command": "pass",
              "command_target_not_resampled_while_latched": "pass", "post_body_apex_wheel_peak_settled_at_first_contact": "pass",
              "height_distance_speed_aids_close_at_body_apex": "pass",
              "other_phase_rewards_not_all_closed_at_apex": "pass",
              "first_takeoff_not_subject_to_relaunch_three_step_filter": "pass",
              "contact_resets_consecutive_air_counter": "pass",
              "one_two_step_bounce_no_relaunch_penalty": "pass", "third_no_contact_step_onward_penalized": "pass",
              "no_duplicate_event_score_after_sustained_reflight": "pass", "no_aux_rearming_after_bounce": "pass",
              "contact_locks_flag_without_detected_apex": "pass", "only_new_command_rearms_flag": "pass",
              "timeout_release_does_not_prove_recovery": "pass",
              "nominal_control_dt_s": cfg["sim"]["dt"] * cfg["control"]["decimation"],
              "physical_contact_dynamics_exercised": False, "termination_loop_exercised": False,
              "stubs": ["Identity base quaternion", "Pre-set mirrored effective lengths and zero symmetry/lateral errors",
                        "Injected contact force, velocity, clearance and leg length, not a simulated state trajectory"],
              "source_method_lines": nominal.method_lines}
    return report, [row for fixture in fixtures for row in fixture.rows]


def arrow(ax, start, end, color, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                 color=color, linewidth=1.3, linestyle=(0, (4, 3)) if dashed else "-",
                                 shrinkA=0, shrinkB=0, zorder=5))


def band(ax, left, right, y, color, height=0.17):
    ax.add_patch(Rectangle((left, y - height / 2), right - left, height,
                           facecolor=color, edgecolor="none", zorder=3))


def create_figure(language):
    words = TEXT[language]
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=140)
    fig.text(0.04, 0.945, words["a"], fontsize=FONT_SIZE, color=style.INK)
    ax = fig.add_axes([0.025, 0.37, 0.95, 0.53])
    events = np.array([3.15, 4.55, 6.0, 7.60, 9.30, 11.05])
    command, _, liftoff, apex, contact, release = events
    for x, label in zip(events, words["events"]):
        ax.text(x, 5.80, label, ha="center", fontsize=FONT_SIZE, color=style.INK)
        ax.plot([x, x], [0.35, 5.43], color="#D8E0E5", linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    arrow(ax, (command - 0.22, 5.35), (release + 0.35, 5.35), style.MUTED)
    ax.scatter(events, np.full(6, 5.35), s=20, color=style.MUTED, zorder=6)
    for left, right, label, color in ((command, liftoff, words["states"][0], "#F0E7D3"),
                                      (liftoff, contact, words["states"][1], "#DDECE8"),
                                      (contact, release, words["states"][2], "#E5E1EB")):
        band(ax, left, right, 4.72, color, height=0.40)
        ax.text((left + right) / 2, 4.72, label, fontsize=FONT_SIZE, color=style.INK,
                ha="center", va="center")
    rows = [("latch", 3.85), ("height_aux", 2.90), ("distance_aux", 1.95), ("outcome", 0.80)]
    for key, y in rows:
        ax.text(0.12, y, words[key], fontsize=FONT_SIZE, color=style.INK, va="center")
        ax.plot([command - 0.15, release + 0.27], [y, y], color="#D0DADF", linewidth=1.0, zorder=1)
    ax.plot([command - 0.15, command, command, release, release, release + 0.28],
             [3.85, 3.85, 4.10, 4.10, 3.85, 3.85], color=LATCH_COLOR, linewidth=1.7)
    band(ax, command, apex, 2.90, AUX_COLOR)
    band(ax, liftoff, apex, 1.95, AUX_COLOR)
    for y in (2.90, 1.95):
        ax.scatter([apex], [y], s=30, facecolors="white", edgecolors=AUX_COLOR, linewidths=1.3, zorder=5)
    ax.text(apex + 0.25, 2.42, words["close"], fontsize=FONT_SIZE, color=AUX_COLOR, va="center")
    ax.plot([contact, contact], [0.80, 1.22], color=RESULT_COLOR, linewidth=2.2, zorder=6)
    ax.scatter([contact], [1.22], marker="D", s=40, color=RESULT_COLOR, zorder=7)
    ax.text(contact + 0.28, 1.22, words["once"], fontsize=FONT_SIZE, color=RESULT_COLOR, va="center")
    ax.set(xlim=(0, 11.65), ylim=(0.25, 6.25))
    ax.axis("off")

    fig.text(0.04, 0.335, words["b"], fontsize=FONT_SIZE, color=style.INK)
    ax2 = fig.add_axes([0.04, 0.105, 0.92, 0.19])
    for i, label in enumerate(words["counts"]):
        ax2.text(2.55 + 0.72 * i, 2.32, label, fontsize=FONT_SIZE, ha="center", color=style.MUTED)
    ax2.text(0, 1.60, words["short"], fontsize=FONT_SIZE, color=PROTECTION_COLOR, va="center")
    ax2.text(0, 0.56, words["long"], fontsize=FONT_SIZE, color=PENALTY_COLOR, va="center")
    for y, long in ((1.60, False), (0.56, True)):
        for i in range(3):
            color = PENALTY_COLOR if long and i >= 2 else PROTECTION_COLOR
            if not long and i >= 2:
                ax2.plot([2.20 + i * 0.72, 2.80 + i * 0.72], [y, y], color="#B5C3CB", linewidth=1.0)
            else:
                band(ax2, 2.25 + i * 0.72, 2.85 + i * 0.72, y, color, height=0.24)
        if long:
            arrow(ax2, (4.48, y), (4.78, y), PENALTY_COLOR)
    ax2.text(4.90, 1.60, words["not_penalized"], fontsize=FONT_SIZE, color=PROTECTION_COLOR, va="center")
    ax2.text(4.90, 0.56, words["penalized"], fontsize=FONT_SIZE, color=PENALTY_COLOR, va="center")
    ax2.set(xlim=(0, 12.0), ylim=(0.02, 2.75))
    ax2.axis("off")
    fig.text(0.5, 0.075, words["no_repeat"], fontsize=FONT_SIZE, color=style.INK, ha="center")
    fig.text(0.5, 0.025, words["footer"], fontsize=FONT_SIZE, color=style.MUTED, ha="center")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [obj for obj in fig.findobj(Text) if obj.get_visible() and obj.get_text()]
    boxes = [obj.get_window_extent(renderer) for obj in texts]
    for obj, box in zip(texts, boxes):
        if not np.isclose(obj.get_fontsize(), FONT_SIZE):
            raise ValueError("Every label must use 14 pt")
        if not all(fig.bbox.contains(*corner) for corner in box.get_points()):
            raise ValueError("Label outside export: " + obj.get_text())
    for i, box in enumerate(boxes):
        for j, other in enumerate(boxes[:i]):
            if box.padded(5).overlaps(other):
                raise ValueError("Overlapping labels: " + texts[i].get_text() + " / " + texts[j].get_text())
    return fig, {"text_count": len(texts), "font_size_pt": FONT_SIZE,
                 "all_text_within_export": True, "text_overlap_pairs": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if not (cfg["env"]["use_jump"] and cfg["env"]["use_jump_distance"]):
        raise ValueError("Figure requires the archived forward-jump task")
    if cfg["rewards"]["scales"]["jump_relaunch"] >= 0:
        raise ValueError("The relaunch term must have a negative scale")
    report, rows = validate_events(cfg, args.env_source)
    style.configure_fonts(args.output_dir, extra_text="".join(str(words) for words in TEXT.values()))
    plt.rcParams.update({"font.size": FONT_SIZE, "font.weight": "normal"})
    layout = {}
    for language in ("zh", "en") if args.language == "both" else (args.language,):
        fig, layout[language] = create_figure(language)
        for extension in ("png", "pdf", "svg"):
            target = args.output_dir / f"single_jump_events_{language}.{extension}"
            fig.savefig(target, dpi=args.dpi, facecolor="white")
            print(target)
        plt.close(fig)
    with (args.output_dir / "injected_source_unit_tests.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report["layout"] = layout
    (args.output_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {"figure_type": "Archived RL task organization/event schematic", "measured_policy_rollout": False,
                "event_positions_are_durations": False, "rolling_jumping_transition_study": False,
                "sources": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                            for name, path in {"archived_config": args.config, "archived_environment": args.env_source}.items()},
                "event_order": ["command_latch", "prepare", "liftoff", "body_vz_zero_crossing", "first_contact", "recovery_or_timeout_release"],
                "height_settlement": "Once at first contact for the entire flight lower-wheel peak, not at body apex",
                "auxiliary_bands": "Eligibility windows, not constant reward values; height aid also requires upward limiting-wheel velocity; distance/speed aids require flight and quality",
                "bounce_guard": "After first contact, 1-2 consecutive no-contact control updates do not trigger relaunch penalty; from step 3 onward the relaunch predicate is true",
                "bounce_guard_is_only_for_relaunch_not_first_liftoff": True,
                "recovery_flag_not_an_empirical_success_label": True,
                "new_command_resets_one_jump_flags": True,
                "relaunch_raw_scale": cfg["rewards"]["scales"]["jump_relaunch"],
                "relaunch_terminal_config": cfg["rewards"].get("jump_relaunch_terminal", False),
                "terminal_behavior_in_figure": "Not depicted or validated; shared base termination loop outside archived environment snapshot",
                "control_dt_s": cfg["sim"]["dt"] * cfg["control"]["decimation"],
                "font_size_pt": FONT_SIZE, "raster_size_px": [round(args.dpi * value) for value in FIGURE_SIZE]}
    (args.output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "source_method_lines"}, indent=2))


if __name__ == "__main__":
    main()
