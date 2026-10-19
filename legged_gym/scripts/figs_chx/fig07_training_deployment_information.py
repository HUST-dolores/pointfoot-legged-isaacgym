#!/usr/bin/env python3
"""Extend the supplied estimation/control graph with a training-only region."""

import argparse
import ast
from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as DrawingPath
from matplotlib.text import Text
import numpy as np
import torch

import fig0_sixd_design_variables as style


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PACKAGED = HERE / "source_snapshot"
SOURCE = (PACKAGED if PACKAGED.is_dir() else
          HERE / "out/paper_picture_collection_2026-10-14/06_解析先验残差修正与融合估计/source_snapshot")
DEFAULT_ENCODER = (PACKAGED / "mlp_encoder.py" if PACKAGED.is_dir()
                   else REPO / "legged_gym/algorithm/mlp_encoder.py")
DEFAULT_OUTPUT = HERE / "out/training_deployment_information"
FIGURE_SIZE = (16.0, 11.4)
FONT_SIZE = 14
CATEGORIES = ("measurement", "analytic", "estimator", "control", "training")
COLORS = {"measurement": ("#487FB7", "#EEF4FA"), "analytic": ("#AB842F", "#FBF5E7"),
          "estimator": ("#238B7B", "#EDF6F2"), "control": ("#85719C", "#F3F0F7"),
          "training": ("#B34959", "#FBF0F2")}
TEXT = {
    "zh": {
        "online_title": "(a)  部署／在线闭环（训练时共享）",
        "training_title": "(b)  仅训练：监督与策略更新",
        "measurement": "本体测量与动作缓存\n姿态、关节、力矩\n上一时刻动作",
        "analytic": "解析负载估计\n平台标定与归一化",
        "history": "原始与因果滤波观测\n10帧历史窗口",
        "encoder": "GRU时序编码\n隐状态128维",
        "load_head": "负载分支\n质量／质心偏移残差",
        "velocity_head": "速度分支\n机体线速度估计",
        "fusion": "当前解析先验＋残差\n显式负载表征（4维）",
        "actor": "运动策略Actor\n当前观测＋任务命令\n估计状态共7维",
        "plant": "动作执行与测量反馈\n机器人、负载及动作缓存",
        "commands": "任务命令",
        "truth": "仿真特权标签\n速度、质量与质心偏移",
        "supervised": "速度／残差监督更新\nGRU与双头参数",
        "rollout": "Rollout、奖励与终止\n特权Critic与优势",
        "ppo": "PPO策略／价值更新\n仅更新Actor／Critic",
        "prior_features": "物理先验特征",
        "current_prior": "当前解析先验",
        "current_observation": "当前观测旁路",
        "estimates": "估计状态7维",
        "action": "动作",
        "privileged_critic": "特权评价输入",
        "legend": ["测量与处理", "解析先验", "学习估计与融合", "策略与执行", "训练专用"],
        "caption": "图 7  训练与部署的信息路径。实线为在线信息流，虚线仅用于训练数据或参数更新。\n负载真值仅供训练监督或测试评估；部署闭环不需要负载真值，部署时网络参数固定。",
    },
    "en": {
        "online_title": "(a)  Deployment / online loop (also used in training)",
        "training_title": "(b)  Training only: supervision and policy updates",
        "measurement": "Proprioceptive inputs\nAttitude, joints, torque\nPrevious action",
        "analytic": "Analytical payload estimate\nCalibration and normalization",
        "history": "Raw / causal-filtered inputs\n10-frame history window",
        "encoder": "GRU temporal encoder\nHidden state: 128D",
        "load_head": "Payload branch\nMass / CoM-shift residuals",
        "velocity_head": "Velocity branch\nBody linear-velocity estimate",
        "fusion": "Current prior + residual\nExplicit payload state (4D)",
        "actor": "Motion policy: actor\nCurrent inputs + commands\n7D estimated state",
        "plant": "Actuation and feedback\nRobot, payload, action cache",
        "commands": "Task commands",
        "truth": "Privileged simulation labels\nVelocity, mass and CoM shift",
        "supervised": "Velocity / residual supervision\nGRU and dual-head updates",
        "rollout": "Rollout, reward, termination\nPrivileged critic / advantages",
        "ppo": "PPO policy / value updates\nActor / critic parameters only",
        "prior_features": "Physics-derived features",
        "current_prior": "Current analytical prior",
        "current_observation": "Current inputs",
        "estimates": "7D estimated state",
        "action": "Action",
        "privileged_critic": "Privileged critic inputs",
        "legend": ["Measurements", "Analytical prior", "Estimation / fusion", "Policy / execution", "Training only"],
        "caption": "Figure 7. Training and deployment information paths. Solid: online signals; dashed: training data or parameter updates.\nGround-truth payload labels support supervision or test evaluation, not deployment inputs; deployment weights are fixed.",
    },
}


@dataclass
class Node:
    name: str
    x: float
    y: float
    width: float
    height: float
    category: str

    @property
    def bounds(self):
        return (self.x-self.width/2, self.y-self.height/2, self.x+self.width/2, self.y+self.height/2)


NODES = [
    Node("measurement", 1.5, 6.15, 2.5, 1.15, "measurement"),
    Node("analytic", 4.7, 8.35, 3.0, .90, "analytic"),
    Node("history", 4.7, 6.15, 3.0, .90, "measurement"),
    Node("encoder", 8.25, 6.15, 2.5, .90, "estimator"),
    Node("load_head", 11.75, 6.65, 3.0, .85, "estimator"),
    Node("velocity_head", 11.75, 5.15, 3.0, .85, "estimator"),
    Node("fusion", 11.75, 8.35, 3.0, .90, "estimator"),
    Node("actor", 8.4, 3.65, 3.1, 1.0, "control"),
    Node("plant", 4.4, 3.65, 3.0, .95, "control"),
    Node("commands", 12.25, 3.65, 3.0, .85, "control"),
    Node("truth", 2.0, 1.2, 3.1, .90, "training"),
    Node("supervised", 6.0, 1.2, 3.1, .90, "training"),
    Node("rollout", 10.0, 1.2, 3.1, .90, "training"),
    Node("ppo", 14.0, 1.2, 3.1, .90, "training"),
]
EDGES = [
    ("measurement", "analytic", False, [(2.75,6.15),(2.93,6.15),(2.93,8.35),(3.2,8.35)]),
    ("measurement", "history", False, [(2.75,6.15),(3.2,6.15)]),
    ("analytic", "history", False, [(4.7,7.90),(4.7,6.60)]),
    ("analytic", "fusion", False, [(6.2,8.35),(10.25,8.35)]),
    ("history", "encoder", False, [(6.2,6.15),(7.0,6.15)]),
    ("encoder", "load_head", False, [(9.5,6.15),(9.9,6.15),(9.9,6.65),(10.25,6.65)]),
    ("encoder", "velocity_head", False, [(9.5,6.15),(9.9,6.15),(9.9,5.15),(10.25,5.15)]),
    ("load_head", "fusion", False, [(11.75,7.075),(11.75,7.90)]),
    ("fusion", "actor", False, [(13.25,8.35),(14.6,8.35),(14.6,4.40),(9.3,4.40),(9.3,4.15)]),
    ("velocity_head", "actor", False, [(10.25,5.15),(10.0,5.15),(10.0,4.40),(9.3,4.40),(9.3,4.15)]),
    ("history", "actor", False, [(4.7,5.70),(4.7,4.65),(7.5,4.65),(7.5,4.15)]),
    ("commands", "actor", False, [(10.75,3.65),(9.95,3.65)]),
    ("actor", "plant", False, [(6.85,3.65),(5.9,3.65)]),
    ("plant", "measurement", False, [(2.9,3.65),(1.5,3.65),(1.5,5.575)]),
    ("truth", "supervised", True, [(3.55,1.2),(4.45,1.2)]),
    ("truth", "rollout", True, [(2.0,.75),(2.0,.35),(10.0,.35),(10.0,.75)]),
    ("plant", "rollout", True, [(4.4,3.175),(4.4,2.65),(10.0,2.65),(10.0,1.65)]),
    ("rollout", "ppo", True, [(11.55,1.2),(12.45,1.2)]),
    ("supervised", "encoder", True, [(6.0,1.65),(6.0,1.85),(6.55,1.85),(6.55,5.2),(8.25,5.2),(8.25,5.70)]),
    ("ppo", "actor", True, [(14.0,1.65),(14.0,2.90),(8.4,2.90),(8.4,3.15)]),
]


def draw_edge(ax, points, training):
    color = COLORS["training"][0] if training else "#43535E"
    path = DrawingPath(np.asarray(points), [DrawingPath.MOVETO] + [DrawingPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.05, color=color, linestyle=(0,(4,3)) if training else "-",
                                 capstyle="butt", zorder=3))


def create_figure(language):
    words = TEXT[language]
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=140)
    ax = fig.add_axes([.015,.22,.97,.75])
    ax.set(xlim=(0,16), ylim=(0,10))
    ax.axis("off")
    ax.add_patch(Rectangle((.1,.15),15.8,2.30,facecolor="#FDF7F8",edgecolor="none",zorder=0))
    ax.plot([.1,15.9],[2.45,2.45],color="#D9C6CC",linewidth=.9)
    ax.text(.25,9.55,words["online_title"],fontsize=FONT_SIZE,color=style.INK)
    ax.text(.25,2.12,words["training_title"],fontsize=FONT_SIZE,color=COLORS["training"][0])
    for source,target,training,points in EDGES:
        draw_edge(ax,points,training)
    box_labels = []
    for node in NODES:
        left,bottom,right,top = node.bounds
        color,fill = COLORS[node.category]
        patch = Rectangle((left,bottom),node.width,node.height,facecolor=fill,edgecolor=color,linewidth=1.1,zorder=4)
        ax.add_patch(patch)
        text = ax.text(node.x,node.y,words[node.name],fontsize=FONT_SIZE,color=style.INK,
                       ha="center",va="center",linespacing=1.35,zorder=5)
        box_labels.append((text,patch,node.name))
    for x,y in ((2.93,6.15),(9.9,6.15),(10.0,4.40)):
        ax.scatter([x],[y],s=10,color="#43535E",zorder=5)
    for x,y,key,color in ((6.0,7.25,"prior_features",COLORS["analytic"][0]),
                          (8.15,8.58,"current_prior",COLORS["analytic"][0]),
                          (5.65,4.92,"current_observation",COLORS["measurement"][0]),
                          (11.75,4.55,"estimates",COLORS["estimator"][0]),
                          (6.22,3.88,"action",COLORS["control"][0]),
                          (7.75,.57,"privileged_critic",COLORS["training"][0])):
        ax.text(x,y,words[key],fontsize=FONT_SIZE,color=color,ha="center",va="center",zorder=5)
    for x,category,label in zip((.12,.31,.50,.69,.88),CATEGORIES,words["legend"]):
        color,fill = COLORS[category]
        fig.add_artist(Rectangle((x-.072,.153),.016,.016,transform=fig.transFigure,facecolor=fill,edgecolor=color,linewidth=1.0))
        fig.text(x-.045,.161,label,fontsize=FONT_SIZE,color=style.INK,ha="left",va="center")
    fig.text(.5,.065,words["caption"],fontsize=FONT_SIZE,color=style.INK,
             ha="center",va="center",linespacing=1.55)
    return fig,box_labels


def check_layout(fig,box_labels):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [obj for obj in fig.findobj(Text) if obj.get_visible() and obj.get_text()]
    boxes = [obj.get_window_extent(renderer) for obj in labels]
    for obj,box in zip(labels,boxes):
        if not np.isclose(obj.get_fontsize(),FONT_SIZE):
            raise ValueError("All labels must use 14 pt")
        if not all(fig.bbox.contains(*corner) for corner in box.get_points()):
            raise ValueError("Text outside export: "+obj.get_text())
    for text,patch,name in box_labels:
        if not all(patch.get_window_extent(renderer).padded(-5).contains(*corner)
                   for corner in text.get_window_extent(renderer).get_points()):
            raise ValueError("Node label exceeds its box: "+name)
    for i,box in enumerate(boxes):
        for j,other in enumerate(boxes[:i]):
            if box.padded(3).overlaps(other):
                raise ValueError("Overlapping text: "+labels[i].get_text()+" / "+labels[j].get_text())
    return {"font_size_pt":FONT_SIZE,"all_text_in_export":True,"all_node_labels_fit":True,
            "text_overlap_pairs":0,"visible_text_count":len(labels)}


def namespace(value):
    return SimpleNamespace(**{key:namespace(item) if isinstance(item,dict) else item for key,item in value.items()})


def extract_methods(path,names,scope):
    tree = ast.parse(path.read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name in names]
    if len(nodes)!=len(names):
        raise ValueError("Missing source methods")
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),scope)
    return {node.name:node.lineno for node in nodes}


def source_audit(cfg,train,env_source,ppo_source,encoder_source):
    if not (cfg["env"]["use_qs_in_obs"] and train["algorithm"]["use_load_residual_estimation"]):
        raise ValueError("Figure requires the model-guided residual-estimation path")
    encoder_cfg = train["MLP_Encoder"]
    if not (encoder_cfg["use_dual_head"] and encoder_cfg["is_gru"] and encoder_cfg["output_detach"]
            and not encoder_cfg["use_hierarchical_com_estimation"] and encoder_cfg["obs_history_length"]==10
            and encoder_cfg["hidden_dims"][-1]==128 and encoder_cfg["num_output_dim"]==7):
        raise ValueError("Diagram must be updated for a different encoder contract")
    tree = ast.parse(env_source.read_text())
    analytical = next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=="_compute_load_estimates")
    attributes = sorted({node.attr for node in ast.walk(analytical) if isinstance(node,ast.Attribute)
                         and isinstance(node.value,ast.Name) and node.value.id=="self"})
    forbidden = {"base_mass","base_mass0","base_com","base_com0","base_lin_vel","load_on_body","has_load","load_mass_per_env"}
    if forbidden.intersection(attributes):
        raise ValueError("Analytical prior unexpectedly reads privileged truth")
    obs_scope = {"torch":torch}
    env_lines = extract_methods(env_source,{"compute_group_observations"},obs_scope)
    env = SimpleNamespace(cfg=namespace(cfg),num_envs=3,device="cpu")
    env.dof_pos=torch.zeros(3,8);env.default_dof_pos=torch.zeros(8);env.dof_vel=torch.zeros(3,8)
    env.base_ang_vel=torch.zeros(3,3);env.projected_gravity=torch.tensor([[0.,0.,-1.]]).repeat(3,1)
    env.torques=torch.ones(3,8);env.actions=torch.zeros(3,8);env.cmd_step=torch.zeros(3,1)
    env.clock_inputs_sin=torch.zeros(3);env.clock_inputs_cos=torch.ones(3)
    env.base_mass=torch.ones(3)*20;env.base_mass0=torch.ones(3)*10
    env.base_com=torch.ones(3,3)*.1;env.base_com0=torch.zeros(3,3);env.base_lin_vel=torch.ones(3,3)
    env.obs_scales=namespace(cfg["normalization"]["obs_scales"])
    prior={key:torch.ones(3)*value for key,value in {"payload_mass":5.,"load_x":.02,"load_y":-.03,"payload_present":1.,
           "sin_lieangle_L_thigh":.6,"cos_lieangle_L_thigh":.8,"sin_lieangle_R_thigh":.6,"cos_lieangle_R_thigh":.8,
           "qs_mass_delta":5.}.items()}
    prior["qs_com_delta"]=torch.ones(3,3)*.01
    env._compute_load_estimates=lambda:prior
    env.obs_buf=torch.zeros(3,cfg["env"]["num_observations"])
    obs1,critic1=obs_scope["compute_group_observations"](env)
    env.base_mass+=17;env.base_com+=.2;env.base_lin_vel+=3
    obs2,critic2=obs_scope["compute_group_observations"](env)
    torch.testing.assert_close(obs1,obs2)
    if torch.equal(critic1[:,:7],critic2[:,:7]):
        raise ValueError("Privileged label perturbation failed to change training-only labels")
    ppo_scope={"torch":torch}
    ppo_lines=extract_methods(ppo_source,{"_get_load_residual_baseline","_apply_load_residual_to_encoder_out"},ppo_scope)
    alg=SimpleNamespace(use_load_residual_estimation=True,load_residual_baseline_obs_start=train["algorithm"]["load_residual_baseline_obs_start"])
    alg._get_load_residual_baseline=lambda obs:ppo_scope["_get_load_residual_baseline"](alg,obs)
    raw=torch.arange(21,dtype=torch.float32).reshape(3,7)/10
    fused=ppo_scope["_apply_load_residual_to_encoder_out"](alg,raw,obs1)
    start=alg.load_residual_baseline_obs_start
    torch.testing.assert_close(fused[:,:3],raw[:,:3])
    torch.testing.assert_close(fused[:,3:7],raw[:,3:7]+obs1[:,start:start+4])
    spec=importlib.util.spec_from_file_location("figure07_encoder_source",encoder_source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.manual_seed(7)
    with redirect_stdout(io.StringIO()):
        encoder=module.MLP_Encoder(**encoder_cfg)
    history=torch.randn(2,encoder_cfg["num_input_dim"])
    output=encoder.encode(history)
    assert output.shape==(2,7) and not output.requires_grad
    supervised_output=encoder.get_encoder_out()
    assert supervised_output.requires_grad
    supervised_output.square().sum().backward()
    assert any(parameter.grad is not None for parameter in encoder.encoder.gru.parameters())
    assert any(parameter.grad is not None for parameter in encoder.encoder.vel_head.parameters())
    assert any(parameter.grad is not None for parameter in encoder.encoder.mass_head.parameters())
    ppo_tree=ast.parse(ppo_source.read_text())
    constructor=next(node for node in ast.walk(ppo_tree) if isinstance(node,ast.FunctionDef) and node.name=="__init__")
    owners={}
    for node in ast.walk(constructor):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Attribute):
            key=node.targets[0].attr
            if key in ("optimizer","extra_optimizer") and isinstance(node.value,ast.Call):
                owners[key]=ast.get_source_segment(ppo_source.read_text(),node.value)
    if "actor_critic.parameters()" not in owners["optimizer"] or "self.encoder.parameters()" not in owners["extra_optimizer"]:
        raise ValueError("Optimizer ownership no longer matches the diagram")
    online_edges=[(source,target) for source,target,training,_ in EDGES if not training]
    training_nodes={node.name for node in NODES if node.category=="training"}
    if any(source in training_nodes or target in training_nodes for source,target in online_edges):
        raise ValueError("A training-only node leaked into the deployment graph")
    return {"scope":"Source-level contracts and isolated tensor/network tests, not hardware deployment validation",
            "online_observations_unchanged_by_truth_perturbation":"pass","privileged_critic_labels_change":"pass",
            "analytical_prior_has_no_privileged_truth_attributes":"pass","analytical_prior_self_attributes":attributes,
            "fusion_preserves_velocity_and_adds_current_4D_prior":"pass","detached_policy_encoder_output":"pass",
            "auxiliary_supervision_reaches_gru_and_both_heads":"pass","optimizer_ownership":owners,
            "deployment_graph_has_no_training_only_nodes":"pass","source_lines":{**env_lines,**ppo_lines},
            "actor_observation_dim":int(obs1.shape[-1]),"history_frames":encoder_cfg["obs_history_length"],
            "history_input_dim":encoder_cfg["num_input_dim"],"estimator_output_dim":7,
            "actor_input_dim":cfg["env"]["num_observations"]+cfg["commands"]["num_commands"]+7,
            "stimulus_notes":"Prior outputs are fixed test tensors for observation isolation; actual prior source is separately audited for truth access"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-cfg",type=Path,default=SOURCE/"env_cfg.json")
    parser.add_argument("--train-cfg",type=Path,default=SOURCE/"train_cfg.json")
    parser.add_argument("--env-source",type=Path,default=SOURCE/"wheelfoot_flat.py")
    parser.add_argument("--ppo-source",type=Path,default=SOURCE/"ppo_20260706.py")
    parser.add_argument("--encoder-source",type=Path,default=DEFAULT_ENCODER)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--language",choices=("zh","en","both"),default="both")
    parser.add_argument("--dpi",type=int,default=300)
    args=parser.parse_args()
    cfg=json.loads(args.env_cfg.read_text());train=json.loads(args.train_cfg.read_text())
    audit=source_audit(cfg,train,args.env_source,args.ppo_source,args.encoder_source)
    corpus="".join(str(words) for words in TEXT.values())
    style.configure_fonts(args.output_dir,extra_text=corpus)
    plt.rcParams.update({"font.size":FONT_SIZE,"font.weight":"normal"})
    layouts={}
    for language in ("zh","en") if args.language=="both" else (args.language,):
        fig,box_labels=create_figure(language)
        layouts[language]=check_layout(fig,box_labels)
        for extension in ("png","pdf","svg"):
            target=args.output_dir/f"training_deployment_information_{language}.{extension}"
            fig.savefig(target,dpi=args.dpi,facecolor="white");print(target)
        plt.close(fig)
    audit["layout"]=layouts
    (args.output_dir/"validation_report.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n")
    manifest={"type":"Extension of user-supplied estimation/control topology, not a second parallel framework",
              "original_reference":"User-attached monochrome block diagram; node roles and online connections retained",
              "sources":{name:{"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
                         for name,path in {"env_cfg":args.env_cfg,"train_cfg":args.train_cfg,"env_source":args.env_source,
                                           "ppo_source":args.ppo_source,"encoder_source":args.encoder_source}.items()},
              "node_categories":{node.name:node.category for node in NODES},
              "color_semantics":COLORS,"solid_online_edges":[[a,b] for a,b,t,_ in EDGES if not t],
              "dashed_training_edges":[[a,b] for a,b,t,_ in EDGES if t],
              "supervision_update_arrow_scope":"GRU and both estimator heads, represented by the encoder update connection",
              "deployment_region_semantics":"Algorithmic inference contract; not a claim that a particular ROS package has implemented all blocks",
              "truth_for_test_only":"May be logged/scored in evaluation; not passed to the online estimator/actor",
              "centroid_semantics":"Payload-induced combined body CoM shift, not a raw payload installation position or whole-robot world CoM",
              "font_size_pt":FONT_SIZE,"raster_size_px":[round(value*args.dpi) for value in FIGURE_SIZE]}
    (args.output_dir/"figure_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    (args.output_dir/"graph_structure.json").write_text(json.dumps({"nodes":[vars(node) for node in NODES],
               "edges":[{"source":a,"target":b,"training_only":t,"route":p} for a,b,t,p in EDGES]},ensure_ascii=False,indent=2)+"\n")
    print("Archived information/gradient contracts verified; no simulator or hardware launched")


if __name__=="__main__":
    main()
