# 自由负载在机器人托板上滑动：独立实验

日期：2026-10-18。分支`payload-slide-lab`，基于`f8d2cf1`，工作树：
`/home/xu/limx_rl/pointfoot-legged-gym-payload-slide`。
这是独立worktree内的新实验文件，尚未提交Git；不要只切换分支HEAD而遗漏本目录未提交的代码。
原`codesign`工作树、标称URDF、训练配置、四次论文补测及其交接包不修改。

## 已做到什么

- 箱体是独立动态刚体，5.93 kg、0.1 m立方体；机器人未固定基座，仍由原冻结平衡策略控制。
- 两种机器人碰撞模型都出现了有支撑的板内滑移，不依赖把箱体抛起或让箱体掉落来冒充滑动。
- 默认演示：机体/托板接触shape与箱体friction输入均为0.05，2.00–2.04 s向箱体施加有限横向力脉冲，
  然后外力归零。箱体相对托板移动约6.89 cm，6 s末仍完整支撑在托板上。
- 第二演示不向箱体施加任何外力，只让机器人在2–3 s接收0.6 m/s前进命令：
  箱体先在板内滑动约18 cm，然后越过边缘滑出。这可作为以后防滑出控制的真实挑战工况。
- 尚未训练“让机器人主动把负载留在板内”的控制器；默认演示留在板内不等于已经解决该控制任务。

## 之前不明显的几个原因

1. 原代码创建箱体时把friction写死为0.5，单改其他配置字段未必影响实际接触材质。
2. 接触双方材质都重要。当前SDK实测只把箱体降至0.02、机体保持原friction=1时，同等脉冲几乎推不动箱体；
   两侧都降低后才出现明显滑移。这里的0.05是SDK输入值，不能不经标定就当作真实试验测得的有效摩擦。
3. 标称机体碰撞盒顶部z=0.023 m，托板顶部z=0.0225 m；居中箱体可能接触机体碰撞盒，
   而不只是视觉上那块板。collapse_fixed_joints=True后它们合并到base_Link，但仍有两个collision shape。
4. 平地、相对静止、没有切向驱动时，即使零摩擦也不会自动产生明显滑动。需明确激励或机器人运动。

处理：通过SDK的body-to-shape索引只设置合并机体/托板body的接触材质，不修改轮/腿摩擦或轮地条件。
规范对照各组的机体/托板及箱体contact_offset都为2 mm，不让接触偏移和摩擦同时变化。
保留`original`几何；另生成`raised`几何，把托板中心从2 cm抬到6 cm，清除机体碰撞盒的干扰。
raised是明确标注的实验几何变体，不是偷偷修改标称模型；会有很小的板质量质心改变，不能混入旧论文比较。
两份派生URDF同时只用碰撞体积做可视化，不载入品牌外观。标称robot.urdf原封不动。

## 本机交互Play

```bash
cd /home/xu/limx_rl/pointfoot-legged-gym-payload-slide
ROBOT_TYPE=WF_TRON1A PYTHONPATH="$PWD" \
 /home/xu/anaconda3/envs/pointfoot_legged_gym/bin/python \
 -m legged_gym.scripts.sliding_lab.robot --viewer --clean-exit
```

默认6 s、raised、mu_0p05、y方向有限脉冲。输出自动保存到本worktree的`logs/sliding_lab_play/时间戳/`。
显示窗口需在本机有NVIDIA驱动和桌面的终端中运行；本轮已在工作站GPU验证物理与离屏画面，
没有在代理受限的本机环境中实际打开交互窗口。

机器人运动诱发滑出，不给箱体外力：

```bash
ROBOT_TYPE=WF_TRON1A PYTHONPATH="$PWD" \
 /home/xu/anaconda3/envs/pointfoot_legged_gym/bin/python \
 -m legged_gym.scripts.sliding_lab.robot --viewer --clean-exit \
 --excitation drive --preset mu_0p05 --direction x
```

可用参数：`--geometry original/raised`、`--preset original/box_only_low/mu_0p1/mu_0p05/mu_0p02/mu_0p005/mu_zero/no_pulse`，
`--direction x/minus_x/y`、`--excitation pulse/drive/none`、`--impulse-dv`、`--mass`、`--duration`。
impulse-dv只是计算40 ms外力的名义输入，实际速度由摩擦/接触求解，不强制达到该速度。
`--preset original`保留原摩擦值，但仍使用实验统一的2 mm接触偏移，不声称逐字复原旧仿真。
同一几何的派生URDF使用固定文件名；本轮同几何录屏按顺序运行，后续大规模并发训练应预生成不可变资产。

## 数据、视频与复算

`logs/sliding_lab_20261018/`：
- `bench_v2/`：51个规范台架接触预检，实际记录plate actor状态作为参照。
- `robot_original_verified/`、`robot_raised_v2/`：统一接触偏移且源码哈希一致的24例/几何规范复验。
- `robot_demo_safe_verified/`：完整彩色相机帧，安全板内滑移演示。
- `robot_demo_drive/`：无箱体外力的机器人运动诱发滑移/滑出演示。
- `media/safe_supported_slide.mp4`、`body_motion_slip.mp4`：原始帧直接编码，不修改箱体位置。
- `media/*measurements.csv/.png/.pdf`：真实板坐标位置、间隙、法向支撑力和边界余量。
- `analysis/robot_slip_cases.csv`、`bench_slip_cases.csv`、`verification.json`：全部测量与验证。

早期`bench/`、`robot_original/`、`robot_raised/`、`robot_demo_safe*`等试探记录保留溯源，
不将重复录屏当独立训练seed或独立统计试次；规范证据以上面明确列出的目录为准。
`robot_original_v2/`缺少最终接触偏移审计字段，未用作规范证据，已在确认源码同步后复验为original_verified。
`robot_drive/`是早期运动激励扫描，方向占位在drive模式不改变机器人前进方向，不作为3-seed鲁棒性。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" python \
 -m legged_gym.scripts.sliding_lab.analyze --input logs/sliding_lab_20261018
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" python \
 -m legged_gym.scripts.sliding_lab.media --input logs/sliding_lab_20261018
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" python \
 -m legged_gym.scripts.sliding_lab.verify --input logs/sliding_lab_20261018
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" python \
 -m unittest legged_gym.scripts.sliding_lab.test_geometry -v
```

板内滑移定义同时要求：相对托板底面中心位移≥2 cm、立方体全部投影在板内、法向间隙<8 mm、
法向接触力>0.2mg、相对倾斜<12°，且在原终止前；不是只看箱体世界坐标移动。
越边与完全滑出分开记录。生成时初始化位姿/速度，此后不回写箱体状态，不加吸附或固定关节。

## 下一阶段的接口与边界

防滑出任务可使用板坐标x/y、相对切向速度、实际接触、板边最小余量和机器人状态，动作仍只控制机器人电机。
在drive工况训练时不向箱体施加控制力，也不能用箱体位姿回写掩盖滑出。
若策略使用箱体真值位置，这是仿真状态反馈；部署前仍需明确传感/估计路径，不能称为只靠本体测量就已实现。
需另行定义成功率、边界裕度、滑出/跌倒、姿态和动作代价，不把本次固定策略的滑动物理验证当作防滑控制结论。

## 物理依据

NVIDIA PhysX 4.1文档说明摩擦依赖接触材质、可采用不同组合方式，且求解器摩擦响应未必严格等同简化解析模型：
[Rigid Body Dynamics](https://nvidiagameworks.github.io/PhysX/4.1/documentation/physxguide/Manual/RigidBodyDynamics.html#friction-and-restitution)，
[PxCombineMode](https://nvidiagameworks.github.io/PhysX/4.1/documentation/physxapi/files/structPxCombineMode.html)。
本实验不假设某个组合规则一定成立，以当前Isaac Gym版本的实测滑移/支撑响应校准参数。
