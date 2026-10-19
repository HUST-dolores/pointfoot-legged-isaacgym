# 图20与图21：中英文图注

## 图20：真实路径高程与连续碰撞模型截图

中文：基础迈步策略与地形适配策略在相同地形网格上的实际路径及连续动作。
分别提供平地、沿+world X升高的连续5 deg坡面与±5 cm粗糙面三个版本。
上图为机体投影实际经过的二维路径下的三角面高程，横轴为累计水平弧长，
不是净前向位移；两条曲线反映不同策略在同一网格上走出的不同路径。
下图为seed12345、env0在预登记2.00/2.24/2.48/2.72 s的真实离屏相机图像，
渲染原仿真中的彩色碰撞体，不使用品牌外观或人为设定姿态。
全部帧使用相同像素裁剪范围，原960 x 720全帧、相机外参和原始状态另存。
截图相机仅随位置平移，不随航向旋转；星号如出现表示该图已在首次终止之后。
粗糙面控制格点0.4 m、三角网格0.1 m、采样支持[-5,5] cm，实测经过路径不必达到两个极值。
这组截图不证明楼梯、台阶或任意地形通过能力；坡面主例的位移方向须结合符号数据解读。

English: Actual terrain profiles and consecutive collision-model images for the
base and terrain-adapted stepping policies. Separate versions show flat ground,
a continuous 5-degree slope rising in +world X, and rough ground sampled within
[-5,5] cm on a 0.4 m control grid and triangulated at 0.1 m resolution. The upper
panel samples the actual triangle surface under each base-projection path. Its
horizontal coordinate is cumulative horizontal arc length, not signed forward
displacement; the policies may follow different paths on the identical mesh.
Lower panels show real offscreen camera frames at the preregistered times
2.00, 2.24, 2.48, and 2.72 s for seed 12345, environment 0. The original
simulation collision geometry is rendered in color, without brand-specific
visual meshes or invented poses. The same pixel crop is applied to every frame;
full 960-by-720 images, camera extrinsics, and raw states are retained. Camera
translation follows the robot, but its viewing direction does not follow yaw.
An asterisk, if present, denotes a post-termination image. These observations
do not establish stair, step, or arbitrary-terrain traversability.

## 图21：接触、净空与姿态同步时序

中文：同地形下两个既有策略的实际轮接触、轮缘竖向净空和机体姿态。
主例与图20完全相同，所有通道来自同一仿真记录、同一采样时钟，覆盖0.02--6.00 s。
轮接触色带按实际Fz>1 N生成，不是期望支撑相位。淡线保留全部有效净空，
深色段表示该轮无接触；最低点由原资产圆柱转胶囊后的碰撞几何、原点偏移及当前姿态计算，
再与其正下方真实三角面作竖向比较，保留负值。不将其解释为全轮体最小分离距离或台阶高度。
pitch/roll为机体世界坐标角，初始化调整保留，初始大净空峰不作为稳定步态能力值。
灰带仅标出连续截图所取的2.00--2.72 s时间段，不是误差带或置信区间。
红线如出现为首次原终止，之后不画有效曲线；失败记录仍完整保留在跨环境统计中。

English: Synchronized measured contact, vertical wheel-rim clearance, and body
attitude for the two existing policies on the same terrain. These are the same
preregistered records as Figure 20, sampled on one clock from 0.02 to 6.00 s.
Contact bands use measured Fz > 1 N rather than desired support phase. Light
curves retain all eligible signed clearance samples; dark segments indicate a
non-contacting wheel. The lowest point is computed from the original
cylinder-to-capsule collision replacement, collision-origin offset, and current
wheel orientation, then compared vertically with the actual triangle below it.
This local vertical measure is neither global minimum wheel-to-ground separation
nor a validated obstacle height. Body pitch and roll use world coordinates;
initialization transients are retained. The gray interval marks only the
2.00--2.72 s image sequence, not uncertainty. A red line, if present, marks first
termination; subsequent samples are not shown as eligible motion, while failed
trials remain in the complete cross-environment statistics.

## 统计表图注

全18批540条轨迹，逐条计算后对每个评测seed的环境等权平均，再对3个评测seed等权汇总。
每种策略每类地形的完成率分母均90。每种策略仅一个训练历程，3个评测seed不是
3个独立训练seed；环境SD只表示本设置下的离散性，不是训练总体置信区间。
初始朝上坡不保证全程航向约束；5 deg设置两策略平均沿坡速度均为负，
不能将短时无终止表述为已验证正向上坡通过。
