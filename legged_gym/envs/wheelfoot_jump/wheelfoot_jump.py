# jump 分支: 解耦跳跃任务。环境类复用 wheelfoot_flat 的 BipedWF(跳跃逻辑靠 use_jump 门控 + 跳跃奖励组)。
# 真正的跳跃代码见 wheelfoot_flat/wheelfoot_flat.py (_update_jump_command / _reward_jump_*)
# 与 wheelfoot_flat_config.py (BipedCfgWFJump)。本文件仅作任务目录入口 + 存档指针。
from legged_gym.envs.wheelfoot_flat.wheelfoot_flat import BipedWF as BipedWFJump  # noqa: F401
