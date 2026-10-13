# jump-forward 分支: 前跳(立定跳远)任务目录入口。环境类复用 wheelfoot_flat 的 BipedWF
# (前跳逻辑靠 use_jump_distance 门控 + 前跳奖励组)。真正代码见 wheelfoot_flat/wheelfoot_flat.py
# 与 wheelfoot_flat_config.py (BipedCfgWFJumpFwd)。本文件仅作任务目录入口 + 存档指针。
from legged_gym.envs.wheelfoot_flat.wheelfoot_flat import BipedWF as BipedWFJumpFwd  # noqa: F401
