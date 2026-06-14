#!/usr/bin/env python3
"""按变体设置 wheelfoot_flat_config.py 的消融标志 + 负载范围（训练脚本用）。

用法: python set_variant_cfg.py <qs> <resid> <torq> <load_max>
  qs/resid/torq : True/False —— use_qs_in_obs / use_residual_learning / use_torques_in_obs
  load_max      : add_load_range 上界（下界固定 2）

只改这几行的赋值；num_observations 等派生量在 config 类定义时会按新标志自动重算（子进程重新 import）。
四个变体的 (qs, resid, torq)：
  Model-guided   = True  True  True
  Estimate-guided= True  False True
  Source-guided  = False False True
  RL-only        = False False False
"""
import re, sys

CFG = "legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py"


def setbool(s, name, val):
    new, n = re.subn(rf'^(\s*{name}\s*=\s*)(True|False)', rf'\g<1>{val}', s, count=1, flags=re.M)
    assert n == 1, f"未能唯一匹配 {name}（匹配 {n} 次）"
    return new


def main():
    qs, resid, torq, load_max = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    s = open(CFG).read()
    s = setbool(s, "use_qs_in_obs", qs)
    s = setbool(s, "use_residual_learning", resid)
    s = setbool(s, "use_torques_in_obs", torq)
    s, n = re.subn(r'^(\s*add_load_range\s*=\s*)\[[^\]]*\]', rf'\g<1>[2, {load_max}]', s, count=1, flags=re.M)
    assert n == 1, f"未能唯一匹配 add_load_range（匹配 {n} 次）"
    open(CFG, "w").write(s)
    print(f"[set_variant_cfg] qs={qs} resid={resid} torq={torq} add_load_range=[2, {load_max}]")


if __name__ == "__main__":
    main()
