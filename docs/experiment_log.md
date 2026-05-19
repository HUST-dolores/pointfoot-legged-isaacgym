# 实验记录

每个实验跑完后**立即**在这里填一条。一个 entry 一行（小调参用同一行更新；重大变化新开一行）。`exp_tag` 和 .mat 文件名一致，便于事后追溯。

## 记录字段说明

| 字段 | 说明 |
|---|---|
| date | 实验时间，例如 2026-05-19 |
| exp_tag | 给 `play.py --exp_tag` 的标签；同时是 .mat 文件名后缀 |
| commit | `git rev-parse --short HEAD` 得到的 hash |
| policy | `<run_name>/model_<iter>.pt` |
| condition | 实验条件简述（速度、载荷、扰动等） |
| key_params | 跟之前不同的关键参数 |
| mat_path | 实际输出的 .mat 路径 |
| notes | 简要观察 / 备注 |

---

## 实验 1：负载估计准确性

| date | exp_tag | commit | policy | condition | key_params | mat_path | notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## 实验 2：控制增益

| date | exp_tag | commit | policy | condition | key_params | mat_path | notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## 实验 3：残差网络抗扰

| date | exp_tag | commit | policy | condition | key_params | mat_path | notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## 实验 4：消融

| date | exp_tag | commit | policy | condition | key_params | mat_path | notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

---

## 推荐的 git 备份流程

每次跑完一个有意义的实验：

```bash
# 1. 看下输出文件
ls -lh logs/wheelfoot_flat/<exp>/exported/play_data_*.mat

# 2. 提交（推荐把 .mat 加 .gitignore，只 commit log）
git add docs/experiment_log.md
git commit -m "exp1 静态 5kg(0.1,0): RMSE m=X.XX kg, com_x=X.XX m, com_y=X.XX m"

# 3. 如果换了 policy/config 也一起 commit
git add legged_gym/envs/...  
git commit -m "..."
```

如果 .mat 文件本身想入版本控制（小文件，单次几 MB）：
```bash
git add logs/wheelfoot_flat/<exp>/exported/play_data_<exp_tag>.mat
git commit -m "..."
```

但更推荐 .mat 不入库，只在 log 表里记 mat_path，避免仓库膨胀。
