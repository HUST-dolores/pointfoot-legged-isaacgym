#!/usr/bin/env python3
"""
figs_chx/prep_data.py  — Co-design 第X章画图 数据准备（npz + BO日志 -> .mat）

把工作站产出的 table2 npz（每个 BO 候选/微调点/Table-II 策略一份，含 per-env 的
scen/rew/fell/length）+ BO 日志（run 时间戳 -> (thigh,shank,motor,fitness)）
抽取成结构化 .mat，供 figs_chx/ 下的 MATLAB 脚本读取出图。

scenario id (代码写死, table2_eval.py:49): 0=obstacle 1=slope 2=load 3=accel

用法:
  python3 prep_data.py --npz_root <scratchpad/npz> --logs <scratchpad/logs> --out data/codesign_figs_data.mat
自带校验: 每个 BO 候选 从 npz 重建的 4 场景等权均值 应 == 日志里的 FITNESS。
"""
import os, re, json, glob, argparse
import numpy as np
from scipy.io import savemat

SCEN_NAMES = ["obstacle", "slope", "load", "accel"]   # id 0/1/2/3


# ---------- BO 日志解析 ----------
CAND_RE = re.compile(
    r"CANDIDATE\s+thigh=([\d.]+)\s+shank=([\d.]+)(?:\s+motor=([\d.]+))?"
    r".*?run=(\S+?)\s+ckpt=(\d+)\s+FITNESS=([\d.\-]+)")
PHASE_RE = re.compile(r"\[bo\]\s+(init|BO)\s+\d+/\d+")


def parse_bo_log(path):
    """返回按评估顺序的候选列表: dict(thigh,shank,motor,run,ckpt,fit,phase,order)."""
    if not os.path.exists(path):
        return []
    cands, pending_phase = [], "init"
    with open(path) as f:
        for line in f:
            mp = PHASE_RE.search(line)
            if mp:
                pending_phase = mp.group(1)   # 'init' or 'BO'
            mc = CAND_RE.search(line)
            if mc:
                thigh, shank, motor, run, ckpt, fit = mc.groups()
                cands.append(dict(
                    thigh=float(thigh), shank=float(shank),
                    motor=(float(motor) if motor else 1.0),
                    run=run, ckpt=int(ckpt), fit=float(fit),
                    phase=pending_phase, order=len(cands) + 1))
    return cands


# ---------- npz 解析 ----------
def npz_index(npz_dir):
    """run 时间戳 -> npz 路径 (table2_<run>_ckpt..._d....npz, run 末尾含 '_')."""
    idx = {}
    for p in glob.glob(os.path.join(npz_dir, "*.npz")):
        b = os.path.basename(p)
        m = re.match(r"table2_(.+?)_ckpt(\d+)_d", b)
        if m:
            idx[m.group(1)] = p          # m.group(1) 形如 'Jul05_06-57-20_'
    return idx


def per_scen_stats(npz_path):
    """返回 (scen_meanrew[4], scen_fall%[4], scen_len[4], scen_eps[4], morph_scales)."""
    d = np.load(npz_path, allow_pickle=True)
    scen, rew, fell, length = d["scen"], d["rew"], d["fell"], d["length"]
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    ms = d["morph_scales"] if "morph_scales" in d.files else np.array([[np.nan, np.nan]])
    mrew = np.full(4, np.nan); mfall = np.full(4, np.nan)
    mlen = np.full(4, np.nan); neps = np.zeros(4, dtype=int)
    for k in range(4):
        sel = scen == k
        neps[k] = int(sel.sum())
        if sel.sum():
            mrew[k] = float(rew[sel].mean())
            mfall[k] = float(fell[sel].mean()) * 100.0
            mlen[k] = float(length[sel].mean())
    return mrew, mfall, mlen, neps, np.asarray(ms, dtype=float), meta


def per_scen_bucketmeans(npz_path):
    """Table-II 用: 每场景 先按 bucket(形态) 求均值, 返回 mean/std across buckets[4]."""
    d = np.load(npz_path, allow_pickle=True)
    scen, rew, bucket, fell = d["scen"], d["rew"], d["bucket"], d["fell"]
    mean4 = np.full(4, np.nan); std4 = np.full(4, np.nan); fall4 = np.full(4, np.nan)
    for k in range(4):
        sel = scen == k
        if not sel.sum():
            continue
        bks = bucket[sel]; rw = rew[sel]; fl = fell[sel]
        bmeans = [rw[bks == b].mean() for b in np.unique(bks)]
        mean4[k] = float(np.mean(bmeans))
        std4[k] = float(np.std(bmeans))
        fall4[k] = float(fl.mean()) * 100.0
    return mean4, std4, fall4


def attach_scen(cands, idx, verbose=True):
    """给每个候选挂 per-scenario 数据 + 校验 fitness."""
    out = []
    maxerr = 0.0
    for c in cands:
        p = idx.get(c["run"])
        if p is None:
            if verbose:
                print(f"  [WARN] 无 npz: run={c['run']}")
            continue
        mrew, mfall, mlen, neps, ms, meta = per_scen_stats(p)
        recon = float(np.nanmean(mrew))          # 4 场景等权 = fitness
        err = abs(recon - c["fit"])
        maxerr = max(maxerr, err)
        c = dict(c)
        c.update(scen_rew=mrew, scen_fall=mfall, scen_len=mlen, scen_eps=neps,
                 npz_thigh=float(ms[0, 0]), npz_shank=float(ms[0, 1]),
                 recon_fit=recon, fit_err=err, difficulty=meta.get("difficulty"))
        out.append(c)
    if verbose:
        print(f"  校验: 重建fitness vs 日志FITNESS 最大误差 = {maxerr:.3f} "
              f"({'OK ✓' if maxerr < 0.5 else '★偏差大, 检查!'})  [{len(out)}/{len(cands)} 候选]")
    return out


def stack(cands, keys):
    return {k: np.array([c[k] for c in cands]) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--npz_root", default=os.path.join(here, "..", "..", "..", "scratchpad", "npz"))
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", default=os.path.join(here, "data", "codesign_figs_data.mat"))
    args = ap.parse_args()

    npz_t2 = os.path.join(args.npz_root, "table2")
    idx = npz_index(npz_t2)
    print(f"table2 npz 索引: {len(idx)} 个 run")

    M = {}   # savemat dict
    M["scen_names"] = np.array(SCEN_NAMES, dtype=object)

    # ---- 2D BO ----
    print("\n[2D BO]")
    bo2d = attach_scen(parse_bo_log(os.path.join(args.logs, "bo.log")), idx)
    if bo2d:
        s = stack(bo2d, ["thigh", "shank", "fit", "order", "recon_fit"])
        M.update(bo2d_thigh=s["thigh"], bo2d_shank=s["shank"], bo2d_fit=s["fit"],
                 bo2d_order=s["order"],
                 bo2d_isinit=np.array([c["phase"] == "init" for c in bo2d]),
                 bo2d_scen=np.vstack([c["scen_rew"] for c in bo2d]),
                 bo2d_fall=np.vstack([c["scen_fall"] for c in bo2d]))

    # ---- 3D BO (full + val 合并, 更多电机点) ----
    print("\n[3D BO full]")
    bo3d = attach_scen(parse_bo_log(os.path.join(args.logs, "bo3d_full.log")), idx)
    print("[3D BO val (额外点)]")
    bo3dv = attach_scen(parse_bo_log(os.path.join(args.logs, "bo3d_val.log")), idx)
    if bo3d:
        s = stack(bo3d, ["thigh", "shank", "motor", "fit", "order", "recon_fit"])
        M.update(bo3d_thigh=s["thigh"], bo3d_shank=s["shank"], bo3d_motor=s["motor"],
                 bo3d_fit=s["fit"], bo3d_order=s["order"],
                 bo3d_isinit=np.array([c["phase"] == "init" for c in bo3d]),
                 bo3d_scen=np.vstack([c["scen_rew"] for c in bo3d]),
                 bo3d_fall=np.vstack([c["scen_fall"] for c in bo3d]))
    # 合并 full+val 用于"per-scenario vs 电机"散点 (去重 run)
    allc = {c["run"]: c for c in (bo3d + bo3dv)}.values()
    allc = list(allc)
    if allc:
        M.update(allc_thigh=np.array([c["thigh"] for c in allc]),
                 allc_shank=np.array([c["shank"] for c in allc]),
                 allc_motor=np.array([c["motor"] for c in allc]),
                 allc_fit=np.array([c["fit"] for c in allc]),
                 allc_scen=np.vstack([c["scen_rew"] for c in allc]),
                 allc_fall=np.vstack([c["scen_fall"] for c in allc]))

    # ---- 微调曲线 ----
    print("\n[微调曲线]")
    ft_map = [("B_zeroshot", 0), ("finetune400", 400), ("ft500", 500),
              ("ft1000", 1000), ("ft1500", 1500), ("ft2000", 2000),
              ("specialist6000", 6000)]
    ft_dir = os.path.join(args.npz_root, "finetune_val")
    ft_steps, ft_fit, ft_scen = [], [], []
    for tag, step in ft_map:
        p = os.path.join(ft_dir, f"eval_{tag}.npz")
        if not os.path.exists(p):
            print(f"  [WARN] 缺 {p}"); continue
        mrew, mfall, mlen, neps, ms, meta = per_scen_stats(p)
        ft_steps.append(step); ft_fit.append(float(np.nanmean(mrew)))
        ft_scen.append(mrew)
        print(f"  {tag:16s} step={step:5d} fit={np.nanmean(mrew):7.2f}  d={meta.get('difficulty')}")
    if ft_steps:
        M.update(ft_steps=np.array(ft_steps), ft_fit=np.array(ft_fit),
                 ft_scen=np.vstack(ft_scen))

    # ---- Table-II (seed1 主 + seed2) ----
    print("\n[Table-II]")
    def load_t2(path):
        mean4, std4, fall4 = per_scen_bucketmeans(path)
        return mean4, std4, fall4
    t2 = os.path.join(args.npz_root, "table2_final")
    t2s2 = os.path.join(args.npz_root, "table2_s2")
    pairs = {
        "B_d05": (t2, "B_spatialDR_29000_d0.5.npz"),
        "noDR_d05": (t2, "noDR_29000_d0.5.npz"),
        "B_d03": (t2, "B_spatialDR_29000_d0.3.npz"),
        "noDR_d03": (t2, "noDR_29000_d0.3.npz"),
        "B_s2_d05": (t2s2, "B_s2_d0.5.npz"),
        "noDR_s2_d05": (t2s2, "nodr_s2_d0.5.npz"),
        "B_s2_d03": (t2s2, "B_s2_d0.3.npz"),
        "noDR_s2_d03": (t2s2, "nodr_s2_d0.3.npz"),
    }
    for key, (d, fn) in pairs.items():
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            print(f"  [WARN] 缺 {p}"); continue
        mean4, std4, fall4 = load_t2(p)
        M[f"t2_{key}_mean"] = mean4
        M[f"t2_{key}_std"] = std4
        M[f"t2_{key}_fall"] = fall4
        print(f"  {key:12s} mean={np.round(mean4,1)} (obstacle/slope/load/accel)")

    # ---- 诊断: per-scenario vs 电机 (短腿子集) ----
    print("\n[诊断] 各场景 mean_rew vs 电机 (3D full+val, 全部 & 短腿子集 thigh<=0.92 & shank<=0.9)")
    if allc:
        arr = sorted(allc, key=lambda c: c["motor"])
        print(f"  {'motor':>6} {'thigh':>5} {'shank':>5} | " +
              " ".join(f"{n:>9}" for n in SCEN_NAMES))
        for c in arr:
            tag = " *短腿" if (c["thigh"] <= 0.92 and c["shank"] <= 0.9) else ""
            print(f"  {c['motor']:6.3f} {c['thigh']:5.2f} {c['shank']:5.2f} | " +
                  " ".join(f"{v:9.1f}" for v in c["scen_rew"]) + tag)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    savemat(args.out, M)
    print(f"\n✓ 写出 {args.out}  ({len(M)} 变量)")


if __name__ == "__main__":
    main()
