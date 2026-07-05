#!/usr/bin/env python3
"""figs_chx/prep_pathB.py — 把路B两个 eval 的产出并入 codesign_figs_data.mat。

读:
  <pathB>/motor_k*.npz   — 定腿 ξ*=(0.914,0.8) 零样本扫电机, table2 schema(bucket/scen/rew/...)
  <pathB>/footheight.npz — 全 obstacle 抬腿峰值净高 (thigh/shank/peak_height/fell)
写入 .mat 的新字段:
  msweep_motor(1×K)  msweep_scen(K×4, 列=obstacle/slope/load/accel)  msweep_fall(K×4)  msweep_fit(1×K)
  footh_thigh/footh_shank/footh_peak/footh_fell(1×M)  footh_target(标量)
scenario id: 0=obstacle 1=slope 2=load 3=accel。
"""
import os, re, glob, json, argparse
import numpy as np
from scipy.io import loadmat, savemat


def per_scen(npz):
    d = np.load(npz, allow_pickle=True)
    scen, rew, fell = d["scen"], d["rew"], d["fell"]
    mr = np.full(4, np.nan); mf = np.full(4, np.nan)
    for k in range(4):
        s = scen == k
        if s.sum():
            mr[k] = float(rew[s].mean()); mf[k] = float(fell[s].mean()) * 100
    return mr, mf


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--pathB", required=True)
    ap.add_argument("--mat", default=os.path.join(here, "data", "codesign_figs_data.mat"))
    args = ap.parse_args()
    M = loadmat(args.mat)   # 保留已有字段, 追加

    # ---- 电机扫描 ----
    ks, scen, fall = [], [], []
    for p in sorted(glob.glob(os.path.join(args.pathB, "motor_k*.npz"))):
        m = re.search(r"motor_k([0-9.]+)\.npz", os.path.basename(p))
        if not m:
            continue
        k = float(m.group(1))
        mr, mf = per_scen(p)
        ks.append(k); scen.append(mr); fall.append(mf)
    if ks:
        order = np.argsort(ks)
        ks = np.array(ks)[order]; scen = np.array(scen)[order]; fall = np.array(fall)[order]
        M["msweep_motor"] = ks
        M["msweep_scen"] = scen
        M["msweep_fall"] = fall
        M["msweep_fit"] = np.nanmean(scen, axis=1)
        print("[msweep] k -> [obstacle slope load accel]  (fit)")
        for i, k in enumerate(ks):
            print(f"  {k:.2f} -> {np.round(scen[i],1)}  ({np.nanmean(scen[i]):.1f})")
        li = int(np.nanargmax(scen[:, 2]))   # load 峰
        print(f"  ★ load 峰值 k={ks[li]:.2f} rew={scen[li,2]:.1f}")

    # ---- 微调电机扫描 (定腿 ξ*=0.914,0.8, 每k微调400) ----
    ftdir = os.path.join(args.pathB, "..", "pathB_ft")
    ftlog = os.path.join(ftdir, "ftmotor.log")
    if os.path.exists(ftlog):
        rmap = {}
        for line in open(ftlog):
            mm = re.search(r"motor=([0-9.]+).*run=(\S+?)\s", line)
            if mm:
                rmap[mm.group(2)] = float(mm.group(1))
        fks, fscen, ffall = [], [], []
        for p in glob.glob(os.path.join(ftdir, "table2_Jul05_12-*.npz")):
            rr = re.search(r"table2_(Jul05_12-[0-9-]+_)_", os.path.basename(p))
            if not rr or rr.group(1) not in rmap:
                continue
            k = rmap[rr.group(1)]
            mr, mf = per_scen(p)
            fks.append(k); fscen.append(mr); ffall.append(mf)
        if fks:
            o = np.argsort(fks)
            fks = np.array(fks)[o]; fscen = np.array(fscen)[o]; ffall = np.array(ffall)[o]
            M["ftmotor_motor"] = fks
            M["ftmotor_scen"] = fscen
            M["ftmotor_fall"] = ffall
            M["ftmotor_fit"] = np.nanmean(fscen, axis=1)
            print("\n[ftmotor 微调] k -> [obstacle slope load accel]  (fit)")
            for i, k in enumerate(fks):
                print(f"  {k:.2f} -> {np.round(fscen[i],1)}  ({np.nanmean(fscen[i]):.1f})")
            bi = int(np.nanargmax(np.nanmean(fscen, axis=1)))
            print(f"  ★ 综合fitness 峰 k={fks[bi]:.2f} = {np.nanmean(fscen[bi]):.1f}")

    # ---- 抬腿高度 ----
    fp = os.path.join(args.pathB, "footheight.npz")
    if os.path.exists(fp):
        d = np.load(fp, allow_pickle=True)
        M["footh_thigh"] = np.asarray(d["thigh"], float)
        M["footh_shank"] = np.asarray(d["shank"], float)
        M["footh_peak"] = np.asarray(d["peak_height"], float)
        M["footh_fell"] = np.asarray(d["fell"], float)
        meta = json.loads(str(d["meta"]))
        M["footh_target"] = float(meta.get("target_m", np.nan))
        # 按形态(bucket)聚合 -> 120 个 (thigh,shank) 点, 供平滑插值热图
        pts = np.round(np.stack([M["footh_thigh"], M["footh_shank"]], 1), 5)
        uniq, inv = np.unique(pts, axis=0, return_inverse=True)
        M["footh_bth"] = uniq[:, 0]
        M["footh_bsh"] = uniq[:, 1]
        M["footh_bpeak"] = np.array([M["footh_peak"][inv == i].mean() for i in range(len(uniq))])
        M["footh_bfall"] = np.array([M["footh_fell"][inv == i].mean() * 100 for i in range(len(uniq))])
        th = M["footh_thigh"]; pk = M["footh_peak"]
        print(f"\n[footh] {len(pk)} envs / {len(uniq)} 形态, target={M['footh_target']:.3f}m, "
              f"peak mean={pk.mean():.3f}m")
        for lo in [0.8, 0.9, 1.0, 1.1]:
            s = (th >= lo) & (th < lo + 0.1)
            if s.sum():
                print(f"  thigh[{lo:.1f},{lo+0.1:.1f}) peak={pk[s].mean():.3f}m")

    savemat(args.mat, M)
    print(f"\n✓ 已并入 {args.mat}")


if __name__ == "__main__":
    main()
