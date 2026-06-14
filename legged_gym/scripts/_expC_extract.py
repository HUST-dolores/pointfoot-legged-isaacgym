#!/usr/bin/env python3
"""纯 CPU 提取 ExpC 三项指标(斜坡存活/静态漂移/急停距离),复刻 plot_expC_*.m 的逻辑。
只读 exported/*.mat,不导入 torch/不碰 GPU。输出 markdown 表格用的数值。"""
import os, glob, re
import numpy as np
from scipy.io import loadmat

ED = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported')
ED = os.path.abspath(ED)
LOAD_EDGES = [2, 8, 14, 20, 26, 30.1]
LOAD_CTR = [(LOAD_EDGES[i] + LOAD_EDGES[i + 1]) / 2 for i in range(len(LOAD_EDGES) - 1)]
VARIANTS = ('Model-guided', 'Estimate-guided', 'Source-guided', 'RL-only')  # 估计信息从多到少


def getmeta(m, f, d):
    try:
        meta = m['meta']
        # struct from loadmat: meta[0,0][f]
        v = meta[f][0, 0] if False else meta[0, 0][f]
        v = np.array(v).squeeze()
        if v.size == 0:
            return d
        return v.item() if v.size == 1 else v
    except Exception:
        return d


def variant_name(m):
    qs = bool(getmeta(m, 'use_qs_in_obs', 1))
    re_ = bool(getmeta(m, 'use_load_residual_estimation', 0))
    tq = bool(getmeta(m, 'use_torques_in_obs', 1))
    if qs and re_:
        return 'Model-guided'
    if (not qs) and (not re_) and (not tq):
        return 'RL-only'
    if qs and (not re_):
        return 'Estimate-guided'
    return 'Source-guided'


def is_wide(m):
    lr = getmeta(m, 'load_run', '')
    return 'wide2-30' in str(lr)


def orient(a):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] < a.shape[1]:
        a = a.T
    return a


def get_dt(m):
    try:
        dt = np.asarray(m['dt']).squeeze()
        return float(dt.flat[0])
    except Exception:
        return 0.02


def binmeans(load, val):
    out = []
    load = np.asarray(load); val = np.asarray(val)
    for i in range(len(LOAD_CTR)):
        s = (load >= LOAD_EDGES[i]) & (load < LOAD_EDGES[i + 1])
        out.append(np.nanmean(val[s]) if s.any() else np.nan)
    return out


# ---------------- 斜坡存活 ----------------
def slope_survival():
    # key (variant, slope) -> 最新文件
    best = {}
    # 仅取「行走斜坡存活」(walk_vx0.5),排除 static_slope(C-2′)与 estop_slope(C-3′),避免污染 C-1/β*
    for f in glob.glob(os.path.join(ED, 'play_data_*walk_vx0.5*_load2-30_flat_slope*.mat')):
        bn = os.path.basename(f)
        if '_corr' in bn or 'estop' in bn or 'static' in bn:
            continue
        m = loadmat(f)
        if 'meta' not in m or not is_wide(m):
            continue
        vn = variant_name(m)
        if vn not in VARIANTS:
            continue
        sl = re.search(r'slope([0-9.]+)', os.path.basename(f))
        if not sl:
            continue
        sl = float(sl.group(1))
        k = (vn, sl)
        t = os.path.getmtime(f)
        if k not in best or best[k][0] < t:
            best[k] = (t, f, m)
    # build per (variant, slope) load-binned survival
    res = {}  # (vn, slope) -> binned list
    slopes = set()
    for (vn, sl), (_, f, m) in best.items():
        slopes.add(sl)
        pit = orient(m['base_pitch_all']) * 180 / np.pi
        roll = orient(m['base_roll_all']) * 180 / np.pi
        ref = orient(m['payload_mass_ref_all'])
        dt = get_dt(m)
        T, N = pit.shape
        t = np.arange(T) * dt
        last = t > (t[-1] - 5.0)  # 末段 5s
        up = (np.abs(pit) < 25) & (np.abs(roll) < 25)
        ld = np.zeros(N); sv = np.zeros(N)
        for j in range(N):
            on = ref[:, j] > 0.3
            ld[j] = np.nanmedian(ref[on, j]) if on.any() else 0.0
            # 末态直立率:末 5s 内 ≥80% 时间直立才算"站住"(抗复位虚高)
            sv[j] = 1.0 if np.nanmean(up[last, j]) >= 0.8 else 0.0
        res[(vn, sl)] = binmeans(ld, sv)
    return res, sorted(slopes)


# ---------------- 静态漂移 ----------------
def static_drift():
    best = {}
    for f in glob.glob(os.path.join(ED, 'play_data_*static*_load2-30_flat.mat')):
        bn = os.path.basename(f)
        if 'estop' in bn or 'slope' in bn:
            continue
        m = loadmat(f)
        if 'meta' not in m or not is_wide(m):
            continue
        vn = variant_name(m)
        if vn not in VARIANTS:
            continue
        t = os.path.getmtime(f)
        if vn not in best or best[vn][0] < t:
            best[vn] = (t, f, m)
    res = {}
    for vn, (_, f, m) in best.items():
        px = orient(m['base_pos_x_all']); py = orient(m['base_pos_y_all'])
        ref = orient(m['payload_mass_ref_all'])
        dt = get_dt(m)
        r = int(round(1.0 / dt)); N = px.shape[1]
        ld = []; dr = []
        for j in range(N):
            on = ref[:, j] > 0.3
            L = np.nanmedian(ref[on, j]) if on.any() else np.nan
            if np.isnan(L):
                continue
            d = np.nanmax(np.sqrt((px[r:, j] - px[r, j]) ** 2 + (py[r:, j] - py[r, j]) ** 2))
            ld.append(L); dr.append(d)
        res[vn] = (binmeans(ld, dr), f)
    return res


# ---------------- 急停距离 ----------------
def estop_distance():
    best = {}
    for f in glob.glob(os.path.join(ED, 'play_data_*_load2-30_flat_estop1.5.mat')):
        m = loadmat(f)
        if 'meta' not in m or not is_wide(m):
            continue
        vn = variant_name(m)
        if vn not in VARIANTS:
            continue
        t = os.path.getmtime(f)
        if vn not in best or best[vn][0] < t:
            best[vn] = (t, f, m)
    res = {}
    for vn, (_, f, m) in best.items():
        cmd = orient(m['command_x_all']); px = orient(m['base_pos_x_all'])
        ref = orient(m['payload_mass_ref_all'])
        dt = get_dt(m)
        T, N = cmd.shape
        pp = int(round(0.3 / dt)); hold = int(round(0.5 / dt)); win = int(round(2.5 / dt))
        ld = []; ds = []
        for j in range(N):
            on = ref[:, j] > 0.3
            L = np.nanmedian(ref[on, j]) if on.any() else np.nan
            if np.isnan(L):
                continue
            for k in range(pp, T - win):
                if (cmd[k - pp:k, j] > 1.0).all() and (cmd[k:k + hold, j] < 0.5).all():
                    ld.append(L); ds.append(np.nanmax(px[k:k + win, j] - px[k, j])); break
        res[vn] = (binmeans(ld, ds), f)
    return res


def fmt(v):
    return f"{v:.2f}" if (v is not None and not np.isnan(v)) else "—"


if __name__ == '__main__':
    print("LOAD bins (kg):", [f"{c:.0f}" for c in LOAD_CTR])
    print("\n===== C-1 斜坡存活率 (变体 × 坡度，行=负载分箱均值) =====")
    surv, slopes = slope_survival()
    for vn in VARIANTS:
        print(f"\n[{vn}]  load-bins -> {[f'{c:.0f}kg' for c in LOAD_CTR]}")
        for sl in slopes:
            row = surv.get((vn, sl))
            if row is None:
                continue
            print(f"  slope {sl:>4g}°: " + "  ".join(fmt(x) for x in row))

    print("\n===== C-2 静态漂移 max|Δpos| [m] =====")
    sd = static_drift()
    for vn in VARIANTS:
        if vn in sd:
            row, f = sd[vn]
            print(f"  [{vn}]: " + "  ".join(fmt(x) for x in row) + f"   ({os.path.basename(f)})")

    print("\n===== C-3 急停停车距离 [m] =====")
    es = estop_distance()
    for vn in VARIANTS:
        if vn in es:
            row, f = es[vn]
            print(f"  [{vn}]: " + "  ".join(fmt(x) for x in row) + f"   ({os.path.basename(f)})")
