#!/usr/bin/env python3
"""BO 外层搜双轮足设计参数 ξ 最大化 f(ξ)=4场景等权 fitness. 维度无关(读 BOUNDS 自适应).
2D = (thigh,shank); 3D = (thigh,shank,motor). 每候选调 finetune_and_eval.sh(从 generalist 微调+eval).
纯 numpy/scipy 的 GP(RBF)+ EI —— 工作站无 sklearn/skopt.
用法: BO_GPU=0 python bo_codesign.py --n_init 6 --n_bo 12 --steps 400
      (先跑通闭环: --n_init 2 --n_bo 1)
输出: /home/ps/mydrive/xu/bo_codesign/history.json + result.json + 最优 ξ*.
"""
import os, re, json, argparse, subprocess, time, itertools
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm
from scipy.spatial.distance import cdist

# ★设计空间: 3D = 腿+电机. 2D 就删掉第三行. eval 顺序必须和 finetune_and_eval.sh 参数一致.
BOUNDS = np.array([[0.8, 1.2], [0.8, 1.2], [0.6, 1.6]])   # thigh, shank, motor
LABELS = ["thigh", "shank", "motor"][: len(BOUNDS)]
DIM = len(BOUNDS)
SH = "/home/ps/mydrive/xu/finetune_and_eval.sh"
OUTDIR = "/home/ps/mydrive/xu/bo_codesign"


def to_unit(X):
    return (X - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
def from_unit(U):
    return U * (BOUNDS[:, 1] - BOUNDS[:, 0]) + BOUNDS[:, 0]


class GP:
    """极简 GP 回归, RBF kernel, 常数噪声, 归一化 [0,1]^D 空间. 维度无关."""
    def __init__(self, length=0.25, sigma_f=1.0, noise=1e-2):
        self.l, self.sf, self.noise = length, sigma_f, noise
    def _k(self, A, B):
        return self.sf**2 * np.exp(-0.5 * cdist(A, B, "sqeuclidean") / self.l**2)
    def fit(self, U, y):
        self.U = U; self.ymean = y.mean(); self.ystd = y.std() + 1e-6
        yc = (y - self.ymean) / self.ystd
        K = self._k(U, U) + self.noise * np.eye(len(U))
        self.L = cho_factor(K, lower=True); self.alpha = cho_solve(self.L, yc)
        return self
    def predict(self, Us):
        Ks = self._k(Us, self.U)
        mu = Ks @ self.alpha
        v = cho_solve(self.L, Ks.T)
        var = np.clip(self.sf**2 - np.einsum("ij,ji->i", Ks, v), 1e-9, None)
        return mu * self.ystd + self.ymean, np.sqrt(var) * self.ystd


def ei(mu, std, ybest, xi=0.01):
    imp = mu - ybest - xi
    Z = imp / std
    return imp * norm.cdf(Z) + std * norm.pdf(Z)


def eval_candidate(xi, tag, steps, gpu):
    """xi = 物理参数 array [DIM]. 调 finetune_and_eval.sh <p1> <p2> [<p3>] <steps> <tag>."""
    args = " ".join(f"{v:.4f}" for v in xi)
    cmd = f"BO_GPU={gpu} bash {SH} {args} {steps} {tag}"
    print(f"[bo]   -> {cmd}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    line = (r.stdout.strip().splitlines() or [""])[-1]
    m = re.search(r"FITNESS=([-\d.]+)", line)
    fit = float(m.group(1)) if m and m.group(1) != "NA" else None
    print(f"[bo]   <- {line}   ({time.time()-t0:.0f}s)", flush=True)
    return fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_init", type=int, default=8)
    ap.add_argument("--n_bo", type=int, default=12)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    gpu = os.environ.get("BO_GPU", "0")
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    pts = 15 if DIM >= 3 else 41
    grid = np.stack(np.meshgrid(*([np.linspace(0, 1, pts)] * DIM), indexing="ij"), -1).reshape(-1, DIM)
    # 初始: 2^DIM 角点(打乱取前 n_init)+ 随机填充
    corners = np.array(list(itertools.product(*BOUNDS)))
    rng.shuffle(corners)
    init = [c.tolist() for c in corners[: args.n_init]]
    while len(init) < args.n_init:
        init.append(from_unit(rng.rand(DIM)).tolist())

    Xphys, yobs, hist = [], [], []
    print(f"[bo] DIM={DIM} labels={LABELS} n_init={args.n_init} n_bo={args.n_bo} steps={args.steps}", flush=True)
    for i in range(args.n_init):
        xi = np.array(init[i])
        print(f"[bo] init {i+1}/{args.n_init}: {dict(zip(LABELS, np.round(xi,3)))}", flush=True)
        f = eval_candidate(xi, f"i{i}", args.steps, gpu)
        if f is not None:
            Xphys.append(xi.tolist()); yobs.append(f)
        hist.append(dict(kind="init", xi=xi.tolist(), fitness=f))
        json.dump(hist, open(f"{OUTDIR}/history.json", "w"), indent=2)

    for j in range(args.n_bo):
        if len(yobs) < 2:
            print("[bo] too few valid evals, stopping"); break
        U = to_unit(np.array(Xphys)); y = np.array(yobs)
        gp = GP().fit(U, y)
        mu, std = gp.predict(grid)
        u_next = grid[int(ei(mu, std, y.max()).argmax())]
        xi = from_unit(u_next)
        print(f"[bo] BO {j+1}/{args.n_bo}: propose {dict(zip(LABELS, np.round(xi,3)))}  (best {y.max():.2f})", flush=True)
        f = eval_candidate(xi, f"b{j}", args.steps, gpu)
        if f is not None:
            Xphys.append(xi.tolist()); yobs.append(f)
        hist.append(dict(kind="bo", xi=xi.tolist(), fitness=f))
        json.dump(hist, open(f"{OUTDIR}/history.json", "w"), indent=2)

    if yobs:
        bi = int(np.argmax(yobs))
        best = dict(zip(LABELS, [round(v, 3) for v in Xphys[bi]]))
        print(f"[bo] ===== BEST xi* = {best}  fitness={yobs[bi]:.2f}  ({len(yobs)} evals) =====")
        json.dump(dict(best=best, best_fitness=yobs[bi], labels=LABELS, n_eval=len(yobs), history=hist),
                  open(f"{OUTDIR}/result.json", "w"), indent=2)
    else:
        print("[bo] no valid evaluations")


if __name__ == "__main__":
    main()
