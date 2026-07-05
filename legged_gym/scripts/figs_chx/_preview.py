#!/usr/bin/env python3
"""
figs_chx/_preview.py — matplotlib 预览（非交付物；交付物是同目录 .m MATLAB 脚本）。
读 data/codesign_figs_data.mat，出 5 张免费/几乎免费图到 preview/。
labels 用中文（Noto Sans CJK）。数值与 MATLAB 脚本完全一致（同一份 .mat）。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "AR PL UMing CN"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 160

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "preview"); os.makedirs(OUT, exist_ok=True)
D = loadmat(os.path.join(HERE, "data", "codesign_figs_data.mat"), squeeze_me=True)

SCEN = ["越障 obstacle", "爬坡 slope", "负载 load", "加速 accel"]
SCOL = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]   # 4 场景配色


def sq(k):
    return np.atleast_1d(D[k])


# ============ Fig 1: BO 收敛 ============
def fig_bo_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, dim, col in [(axes[0], "2d", "#2166ac"), (axes[1], "3d", "#b2182b")]:
        fit = sq(f"bo{dim}_fit"); order = sq(f"bo{dim}_order")
        isinit = sq(f"bo{dim}_isinit").astype(bool)
        o = np.argsort(order); fit, isinit = fit[o], isinit[o]
        x = np.arange(1, len(fit) + 1)
        best = np.maximum.accumulate(fit)
        ax.scatter(x[isinit], fit[isinit], c="#888", s=42, zorder=3,
                   label="初始采样 (init)", edgecolor="k", linewidth=.4)
        ax.scatter(x[~isinit], fit[~isinit], c=col, s=48, zorder=3,
                   label="BO 提议 (EI)", edgecolor="k", linewidth=.4)
        ax.step(x, best, where="post", color=col, lw=2, alpha=.8,
                label="best-so-far")
        ni = int(isinit.sum())
        ax.axvline(ni + .5, ls="--", c="gray", lw=1)
        bi = int(np.argmax(fit))
        ax.annotate(f"ξ*  fit={fit[bi]:.1f}", (x[bi], fit[bi]),
                    xytext=(x[bi], fit[bi] + (12 if dim == "3d" else 8)),
                    ha="center", fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=col))
        lbl = "2D BO (仅腿长)" if dim == "2d" else "3D BO (腿长 + 电机)"
        ax.set_title(lbl, fontsize=11)
        ax.set_xlabel("评估序号 (每候选微调400步)")
        ax.set_ylabel("fitness  f(ξ) = 4场景等权 mean_rew")
        ax.grid(alpha=.25); ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("BO 收敛曲线 — 预训练generalist 微调400步/候选", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_bo_convergence.png"), bbox_inches="tight")
    plt.close(fig)


# ============ Fig 2: Table-II 柱状 (spatial-DR vs No-DR) ============
def fig_table2():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (tag, ttl) in zip(axes, [("s2_d05", "seed2 · 难度0.5 · 重标定slope[1,10]"),
                                      ("s2_d03", "seed2 · 难度0.3 · 重标定slope[1,10]")]):
        B = sq(f"t2_B_{tag}_mean"); Bs = sq(f"t2_B_{tag}_std")
        N = sq(f"t2_noDR_{tag}_mean"); Ns = sq(f"t2_noDR_{tag}_std")
        Bm, Nm = float(np.mean(B)), float(np.mean(N))
        cats = SCEN + ["综合 fitness"]
        Bv = list(B) + [Bm]; Nv = list(N) + [Nm]
        Be = list(Bs) + [0]; Ne = list(Ns) + [0]
        x = np.arange(len(cats)); w = .38
        b1 = ax.bar(x - w/2, Bv, w, yerr=Be, capsize=3, color="#2166ac",
                    label="spatial-DR (B, 训练时64随机腿形态)", edgecolor="k", lw=.4)
        b2 = ax.bar(x + w/2, Nv, w, yerr=Ne, capsize=3, color="#d6604d",
                    label="No-DR (单一标称形态1.0/1.0)", edgecolor="k", lw=.4)
        for i in range(len(cats)):
            d = Bv[i] - Nv[i]
            ax.annotate(f"+{d:.0f}" if d >= 0 else f"{d:.0f}",
                        (x[i], max(Bv[i], Nv[i])), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8,
                        color="#178217" if d > 0 else "#b00",
                        fontweight="bold")
        ax.axhline(0, color="k", lw=.6)
        ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9, rotation=12)
        ax.set_ylabel("100 留出形态 mean-of-bucket-means")
        ax.set_title(ttl, fontsize=10)
        ax.grid(axis="y", alpha=.25)
        if ax is axes[0]:
            ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(1.05, 1.22), ncol=2)
    fig.suptitle("Table-II 复现: spatial-DR 泛化 击败 No-DR (逐场景 + 综合)", fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_table2_bars.png"), bbox_inches="tight")
    plt.close(fig)


# ============ Fig 3: 微调效率曲线 ============
def fig_finetune():
    steps = sq("ft_steps"); fit = sq("ft_fit"); scen = np.atleast_2d(D["ft_scen"])
    o = np.argsort(steps); steps, fit, scen = steps[o], fit[o], scen[o]
    is_spec = steps >= 6000
    ft_s, ft_f = steps[~is_spec], fit[~is_spec]
    spec = float(fit[is_spec][0]) if is_spec.any() else np.nan
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    # 左: 综合 fitness
    ax.axhline(spec, ls="--", color="#444", lw=1.5,
               label=f"specialist 上界 (+6000步) = {spec:.1f}")
    ax.plot(ft_s, ft_f, "-o", color="#b2182b", lw=2, ms=7,
            label="generalist 微调 (从 B)", zorder=3)
    for xs, fs in zip(ft_s, ft_f):
        pct = fs / spec * 100
        ax.annotate(f"{pct:.0f}%", (xs, fs), textcoords="offset points",
                    xytext=(9, -4), ha="left", fontsize=8, color="#b2182b")
    ax.axvline(400, ls=":", color="gray", lw=1)
    ax.set_xlabel("微调步数 (iter)"); ax.set_ylabel("fitness (单形态 ξ=(1.2,0.8), 难度0.3)")
    ax.set_title("微调效率: +400步 = specialist 89%, +1500步 追平", fontsize=10)
    ax.grid(alpha=.25); ax.legend(fontsize=8.5, loc="lower right")
    # 右: 逐场景
    for k in range(4):
        ax2.plot(ft_s, scen[~is_spec, k], "-o", color=SCOL[k], lw=1.6, ms=5, label=SCEN[k])
        ax2.axhline(scen[is_spec, k][0] if is_spec.any() else np.nan,
                    ls=":", color=SCOL[k], lw=1, alpha=.6)
    ax2.set_xlabel("微调步数 (iter)"); ax2.set_ylabel("逐场景 mean_rew")
    ax2.set_title("逐场景微调轨迹 (虚线=specialist)", fontsize=10)
    ax2.grid(alpha=.25); ax2.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_finetune_curve.png"), bbox_inches="tight")
    plt.close(fig)


# ============ Fig 4: 逐场景 mean_rew vs 电机 (局部极值揭示) ============
def fig_scen_vs_motor():
    # 定腿 ξ*=(0.914,0.8) 电机扫描: 微调(每k+400) vs 零样本(固定 B')。
    mk, mfit = sq("msweep_motor"), sq("msweep_fit")            # 零样本 11 点
    fk, ffit = sq("ftmotor_motor"), sq("ftmotor_fit")          # 微调 6 点
    fscen = np.atleast_2d(D["ftmotor_scen"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # A: 综合 fitness 内部最优
    ax = axes[0]
    ax.plot(fk, ffit, "-o", color="#b2182b", lw=2.2, ms=8, label="微调 (为每个设计 +400步)")
    ax.plot(mk, mfit, "-s", color="#4393c3", lw=1.8, ms=5, label="零样本 (固定 B' generalist)")
    bi = int(np.argmax(ffit))
    ax.axvline(fk[bi], ls=":", color="#b2182b", alpha=.6)
    ax.annotate(f"内部最优 k≈{fk[bi]:.1f}\n(=3D BO 的 ξ*电机)", (fk[bi], ffit[bi]),
                xytext=(fk[bi] + 0.12, ffit[bi] - 14), fontsize=9, color="#b2182b",
                arrowprops=dict(arrowstyle="->", color="#b2182b"))
    ax.set_xlabel("电机扭矩缩放 k (+ ENCOS 物理质量代价)")
    ax.set_ylabel("综合 fitness (4场景等权, 定腿 ξ*=(0.914,0.8))")
    ax.set_title("综合 fitness 有内部最优 (仅微调后显现)", fontsize=10)
    ax.grid(alpha=.25); ax.legend(fontsize=9, loc="lower right")
    # B: 逐场景机制 (相对 k=0.6 的 % 变化, 因绝对量级差大)
    ax2 = axes[1]
    rel = (fscen / fscen[0:1, :] - 1.0) * 100
    for k in range(4):
        style = "-o" if k in (1, 2) else "--o"   # 扭矩型实线, 质量型虚线
        ax2.plot(fk, rel[:, k], style, color=SCOL[k], lw=1.8, ms=6, label=SCEN[k])
    ax2.axhline(0, color="k", lw=.6)
    ax2.set_xlabel("电机扭矩缩放 k")
    ax2.set_ylabel("逐场景 相对 k=0.6 的变化 (%)")
    ax2.set_title("机制: 质量型 obstacle/accel↓ · 扭矩型 slope/load↑饱和", fontsize=10)
    ax2.grid(alpha=.25); ax2.legend(fontsize=8, ncol=2)
    fig.suptitle("电机的 \"no free torque\": 弱电机差、最大电机不更好、最优 k≈1.0-1.1 "
                 "(质量代价 vs 扭矩收益饱和 的交汇)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_scenario_vs_motor.png"), bbox_inches="tight")
    plt.close(fig)


# ============ Fig 5: 逐场景 vs (大腿,小腿) 地形 (2D BO) ============
def fig_scen_landscape():
    import matplotlib.tri as mtri
    th, sh = sq("bo2d_thigh"), sq("bo2d_shank")
    scen = np.atleast_2d(D["bo2d_scen"]); fit = sq("bo2d_fit")
    bi = int(np.argmax(fit))
    panels = [(0, SCEN[0]), (1, SCEN[1]), (2, SCEN[2]), (3, SCEN[3]), (-1, "综合 fitness")]
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.9))
    tri = mtri.Triangulation(th, sh)
    for ax, (k, ttl) in zip(axes, panels):
        z = fit if k == -1 else scen[:, k]
        tcf = ax.tricontourf(tri, z, levels=12, cmap="viridis")
        ax.scatter(th, sh, c="w", s=14, edgecolor="k", lw=.4, zorder=3)
        ax.scatter(th[bi], sh[bi], marker="*", c="red", s=180, edgecolor="k",
                   lw=.6, zorder=4, label="ξ*")
        fig.colorbar(tcf, ax=ax, fraction=.046, pad=.04)
        ax.set_title(ttl, fontsize=10); ax.set_xlabel("大腿缩放 s_t")
        if ax is axes[0]:
            ax.set_ylabel("小腿缩放 s_s")
        ax.set_xlim(.78, 1.22); ax.set_ylim(.78, 1.22)
    fig.suptitle("逐场景性能地形 vs 腿长 (2D BO 12点, 三角插值) — 各场景均偏短腿(低CoM稳); "
                 "负载/爬坡对腿长最敏感, 越障相对不敏感(固定净高proxy无\"够得着\"激励) → ξ*落短腿",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_scenario_landscape.png"), bbox_inches="tight")
    plt.close(fig)


# ============ Fig 6: 抬腿峰值净高 vs (大腿,小腿) [路B-A] ============
def fig_foot_height():
    import matplotlib.tri as mtri
    tu, su = sq("footh_bth"), sq("footh_bsh")
    peak_u, fall_u = sq("footh_bpeak"), sq("footh_bfall")
    target = float(np.atleast_1d(D["footh_target"])[0])
    tri = mtri.Triangulation(tu, su)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, z, ttl, cb, cmap in [
        (axes[0], peak_u, f"峰值抬脚净高 (m) — 命令目标 {target:.2f}m", "峰值净高 (m)", "viridis"),
        (axes[1], fall_u, "摔倒率 (%)", "fall %", "magma")]:
        tcf = ax.tricontourf(tri, z, levels=14, cmap=cmap)
        ax.scatter(tu, su, c="w", s=6, edgecolor="k", lw=.3, zorder=3)
        fig.colorbar(tcf, ax=ax, fraction=.046, pad=.04).set_label(cb)
        ax.set_xlabel("大腿缩放 s_t"); ax.set_title(ttl, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("小腿缩放 s_s")
        ax.set_xlim(.8, 1.2); ax.set_ylim(.8, 1.2); ax.set_aspect("equal")
    fig.suptitle("抬腿净高 vs 腿长 (全obstacle场景, B'零样本, 1024env×120形态) — "
                 "长腿抬得更高(0.23→0.30m)但更易摔 → proxy漏掉的\"够得着 vs 稳\"权衡",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_foot_height.png"), bbox_inches="tight")
    plt.close(fig)


# ============ Fig 7: ξ* 鲁棒性 (平坦盆地 + 权重敏感) ============
def fig_robustness():
    fit = sq("fig7_fit"); mo = sq("fig7_mo")
    schemes = np.atleast_2d(D["fig7_schemes"])          # 5x3 thigh/shank/motor
    names = [str(x) for x in np.atleast_1d(D["fig7_scheme_names"])]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    # A: 平坦盆地
    ax = axes[0]
    order = np.argsort(-fit); fs = fit[order]; ms = mo[order]
    x = np.arange(1, len(fs) + 1)
    best = fs[0]; band = fs >= best - 3.0
    ax.bar(x[band], fs[band], color="#b2182b", label="最优盆地 (差<3≈噪声)")
    ax.bar(x[~band], fs[~band], color="#bbbbbb", label="其余候选")
    ax.axhline(best - 3.0, ls="--", color="#b2182b", lw=1)
    for xi, fi, mi in zip(x[band], fs[band], ms[band]):
        ax.annotate(f"k={mi:.2f}", (xi, fi), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7.5, rotation=90, color="#7a0a0a")
    ax.set_xlabel("候选排名 (fitness 降序)"); ax.set_ylabel("综合 fitness")
    ax.set_ylim(110, 235)
    ax.set_title(f"最优是平坦盆地: 前{int(band.sum())}名差<3, 却横跨 motor 0.89-1.46", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=.25)
    # B: 权重敏感
    ax2 = axes[1]
    w = 0.25; xb = np.arange(len(names))
    cols = ["#1b9e77", "#7570b3", "#e6194b"]
    labs = ["大腿 thigh", "小腿 shank", "电机 motor"]
    for j in range(3):
        ax2.bar(xb + (j - 1) * w, schemes[:, j], w, color=cols[j], label=labs[j], edgecolor="k", lw=.3)
    ax2.axhline(1.0, ls=":", color="gray", lw=1)
    ax2.set_xticks(xb); ax2.set_xticklabels(names, fontsize=8.5, rotation=12)
    ax2.set_ylabel("最优设计值 (缩放)"); ax2.set_ylim(0.7, 1.25)
    ax2.set_title("权重敏感: shank恒0.8, thigh 0.8-0.91, motor 0.96-1.1", fontsize=9.5)
    ax2.legend(fontsize=8, ncol=3, loc="upper center"); ax2.grid(axis="y", alpha=.25)
    fig.suptitle("ξ* 鲁棒性: 报成\"区域\"(短腿 shank≈0.8 + 中等电机~1.0), 非过度精确的点 — "
                 "电机维弱约束(盆地平坦) + 权重稳健", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_robustness.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_robustness();    print("✓ fig7 鲁棒性")
    fig_bo_convergence(); print("✓ fig1 BO收敛")
    fig_table2();         print("✓ fig2 Table-II")
    fig_finetune();       print("✓ fig3 微调曲线")
    fig_scen_vs_motor();  print("✓ fig4 场景vs电机")
    fig_scen_landscape(); print("✓ fig5 场景地形")
    fig_foot_height();    print("✓ fig6 抬腿净高")
    print("→", OUT)
