#!/usr/bin/env python3
"""Compare two table2_eval.py outputs side by side (pure numpy, no Isaac needed).

Usage: python legged_gym/scripts/table2_compare.py <spatialDR.npz> <noDR.npz> [name1 name2]
Prints per-scenario mean reward / fall% and the Table-II headline (mean-of-bucket-means).
Success criterion (plan / Chen et al. Table II): spatial-DR beats No-DR on held-out morphologies.
"""
import sys
import json
import numpy as np

SCEN_NAMES = ["obstacle", "slope", "load", "accel"]


def load(path):
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    return d, meta


def bucket_means(d):
    b, r = d["bucket"], d["rew"]
    return np.array([r[b == x].mean() for x in np.unique(b)])


def main():
    p1, p2 = sys.argv[1], sys.argv[2]
    n1 = sys.argv[3] if len(sys.argv) > 3 else "spatial-DR"
    n2 = sys.argv[4] if len(sys.argv) > 4 else "No-DR"
    d1, m1 = load(p1)
    d2, m2 = load(p2)
    for m, p in ((m1, p1), (m2, p2)):
        print(f"[cmp] {p}: run={m['load_run']} ckpt={m['checkpoint']} buckets={m['buckets']} "
              f"difficulty={m['difficulty']} eps={len(np.load(p)['rew'])}")
    if m1["difficulty"] != m2["difficulty"] or m1["morph_seed"] != m2["morph_seed"]:
        print("[cmp] !! WARNING: difficulty or morph_seed differ — comparison is NOT apples-to-apples")
    print(f"[cmp] {'scenario':<10} | {n1:>16} | {n2:>16} | {'Δrew':>8}")
    print(f"[cmp] {'':<10} | {'rew':>8} {'fall%':>6} | {'rew':>8} {'fall%':>6} |")
    for k, name in enumerate(SCEN_NAMES):
        a, b = d1["scen"] == k, d2["scen"] == k
        if a.sum() == 0 or b.sum() == 0:
            print(f"[cmp] {name:<10} | (missing episodes)"); continue
        r1, f1 = d1["rew"][a].mean(), 100 * d1["fell"][a].mean()
        r2, f2 = d2["rew"][b].mean(), 100 * d2["fell"][b].mean()
        print(f"[cmp] {name:<10} | {r1:>8.2f} {f1:>5.1f}% | {r2:>8.2f} {f2:>5.1f}% | {r1-r2:>+8.2f}")
    r1, r2 = d1["rew"].mean(), d2["rew"].mean()
    f1, f2 = 100 * d1["fell"].mean(), 100 * d2["fell"].mean()
    print(f"[cmp] {'ALL':<10} | {r1:>8.2f} {f1:>5.1f}% | {r2:>8.2f} {f2:>5.1f}% | {r1-r2:>+8.2f}")
    bm1, bm2 = bucket_means(d1), bucket_means(d2)
    print(f"[cmp] ★ mean-of-bucket-means: {n1}={bm1.mean():.2f}  {n2}={bm2.mean():.2f}  Δ={bm1.mean()-bm2.mean():+.2f}")
    if len(bm1) == len(bm2):
        wins = int((bm1 > bm2).sum())
        print(f"[cmp] per-morphology wins: {n1} wins {wins}/{len(bm1)} held-out morphologies")
    verdict = "SPATIAL-DR WINS (Table-II criterion met)" if bm1.mean() > bm2.mean() else "spatial-DR does NOT beat No-DR"
    print(f"[cmp] VERDICT: {verdict}")


if __name__ == "__main__":
    main()
