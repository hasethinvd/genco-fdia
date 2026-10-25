"""
python scripts/make_figures.py results/results_nr_ramp1.json [results/results_nr_ramp5.json]
Writes results/fig1_detection_vs_knowledge.png and results/fig2_bus_map.png
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grid import load_matpower

argv = sys.argv[1:]
tag = ""
if "--tag" in argv:                       # optional suffix so runs don't overwrite each other's figures
    i = argv.index("--tag"); tag = "_" + argv[i + 1]; del argv[i:i + 2]
files = argv or ["results/results_nr_ramp1.json"]
R = [json.load(open(f)) for f in files]
os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------- figure 1
fig, axes = plt.subplots(1, len(R), figsize=(5.2 * len(R), 4), sharey=True, squeeze=False)
for ax, r in zip(axes[0], R):
    k = [l["k"] for l in r["levels"]]
    style = dict(marker="o", lw=2)
    ax.plot(k, [l["det_phys"] for l in r["levels"]], label="physics check (oracle named in panel title)", color="#1f77b4", **style)
    ax.plot(k, [l["det_dc"] for l in r["levels"]], label="physics check (DC baseline)", color="#7f7f7f", ls="--", **style)
    ax.plot(k, [l["det_pmu"] for l in r["levels"]], label=f"PMU cross-check ({int(r['pmu_frac']*100)}% buses)", color="#ff7f0e", **style)
    ax.plot(k, [l["det_jump"] for l in r["levels"]], label="temporal state-jump", color="#bcbd22", ls=":", **style)
    ax.plot(k, [l["det_fused"] for l in r["levels"]], label="physics + PMU (fused)", color="#2ca02c", lw=3, marker="s")
    ax.axhline(r["fpr_target"], color="k", lw=0.8, ls="-.")
    ax.text(0.02, r["fpr_target"] + 0.02, f"FPR target {r['fpr_target']:.0%}", fontsize=8)
    ax.set_xlabel("attacker knowledge of grid model  k")
    ramp = r.get("ramp", 1)
    ax.set_title(f"oracle: {r['oracle']}\n"
                 f"{r['case']}: {'step attack' if ramp == 1 else f'attack ramped over {ramp} scans'}, "
                 f"hidden load ≈ {np.mean([l['hidden_mw'] for l in r['levels'][1:]]):.0f} MW", fontsize=10)
    ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.3)
axes[0][0].set_ylabel("detection rate")
axes[0][-1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
fig.savefig(f"results/fig1_detection_vs_knowledge{tag}.png", dpi=160, bbox_inches="tight")
print(f"wrote results/fig1_detection_vs_knowledge{tag}.png")

# ---------------------------------------------------------------- figure 2
r = R[0]
ex = r["example"]
if ex:
    g = load_matpower("data/case118.m")
    f, t = g.from_to
    # spring-ish layout via spectral embedding of the Laplacian (no networkx)
    n = g.n
    A = np.zeros((n, n)); A[f, t] = 1; A[t, f] = 1
    L = np.diag(A.sum(1)) - A
    w, v = np.linalg.eigh(L)
    xy = v[:, 1:3]
    phys = np.array(ex["per_bus"]); pmu = np.array(ex["per_bus_pmu"])
    thr_p, thr_m = r["thresholds"]["phys"], r["thresholds"]["pmu"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, score, thr, title in [(axes[0], phys, thr_p, "physics-consistency residual"),
                                  (axes[1], pmu, thr_m, "PMU line-flow cross-check")]:
        for a, b in zip(f, t):
            ax.plot(xy[[a, b], 0], xy[[a, b], 1], color="#cccccc", lw=0.6, zorder=1)
        c = np.clip(score / max(thr, 1e-9), 0, 3)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap="Reds", vmin=0, vmax=3, s=28, zorder=2, edgecolor="k", lw=0.3)
        ax.scatter(xy[ex["interior"], 0], xy[ex["interior"], 1], s=140, facecolor="none",
                   edgecolor="#d62728", lw=1.8, zorder=3, label="attacker-controlled buses")
        ax.scatter(xy[ex["boundary"], 0], xy[ex["boundary"], 1], s=140, facecolor="none",
                   edgecolor="#1f77b4", lw=1.2, ls="--", zorder=3, label="boundary buses")
        if "pmu_buses" in r and "PMU" in title:
            pb = r["pmu_buses"]
            ax.scatter(xy[pb, 0], xy[pb, 1], marker="^", s=50, facecolor="none",
                       edgecolor="#ff7f0e", lw=1.2, zorder=4, label="PMU")
        ax.set_title(f"{title}\nmax = {score.max() / max(thr, 1e-9):.1f} × threshold")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, label="score / threshold")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Full-knowledge (k=1) stealthy FDIA on {r['case']}: invisible to the physics check, "
                 f"visible on the independent PMU channel", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"results/fig2_bus_map{tag}.png", dpi=160, bbox_inches="tight")
    print(f"wrote results/fig2_bus_map{tag}.png")
