"""
Figure 3: clean-data physics threshold per oracle (left) and the fine-tuning curve (right).

  python scripts/make_fig3.py [--log results/finetune_log.json] [--out results/fig3_threshold_and_finetune.png]

Missing result files are skipped, so it can be run on a partial set of results.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/fig3_threshold_and_finetune.png")
args = ap.parse_args()

# (label, results file, colour); step-attack runs, thresholds are calibrated on clean scans only
BARS = [
    ("AC-NR\n(datakit)", "results/results_nr_ramp1_datakit.json", "#2ca02c"),
    ("GENCO\nreleased\n(datakit)", "results/results_genco_ramp1_datakit.json", "#1f77b4"),
    ("GENCO ft\ncase-file only\n(datakit)", "results/results_genco_ftwalk_ramp1_datakit.json", "#c5b0d5"),
    ("GENCO ft\nboth regimes\n(datakit)", "results/results_genco_ft_mix_ramp1_datakit.json", "#9467bd"),
    ("AC-NR\n(case file)", "results/results_nr_ramp1.json", "#98df8a"),
    ("GENCO\nreleased\n(case file)", "results/results_genco_ramp1.json", "#aec7e8"),
    ("GENCO ft\ncase-file only\n(case file)", "results/results_genco_ftwalk_ramp1.json", "#c5b0d5"),
    ("GENCO ft\nboth regimes\n(case file)", "results/results_genco_ft_mix_ramp1.json", "#9467bd"),
    ("DC\n(datakit)", "results/results_nr_ramp1_datakit.json", "#7f7f7f"),
]
LOGS = [("case-file walks only", "results/finetune_log_walk.json", "#c5b0d5"),
        ("both regimes", "results/finetune_log_mix.json", "#9467bd")]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
xs, vals, labels, cols = [], [], [], []
for lab, f, c in BARS:
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    key = "dc" if lab.startswith("DC") else "phys"
    vals.append(r["thresholds"][key]); labels.append(lab); cols.append(c)
x = np.arange(len(vals))
ax.bar(x, vals, color=cols, edgecolor="k", lw=0.5)
for xi, v in zip(x, vals):
    ax.text(xi, v * 1.05, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", fontsize=9)
ax.set_yscale("log"); ax.set_ylim(5, max(vals) * 1.8); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("clean-data physics threshold at 1% FPR\n(noise-normalised units, log scale)")
ax.set_title("Oracle error floor sets the detection threshold", fontsize=10)
ax.grid(axis="y", alpha=0.3)

plotted = False
for lab, f, col in LOGS:
    if not os.path.exists(f):
        continue
    L = json.load(open(f)); h = L["history"]
    t = [0.0] + [e["minutes"] for e in h]
    ax2.plot(t, [L["before"]["p99"]] + [e["p99"] for e in h], color=col, lw=2, label=f"{lab}: p99")
    ax2.plot(t, [L["before"]["median"]] + [e["median"] for e in h], color=col, lw=1.5, ls="--", label=f"{lab}: median")
    plotted = True
if plotted:
    ax2.set_yscale("log"); ax2.set_xlabel("fine-tuning wall-clock (minutes, one GB200)")
    ax2.set_ylabel("held-out per-scan max |residual| (sensor-sigma units)")
    ax2.set_title("Fine-tuning curves (held-out scans)", fontsize=10)
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3, which="both")
else:
    ax2.text(0.5, 0.5, "fine-tuning logs not found", ha="center", transform=ax2.transAxes); ax2.axis("off")

fig.tight_layout()
fig.savefig(args.out, dpi=160, bbox_inches="tight")
print("wrote", args.out, "| bars:", [l.replace("\n", " ") for l in labels])
