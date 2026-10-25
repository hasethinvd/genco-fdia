"""
T3: IEEE 118 load scenarios through gridfm-datakit -> data/scenarios_118.npz
(arrays P[T,n], Q[T,n]: net bus injections in pu on baseMVA, bus order = case118.m).

  # generate with datakit itself (needs Julia/PowerModels: run `gridfm_datakit setup_pm` once)
  python scripts/make_datakit_scenarios.py --n_scen 720

  # or convert an existing datakit raw directory (bus_data/gen_data parquet)
  python scripts/make_datakit_scenarios.py --from_raw /path/to/raw

Datakit's default load perturbation (agg_load_profile) is kept. Topology, generation
and admittance perturbations are switched OFF so every scenario is the same grid as
data/case118.m with different loads; the experiment's truth is always AC-NR on that grid.
Net injection per bus = sum of in-service generator p_mw at the bus - Pd (P) and
bus Qg - Qd (Q), exactly the columns graphkit consumes.
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys, tempfile
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grid import load_matpower, NewtonRaphsonOracle

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="data/case118.m")
ap.add_argument("--n_scen", type=int, default=720, help="scenarios to generate (>= n_traj*T avoids overlapping windows)")
ap.add_argument("--from_raw", default=None, help="existing datakit raw dir; skips generation")
ap.add_argument("--work_dir", default="data/datakit_out")
ap.add_argument("--network", default="case118_ieee")
ap.add_argument("--setup_pm", action="store_true", help="run `gridfm_datakit setup_pm` first (installs Julia deps)")
ap.add_argument("--base_config", default="data/datakit_default_v1.1.0.yaml",
                help="datakit's default.yaml for the installed version (all keys are required by datakit)")
ap.add_argument("--processes", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/scenarios_118.npz")
args = ap.parse_args()

# ------------------------------------------------------------------ generate
raw = args.from_raw
if raw is None:
    exe = shutil.which("gridfm_datakit") or shutil.which("gridfm-datakit")
    if exe is None:
        sys.exit("gridfm_datakit not on PATH: activate the venv with gridfm-datakit installed")
    if args.setup_pm:
        subprocess.run([exe, "setup_pm"], check=True)
    # start from datakit's own default config (datakit has no defaults for missing keys) and
    # override only what this entry needs; the load perturbation stays at datakit's default.
    cfg = yaml.safe_load(open(args.base_config))
    cfg["network"].update(name=args.network, source="pglib")
    cfg["load"]["scenarios"] = args.n_scen
    for k in ("topology_perturbation", "generation_perturbation", "admittance_perturbation"):
        cfg[k]["type"] = "none"
    cfg["settings"].update(data_dir=os.path.abspath(args.work_dir), mode="pf", num_processes=args.processes,
                           seed=args.seed, overwrite=True)
    cfg_path = os.path.join(tempfile.mkdtemp(), "datakit_118.yaml")
    yaml.safe_dump(cfg, open(cfg_path, "w"))
    print("datakit config overrides: network", cfg["network"], "| load", cfg["load"], "| settings", cfg["settings"])
    subprocess.run([exe, "generate", cfg_path], check=True)
    raw = os.path.join(args.work_dir, args.network, "raw")

# ------------------------------------------------------------------- convert
bus = pd.read_parquet(os.path.join(raw, "bus_data.parquet"))
gen = pd.read_parquet(os.path.join(raw, "gen_data.parquet"))
g = load_matpower(args.case)
n = g.n
assert bus.groupby("scenario")["bus"].count().eq(n).all(), \
    f"datakit grid has a different bus count than {args.case} ({n})"
bus = bus.sort_values(["scenario", "bus"])
scen = np.sort(bus["scenario"].unique())

gen = gen[gen["in_service"] > 0]
pg = (gen.groupby(["scenario", "bus"])["p_mw"].sum()
         .unstack("bus", fill_value=0.0).reindex(index=scen, columns=range(n), fill_value=0.0))
piv = lambda col: bus.pivot(index="scenario", columns="bus", values=col).reindex(index=scen, columns=range(n))
Pd, Qd, Qg = piv("Pd").to_numpy(), piv("Qd").to_numpy(), piv("Qg").to_numpy()
P = (pg.to_numpy() - Pd) / g.baseMVA
Q = (Qg - Qd) / g.baseMVA

# sanity: how does the datakit load level compare with the case file, and does our NR converge?
print(f"{len(scen)} scenarios, n={n}")
print(f"total load: case file {g.bus[:, 2].sum():.0f} MW | datakit mean {Pd.sum(1).mean():.0f} MW "
      f"(min {Pd.sum(1).min():.0f}, max {Pd.sum(1).max():.0f})")
nr = NewtonRaphsonOracle(g)
# Per-scenario voltage setpoints at generator (PV) and slack buses: in datakit these are
# OPF outputs that vary per scenario, and they are inputs to both NR and GENCO.
Vm = piv("Vm").to_numpy()
Vset = np.where(g.bus_type == 1, 1.0, Vm)
print(f"slack Vm: mean {Vset[:, g.bus_type == 3].mean():.4f} (case file {g.vset()[g.bus_type == 3][0]:.4f}) | "
      f"PV Vm range {Vset[:, g.bus_type == 2].min():.3f}-{Vset[:, g.bus_type == 2].max():.3f}")
conv = np.array([nr.solve(P[i], Q[i], Vset[i])["converged"] for i in range(len(P))])
print(f"AC-NR (truth) converges on {conv.mean() * 100:.1f}% of scenarios")
if not conv.all():
    print("WARNING: non-converged scenarios will truncate trajectories in run_experiment.py")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
np.savez(args.out, P=P, Q=Q, Vset=Vset, converged=conv, scenario=scen)
print("saved", args.out, "P", P.shape, "Q", Q.shape, "Vset", Vset.shape)
