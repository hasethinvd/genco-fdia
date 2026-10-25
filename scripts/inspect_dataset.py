"""
Read-only diagnosis: is datakit's case118_ieee the same grid as data/case118.m,
and how accurate is GENCO against datakit's own ground truth?

  python scripts/inspect_dataset.py <dir containing bus_data.parquet etc.> [--scenario 0]

1. grid comparison : slack/PV voltage setpoints, slack angle, loads, generators,
                     shunts, branch order + admittances + taps
2. our NR on the dataset's own injections vs the dataset's solved Vm/Va
                   (grid equivalence test, independent of GENCO)
3. GENCO on the dataset's own injections vs the dataset's solved Vm/Va
                   (does the adapter reproduce the model's in-distribution accuracy?)
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import pyarrow.dataset as ds

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grid import load_matpower, NewtonRaphsonOracle, build_ybus, PQ, PV, REF

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--case", default="data/case118.m")
ap.add_argument("--scenario", type=int, default=0)
ap.add_argument("--no_genco", action="store_true")
args = ap.parse_args()

hits = glob.glob(os.path.join(args.root, "**", "bus_data.parquet"), recursive=True)
if not hits:
    sys.exit(f"no bus_data.parquet under {args.root} (download may still be running)")
raw = os.path.dirname(hits[0]); print("raw dir:", raw)


def table(name):
    d = ds.dataset(os.path.join(raw, name + ".parquet"), partitioning="hive")
    return d.to_table(filter=ds.field("scenario") == args.scenario).to_pandas()   # keep file order (branch order matters)

bus, gen, br = table("bus_data"), table("gen_data"), table("branch_data")
g = load_matpower(args.case)
n = g.n
print(f"dataset: {len(bus)} buses, {len(gen)} gens, {len(br)} branches | ours: {n}, {int((g.gen[:,7]>0).sum())}, {len(g.branch)}")
bus = bus.sort_values("bus").reset_index(drop=True)
bt_ds = np.where(bus["REF"] > 0, REF, np.where(bus["PV"] > 0, PV, PQ))

print("\n== 1. grid comparison ==")
print("bus types equal:", bool((bt_ds == g.bus_type).all()),
      "| ref bus ds/ours:", int(np.where(bt_ds == REF)[0][0]), int(np.where(g.bus_type == REF)[0][0]))
ref = int(np.where(g.bus_type == REF)[0][0])
print(f"slack Vm   dataset {bus['Vm'][ref]:.4f}   ours (gen Vg) {g.vset()[ref]:.4f}")
print(f"slack Va   dataset {bus['Va'][ref]:.4f} deg ours 0 (case file bus Va {g.bus[ref,8]:.2f})")
pvm = bt_ds == PV
print(f"PV Vm      max |dataset - ours| = {np.abs(bus['Vm'].to_numpy()[pvm] - g.vset()[pvm]).max():.4f}")
print(f"total Pd   dataset {bus['Pd'].sum():.1f} MW   ours {g.bus[:,2].sum():.1f} MW")
print(f"total Qd   dataset {bus['Qd'].sum():.1f}      ours {g.bus[:,3].sum():.1f}")
print(f"shunt GS/BS max |diff| (pu on 100): {np.abs(bus['GS'] - g.bus[:,4]/g.baseMVA).max():.4f} / "
      f"{np.abs(bus['BS'] - g.bus[:,5]/g.baseMVA).max():.4f}")
gen_on = gen[gen["in_service"] > 0]
print("gen buses equal:", sorted(gen_on["bus"].tolist()) == sorted(g.gen_bus_idx().tolist()),
      f"| total Pg dataset {gen_on['p_mw'].sum():.1f} MW, ours {g.gen[g.gen[:,7]>0,1].sum():.1f} MW")
f, t = g.from_to
same_order = (br["from_bus"].to_numpy() == f).all() and (br["to_bus"].to_numpy() == t).all()
print("branch from/to order equal:", bool(same_order), f"| tap dataset unique {np.unique(np.round(br['tap'],4))[:6]} ours "
      f"{np.unique(np.where(g.branch[:,8]==0,1,g.branch[:,8]))[:6]}")
if same_order:
    _, (Yff, Yft, Ytf, Ytt) = build_ybus(g)
    print(f"Yff max |diff| {np.abs(br['Yff_r'].to_numpy() + 1j*br['Yff_i'].to_numpy() - Yff).max():.4f}   "
          f"Yft {np.abs(br['Yft_r'].to_numpy() + 1j*br['Yft_i'].to_numpy() - Yft).max():.4f}")

print("\n== 2. our AC-NR on the dataset's injections vs dataset truth ==")
base = g.baseMVA
pg = np.zeros(n); pg[gen_on["bus"].to_numpy().astype(int)] += gen_on["p_mw"].to_numpy()
P = (pg - bus["Pd"].to_numpy()) / base
Q = (bus["Qg"].to_numpy() - bus["Qd"].to_numpy()) / base
Vset = np.where(bt_ds == PQ, 1.0, bus["Vm"].to_numpy())          # use the dataset's own setpoints
nr = NewtonRaphsonOracle(g)
r = nr.solve(P, Q, Vset)
dVm = np.abs(r["Vm"] - bus["Vm"].to_numpy())
dVa = np.abs(np.rad2deg(r["Va"]) - (bus["Va"].to_numpy() - bus["Va"][ref]))
print(f"NR converged={r['converged']}  max|dVm|={dVm.max():.2e}  max|dVa|={dVa.max():.3f} deg   (small => same physical grid)")

if not args.no_genco:
    print("\n== 3. GENCO on the dataset's injections vs dataset truth ==")
    from genco_oracle import GencoOracle
    orc = GencoOracle(g, device="cpu", va_ref_deg=float(bus["Va"][ref]))
    o = orc.solve(P, Q, Vset)
    eVm = np.abs(o["Vm"] - bus["Vm"].to_numpy()); eVa = np.abs(np.rad2deg(o["Va"]) - bus["Va"].to_numpy())
    print(f"GENCO  max|dVm|={eVm.max():.2e} (slack {eVm[ref]:.2e}, PQ {eVm[bt_ds==PQ].max():.2e})  "
          f"max|dVa|={eVa.max():.3f} deg mean {eVa.mean():.3f}")
    print(f"GENCO slack Vm predicted {o['Vm'][ref]:.4f} vs dataset {bus['Vm'][ref]:.4f}")
