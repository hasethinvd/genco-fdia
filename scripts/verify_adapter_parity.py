"""
Adapter parity on a datakit scenario (uses the dataset's OWN admittances/loads, so the
only thing under test is the adapter + model, not the case file).

  A. graphkit's exact pipeline on the raw tables -> model -> compare with datakit's solved
     Vm/Va  (= the model's in-distribution accuracy in this environment)
  B. my adapter's dynamic inputs vs graphkit's input tensors  (feature parity)
  C. my adapter end to end (solve) vs datakit truth

  python scripts/verify_adapter_parity.py <dir with bus_data.parquet> [--scenario 0 1 2]
Run inside a job (needs the GPU node's CUDA libraries).
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import pyarrow.dataset as ds
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grid import load_matpower, PQ, PV, REF
from genco_oracle import GencoOracle
from gridfm_graphkit.datasets.hetero_preprocess import build_hetero_data_for_scenario, merge_agg_gen_into_bus
from gridfm_graphkit.io.param_handler import get_task_transforms
from gridfm_graphkit.datasets import globals as G

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--case", default="data/case118.m")
ap.add_argument("--scenario", type=int, nargs="+", default=[0, 1, 2])
args = ap.parse_args()

raw = os.path.dirname(glob.glob(os.path.join(args.root, "**", "bus_data.parquet"), recursive=True)[0])
g = load_matpower(args.case)
orc = GencoOracle(g)
dev = orc.device
base = g.baseMVA


def tab(name, sc):
    d = ds.dataset(os.path.join(raw, name + ".parquet"), partitioning="hive")
    return d.to_table(filter=ds.field("scenario") == sc).to_pandas()


for sc in args.scenario:
    bus, gen, br = tab("bus_data", sc), tab("gen_data", sc), tab("branch_data", sc)
    bus = bus.sort_values("bus").reset_index(drop=True)
    ref = int(np.where(bus["REF"] > 0)[0][0])
    truth_vm = bus["Vm"].to_numpy(); truth_va = np.deg2rad(bus["Va"].to_numpy())

    # ---- A. graphkit exact pipeline -> model -------------------------------
    bus_m = merge_agg_gen_into_bus(bus, gen)
    data = build_hetero_data_for_scenario(sc, bus_m, gen, br)
    orc.norm.transform(data)
    data = get_task_transforms(orc.args)(data)
    with torch.no_grad():
        out = orc.model({k: v.to(dev) for k, v in data.x_dict.items()},
                        {k: v.to(dev) for k, v in data.edge_index_dict.items()},
                        {k: v.to(dev) for k, v in data.edge_attr_dict.items()},
                        {k: v.to(dev) for k, v in data.mask_dict.items() if k in ("bus", "gen", "PQ", "PV", "REF")})
    o = out["bus"].cpu().numpy()
    print(f"\n--- scenario {sc} (slack Vm {truth_vm[ref]:.4f}, load {bus['Pd'].sum():.0f} MW)")
    print(f"A. graphkit pipeline vs datakit truth : max|dVm|={np.abs(o[:,0]-truth_vm).max():.2e}  "
          f"max|dVa|={np.rad2deg(np.abs(o[:,1]-truth_va)).max():.3f} deg  mean {np.rad2deg(np.abs(o[:,1]-truth_va)).mean():.3f}")

    # ---- B. feature parity: my dynamic inputs vs graphkit's input tensors ---
    gen_on = gen[gen["in_service"] > 0]
    pg = np.zeros(g.n); np.add.at(pg, gen_on["bus"].to_numpy().astype(int), gen_on["p_mw"].to_numpy())
    P = (pg - bus["Pd"].to_numpy()) / base
    Q = (bus["Qg"].to_numpy() - bus["Qd"].to_numpy()) / base
    orc.Pd_base_pu = bus["Pd"].to_numpy() / base            # adapter's PV/REF-bus load = scenario load
    orc.Qd_base_pu = bus["Qd"].to_numpy() / base
    xb, xg, _ = orc._dynamic_inputs(P[None], Q[None], bus["Vm"].to_numpy())
    gk = data.x_dict["bus"].numpy()
    cols = [G.PD_H, G.QD_H, G.QG_H, G.VM_H, G.VA_H, G.PQ_H, G.PV_H, G.REF_H, G.GS, G.BS]
    names = ["Pd", "Qd", "Qg", "Vm", "Va", "PQ", "PV", "REF", "GS", "BS"]
    diffs = {n_: float(np.abs(xb.numpy()[:, c] - gk[:, c]).max()) for n_, c in zip(names, cols)}
    print("B. bus-input parity max|diff| :", {k: f"{v:.1e}" for k, v in diffs.items()})
    # generator Pg aggregated per bus (REF gens masked in both)
    gk_pg = np.zeros(g.n); ge = data.edge_index_dict[("gen", "connected_to", "bus")].numpy()
    np.add.at(gk_pg, ge[1], data.x_dict["gen"].numpy()[ge[0], G.PG_H])
    my_pg = np.zeros(g.n); np.add.at(my_pg, orc.gen_bus, xg.numpy()[:, G.PG_H])
    print(f"   gen Pg per bus parity max|diff| = {np.abs(my_pg - gk_pg).max():.1e}")

    # ---- C. my adapter end to end ------------------------------------------
    orc.t_eattr = data.edge_attr_dict[("bus", "connects", "bus")].clone()    # dataset's (perturbed) admittances
    orc._batched.clear()
    r = orc.solve(P, Q, bus["Vm"].to_numpy())
    print(f"C. my adapter vs datakit truth        : max|dVm|={np.abs(r['Vm']-truth_vm).max():.2e}  "
          f"max|dVa|={np.rad2deg(np.abs(r['Va']-truth_va)).max():.3f} deg")
    print(f"   my adapter vs graphkit pipeline    : max|dVm|={np.abs(r['Vm']-o[:,0]).max():.2e}  "
          f"max|dVa|={np.rad2deg(np.abs(r['Va']-o[:,1])).max():.4f} deg")
