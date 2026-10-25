"""
T2 acceptance: GencoOracle vs NewtonRaphsonOracle.

  Base case:   Vm within ~1e-3 pu, Va within ~0.1 deg (task spec).
  Perturbed:   same metrics on load-trajectory scans (what the experiment feeds it).
  Timing:      ms per solve at batch 1 and batch 64.
Two variants are reported:
  masked-slack : faithful to the checkpoint (PF task masks the slack Vm; model infers it)
  slack-given  : slack Vm unmasked and set to the case setpoint (operator knows it)

  python scripts/test_genco_oracle.py [--n_scans 64] [--device cuda]
Writes results/genco_acceptance.json.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grid import load_matpower, NewtonRaphsonOracle, PQ, PV, REF
from attacks import load_trajectory
from genco_oracle import GencoOracle

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="data/case118.m")
ap.add_argument("--n_scans", type=int, default=64)
ap.add_argument("--device", default=None)
ap.add_argument("--va_ref_deg", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

g = load_matpower(args.case)
nr = NewtonRaphsonOracle(g)
Vset = g.vset()
bt = g.bus_type
ref = int(np.where(bt == REF)[0][0])
rng = np.random.default_rng(args.seed)
P, Q = load_trajectory(g, args.n_scans, rng, step=0.003)
P0, Q0 = g.base_injections()


def errors(orc, Pi, Qi):
    t, o = nr.solve(Pi, Qi, Vset), orc.solve(Pi, Qi, Vset)
    dVm = np.abs(o["Vm"] - t["Vm"])
    dVa = np.abs(np.rad2deg(o["Va"] - t["Va"]))
    dPf = np.abs(o["Sf"].real - t["Sf"].real) * g.baseMVA
    return dict(vm_max=float(dVm.max()), vm_mean=float(dVm.mean()),
                va_max_deg=float(dVa.max()), va_mean_deg=float(dVa.mean()),
                pf_max_mw=float(dPf.max()), pf_mean_mw=float(dPf.mean()),
                vm_max_pq=float(dVm[bt == PQ].max()), vm_ref=float(dVm[ref]),
                slack_vm_pred=float(o["Vm"][ref]), slack_vm_nr=float(t["Vm"][ref]))


def timeit(fn, reps):
    fn(); t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e3


res = {"variants": {}}
for label, known in (("masked-slack", False), ("slack-given", True)):
    orc = GencoOracle(g, device=args.device, va_ref_deg=args.va_ref_deg, ref_vm_known=known)
    print(f"\n=== {label}   device={orc.device}  baseMVA(normalizer)={orc.b:.3f}  n_gen={orc.n_gen}")
    base = errors(orc, P0, Q0)
    print("BASE CASE ", {k: round(v, 5) for k, v in base.items()})
    ok = base["vm_max"] < 1.5e-3 and base["va_max_deg"] < 0.15
    print("ACCEPTANCE (Vm<~1e-3, Va<~0.1deg):", "PASS" if ok else "FAIL")
    errs = [errors(orc, P[i], Q[i]) for i in range(args.n_scans)]
    pert = {k: dict(mean=float(np.mean([e[k] for e in errs])), max=float(np.max([e[k] for e in errs])))
            for k in errs[0]}
    print("PERTURBED (mean,max)", {k: (round(v["mean"], 5), round(v["max"], 5)) for k, v in pert.items()})
    bres = orc.solve_batch(P, Q, Vset)
    sing = [orc.solve(P[i], Q[i], Vset) for i in range(min(8, args.n_scans))]
    bvs = float(max(np.abs(bres[i]["Vm"] - sing[i]["Vm"]).max() for i in range(len(sing))))
    Pb, Qb = np.tile(P[:1], (64, 1)), np.tile(Q[:1], (64, 1))
    t1 = timeit(lambda: orc.solve(P[0], Q[0], Vset), 50)
    t64 = timeit(lambda: orc.solve_batch(Pb, Qb, Vset), 20)
    print(f"batch-vs-single max|dVm|={bvs:.1e}   ms/solve: batch1={t1:.2f}  batch64={t64:.2f} total ({t64/64:.3f}/scan)")
    res["variants"][label] = dict(base_case=base, perturbed=pert, accept=bool(ok), batch_vs_single_vm_max=bvs,
                                  timing_ms=dict(batch1=t1, batch64_total=t64, batch64_per_scan=t64 / 64))
res["timing_ms_nr_numpy_cpu"] = timeit(lambda: nr.solve(P[0], Q[0], Vset), 50)
print(f"\nNR (numpy, CPU) {res['timing_ms_nr_numpy_cpu']:.2f} ms/solve")

os.makedirs("results", exist_ok=True)
json.dump(res, open("results/genco_acceptance.json", "w"), indent=2)
print("wrote results/genco_acceptance.json")
