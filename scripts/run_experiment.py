"""
Run the full experiment.

  python scripts/run_experiment.py --oracle nr      # pandapower-free NR stand-in
  python scripts/run_experiment.py --oracle genco   # GENCO PF checkpoint (GPU)

Outputs results/results_<oracle>.json with, per knowledge level k:
  detection rate / FPR for  (a) physics residual with chosen oracle
                            (b) physics residual with DC oracle (baseline)
                            (c) temporal state-jump on top of (a)
plus a saved per-bus residual map for one example attack (for the figure).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grid import load_matpower, NewtonRaphsonOracle, DCOracle
from attacks import Attacker, load_trajectory, true_measurements, add_noise
from detect import physics_residual, state_jump, calibrate, evaluate, pmu_line_residual, place_pmus

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="data/case118.m")
ap.add_argument("--oracle", default="nr", choices=["nr", "genco"])
ap.add_argument("--n_traj", type=int, default=40, help="trajectories")
ap.add_argument("--T", type=int, default=12, help="scans per trajectory")
ap.add_argument("--attack_at", type=int, default=6, help="scan index attack starts")
ap.add_argument("--knowledge", default="0,0.25,0.5,0.75,0.9,1.0")
ap.add_argument("--magnitude", type=float, default=0.2, help="interior angle shift, degrees")
ap.add_argument("--ramp", type=int, default=1, help="scans over which the attack ramps to full magnitude")
ap.add_argument("--region", type=int, default=8)
ap.add_argument("--fpr", type=float, default=0.01)
ap.add_argument("--pmu_frac", type=float, default=0.2, help="fraction of buses with a PMU")
ap.add_argument("--pmu_place", default="degree", choices=["degree", "random"])
ap.add_argument("--load_step", type=float, default=0.003, help="per-scan load random-walk std (fraction)")
ap.add_argument("--scenarios", default=None,
                help="npz with P[T,n], Q[T,n] from make_datakit_scenarios.py; trajectories are windows "
                     "of consecutive samples instead of load_trajectory()")
ap.add_argument("--genco_ckpt", default=None,
                help="GENCO checkpoint dir (yaml + best_model_state_dict.pt + normalizer_stats.pt); "
                     "default $GENCO_CKPT_DIR. Point at a fine-tuned dir together with --tag")
ap.add_argument("--tag", default="", help="suffix for the results filename, e.g. _ft")
ap.add_argument("--label", default=None, help="oracle label stored in the results (default: GENCO-PF (fine-tuned) when --tag is given)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
g = load_matpower(args.case)
truth = NewtonRaphsonOracle(g)           # the real world
dc = DCOracle(g)
if args.oracle == "nr":
    oracle = NewtonRaphsonOracle(g)      # stand-in for GENCO
else:
    from genco_oracle import GencoOracle
    oracle = GencoOracle(g, checkpoint=args.genco_ckpt,
                         label=args.label or ("GENCO-PF (fine-tuned)" if args.tag else None))
Vset = g.vset()
pmu_buses = place_pmus(g, args.pmu_frac, rng, args.pmu_place)
ks = [float(x) for x in args.knowledge.split(",")]

# ---------------------------------------------------------------- clean phase
t0 = time.time()
clean_phys, clean_dc, clean_jump, clean_pmu = [], [], [], []
trajs = []
if args.scenarios:
    sc = np.load(args.scenarios)
    SP, SQ = sc["P"], sc["Q"]
    # datakit's slack / PV voltage setpoints vary per scenario and are inputs to both NR and GENCO
    SV = sc["Vset"] if "Vset" in sc.files else None
    assert len(SP) >= args.T, f"{args.scenarios} has {len(SP)} samples, need >= T={args.T}"
    # evenly spread windows of T consecutive samples (disjoint when n_traj*T <= samples)
    starts = np.linspace(0, len(SP) - args.T, args.n_traj).astype(int)
    print(f"scenarios: {len(SP)} datakit samples, {args.n_traj} windows of {args.T}"
          f"{' (overlapping)' if args.n_traj * args.T > len(SP) else ''}")
for i in range(args.n_traj):
    if args.scenarios:
        sl = slice(starts[i], starts[i] + args.T)
        P, Q = SP[sl], SQ[sl]
        V = SV[sl] if SV is not None else np.tile(Vset, (args.T, 1))
    else:
        P, Q = load_trajectory(g, args.T, rng, step=args.load_step)
        V = np.tile(Vset, (args.T, 1))
    zs, prev = [], None
    for t in range(args.T):
        zt = true_measurements(g, truth, P[t], Q[t], V[t])
        if not zt["converged"]:
            break
        z = add_noise(zt, rng)
        zs.append(zt)                       # keep noise-free; noise added per use
        s_phys, st = physics_residual(g, oracle, z, V[t])
        s_dc, _ = physics_residual(g, dc, z, V[t])
        clean_phys.append(s_phys.max()); clean_dc.append(s_dc.max())
        clean_jump.append(state_jump(prev, st).max()); prev = st
        clean_pmu.append(pmu_line_residual(g, truth.Yb, zt["Vm"], zt["Va"], z, pmu_buses, rng)[0].max())
    trajs.append((P, Q, zs, V))
clean_phys, clean_dc, clean_jump, clean_pmu = map(np.array, (clean_phys, clean_dc, clean_jump, clean_pmu))
thr_pmu = calibrate(clean_pmu, args.fpr)
thr_phys = calibrate(clean_phys, args.fpr)
thr_dc = calibrate(clean_dc, args.fpr)
thr_jump = calibrate(clean_jump[clean_jump > 0], args.fpr)
print(f"clean: {len(clean_phys)} scans in {time.time()-t0:.1f}s  "
      f"thr phys={thr_phys:.2f} dc={thr_dc:.2f} jump={thr_jump:.2f} pmu={thr_pmu:.2f}")

# --------------------------------------------------------------- attack phase
results = dict(case=g.name, oracle=oracle.name, n_traj=args.n_traj, T=args.T,
               magnitude=args.magnitude, ramp=args.ramp, region=args.region, fpr_target=args.fpr,
               scenarios=args.scenarios,
               thresholds=dict(phys=thr_phys, dc=thr_dc, jump=thr_jump, pmu=thr_pmu),
               pmu_buses=pmu_buses.tolist(), pmu_frac=args.pmu_frac,
               clean=dict(phys=clean_phys.tolist(), dc=clean_dc.tolist(),
                          jump=clean_jump.tolist(), pmu=clean_pmu.tolist()),
               levels=[], example=None)

for k in ks:
    att = Attacker(g, k, rng, region_size=args.region)
    sp, sd, sj, spmu, sfused, hidden_mw = [], [], [], [], [], []
    for (P, Q, zs, V) in trajs:
        prev = None
        reg = att.pick_region()
        for t, zt in enumerate(zs):
            if t < args.attack_at:
                _, st = physics_residual(g, oracle, add_noise(zt, rng), V[t]); prev = st
                continue
            ramp = min(1.0, (t - args.attack_at + 1) / args.ramp)
            za, info = att.attack(zt, P[t], Q[t], V[t], magnitude=args.magnitude,
                                  reg=reg, ramp=ramp)
            za = add_noise(za, rng)
            z = zt
            s_phys, st = physics_residual(g, oracle, za, V[t])
            s_dc, _ = physics_residual(g, dc, za, V[t])
            jump_vec = state_jump(prev, st); jump = jump_vec.max(); prev = st
            pmu_vec, _ = pmu_line_residual(g, truth.Yb, zt["Vm"], zt["Va"], za, pmu_buses, rng)
            sp.append(s_phys.max()); sd.append(s_dc.max()); sj.append(jump)
            spmu.append(pmu_vec.max())
            sfused.append(max(s_phys.max() / thr_phys, pmu_vec.max() / thr_pmu))
            # consequence: how much real load in the region is hidden from the
            # operator (MW), i.e. true minus reported net load
            Pd = g.bus[:, 2] / g.baseMVA
            reported_load = -(za["P"][info["interior"]].sum())
            true_load = -(z["P"][info["interior"]].sum())
            hidden_mw.append(float((true_load - reported_load) * g.baseMVA))
            if k == 1.0 and t == len(zs) - 1:
                cand = dict(k=k, interior=info["interior"].tolist(),
                            boundary=info["boundary"].tolist(),
                            per_bus=s_phys.tolist(), per_bus_pmu=pmu_vec.tolist(),
                            per_bus_jump=jump_vec.tolist(), pmu_score=float(pmu_vec.max()))
                if results["example"] is None or cand["pmu_score"] > results["example"]["pmu_score"]:
                    results["example"] = cand
    sp, sd, sj, spmu, sfused = map(np.array, (sp, sd, sj, spmu, sfused))
    tp_p, _ = evaluate(clean_phys, sp, thr_phys)
    tp_d, _ = evaluate(clean_dc, sd, thr_dc)
    tp_j = float(np.mean(sj > thr_jump))
    tp_pmu = float(np.mean(spmu > thr_pmu))
    tp_f = float(np.mean(sfused > 1.0))
    lvl = dict(k=k, det_phys=tp_p, det_dc=tp_d, det_jump=tp_j, det_pmu=tp_pmu, det_fused=tp_f,
               mean_resid=float(sp.mean()), hidden_mw=float(np.mean(hidden_mw)))
    results["levels"].append(lvl)
    print(f"k={k:4.2f}  phys={tp_p:5.2f}  DC={tp_d:5.2f}  jump={tp_j:5.2f}  pmu={tp_pmu:5.2f}  "
          f"fused={tp_f:5.2f}  mean resid={sp.mean():7.1f}  hidden {np.mean(hidden_mw):6.1f} MW")

# runtime per scan
t0 = time.time()
for _ in range(20):
    oracle.solve(trajs[0][0][0], trajs[0][1][0], trajs[0][3][0])
results["ms_per_solve"] = (time.time() - t0) / 20 * 1e3
t0 = time.time()
for _ in range(20):
    dc.solve(trajs[0][0][0], trajs[0][1][0], trajs[0][3][0])
results["ms_per_solve_dc"] = (time.time() - t0) / 20 * 1e3
print(f"runtime: {oracle.name} {results['ms_per_solve']:.1f} ms/solve, "
      f"DC {results['ms_per_solve_dc']:.1f} ms/solve")

os.makedirs("results", exist_ok=True)
out = f"results/results_{args.oracle}{args.tag}_ramp{args.ramp}{'_datakit' if args.scenarios else ''}.json"
json.dump(results, open(out, "w"), indent=1)
print("saved", out)
