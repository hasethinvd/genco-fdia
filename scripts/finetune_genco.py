"""
Fine-tune the GENCO-PF checkpoint on the operator's own regime ("extending GENCO").

Data   : fresh datakit scenarios (a seed different from the experiment's test scenarios)
         + fresh case118.m load walks, injections carrying the same sensor noise the
         detector sees.
Labels : AC Newton-Raphson on those same noisy inputs (the oracle GENCO imitates, and
         the experiment's truth).
Loss   : Huber on Vm, Va and branch P/Q flows, each divided by the detector's sensor
         sigma, so the model is trained on exactly what the physics check measures.
         Flows come from graphkit's own ComputeBranchFlow layer.
Report : per-scan max |z| (what sets the clean threshold) before vs after, on held-out
         scenarios.

  python scripts/finetune_genco.py --datakit_train data/scenarios_train.npz --minutes 25 \
         --out_dir $W/ckpt_ft
Run inside a GPU job.
"""
from __future__ import annotations
import argparse, json, math, os, shutil, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grid import load_matpower, NewtonRaphsonOracle
from attacks import load_trajectory, NOISE
from genco_oracle import GencoOracle
from gridfm_graphkit.models.utils import ComputeBranchFlow

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="data/case118.m")
ap.add_argument("--datakit_train", default=None, help="npz from make_datakit_scenarios.py (seed != test seed)")
ap.add_argument("--datakit_test", default="data/scenarios_118.npz", help="used only to check train/test are disjoint")
ap.add_argument("--n_walk", type=int, default=600, help="case118.m load-walk trajectories (x10 scans)")
ap.add_argument("--dk_draws", type=int, default=4, help="noise draws per datakit scenario")
ap.add_argument("--walk_draws", type=int, default=1)
ap.add_argument("--minutes", type=float, default=25.0, help="wall-clock training budget")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--val_frac", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--log", default="results/finetune_log.json")
args = ap.parse_args()

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
g = load_matpower(args.case)
n = g.n
f_idx, _ = g.from_to
m = len(f_idx)
SIG = dict(vm=NOISE["Vm"], va=0.002, p=NOISE["Pf"], q=NOISE["Qf"])   # detector sigmas (pu / rad)

# ------------------------------------------------------------------ data
blocks = []   # (P, Q, V, draws, name, scans per group)  -- val split is by group so a trajectory never straddles train/val
if args.datakit_train:
    d = np.load(args.datakit_train)
    if args.datakit_test and os.path.exists(args.datakit_test):
        t = np.load(args.datakit_test)
        dmin = np.abs(d["P"][:, None, :] - t["P"][None, :, :]).sum(-1).min()
        print(f"train/test datakit scenarios: min L1 distance {dmin:.4f} (must be > 0, they are different samples)")
        assert dmin > 1e-6, "datakit train set overlaps the test set"
    blocks.append((d["P"], d["Q"], d["Vset"], args.dk_draws, "datakit", 1))
wr = np.random.default_rng(1000 + args.seed)
wP, wQ = [], []
for _ in range(args.n_walk):
    P, Q = load_trajectory(g, 10, wr, step=0.003)
    wP.append(P); wQ.append(Q)
wP, wQ = np.concatenate(wP), np.concatenate(wQ)
blocks.append((wP, wQ, np.tile(g.vset(), (len(wP), 1)), args.walk_draws, "walk", 10))   # 10 scans per trajectory

P_all, Q_all, V_all, grp = [], [], [], []
gid = 0
for P, Q, V, draws, name, gsz in blocks:
    ids = gid + np.arange(len(P)) // gsz; gid += -(-len(P) // gsz)
    for _ in range(draws):
        P_all.append(P + rng.normal(0, NOISE["P"], P.shape))
        Q_all.append(Q + rng.normal(0, NOISE["Q"], Q.shape))
        V_all.append(V); grp.append(ids)
P_all, Q_all, V_all, grp = map(np.concatenate, (P_all, Q_all, V_all, grp))
print(f"training samples: {len(P_all)} from {gid} groups "
      f"({', '.join(f'{b[4]} x{b[3]}' for b in blocks)})")

# NR labels on the noisy inputs
nr = NewtonRaphsonOracle(g)
t0 = time.time()
keep, tVm, tVa, tSf, tSt = [], [], [], [], []
for i in range(len(P_all)):
    r = nr.solve(P_all[i], Q_all[i], V_all[i])
    if r["converged"]:
        keep.append(i); tVm.append(r["Vm"]); tVa.append(r["Va"]); tSf.append(r["Sf"]); tSt.append(r["St"])
keep = np.array(keep)
P_all, Q_all, V_all, grp = P_all[keep], Q_all[keep], V_all[keep], grp[keep]
tVm, tVa = np.array(tVm), np.array(tVa)
tP = np.concatenate([np.array(tSf).real, np.array(tSt).real], axis=1)     # [S, 2m] (forward then reverse)
tQ = np.concatenate([np.array(tSf).imag, np.array(tSt).imag], axis=1)
print(f"NR labels: {len(keep)} converged in {time.time()-t0:.0f}s")

# split by scenario group so noise draws of one scenario never straddle train/val
ug = np.unique(grp); rng.shuffle(ug)
val_g = set(ug[: int(len(ug) * args.val_frac)].tolist())
is_val = np.array([x in val_g for x in grp])
tr_idx, va_idx = np.where(~is_val)[0], np.where(is_val)[0]
print(f"train {len(tr_idx)}  val {len(va_idx)}")

# ------------------------------------------------------------------ model / loss
orc = GencoOracle(g)
dev, model, b = orc.device, orc.model, orc.b
flow = ComputeBranchFlow()
base = g.baseMVA
to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
tVm_t, tVa_t, tP_t, tQ_t = map(to_t, (tVm, tVa, tP * base / b, tQ * base / b))   # flows in model units
unit = b / base                                                                      # model flow units -> pu


def run_batch(idx, grad):
    B = len(idx)
    # per-scan Vset: GencoOracle._dynamic_inputs takes a single Vset; apply the PV setpoints per scan here
    xb, xg, _ = orc._dynamic_inputs(P_all[idx], Q_all[idx], V_all[idx[0]])
    xb = xb.view(B, n, -1).clone()
    xb[:, orc.pv_idx, orc.G.VM_H] = torch.from_numpy(V_all[idx][:, orc.pv_idx]).float()
    xb = xb.view(B * n, -1)
    eidx, eattr, mask = orc._static_batch(B)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = model({"bus": xb.to(dev), "gen": xg.to(dev)}, eidx, eattr, mask)["bus"]
        Pft, Qft = flow(out, eidx[("bus", "connects", "bus")], eattr[("bus", "connects", "bus")])
    Vm, Va = out[:, 0].view(B, n), out[:, 1].view(B, n)
    ii = torch.as_tensor(idx, device=dev)
    zVm = (Vm - tVm_t[ii]) / SIG["vm"]
    zVa = (Va - tVa_t[ii]) / SIG["va"]
    zP = (Pft.view(B, -1) - tP_t[ii]) * unit / SIG["p"]
    zQ = (Qft.view(B, -1) - tQ_t[ii]) * unit / SIG["q"]
    return zVm, zVa, zP, zQ


def loss_fn(z):
    return sum(F.huber_loss(x, torch.zeros_like(x), delta=1.0) for x in z)


def evaluate(max_n=2048):
    """per-scan max |z| over Vm / line P / line Q (what the physics detector thresholds)."""
    model.eval()
    idx_all = va_idx[:max_n]
    scan_max, comp = [], {"vm": [], "p": [], "q": []}
    for s in range(0, len(idx_all), args.bs):
        idx = idx_all[s:s + args.bs]
        if len(idx) < 2:
            continue
        zVm, zVa, zP, zQ = run_batch(idx, grad=False)
        mv, mp, mq = zVm.abs().amax(1), zP.abs().amax(1), zQ.abs().amax(1)
        scan_max.append(torch.maximum(torch.maximum(mv, mp), mq).cpu())
        comp["vm"].append(mv.cpu()); comp["p"].append(mp.cpu()); comp["q"].append(mq.cpu())
    sm = torch.cat(scan_max).numpy()
    out = dict(median=float(np.median(sm)), p99=float(np.quantile(sm, 0.99)))
    for k, v in comp.items():
        out[f"{k}_median"] = float(np.median(torch.cat(v).numpy()))
    model.train()
    return out


# ------------------------------------------------------------------ train
before = evaluate()
print("BEFORE fine-tuning (held-out, per-scan max |z|, sigma units):", {k: round(v, 2) for k, v in before.items()})

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
model.train()
budget = args.minutes * 60
t_start = time.time()
best, best_state, step, epoch = float("inf"), None, 0, 0
hist = []
while time.time() - t_start < budget:
    perm = rng.permutation(tr_idx)
    for s in range(0, len(perm) - args.bs + 1, args.bs):
        frac = (time.time() - t_start) / budget
        if frac >= 1:
            break
        lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        for gp in opt.param_groups:
            gp["lr"] = lr
        z = run_batch(perm[s:s + args.bs], grad=True)
        loss = loss_fn(z)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
    epoch += 1
    ev = evaluate()
    hist.append(dict(epoch=epoch, step=step, minutes=(time.time() - t_start) / 60, lr=lr, loss=float(loss), **ev))
    print(f"epoch {epoch:3d} step {step:6d} {(time.time()-t_start)/60:5.1f} min  loss {float(loss):.3f}  "
          f"val max|z| median {ev['median']:.2f}  p99 {ev['p99']:.2f}", flush=True)
    if ev["p99"] < best:
        best = ev["p99"]
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
after = evaluate()
print("AFTER  fine-tuning (held-out, per-scan max |z|, sigma units):", {k: round(v, 2) for k, v in after.items()})

# ------------------------------------------------------------------ save
src = os.environ.get("GENCO_CKPT_DIR", "ckpt/case118_ieee/base/seed0")
os.makedirs(args.out_dir, exist_ok=True)
for fn in os.listdir(src):
    if fn.endswith(".yaml") or fn == "normalizer_stats.pt":
        shutil.copy(os.path.join(src, fn), args.out_dir)
torch.save(best_state, os.path.join(args.out_dir, "best_model_state_dict.pt"))
os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
json.dump(dict(args=vars(args), n_train=int(len(tr_idx)), n_val=int(len(va_idx)), steps=step, epochs=epoch,
               before=before, after=after, history=hist), open(args.log, "w"), indent=1)
print("saved", args.out_dir, "and", args.log)
