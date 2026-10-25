"""
GENCO oracle: drop-in replacement for NewtonRaphsonOracle.

Interface contract (the only thing run_experiment.py relies on):
    oracle.name                        -> str
    oracle.solve(P, Q, Vset) -> dict(Vm[n], Va[n], Sf[m] complex, St[m] complex,
                                     converged bool, iters int, ms float)
P, Q are *net* bus injections in pu on baseMVA; Vset generator setpoints.
`solve_batch(P[B,n], Q[B,n], Vset)` does B scans in one forward pass.

How GENCO is fed (read from gridfm-graphkit, branch genco-paper-repro; not guessed):
  * heterogeneous graph: bus nodes (15 features), gen nodes (6), bus-bus edges
    (10 features, both directions per branch), gen<->bus edges.
  * we build the raw tables from OUR Grid (true admittances), push them once through
    graphkit's own pipeline (HeteroDataMVANormalizer -> PowerFlowTransforms: drop
    inactive branches/gens, AddPFHeteroMask, ApplyMasking) and cache the result.
  * PF task leaves as known inputs: Pd, Qd (all buses), Vm at PV buses, Va at the
    REF bus, Pg of non-REF generators, bus-type flags, shunts GS/BS, branch
    Yff/Yft/tap. Everything else (Vm/Va at PQ, Va/Qg at PV, Vm/Qg/Pg at REF, limits,
    costs, branch flows) is masked to 0 and predicted or ignored.
  * per solve only the dynamic inputs are rewritten (normalised the same way the
    normalizer does): Pd, Qd, Vm[PV], Pg[non-REF gens].
  * model output [Vm, Va(rad)] per bus; flows are recomputed with branch_flows()
    on the TRUE admittances (the operator knows the grid).

Net injection -> (Pd, Pg) split (the oracle only receives P = Pg - Pd):
    PQ buses (no generator): Pd = -P, Qd = -Q
    PV / REF buses:          Pd = case load, Pg = P + Pd
This split is an adapter choice and is stated in the write-up.

Environment (HSG): module `torch_scatter` is a pure-torch stand-in, see
scripts/torch_scatter_shim.  Checkpoint dir is $GENCO_CKPT_DIR (default
./ckpt/case118_ieee/base/seed0) holding the yaml config, best_model_state_dict.pt
and normalizer_stats.pt downloaded from gridfm/genco-pf-datakit.
"""
from __future__ import annotations
import glob
import os
import time
import numpy as np
from grid import Grid, build_ybus, branch_flows, PQ, PV, REF


class GencoOracle:
    name = "GENCO-PF"

    def __init__(self, g: Grid, checkpoint: str | None = None,
                 device: str | None = None, va_ref_deg: float = 0.0,
                 ref_vm_known: bool = False, label: str | None = None):
        """ref_vm_known=False: faithful to the checkpoint (PF task masks the slack Vm and
        the model infers it; datakit's slack Vm is an OPF output that varies per scenario).
        ref_vm_known=True: unmask the slack Vm and feed Vset[ref] (the operator knows the
        slack setpoint). Inference-time conditioning the model was not trained with."""
        import torch
        import yaml
        from gridfm_graphkit.io.param_handler import (
            NestedNamespace, load_model, get_task_transforms)
        from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
        from gridfm_graphkit.datasets import globals as G
        from torch_geometric.data import HeteroData

        self.torch, self.G = torch, G
        self.g = g
        self.Y, self.Yb = build_ybus(g)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        ckpt = checkpoint or os.environ.get("GENCO_CKPT_DIR", "ckpt/case118_ieee/base/seed0")
        cfg_path = glob.glob(os.path.join(ckpt, "*.yaml"))[0]
        with open(cfg_path) as f:
            self.args = args = NestedNamespace(**yaml.safe_load(f))
        self.network = args.data.networks[0]

        # ---- model ----------------------------------------------------------
        self.model = load_model(args)
        sd = torch.load(os.path.join(ckpt, "best_model_state_dict.pt"), map_location="cpu")
        sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint/model mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        self.model.to(self.device).eval()

        # ---- normalizer (global baseMVA fitted on the training run) ----------
        self.norm = HeteroDataMVANormalizer(args)
        stats = torch.load(os.path.join(ckpt, "normalizer_stats.pt"), map_location="cpu", weights_only=True)
        self.norm.fit_from_dict(stats[self.network])
        self.b = float(self.norm.baseMVA)

        # ---- static graph through graphkit's own pipeline --------------------
        bt = g.bus_type
        self.bt = bt
        self.pq_idx, self.pv_idx = np.where(bt == PQ)[0], np.where(bt == PV)[0]
        self.ref = int(np.where(bt == REF)[0][0])
        n = g.n
        base = g.baseMVA

        Pd_mw, Qd_mw = g.bus[:, 2].copy(), g.bus[:, 3].copy()
        on = g.gen[:, 7] > 0
        gen_rows = g.gen[on]
        self.gen_bus = np.array([g.ext2int[int(r[0])] for r in gen_rows])
        self.n_gen = len(self.gen_bus)
        bus_has_gen = np.zeros(n, bool); bus_has_gen[self.gen_bus] = True
        # PQ buses that also carry a generator are handled like PV buses below
        self.inj_from_load = (bt == PQ) & ~bus_has_gen
        # per-generator share of its bus's Pg (equal split if base dispatch is zero)
        pg_base = gen_rows[:, 1].astype(float)
        share = np.zeros(self.n_gen)
        for bi in np.unique(self.gen_bus):
            idx = np.where(self.gen_bus == bi)[0]
            tot = pg_base[idx].sum()
            share[idx] = pg_base[idx] / tot if abs(tot) > 1e-9 else 1.0 / len(idx)
        self.gen_share = share
        self.Pd_base_pu, self.Qd_base_pu = Pd_mw / base, Qd_mw / base

        f_idx, t_idx = g.from_to
        Yff, Yft, Ytf, Ytt = self.Yb
        tap = np.where(g.branch[:, 8] == 0, 1.0, g.branch[:, 8])
        m = len(f_idx)

        bus_x = np.zeros((n, 15), dtype=np.float32)
        bus_x[:, G.PD_H], bus_x[:, G.QD_H] = Pd_mw, Qd_mw
        bus_x[:, G.VM_H] = g.vset()
        bus_x[:, G.VA_H] = 0.0
        bus_x[:, G.PQ_H], bus_x[:, G.PV_H], bus_x[:, G.REF_H] = bt == PQ, bt == PV, bt == REF
        bus_x[self.ref, G.VA_H] = va_ref_deg
        bus_x[:, G.MIN_VM_H], bus_x[:, G.MAX_VM_H] = 0.9, 1.1       # masked by the PF mask
        bus_x[:, G.GS], bus_x[:, G.BS] = g.bus[:, 4] / base, g.bus[:, 5] / base
        bus_x[:, G.VN_KV] = g.bus[:, 9]

        gen_x = np.zeros((self.n_gen, 7), dtype=np.float32)
        gen_x[:, G.PG_H] = pg_base
        gen_x[:, G.G_ON] = 1.0

        def edge_attr(y_self, y_cross):
            """forward edge: (Yff, Yft); reverse edge: (Ytt, Ytf), same column slots."""
            a = np.zeros((m, 11), dtype=np.float32)
            a[:, G.YFF_TT_R], a[:, G.YFF_TT_I] = y_self.real, y_self.imag
            a[:, G.YFT_TF_R], a[:, G.YFT_TF_I] = y_cross.real, y_cross.imag
            a[:, G.TAP] = tap
            a[:, G.ANG_MIN], a[:, G.ANG_MAX], a[:, G.RATE_A] = -360.0, 360.0, 9999.0  # masked
            a[:, G.B_ON] = 1.0
            return a
        fwd = edge_attr(Yff, Yft)
        rev = edge_attr(Ytt, Ytf)

        T = torch.tensor
        d = HeteroData()
        d["bus"].x = T(bus_x)
        d["gen"].x = T(gen_x)
        d["bus"].y = d["bus"].x[:, : G.VA_H + 1].clone()
        d["gen"].y = d["gen"].x[:, : G.PG_H + 1].clone()
        ei = np.concatenate([np.stack([f_idx, t_idx]), np.stack([t_idx, f_idx])], axis=1)
        d["bus", "connects", "bus"].edge_index = T(ei, dtype=torch.long)
        d["bus", "connects", "bus"].edge_attr = T(np.concatenate([fwd, rev]))
        d["bus", "connects", "bus"].y = torch.zeros(2 * m, 2)
        gi = np.arange(self.n_gen)
        d["gen", "connected_to", "bus"].edge_index = T(np.stack([gi, self.gen_bus]), dtype=torch.long)
        d["bus", "connected_to", "gen"].edge_index = T(np.stack([self.gen_bus, gi]), dtype=torch.long)

        self.norm.transform(d)
        d = get_task_transforms(args)(d)

        # cached, already normalised + masked tensors (CPU, float32)
        self.t_bus = d.x_dict["bus"].clone()
        self.t_gen = d.x_dict["gen"].clone()
        self.t_eidx = {k: v.clone() for k, v in d.edge_index_dict.items()}
        self.t_eattr = d.edge_attr_dict[("bus", "connects", "bus")].clone()
        self.t_mask = {k: v.clone() for k, v in d.mask_dict.items()
                       if k in ("bus", "gen", "PQ", "PV", "REF")}
        self.ref_vm_known = ref_vm_known
        if ref_vm_known:
            self.t_mask["bus"][self.ref, G.VM_H] = False     # slack Vm becomes a fixed input
            self.name = "GENCO-PF (slack Vm given)"
        if label:
            self.name = label
        self.nb = n
        self.ref_gen = np.isin(self.gen_bus, [self.ref])
        self._batched: dict[int, tuple] = {}

    # ------------------------------------------------------------------ helpers
    def _static_batch(self, B: int):
        """Replicate the static graph B times (cached per B), on device."""
        if B in self._batched:
            return self._batched[B]
        torch = self.torch
        dev, n, ng = self.device, self.nb, self.n_gen
        off_b = (torch.arange(B) * n)
        off_g = (torch.arange(B) * ng)
        def rep_idx(ei, src_off, dst_off):
            e = ei.unsqueeze(0).repeat(B, 1, 1)
            e[:, 0, :] += src_off.view(B, 1)
            e[:, 1, :] += dst_off.view(B, 1)
            return e.permute(1, 0, 2).reshape(2, -1).to(dev)
        eidx = {
            ("bus", "connects", "bus"): rep_idx(self.t_eidx[("bus", "connects", "bus")], off_b, off_b),
            ("gen", "connected_to", "bus"): rep_idx(self.t_eidx[("gen", "connected_to", "bus")], off_g, off_b),
            ("bus", "connected_to", "gen"): rep_idx(self.t_eidx[("bus", "connected_to", "gen")], off_b, off_g),
        }
        # as in graphkit batches: only the bus-bus relation carries edge attributes
        eattr = {("bus", "connects", "bus"): self.t_eattr.repeat(B, 1).to(dev)}
        mask = {k: v.repeat(B, *([1] * (v.dim() - 1))).to(dev) for k, v in self.t_mask.items()}
        out = (eidx, eattr, mask)
        self._batched[B] = out
        return out

    def _dynamic_inputs(self, P, Q, Vset):
        """P, Q: [B, n] pu. Returns normalised x_bus [B*n,15], x_gen [B*ng,6]."""
        torch, G = self.torch, self.G
        base, b = self.g.baseMVA, self.b
        P, Q = np.atleast_2d(P), np.atleast_2d(Q)
        B, n = P.shape
        Pd = np.where(self.inj_from_load, -P, self.Pd_base_pu)           # [B,n] pu
        Qd = np.where(self.inj_from_load, -Q, self.Qd_base_pu)
        Pg_bus = P + Pd                                                    # pu, valid at gen buses
        xb = self.t_bus.numpy().copy()[None].repeat(B, 0)
        xb[:, :, G.PD_H] = Pd * base / b
        xb[:, :, G.QD_H] = Qd * base / b
        xb[:, self.pv_idx, G.VM_H] = np.asarray(Vset)[self.pv_idx]
        if self.ref_vm_known:
            xb[:, self.ref, G.VM_H] = np.asarray(Vset)[self.ref]
        xg = self.t_gen.numpy().copy()[None].repeat(B, 0)
        pg_gen = Pg_bus[:, self.gen_bus] * self.gen_share * base / b       # [B,ng] normalised
        keep = ~self.ref_gen                                               # REF gens stay masked (0)
        xg[:, keep, G.PG_H] = pg_gen[:, keep]
        return (torch.from_numpy(xb.reshape(B * n, -1)), torch.from_numpy(xg.reshape(B * self.n_gen, -1)), B)

    @staticmethod
    def _sync(torch, dev):
        if dev.type == "cuda":
            torch.cuda.synchronize()

    # ------------------------------------------------------------------- solving
    def solve_batch(self, P, Q, Vset):
        torch = self.torch
        t0 = time.perf_counter()
        xb, xg, B = self._dynamic_inputs(P, Q, Vset)
        eidx, eattr, mask = self._static_batch(B)
        with torch.no_grad():
            out = self.model({"bus": xb.to(self.device), "gen": xg.to(self.device)},
                             eidx, eattr, mask)
            self._sync(torch, self.device)
        res = out["bus"].float().cpu().numpy().reshape(B, self.nb, -1)
        ms = (time.perf_counter() - t0) * 1e3
        Vm, Va = res[:, :, 0].astype(float), res[:, :, 1].astype(float)   # Va in radians
        results = []
        for i in range(B):
            V = Vm[i] * np.exp(1j * Va[i])
            Sf, St = branch_flows(self.g, V, self.Yb)
            results.append(dict(Vm=Vm[i], Va=Va[i], Sf=Sf, St=St, converged=True,
                                iters=1, ms=ms / B))
        return results

    def solve(self, P, Q, Vset):
        return self.solve_batch(np.asarray(P)[None], np.asarray(Q)[None], Vset)[0]
