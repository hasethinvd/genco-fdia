"""
Scenarios and false-data-injection attacks.

Measurement vector reported to the control room, per scan:
    P_inj[n], Q_inj[n], Vm[n], Pf[m], Qf[m]      (pu)
all carrying Gaussian sensor noise.

Attacker model (one knob, k in [0,1] = "knowledge"):
  - The attacker owns a geographically contiguous set of substations ("region")
    and can rewrite every measurement there.
  - The attacker wants the control room to believe a *target state* that hides
    a real overload / shifts estimated load by `magnitude`.
  - To stay stealthy the attacker recomputes region measurements with an AC
    power flow on THEIR OWN model of the grid. With k = 1 that model equals the
    true one and the attack is perfectly physics-consistent. With k < 1 a
    fraction (1-k) of branch parameters in their model are wrong (multiplicative
    error), so the injected numbers are only approximately consistent.
  - k = 0 degenerates to a naive injection (scale measurements, no physics).

This mirrors the knowledge-of-H parameterisation used in the FDIA literature
(e.g. GenAI-FDIA, 2026) but needs no external dependency.
"""
from __future__ import annotations
import numpy as np
from grid import Grid, NewtonRaphsonOracle, build_ybus, branch_flows


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
NOISE = dict(P=0.005, Q=0.005, Vm=0.002, Pf=0.005, Qf=0.005)


def true_measurements(g: Grid, oracle: NewtonRaphsonOracle, P, Q, Vset):
    r = oracle.solve(P, Q, Vset)
    V = r["Vm"] * np.exp(1j * r["Va"])
    S = V * np.conj(oracle.Y @ V)
    return dict(P=S.real, Q=S.imag, Vm=r["Vm"], Pf=r["Sf"].real, Qf=r["Sf"].imag,
                Va=r["Va"], converged=r["converged"])


def add_noise(z: dict, rng) -> dict:
    out = dict(z)
    for key, s in NOISE.items():
        out[key] = z[key] + rng.normal(0, s, size=z[key].shape)
    return out


# ---------------------------------------------------------------------------
# Scenarios: smooth load trajectories so a temporal detector is meaningful
# ---------------------------------------------------------------------------
def load_trajectory(g: Grid, T: int, rng, level=(0.8, 1.2), step=0.01):
    """Return P[T,n], Q[T,n] net injections. Loads follow a bounded random walk
    around a scenario-level scaling; generators share the slack via the ref bus
    (NR handles that), PV buses keep their scheduled P."""
    P0, Q0 = g.base_injections()
    Pd = g.bus[:, 2] / g.baseMVA
    Qd = g.bus[:, 3] / g.baseMVA
    Pg = P0 + Pd
    Qg = Q0 + Qd
    scale = rng.uniform(*level)
    s = np.full(g.n, scale)
    P = np.zeros((T, g.n)); Q = np.zeros((T, g.n))
    for t in range(T):
        s = np.clip(s + rng.normal(0, step, g.n), level[0] * 0.9, level[1] * 1.1)
        # scale generation uniformly to roughly track load (slack absorbs rest)
        gscale = s.mean()
        P[t] = Pg * gscale - Pd * s
        Q[t] = Qg * gscale - Qd * s
    return P, Q


# ---------------------------------------------------------------------------
# Attacker
# ---------------------------------------------------------------------------
def contiguous_region(g: Grid, seed_bus: int, size: int) -> np.ndarray:
    """BFS over branches from a seed bus; returns bus indices."""
    f, t = g.from_to
    adj = {i: set() for i in range(g.n)}
    for a, b in zip(f, t):
        adj[a].add(b); adj[b].add(a)
    seen, frontier = [seed_bus], [seed_bus]
    while frontier and len(seen) < size:
        nxt = []
        for u in frontier:
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.append(v); nxt.append(v)
                    if len(seen) >= size:
                        break
            if len(seen) >= size:
                break
        frontier = nxt
    return np.array(seen)


class Attacker:
    """
    Local stealthy AC FDIA (Hug & Giampapa 2012 construction).

    interior  = buses the attacker fully controls (contiguous region)
    boundary  = neighbours of interior
    The attacker perturbs the *state* only on interior buses, keeps boundary
    voltages exactly as measured, and rewrites:
        - P,Q injections at interior + boundary buses
        - Vm at interior buses
        - P,Q flows on every line with at least one interior end
    computed with h(x') on THEIR model of the grid. Lines boundary->outside are
    untouched and stay consistent because boundary voltages did not change.
    With k = 1 (perfect model) the result satisfies the AC equations exactly.
    """

    def __init__(self, g: Grid, knowledge: float, rng, region_size=8,
                 model_error=0.30):
        self.g, self.k, self.rng = g, float(knowledge), rng
        self.region_size = region_size
        m = len(g.branch)
        wrong = rng.random(m) >= self.k
        scale = np.ones(m)
        scale[wrong] = rng.uniform(1 - model_error, 1 + model_error, wrong.sum())
        self.branch_scale = scale
        self.Y_model, self.Yb_model = build_ybus(g, scale)
        f, t = g.from_to
        self.adj = {i: set() for i in range(g.n)}
        for a, b in zip(f, t):
            self.adj[a].add(b); self.adj[b].add(a)

    def pick_region(self, seed_bus=None):
        g, rng = self.g, self.rng
        if seed_bus is None:
            pq = np.where(g.bus_type == 1)[0]
            seed_bus = int(rng.choice(pq))
        interior = contiguous_region(g, seed_bus, self.region_size)
        interior = interior[g.bus_type[interior] != 3]   # never the slack
        boundary = np.array(sorted(set().union(*[self.adj[i] for i in interior])
                                   - set(interior)))
        f, t = g.from_to
        lines = np.isin(f, interior) | np.isin(t, interior)
        return dict(seed=seed_bus, interior=interior, boundary=boundary,
                    region=np.r_[interior, boundary], branches=lines)

    def attack(self, z_true: dict, P, Q, Vset, magnitude=0.15, reg=None,
               ramp=1.0):
        """
        magnitude : angle shift (degrees) applied to interior buses; sign chosen
                    to make the region appear LESS loaded. ramp in (0,1] scales
                    it so a slow attacker can creep in over several scans.
        Returns (z_attacked, info).
        """
        g = self.g
        if reg is None:
            reg = self.pick_region()
        interior, boundary, lines = reg["interior"], reg["boundary"], reg["branches"]
        z = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in z_true.items()}

        if self.k == 0:
            # naive: report half the interior load as gone, scale incident
            # flows the same way, no physics anywhere
            Pd = g.bus[:, 2] / g.baseMVA
            Qd = g.bus[:, 3] / g.baseMVA
            frac = 0.5 * ramp
            z["P"][interior] += frac * Pd[interior]
            z["Q"][interior] += frac * Qd[interior]
            z["Pf"][lines] *= (1 - frac)
            z["Qf"][lines] *= (1 - frac)
            return z, reg

        # target state: shift interior angles by +/-d (sign chosen so the region
        # appears LESS loaded than it is) and lift interior PQ voltages slightly
        d = np.deg2rad(magnitude) * ramp
        pq_int = interior[g.bus_type[interior] == 1]   # PV buses hold Vm
        best = None
        for sign in (+1.0, -1.0):
            Va, Vm = z_true["Va"].copy(), z_true["Vm"].copy()
            Va[interior] += sign * d
            Vm[pq_int] *= (1 + 0.002 * ramp)
            V = Vm * np.exp(1j * Va)
            S = V * np.conj(self.Y_model @ V)
            apparent_load = -S.real[interior].sum()
            if best is None or apparent_load < best[0]:
                best = (apparent_load, Va, Vm, V, S)
        _, Va, Vm, V, S = best
        Sf, _ = branch_flows(g, V, self.Yb_model)
        touched = reg["region"]
        z["P"][touched] = S.real[touched]
        z["Q"][touched] = S.imag[touched]
        z["Vm"][interior] = Vm[interior]
        z["Pf"][lines] = Sf.real[lines]
        z["Qf"][lines] = Sf.imag[lines]
        return z, reg
