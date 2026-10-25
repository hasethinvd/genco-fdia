"""
Grid model and power-flow oracles.

The experiment needs a *physics oracle*: given reported bus injections, return the
physically implied voltages and line flows. Three oracles implement the same
interface so results are directly comparable:

  - NewtonRaphsonOracle : exact AC power flow (ground truth / reference)
  - DCOracle            : linear DC approximation (the "classical fast" baseline)
  - GencoOracle         : wraps a GENCO PF checkpoint via gridfm-graphkit
                          (see genco_oracle.py; needs GPU + HF access)

All power quantities are per-unit on baseMVA. Buses are re-indexed 0..n-1.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
import numpy as np

PQ, PV, REF = 1, 2, 3


@dataclass
class Grid:
    baseMVA: float
    bus: np.ndarray       # MATPOWER bus matrix (external numbering)
    gen: np.ndarray
    branch: np.ndarray
    ext2int: dict = field(default_factory=dict)
    name: str = "case"

    # ---- derived -----------------------------------------------------------
    @property
    def n(self) -> int:
        return self.bus.shape[0]

    @property
    def bus_type(self) -> np.ndarray:
        return self.bus[:, 1].astype(int)

    @property
    def from_to(self):
        f = np.array([self.ext2int[int(b)] for b in self.branch[:, 0]])
        t = np.array([self.ext2int[int(b)] for b in self.branch[:, 1]])
        return f, t

    def gen_bus_idx(self) -> np.ndarray:
        on = self.gen[:, 7] > 0
        return np.array([self.ext2int[int(b)] for b in self.gen[on, 0]])

    def base_injections(self):
        """Net P, Q injection per bus (pu) from the case file."""
        Pd = self.bus[:, 2] / self.baseMVA
        Qd = self.bus[:, 3] / self.baseMVA
        Pg = np.zeros(self.n); Qg = np.zeros(self.n)
        on = self.gen[:, 7] > 0
        for row in self.gen[on]:
            i = self.ext2int[int(row[0])]
            Pg[i] += row[1] / self.baseMVA
            Qg[i] += row[2] / self.baseMVA
        return Pg - Pd, Qg - Qd

    def vset(self) -> np.ndarray:
        v = np.ones(self.n)
        on = self.gen[:, 7] > 0
        for row in self.gen[on]:
            v[self.ext2int[int(row[0])]] = row[5]
        return v

    def line_limits(self) -> np.ndarray:
        r = self.branch[:, 5].copy()
        r[r <= 0] = 9999.0
        return r / self.baseMVA


def load_matpower(path: str) -> Grid:
    txt = open(path).read()

    def block(name):
        m = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", txt, re.S)
        rows = []
        for line in m.group(1).splitlines():
            line = line.split("%")[0].strip().rstrip(";")
            if line:
                rows.append([float(x) for x in line.split()])
        w = max(len(r) for r in rows)
        return np.array([r + [0.0] * (w - len(r)) for r in rows])

    base = float(re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)", txt).group(1))
    bus, gen, br = block("bus"), block("gen"), block("branch")
    br = br[br[:, 10] > 0]  # in-service branches only
    g = Grid(base, bus, gen, br, name=path.split("/")[-1].replace(".m", ""))
    g.ext2int = {int(b): i for i, b in enumerate(bus[:, 0])}
    return g


# ---------------------------------------------------------------------------
# Admittance matrix
# ---------------------------------------------------------------------------
def build_ybus(g: Grid, branch_scale: np.ndarray | None = None):
    """
    branch_scale: optional per-branch multiplier on series admittance.
    Used to give the *attacker* an imperfect grid model.
    """
    n = g.n
    f, t = g.from_to
    br = g.branch
    r, x, b = br[:, 2], br[:, 3], br[:, 4]
    tap = np.where(br[:, 8] == 0, 1.0, br[:, 8])
    shift = np.deg2rad(br[:, 9])
    ys = 1.0 / (r + 1j * x)
    if branch_scale is not None:
        ys = ys * branch_scale
    tc = tap * np.exp(1j * shift)
    Ytt = ys + 1j * b / 2
    Yff = Ytt / (tc * np.conj(tc))
    Yft = -ys / np.conj(tc)
    Ytf = -ys / tc
    Ysh = (g.bus[:, 4] + 1j * g.bus[:, 5]) / g.baseMVA
    Y = np.zeros((n, n), dtype=complex)
    np.add.at(Y, (f, f), Yff); np.add.at(Y, (t, t), Ytt)
    np.add.at(Y, (f, t), Yft); np.add.at(Y, (t, f), Ytf)
    Y[np.arange(n), np.arange(n)] += Ysh
    return Y, (Yff, Yft, Ytf, Ytt)


def branch_flows(g: Grid, V: np.ndarray, Yb):
    Yff, Yft, Ytf, Ytt = Yb
    f, t = g.from_to
    If = Yff * V[f] + Yft * V[t]
    It = Ytf * V[f] + Ytt * V[t]
    Sf = V[f] * np.conj(If)
    St = V[t] * np.conj(It)
    return Sf, St


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------
class Oracle:
    """Map reported injections -> implied |V|, theta, branch flows."""
    name = "base"

    def solve(self, P: np.ndarray, Q: np.ndarray, Vset: np.ndarray):
        raise NotImplementedError


class NewtonRaphsonOracle(Oracle):
    name = "AC-NR"

    def __init__(self, g: Grid, branch_scale=None, tol=1e-8, max_it=30):
        self.g = g
        self.Y, self.Yb = build_ybus(g, branch_scale)
        self.tol, self.max_it = tol, max_it
        bt = g.bus_type
        self.ref = int(np.where(bt == REF)[0][0])
        self.pv = np.where(bt == PV)[0]
        self.pq = np.where(bt == PQ)[0]
        self.pvpq = np.r_[self.pv, self.pq]

    def solve(self, P, Q, Vset):
        g, Y = self.g, self.Y
        n = g.n
        Vm = np.where(g.bus_type == PQ, 1.0, Vset)
        Va = np.zeros(n)
        pvpq, pq = self.pvpq, self.pq
        npv, npq = len(self.pv), len(pq)
        conv = False
        for it in range(self.max_it):
            V = Vm * np.exp(1j * Va)
            S = V * np.conj(Y @ V)
            mis = np.r_[(S.real - P)[pvpq], (S.imag - Q)[pq]]
            if np.max(np.abs(mis)) < self.tol:
                conv = True
                break
            # Jacobian
            Ibus = Y @ V
            diagV = np.diag(V); diagI = np.diag(Ibus)
            diagVn = np.diag(V / np.abs(V))
            dS_dVa = 1j * diagV @ np.conj(diagI - Y @ diagV)
            dS_dVm = diagV @ np.conj(Y @ diagVn) + np.conj(diagI) @ diagVn
            J11 = dS_dVa[np.ix_(pvpq, pvpq)].real
            J12 = dS_dVm[np.ix_(pvpq, pq)].real
            J21 = dS_dVa[np.ix_(pq, pvpq)].imag
            J22 = dS_dVm[np.ix_(pq, pq)].imag
            J = np.block([[J11, J12], [J21, J22]])
            dx = np.linalg.solve(J, -mis)
            Va[pvpq] += dx[:npv + npq]
            Vm[pq] += dx[npv + npq:]
        V = Vm * np.exp(1j * Va)
        Sf, St = branch_flows(g, V, self.Yb)
        return dict(Vm=Vm, Va=Va, Sf=Sf, St=St, converged=conv, iters=it)


class DCOracle(Oracle):
    name = "DC"

    def __init__(self, g: Grid):
        self.g = g
        f, t = g.from_to
        x = g.branch[:, 3]
        tap = np.where(g.branch[:, 8] == 0, 1.0, g.branch[:, 8])
        b = 1.0 / (x * tap)
        n = g.n
        B = np.zeros((n, n))
        np.add.at(B, (f, f), b); np.add.at(B, (t, t), b)
        np.add.at(B, (f, t), -b); np.add.at(B, (t, f), -b)
        self.B, self.b, self.f, self.t = B, b, f, t
        self.shift = np.deg2rad(g.branch[:, 9])
        self.ref = int(np.where(g.bus_type == REF)[0][0])

    def solve(self, P, Q, Vset):
        n = self.g.n
        keep = np.array([i for i in range(n) if i != self.ref])
        Va = np.zeros(n)
        Va[keep] = np.linalg.solve(self.B[np.ix_(keep, keep)], P[keep])
        Pf = self.b * (Va[self.f] - Va[self.t] - self.shift)
        return dict(Vm=np.ones(n), Va=Va, Sf=Pf + 0j, St=-Pf + 0j,
                    converged=True, iters=1)
