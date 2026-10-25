"""
Detectors.

1. Physics-consistency residual (the GENCO role)
   Take the reported injections (P, Q, with Vset), ask the oracle what voltages
   and flows they physically imply, compare with the reported Vm / Pf / Qf.
   Per-bus score = max normalised mismatch touching that bus. Scan score = max
   over buses. Works with any Oracle (AC-NR, DC, GENCO).

2. Temporal state-jump
   Consecutive scans should move smoothly. Score = max |ΔVm| and |ΔVa| of the
   oracle-implied state between scan t-1 and t, normalised by the clean
   distribution. An attacker that is perfectly consistent *within* a scan still
   has to make the state jump when the attack switches on.

Thresholds are calibrated on clean scans at a target false-positive rate.
"""
from __future__ import annotations
import numpy as np
from grid import Grid, Oracle
from attacks import NOISE


def physics_residual(g: Grid, oracle: Oracle, z: dict, Vset):
    r = oracle.solve(z["P"], z["Q"], Vset)
    f, t = g.from_to
    dV = (z["Vm"] - r["Vm"]) / NOISE["Vm"]
    dPf = (z["Pf"] - r["Sf"].real) / NOISE["Pf"]
    if oracle.name == "DC":
        dV = np.zeros_like(dV)               # DC has no voltage magnitude
        dQf = np.zeros_like(dPf)
    else:
        dQf = (z["Qf"] - r["Sf"].imag) / NOISE["Qf"]
    per_bus = np.abs(dV).copy()
    br = np.maximum(np.abs(dPf), np.abs(dQf))
    np.maximum.at(per_bus, f, br)
    np.maximum.at(per_bus, t, br)
    return per_bus, r


def state_jump(prev_state: dict | None, state: dict):
    if prev_state is None:
        return np.zeros_like(state["Vm"])
    dVm = np.abs(state["Vm"] - prev_state["Vm"]) / 0.002
    dVa = np.abs(np.angle(np.exp(1j * (state["Va"] - prev_state["Va"])))) / 0.005
    return np.maximum(dVm, dVa)


def calibrate(clean_scores: np.ndarray, fpr=0.01) -> float:
    return float(np.quantile(clean_scores, 1 - fpr))


def evaluate(scores_clean, scores_attack, thr):
    tpr = float(np.mean(scores_attack > thr))
    fpr = float(np.mean(scores_clean > thr))
    return tpr, fpr


# ---------------------------------------------------------------------------
# 3. Independent-channel check (PMU line flows)
#    PMUs on a subset of buses report phasors directly over a separate network
#    path (C37.118, not the SCADA channel the attacker owns). For every line
#    with a PMU at BOTH ends the operator can compute the line flow from the
#    phasors alone and compare it with the SCADA-reported flow. No global
#    power-flow solve is involved, so the check is as precise as the PMUs.
#    A SCADA-only attacker who is perfectly physics-consistent still cannot
#    make the phasors agree with the fake flows.
# ---------------------------------------------------------------------------
PMU_SIGMA_DEG = 0.01     # angle accuracy
PMU_SIGMA_VM = 0.001     # magnitude accuracy (pu)


def pmu_line_residual(g: Grid, Yb, true_Vm, true_Va, z_reported, pmu_buses, rng):
    f, t = g.from_to
    has = np.zeros(g.n, bool); has[pmu_buses] = True
    lines = np.where(has[f] & has[t])[0]
    per_bus = np.zeros(g.n)
    if len(lines) == 0:
        return per_bus, lines
    Vm = true_Vm + rng.normal(0, PMU_SIGMA_VM, g.n)
    Va = true_Va + rng.normal(0, np.deg2rad(PMU_SIGMA_DEG), g.n)
    V = Vm * np.exp(1j * Va)
    Yff, Yft, Ytf, Ytt = Yb
    Sf = V[f[lines]] * np.conj(Yff[lines] * V[f[lines]] + Yft[lines] * V[t[lines]])
    # per-line uncertainty of the PMU-derived real-power flow: angle and
    # magnitude errors at both ends propagate through the line susceptance
    x = g.branch[lines, 3]
    b = 1.0 / np.maximum(x, 1e-3)
    sig = np.sqrt(NOISE["Pf"] ** 2
                  + 2 * (b * np.deg2rad(PMU_SIGMA_DEG)) ** 2
                  + 2 * (b * PMU_SIGMA_VM * 0.2) ** 2)   # Vm enters P weakly
    score = np.abs(z_reported["Pf"][lines] - Sf.real) / sig
    np.maximum.at(per_bus, f[lines], score)
    np.maximum.at(per_bus, t[lines], score)
    return per_bus, lines


def place_pmus(g: Grid, frac: float, rng, strategy="degree"):
    """'degree': PMUs at the best-connected buses (how utilities actually place
    them); 'random': uniform."""
    m = int(round(frac * g.n))
    if strategy == "random":
        return np.sort(rng.choice(g.n, m, replace=False))
    f, t = g.from_to
    deg = np.bincount(np.r_[f, t], minlength=g.n).astype(float)
    deg += rng.random(g.n) * 1e-3          # tie-break
    return np.sort(np.argsort(-deg)[:m])
