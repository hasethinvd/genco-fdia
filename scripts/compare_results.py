"""
Compare a regenerated results file with the committed reference.

  python scripts/compare_results.py reference.json regenerated.json [--tol_det 0.0 --tol_thr 0.0]

Prints per-k detection rates and thresholds side by side and exits 1 if any difference
exceeds the tolerances. CPU oracles (AC-NR, DC) are deterministic and should match exactly;
GPU inference may differ in the last bits, so the GENCO checks use a small tolerance.
"""
import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("reference")
ap.add_argument("regenerated")
ap.add_argument("--tol_det", type=float, default=0.0, help="max |difference| in a detection rate")
ap.add_argument("--tol_thr", type=float, default=0.0, help="max relative difference in a clean threshold")
args = ap.parse_args()

A, B = json.load(open(args.reference)), json.load(open(args.regenerated))
bad = 0
print(f"oracle: {A['oracle']}  vs  {B['oracle']}")
for key in ("phys", "dc", "pmu", "jump"):
    a, b = A["thresholds"][key], B["thresholds"][key]
    rel = abs(a - b) / max(abs(a), 1e-12)
    flag = "" if rel <= args.tol_thr + 1e-12 else "  <-- DIFF"
    bad += bool(flag)
    print(f"threshold {key:5s} {a:10.4f} {b:10.4f}  rel {rel:.2e}{flag}")
for la, lb in zip(A["levels"], B["levels"]):
    assert la["k"] == lb["k"]
    row = f"k={la['k']:4.2f}"
    for m in ("det_phys", "det_dc", "det_pmu", "det_fused", "det_jump"):
        d = abs(la[m] - lb[m])
        flag = "*" if d > args.tol_det + 1e-12 else " "
        bad += flag == "*"
        row += f"  {m[4:]:5s} {la[m]:.3f}/{lb[m]:.3f}{flag}"
    print(row)
print("PASS: regenerated results match the reference within tolerance" if bad == 0
      else f"FAIL: {bad} value(s) differ beyond tolerance (marked * or DIFF)")
sys.exit(1 if bad else 0)
