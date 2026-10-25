"""
Print the headline tables of the write-up from results/*.json, and (--check) verify that
every row appears verbatim in submission/writeup.md.

  python scripts/headline_numbers.py            # print the tables
  python scripts/headline_numbers.py --check    # exit 1 if the write-up disagrees with the results

Rows whose result file does not exist yet are skipped (so it works on a partial set).
"""
import json, os, sys

R = "results/results_{}.json"
KS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]

def load(name):
    f = R.format(name)
    return json.load(open(f)) if os.path.exists(f) else None

def det(name, key):
    r = load(name)
    return None if r is None else [l[key] for l in r["levels"]]

def thr(name, key):
    r = load(name)
    return None if r is None else r["thresholds"][key]

# (row label in the write-up, values)
DET_DATAKIT = [
    ("physics check, AC-NR (reference)", det("nr_ramp1_datakit", "det_phys")),
    ("physics check, GENCO as released", det("genco_ramp1_datakit", "det_phys")),
    ("physics check, GENCO fine-tuned on case-file walks only", det("genco_ftwalk_ramp1_datakit", "det_phys")),
    ("physics check, GENCO fine-tuned on both regimes", det("genco_ft_mix_ramp1_datakit", "det_phys")),
    ("physics check, DC", det("nr_ramp1_datakit", "det_dc")),
    ("PMU cross-check", det("nr_ramp1_datakit", "det_pmu")),
    ("physics (AC-NR) + PMU", det("nr_ramp1_datakit", "det_fused")),
]
DET_CASE = [
    ("physics check, AC-NR (reference)", det("nr_ramp1", "det_phys")),
    ("physics check, GENCO as released", det("genco_ramp1", "det_phys")),
    ("physics check, GENCO fine-tuned on case-file walks only", det("genco_ftwalk_ramp1", "det_phys")),
    ("physics check, GENCO fine-tuned on both regimes", det("genco_ft_mix_ramp1", "det_phys")),
]
def thr_row(label, key, names):
    vals = [thr(n, key) for n in names]
    return (label, None if any(v is None for v in vals) else vals)

NAMES = {"nr": ["nr_ramp1_datakit", "nr_ramp5_datakit", "nr_ramp1", "nr_ramp5"],
         "genco": ["genco_ramp1_datakit", "genco_ramp5_datakit", "genco_ramp1", "genco_ramp5"],
         "ftwalk": ["genco_ftwalk_ramp1_datakit", "genco_ftwalk_ramp5_datakit", "genco_ftwalk_ramp1", "genco_ftwalk_ramp5"],
         "ftmix": ["genco_ft_mix_ramp1_datakit", "genco_ft_mix_ramp5_datakit", "genco_ft_mix_ramp1", "genco_ft_mix_ramp5"]}
THR = [thr_row("AC-NR", "phys", NAMES["nr"]),
       thr_row("GENCO as released", "phys", NAMES["genco"]),
       thr_row("GENCO fine-tuned on case-file walks only", "phys", NAMES["ftwalk"]),
       thr_row("GENCO fine-tuned on both regimes", "phys", NAMES["ftmix"]),
       thr_row("DC", "dc", NAMES["nr"])]

def fmt(vals, nd):
    return " | ".join(f"{v:.{nd}f}" for v in vals)

tables = [("detection, step attack, datakit scenarios", DET_DATAKIT, 2),
          ("detection, step attack, case-file operating points", DET_CASE, 2),
          ("clean physics threshold (datakit step, datakit ramp, case step, case ramp)", THR, 1)]

check = "--check" in sys.argv
text = open("submission/writeup.md").read() if check else ""
bad = 0
for title, rows, nd in tables:
    print(f"\n## {title}")
    for label, vals in rows:
        if vals is None:
            print(f"  {label:58s} (no results file yet)")
            continue
        line = fmt(vals, nd)
        ok = ""
        if check:
            hit = f"| {label} | {line} |" in text
            ok = "  OK" if hit else "  MISMATCH vs write-up"
            bad += 0 if hit else 1
        print(f"  {label:58s} {line}{ok}")
if check:
    print("\nALL ROWS MATCH" if bad == 0 else f"\n{bad} ROW(S) DO NOT MATCH THE WRITE-UP")
    sys.exit(1 if bad else 0)
