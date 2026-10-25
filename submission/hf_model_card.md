---
license: apache-2.0
base_model: gridfm/genco-pf-datakit
tags: [power-flow, grid-foundation-model, genco, false-data-injection]
---

# GENCO-PF fine-tuned as a physics oracle for false-data-injection detection (IEEE 118)

Two fine-tunes of the `gridfm/genco-pf-datakit` IEEE 118 Base checkpoint (`case118_ieee/base/seed0`),
produced for a GENCO Challenge entry. Code, data and the write-up: https://github.com/hasethinvd/genco-fdia.

Each folder holds `best_model_state_dict.pt`, the training `*.yaml` and `normalizer_stats.pt`, in the same
layout as the base checkpoint, and loads with the entry's `src/genco_oracle.py` (`GENCO_CKPT_DIR=<folder>`).

| folder | training data | steps / time (one GB200) | held-out per-scan max residual, p99 (sensor-noise units) |
|---|---|---|---|
| `mixed` | 1,500 gridfm-datakit IEEE 118 load scenarios x 4 noise draws + 600 case-file load-walk trajectories (10,854 scans) | 8,313 / 30 min | 294 -> 6.6 |
| `case_file_only` | 600 case-file load-walk trajectories (5,400 scans) | 7,155 / 25 min | 294 -> 13.8 |

Training: inputs carry Gaussian sensor noise (0.5 MW on injections); labels are AC Newton-Raphson solutions of the
same noisy inputs; loss is a Huber loss on Vm, Va and branch P/Q flows, each divided by the detector's sensor noise.

Intended use: research artefact for the experiment in the linked repository (a physics-consistency check of SCADA
telemetry on IEEE 118). Limitations: one grid, one noise model, one training seed per model; the `case_file_only`
model is accurate on the case file's operating points and poor on datakit scenarios (see the write-up); detections
above the exact oracle's at full attacker knowledge are surrogate error, not physical detection. Not validated for
operational use.
