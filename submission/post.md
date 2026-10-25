# Stealthy false-data injection against a grid foundation model used as a physics oracle

**Participant:** Harpreet Sethi (NVIDIA)  
**Contact:** hasethi@nvidia.com

## Summary

A power-flow check lets a control room test whether the telemetry it receives is physically consistent, and a false-data injection (FDI) that is to pass the check must be computed with a model of the grid. Running that check for every substation, scan and hypothesis is a large power-flow workload, which a grid foundation model such as GENCO can evaluate in bulk on a GPU at the price of approximation error. We ask how well an attacker must know the grid to produce data that still passes, and how small a surrogate's error must be, relative to sensor noise, for its disagreement with the data to carry information. In simulation on IEEE 118, an exact AC power flow (Newton-Raphson, NR) detects the attack for attacker knowledge `k` up to 0.75 (detection 0.91 to 1.00) and fails as the attacker's model approaches the true one (0.06 at `k = 0.9`, 0.01 at `k = 1`), where the attack hides 40 to 70 MW of load. A cross-check of SCADA flows against PMU phasors on 20% of buses does not depend on `k` and detects 0.64 of the `k = 1` step attacks and 0.49 of the ramped ones on datakit scenarios.

Used as the oracle, the released GENCO checkpoint has a clean-data detection threshold 7 times that of NR on the datakit scenarios it was trained on (107 against 15, in sensor-noise units) and 19 times on the standard case file (294 against 15), and its detection falls accordingly. We trace the second case to the PF task's treatment of the slack voltage and to case-file generator setpoints that lie outside the training distribution. Fine-tuning on a loss written in the detector's own units, with NR labels, closes the threshold gap: a model trained on both regimes reaches thresholds of 13 to 14, comparable to the exact oracle's, whereas a fine-tune on one regime raises the threshold outside it (107 to 549). The fine-tuned model also reports detections at `k = 0.9` and `k = 1` that the exact oracle does not (0.35 against 0.01 at `k = 1`); these reflect surrogate error on attacked inputs, not physical inconsistency. We therefore regard a fine-tuned GENCO as a candidate screen in front of an exact solver, a pipeline we did not measure.

## Motivation

State estimators reject measurements that are inconsistent with the network model, but Liu et al. (ACM CCS 2009) showed that an attacker who knows the model can alter measurements so that the injected error passes bad-data detection, and Hug and Giampapa (IEEE Transactions on Smart Grid, 2012) assessed the vulnerability of AC state estimation to such injections. A direct defence solves the power flow implied by the reported injections and compares the result with the reported voltages and flows. Doing so for every substation, every scan and every hypothesis that a report is false is a large workload. A grid foundation model can evaluate many cases in one batched pass, but it returns a state for every input, including inputs for which no consistent state exists (entries #7 and #9 of this challenge report this; we did not test it). We do not claim that GENCO is faster than a tuned solver on a single 118-bus scan; the Results show that it is not.

Whether an approximate oracle can serve depends on the ratio of its own error to the sensor noise. Branch flows follow from voltage angles through line susceptances of order 10 to 20 per unit, so an angle error of 0.1 degrees, comparable to the error we measure for the released checkpoint on a datakit scenario, corresponds to a flow error of roughly 2 to 3 MW against a sensor noise of 0.5 MW. The detection threshold must exceed the oracle's own error to hold the false-alarm rate fixed, so the oracle's error, and not only the attack, can set the smallest detectable attack. We therefore ask two questions: how well must an attacker know the grid's electrical parameters to produce measurements that still satisfy the power-flow equations, and how small must a surrogate's error be, compared with sensor noise, for its disagreement with the data to carry information?

## Method

1. **Grid and scenarios.** IEEE 118 (MATPOWER data; identical in topology, impedances, loads and shunts to the PGLib case that datakit uses). Operating points are 720 load scenarios generated with `gridfm-datakit` (default `agg_load_profile` load perturbation, with topology, generation and admittance perturbations switched off), each including its slack and PV voltage setpoints, which datakit's OPF-derived states vary and which are inputs to every oracle. Trajectories are windows of consecutive samples. A second set, the case-file operating points, uses the case file's own dispatch with a bounded random walk in load. Each scan reports bus P, Q and |V| and line P and Q with Gaussian sensor noise (0.5 MW, 0.2% in |V|). The ground truth is AC NR on this grid in every experiment; only the oracle changes.
2. **Attacker.** The attacker controls a contiguous set of 8 substations and their boundary, perturbs the state inside, keeps boundary voltages fixed, and recomputes every measurement it owns with an AC power flow on its own grid model, whose branch parameters are correct with probability `k`. At `k = 1` the attack is consistent with the power-flow equations; at `k = 0` it is a naive scaling. The attack hides roughly 40 to 70 MW of real load from the operator (we report MW, not degrees); a ramped variant spreads it over 5 scans.
3. **Detectors.** Each is thresholded on clean scans of the same experiment at a 1% false-positive rate, and detection is the fraction of attacked scans whose score exceeds that threshold.
   - *Physics check:* solve power flow from the reported injections with the oracle (NR, DC or GENCO), compare the implied |V| and line flows with the reported ones, and take the per-scan maximum of the normalised residual. The clean threshold is this maximum's 99th percentile on clean scans, in units of the sensor noise.
   - *Temporal:* jump in the oracle-implied state between consecutive scans.
   - *PMU cross-check:* for lines with PMUs at both ends (20% of buses, highest degree first, on a separate channel), compute the flow from phasors and compare it with the SCADA flow. Detection here is limited by PMU coverage, not by attacker knowledge.
   - *Fused:* the maximum of the physics and PMU scores in threshold units.
4. **GENCO adapter.** We use `gridfm-graphkit` (paper-repro branch) with the `gridfm/genco-pf-datakit` IEEE 118 Base checkpoint. The graph is built from our own grid (true admittances) and passed once through graphkit's normalisation and masking pipeline; per scan only the dynamic inputs are rewritten. On datakit scenarios the adapter's input tensors match graphkit's own pipeline to below 3e-7, and the checkpoint reproduces datakit's solved state to 0.10 degrees in angle (maximum over buses, one scenario). The oracle receives net injections, which we split into load and generation by a stated rule: load buses take `Pd = -P`; generator and slack buses keep the case-file load and `Pg = P + Pd`. Branch flows are recomputed from the predicted state with the true admittances.
5. **Toolkits.** Scenarios come from `gridfm-datakit` and GENCO inference and fine-tuning from `gridfm-graphkit` and the released checkpoint. The repository also contains a dependency-free NR oracle, so every result that does not involve GENCO reproduces on a CPU.

### Fine-tuning

The released `gridfm/genco-pf-transfer` repository already contains from-scratch and fine-tuned IEEE 118 checkpoints at seven training-set sizes (the paper's transfer experiment), so fine-tuning GENCO on this grid is not new in itself. Our fine-tune differs in its target: the loss is the detector's own residual, the labels are NR solutions of noisy inputs, and we test whether one model holds two operating regimes. We fine-tune the released checkpoint twice. In both runs the inputs carry the sensor noise that the detector sees, and the labels are NR solutions of the same noisy inputs. The loss is a Huber loss on Vm, Va and branch P and Q flows, each divided by the detector's sensor noise, so the model is trained on the quantity the physics check measures. Held-out scans are grouped by trajectory, and the experiment's own scenarios are never used in training (for the datakit data this is checked: the training and test samples come from different generator seeds and are different draws).

- *Case-file only:* 600 fresh load-walk trajectories of 10 scans (5,400 training and 600 held-out scans), 7,155 steps in 25 minutes on one GB200.
- *Mixed:* 1,500 fresh datakit scenarios with 4 noise draws each, plus the 600 case-file walk trajectories (10,854 training and 1,146 held-out scans), 8,313 steps in 30 minutes.

## Results

![fig1](https://raw.githubusercontent.com/hasethinvd/genco-fdia/main/results/fig1_detection_vs_knowledge_datakit.png)

*Figure 1 — Detection by the physics check against attacker knowledge `k`, step attack on datakit scenarios, for three oracles: exact NR (left), GENCO as released (middle) and GENCO fine-tuned on both regimes (right). The exact curve falls only beyond `k = 0.75`; the released GENCO falls from `k = 0.25`; the fine-tuned model follows the exact curve up to `k = 0.75` and stays above it beyond, which we attribute to surrogate error (see text). The PMU curve is the same in all panels.*

### Detection against attacker knowledge

With NR as the oracle, detection stays between 0.91 and 1.00 for `k` up to 0.75 and falls to 0.06 at `k = 0.9` and 0.01 at `k = 1` (ramped attack: 0.52 at `k = 0.9`). DC power flow does not serve as an oracle here: its approximation error sets a clean threshold of 194 against 15 for NR, and only naive attacks exceed it. The PMU cross-check is flat in `k` because it does not use the attacker's model; it detects an attack when a PMU pair spans an attacked line. Fusing it with the NR physics check gives 0.65 at `k = 1`, never lower than either check alone in this table. These results hold for this attacker construction and noise model.

| attacker knowledge k (step attack, datakit scenarios) | 0 | 0.25 | 0.5 | 0.75 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| physics check, AC-NR (reference) | 1.00 | 0.95 | 0.91 | 0.95 | 0.06 | 0.01 |
| physics check, GENCO as released | 0.90 | 0.37 | 0.21 | 0.22 | 0.13 | 0.18 |
| physics check, GENCO fine-tuned on case-file walks only | 0.08 | 0.08 | 0.03 | 0.02 | 0.07 | 0.03 |
| physics check, GENCO fine-tuned on both regimes | 1.00 | 0.96 | 0.93 | 0.98 | 0.37 | 0.35 |
| physics check, DC | 0.97 | 0.28 | 0.08 | 0.04 | 0.02 | 0.02 |
| PMU cross-check | 0.79 | 0.73 | 0.67 | 0.69 | 0.68 | 0.64 |
| physics (AC-NR) + PMU | 1.00 | 0.95 | 0.96 | 1.00 | 0.69 | 0.65 |

A detection rate above the exact oracle's at `k = 1` cannot arise from physical inconsistency, because a perfectly consistent attack leaves none. The GENCO rows at `k = 1` (0.18 for the released checkpoint, 0.35 for the mixed fine-tune, against 0.01 for NR) therefore reflect GENCO's own error growing on the shifted inputs of an attacked scan, and we do not count them as detection.

![fig2](https://raw.githubusercontent.com/hasethinvd/genco-fdia/main/results/fig2_bus_map_datakit.png)

*Figure 2 — One full-knowledge attack on datakit scenarios. Left: the physics residual stays below the threshold at every bus. Right: the PMU line check exceeds its threshold at the attacked boundary.*

### The error floor of the oracle

The clean threshold is the quantity that connects oracle error to detection, and the table below reports it for every oracle.

| clean physics threshold at 1% FPR (noise-normalised units) | datakit, step | datakit, ramped | case file, step | case file, ramped |
|---|---|---|---|---|
| AC-NR | 15.1 | 14.9 | 15.5 | 15.8 |
| GENCO as released | 106.9 | 105.6 | 294.2 | 291.8 |
| GENCO fine-tuned on case-file walks only | 548.6 | 499.1 | 11.4 | 12.4 |
| GENCO fine-tuned on both regimes | 13.1 | 13.8 | 12.9 | 14.0 |
| DC | 193.8 | 196.1 | 172.2 | 171.4 |

The released checkpoint's threshold is 7 times that of NR on datakit scenarios and 19 times on the case-file operating points, and its detection on the latter falls to 0.05 to 0.15 for `k` of 0.25 and above.

| attacker knowledge k (step attack, case-file operating points) | 0 | 0.25 | 0.5 | 0.75 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| physics check, AC-NR (reference) | 1.00 | 0.97 | 0.91 | 0.81 | 0.35 | 0.01 |
| physics check, GENCO as released | 0.50 | 0.05 | 0.11 | 0.11 | 0.12 | 0.15 |
| physics check, GENCO fine-tuned on case-file walks only | 1.00 | 1.00 | 0.95 | 0.96 | 0.81 | 0.38 |
| physics check, GENCO fine-tuned on both regimes | 1.00 | 1.00 | 0.92 | 0.87 | 0.67 | 0.26 |

On the case file's own dispatch the released checkpoint predicts a slack voltage of 0.987 where the case has 1.035, and its angle error against NR is 2.9 degrees on average and up to 4.4 degrees. The adapter is not the cause: on datakit's own scenarios the checkpoint is accurate to 0.1 degrees. In datakit's data the slack voltage is an OPF output that varies from scenario to scenario, and the PF task masks it, so the model infers it from the other inputs (the released configuration sets `mask_ref_vm: true`, and the checkpoint repository's commit of 14 October 2026 keeps the slack voltage masked for the paper's PF checkpoints, so this is the intended input pattern rather than an adapter choice); the case file's generator setpoints lie outside the distribution it learned. Supplying the slack voltage as an input does not help (angle error rises to 10 degrees), which suggests that the model has not seen that input pattern in training.

### Fine-tuning: what it fixes and what it breaks

Fine-tuning on one regime lowers the threshold in that regime and raises it elsewhere. On held-out case-file scans the 99th percentile of the per-scan maximum oracle-minus-NR error, in sensor-noise units (the quantity that sets the clean threshold), falls from 293.5 to 13.8 (median 250 to 3.9) in 25 minutes of training, and the experiment's clean threshold on fresh case-file trajectories falls from 294 to 11, below the 15.5 of NR. The same model has a threshold of 549 on datakit scenarios that it did not see, and its detection there falls to 0.02 to 0.08.

Training on both regimes removes the trade-off. The mixed model's held-out error falls from a 99th percentile of 294 to 6.6 (median 85 to 1.9) in 30 minutes, and its clean thresholds on fresh trajectories are 13.1 and 13.8 on datakit scenarios and 12.9 and 14.0 on case-file operating points (step and ramped attacks), against about 15 for NR. Its detection tracks the exact oracle's for `k` up to 0.75 (0.93 to 0.98 against 0.91 to 0.95 on datakit scenarios). Above that it reports more than NR, as discussed above. Each fine-tune is a single training seed, and we measured no variation between seeds.

![fig3](https://raw.githubusercontent.com/hasethinvd/genco-fdia/main/results/fig3_threshold_and_finetune.png)

*Figure 3 — Left: clean physics threshold per oracle at a 1% false-positive rate (log scale). The case-file-only fine-tune (light purple) is lowest on the case file and highest on datakit scenarios; the model trained on both regimes is close to NR on both. Right: held-out oracle error, in sensor-noise units, during the two fine-tunes.*

### Runtime

Per scan on IEEE 118:
- GENCO: 44 ms at batch size 1 and 0.8 ms per scan at batch size 64 (one GB200 GPU).
- AC NR (numpy, one CPU core of the GPU node): 11 ms.
- DC: 0.8 ms.

GENCO is slower than a tuned CPU solver for a single scan; its advantage, if any, lies in checking many scans together.

## Conclusion

For the attacker we simulated, an exact power-flow check detects the injection until the attacker's grid model is nearly exact, and a PMU cross-check, which does not depend on that model, recovers about half or more of the attacks that remain. The error of the oracle sets how much of this a surrogate retains: the released GENCO checkpoint raises the clean detection threshold 7 to 19 times above that of the exact oracle and loses most of the detection, whereas a fine-tune on both operating regimes brings the threshold to 13 to 14 against 15. Operators or researchers who consider a grid foundation model as a screening oracle can use the clean threshold, measured in sensor-noise units, as a direct test of its fitness, and can expect that fine-tuning on the operator's own operating points matters. If such a model screens scans in bulk and an exact solver confirms those it flags, the exact solver would be called on only a fraction of the scans; we did not measure this pipeline, nor the referral rate it would imply.

The results are those of a simulation study. The clean threshold is calibrated at a 1% per-scan false-positive rate; at a typical 2 to 4 second SCADA scan that is hundreds of false alarms a day from one check, so an operational threshold would be set lower and detection would be lower than reported. Real telemetry errors are not Gaussian, the grid model has errors of its own, and the attacker is fixed and does not adapt to the detector or to the surrogate's blind spots. The analysis is steady-state, covers one transmission grid and one noise model, assumes the PMU channel is uncompromised, and includes no network-layer evidence. The fine-tunes are trained and tested against NR on the same grid, each with a single seed, so we make no claim about other grids or about variance between seeds. The extra detections of the fine-tuned model at high attacker knowledge are surrogate error and not physical detection.

## Next steps

Measure the screening pipeline directly (the fraction of clean scans referred to the exact solver against the fraction of exact-oracle detections retained); replace the single-scan flow comparison with a state-estimation detector, for example GENCO fine-tuned for state estimation on noisy data as entry #12 does for tap identification, so that oracle error is not amplified through line susceptances; start from the subgrid-pretrained checkpoint in `gridfm/genco-pf-transfer` to test whether broader pretraining reduces the specialisation we observed; evaluate runtime on case2000; test whether attacked scans separate in GENCO's latent space; fuse a network-layer signal such as substation traffic anomaly scores; and replace the fixed-threshold fusion with a calibrated joint likelihood.

Code: https://github.com/hasethinvd/genco-fdia (Apache-2.0). Results JSON and figures: https://github.com/hasethinvd/genco-fdia/tree/main/results.

**Tooling and AI assistance.** The analysis code, experiments and this write-up were developed with Claude Code (Anthropic). The experiments were run by the author on a GPU cluster, and every number in the tables is checked against the results files by `scripts/headline_numbers.py`.
