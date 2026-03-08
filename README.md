# Contextual Forecasting Integrator (CFI) v0.2

A continuous, context-weighted risk steering architecture for conversational AI. Model-agnostic middleware that replaces binary safety gates with adaptive control-theoretic steering.

## Features
- Multi-horizon risk forecasting with causal inference
- Closed-loop meta-learning for self-calibration
- Smoothing (Kalman filters) and dynamic hysteresis for reduced jitter
- Stabilizers: CCI, VOL, AMB, ICI, CSS
- Telemetry logging for audits
- Eval suite with benchmarks (ERR, FPS, FPR) and ablations
- Deployment examples for HuggingFace pipelines

BenchmarksRun python evals/run_benchmark.py --dataset harmbench for results.LicenseMIT (see LICENSE)ContributionsPRs welcome. All changes must include:Ablations (e.g., no_hysteresis vs baseline)
Evals on at least HarmBench/WildChat
Updated docs

Contact: DM on X @D_McMillan76

## Related: EPIC (Epistemic Predictive Integrity Core)
Full epistemic governor built on CFI principles:  
https://github.com/PsychoFrogMultimedia/EPIC-truth-calibrated-architecture-

## Installation
```bash
pip install -r requirements.txt

UsageSee examples/wrap_llama.py for integration:python

from cfi.core import CFI

cfi = CFI(params={ 'beta': 0.7, 'w_short': 0.5 })  # Custom params
u_t = "Current user prompt"
H_t = ["History turn 1", "History turn 2"]
S_prev = cfi.get_initial_state()
band, guidance, S_new = cfi(u_t, H_t, S_prev)
if band == 'steer':
    # Apply guidance to model generation
    pass
