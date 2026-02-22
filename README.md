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

## Installation
```bash
pip install -r requirements.txt
