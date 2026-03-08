Contextual Forecasting Integrator (CFI) v0.2


![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)


Continuous, context-weighted risk forecasting and adaptive steering middleware for conversational AI model-agnostic layer that replaces binary safety gates with control-theoretic posture regulation — forecasting multi-horizon risk and gently steering generation to reduce confident unsupported output while preserving natural flow.

CFI is the core forecasting & steering engine that powers EPIC v1.0 — the full epistemic governance architecture.


Core Idea

CFI runs continuously across turns:

Forecasts short / mid / far horizon risk across 10 dimensions
(ambiguity load, bluff pressure, anchor thinness, calibration mismatch, narrative leakage, capture risk, …)

Outputs a steering band + guidance + updated state

Applies smoothing (Kalman-like) + dynamic hysteresis to avoid jitter

Self-calibrates from outcome signals (closed-loop)

No hard refusals by default — prefers steering over blocking.


Features

Stateful multi-horizon forecasting (short: current turn, mid: exchange, far: cross-exchange drift)

10 normalized [0–1] forecast dimensions

5 named stabilizers: CCI (coherence), VOL (volatility), AMB (ambiguity), ICI (integrity), CSS (calibration stability)

7 discrete steering bands (normal → cautious → retrieve-first → abstain-cleanly, etc.)

Dynamic hysteresis on lane/band switches

Full telemetry output (forecast scores, band history, oscillation index, …)

Designed for integration into prompt wrappers, inference loops, agent toolchains


Related: EPIC v1.0 — Epistemic Predictive Integrity Core

CFI has been extended into a complete epistemic governor:

EPIC-truth-calibrated-architecture

→ Adds Platonic 5+1 processing spine

→ Adds ARC reality-binding & claim-state routing

→ Enforces legible output lanes (verified / inference / weak / speculative / unknown / narrative)

→ Includes runtime JSON spec + companion guideIf you want the full truth-calibrated system (upstream routing + mandatory disclosure), start with EPIC.

CFI remains the beating heart of its steering logic.


Basic Usage

python

from cfi.core import CFI

# Initialize once (tune params to your tolerance / domain)

cfi = CFI(
    params={
        'beta': 0.7,                # smoothing strength
        'w_short': 0.5,             # short-horizon weight
        'w_mid': 0.3,
        'w_far': 0.2,
        # ... hysteresis margins, stabilizer weights, etc.
    }
)


# Per-turn call (stateful — pass previous state)

user_prompt = "Current user message here"

history = ["turn 1 assistant", "turn 1 user", "turn 2 assistant"]

prev_state = cfi.get_initial_state()  # or from previous turn

steering_band, guidance, new_state = cfi(
    prompt=user_prompt,
    history=history,
    prev_control_state=prev_state
)


# Act on the steering decision

if steering_band == 'normal':
    # proceed with normal generation
    pass
elif steering_band == 'cautious':
    # inject guidance into system prompt or sampler kwargs
    print("Guidance:", guidance)
elif steering_band in ['retrieve_first', 'clarify_first']:
    # trigger RAG / ask clarifying question
    pass
elif steering_band == 'abstain_cleanly':
    # safe, non-defensive refusal path
    pass


# Carry state forward

next_prev_state = new_state


Status

Reference implementation + operational spec

Primary role: reliable steering layer inside EPIC

Open for: bug reports, tuning suggestions, evaluation ideas, integration patterns


License

MIT


Contributing

Issues, PRs, and discussions welcome — especially around:

Stabilizer behavior improvements

New forecast dimensions or telemetry views

Integration examples with popular LLM frameworks

Ablation experiments on smoothing/hysteresis impact

Feel free to open an issue first to discuss direction.

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
