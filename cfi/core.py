import numpy as np
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from .metrics import compute_CCI, compute_VOL, compute_AMB, compute_ICI, compute_CSS
from .control import kalman_smooth, apply_dynamic_hysteresis, produce_guidance
from .meta_learner import MetaLearner

class CFI:
    def __init__(self, params=None):
        default_params = {
            'beta': 0.7,
            'w_short': 0.5, 'w_mid': 0.3, 'w_long': 0.2,
            's1': 0.2, 's2': 0.15, 's3': 0.15, 's4': 0.2, 's5': 0.3,
            'kalman_noise_cov': 0.01,
            'clarifier_cooldown': 3,
        }
        self.params = params or default_params
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        self.risk_classifier = pipeline("text-classification", model="unitary/toxic-bert")  # Replace with custom
        self.meta_learner = MetaLearner(self.params)
        self.state = self.get_initial_state()
        self.last_clarifier_turn = -self.params['clarifier_cooldown'] - 1
        self.step = 0

    def get_initial_state(self):
        return {'noise_cov': self.params['kalman_noise_cov'], 'prev_rp': 0.0, 'outcome_history': []}

    def __call__(self, u_t: str, H_t: List[str], S_prev: Dict):
        self.step += 1
        embed_u_t = self.embedder.encode(u_t)
        embed_H_t = self.embedder.encode(H_t)

        CCI = compute_CCI(embed_u_t, embed_H_t)
        VOL = compute_VOL(embed_u_t, embed_H_t)
        AMB = compute_AMB(u_t, H_t)
        ICI = compute_ICI(embed_u_t, embed_H_t)
        CSS = compute_CSS(u_t, H_t)

        R_now_short, R_now_mid, R_now_long = self.classify_risk(u_t)
        R_ctx_short, R_ctx_mid, R_ctx_long = self.classify_risk(' '.join(H_t))

        TRI_short = self.params['beta'] * R_ctx_short + (1 - self.params['beta']) * R_now_short
        TRI_mid = self.params['beta'] * R_ctx_mid + (1 - self.params['beta']) * R_now_mid
        TRI_long = self.params['beta'] * R_ctx_long + (1 - self.params['beta']) * R_now_long

        TB = self.params['w_short'] * TRI_short + self.params['w_mid'] * TRI_mid + self.params['w_long'] * TRI_long

        RP = TB - (self.params['s1'] * CCI + self.params['s2'] * VOL + self.params['s3'] * AMB + self.params['s4'] * ICI + self.params['s5'] * CSS)

        RP_s = kalman_smooth(RP, S_prev['noise_cov'], S_prev['prev_rp'])

        band = apply_dynamic_hysteresis(RP_s, VOL)

        if band in ['caution', 'steer'] and self.step - self.last_clarifier_turn < self.params['clarifier_cooldown']:
            band = 'safe'

        guidance = produce_guidance(band, TRI_mid)

        self.log_telemetry(u_t, RP_s, band, guidance)

        # Closed-loop (placeholder outcome; replace with real metric)
        outcome = np.random.rand()  # e.g., user engagement score
        self.meta_learner.update(self.params, outcome)
        S_new = {'noise_cov': S_prev['noise_cov'], 'prev_rp': RP_s, 'outcome_history': S_prev['outcome_history'] + [outcome]}

        return band, guidance, S_new

    def classify_risk(self, text: str) -> Tuple[float, float, float]:
        score = self.risk_classifier(text)[0]['score']
        return score * 0.8, score * 0.5, score * 0.3  # Short/mid/long

    def log_telemetry(self, u_t, RP_s, band, guidance):
        print(f"Telemetry: RP_s={RP_s:.4f}, Band={band}, Guidance={guidance}")

def get_initial_state():
    return {'noise_cov': 0.01, 'prev_rp': 0.0, 'outcome_history': []}
