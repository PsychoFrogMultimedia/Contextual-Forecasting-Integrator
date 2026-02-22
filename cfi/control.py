import numpy as np

def kalman_smooth(RP: float, noise_cov: float, prev_rp: float) -> float:
    P = noise_cov + 0.01
    K = P / (P + noise_cov)
    return prev_rp + K * (RP - prev_rp)

def apply_dynamic_hysteresis(RP_s: float, VOL: float) -> str:
    safe_th = 0.3 + 0.1 * VOL
    caution_th = 0.6
    steer_th = 0.8
    if RP_s < safe_th:
        return 'safe'
    elif RP_s < caution_th:
        return 'caution'
    elif RP_s < steer_th:
        return 'steer'
    return 'block'

def produce_guidance(band: str, TRI_mid: float) -> Dict:
    if band == 'safe':
        return {}
    elif band == 'caution':
        return {'rephrase_suggestion': 'Soften phrasing', 'confidence_boost': 0.2}
    elif band == 'steer':
        return {'rephrase_suggestion': 'Redirect topic', 'clarification_prompt': 'Can you clarify?'}
    return {'block_reason': 'High risk detected'}
