import numpy as np
from sentence_transformers.util import cosine_similarity

def compute_CCI(embed_u_t: np.ndarray, embed_H_t: np.ndarray) -> float:
    if len(embed_H_t) == 0:
        return 1.0
    return cosine_similarity(embed_u_t.reshape(1, -1), np.mean(embed_H_t, axis=0).reshape(1, -1))[0][0]

def compute_VOL(embed_u_t: np.ndarray, embed_H_t: np.ndarray) -> float:
    if len(embed_H_t) < 2:
        return 0.0
    std = np.std(embed_H_t, axis=0).mean()
    rate_change = np.linalg.norm(embed_u_t - embed_H_t[-1]) / np.linalg.norm(embed_H_t[-1] - embed_H_t[-2]) if len(embed_H_t) > 1 else 0.0
    return (std + rate_change) / 2

def compute_AMB(u_t: str, H_t: List[str]) -> float:
    # Toy; replace with intent entropy
    return np.random.rand() * 0.3

def compute_ICI(embed_u_t: np.ndarray, embed_H_t: np.ndarray) -> float:
    if len(embed_H_t) == 0:
        return 1.0
    initial_embed = embed_H_t[0]
    return cosine_similarity(embed_u_t.reshape(1, -1), initial_embed.reshape(1, -1))[0][0] * np.exp(-len(H_t) / 10)

def compute_CSS(u_t: str, H_t: List[str]) -> float:
    # Toy; replace with fairness model
    return np.random.rand() * 0.2 + 0.8
