from typing import Tuple
import numpy as np
from src.util.util import l2norm

def candidate_similarities(
    Z_audio_scaled: np.ndarray,        # [N, A]
    song_text_vecs: np.ndarray,        # [N, D]
    q_vec: np.ndarray,                 # [1, D]
    y_pred_audio: np.ndarray,          # [1, A]
) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine similarities for audio and text."""
    audio_sim = (l2norm(Z_audio_scaled) @ l2norm(y_pred_audio).T).ravel()
    text_sim = (song_text_vecs @ q_vec.T).ravel()
    return audio_sim, text_sim

def blend_score(
    audio_sim: np.ndarray,
    text_sim: np.ndarray,
    pop: np.ndarray,
    audio_weight: float,
    text_weight: float,
    pop_weight: float,
    lambda_eff: float,
    genre_align: np.ndarray,
) -> np.ndarray:
    """Linear blend with learned signals + confidence scaling."""
    return (
        audio_weight * audio_sim
        + text_weight * text_sim
        + pop_weight * pop
        + lambda_eff * genre_align
    ).astype(np.float32)
