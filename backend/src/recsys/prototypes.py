from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

def _row_l2norm(X: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(X, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return X / denom

def _softmax(x: np.ndarray, temp: float = 1.0, axis: int = -1) -> np.ndarray:
    z = x / max(temp, 1e-6)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-12, None)

def _entropy(p: np.ndarray, axis: int = -1, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum(axis=axis))

# --- core prototype logic ---

def build_label_prototypes(
    df: pd.DataFrame,
    song_text_vecs: np.ndarray,
    label_field: str,
    min_label_count: int,
    top_k_fallback: int = 25,
) -> Tuple[List[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build data-driven label prototypes and per-track logits:
      returns (labels[G], protos[G,D], track_logits[N,G])
      protos/logits can be None if insufficient labels
    """
    if label_field not in df.columns or song_text_vecs is None or len(df) == 0:
        return [], None, None

    labels_arr = df[label_field].astype(str).str.lower().fillna("").values
    vc = pd.Series(labels_arr)[lambda s: s != ""].value_counts()

    keep = set(vc[vc >= min_label_count].index.tolist())
    if len(keep) < 3:
        top_k = min(top_k_fallback, len(vc))
        keep = set(vc.head(top_k).index.tolist())

    keep_labels = sorted(list(keep))
    if not keep_labels:
        return [], None, None

    G = len(keep_labels)
    D = song_text_vecs.shape[1]
    sums = np.zeros((G, D), dtype=np.float32)
    counts = np.zeros((G,), dtype=np.int32)
    label_to_ix = {lab: i for i, lab in enumerate(keep_labels)}

    for i, lab in enumerate(labels_arr):
        gi = label_to_ix.get(lab)
        if gi is not None:
            sums[gi] += song_text_vecs[i]
            counts[gi] += 1

    counts = np.maximum(counts, 1)[:, None]
    protos = _row_l2norm((sums / counts).astype(np.float32))  # [G, D]
    track_logits = (song_text_vecs @ protos.T).astype(np.float32)        # [N, G]
    return keep_labels, protos, track_logits

def genre_alignment(
    q_vec: np.ndarray,
    idx: np.ndarray,
    genre_protos: Optional[np.ndarray],
    track_genre_logits: Optional[np.ndarray],
    proto_temp: float,
    genre_align_weight: float,
) -> Tuple[float, np.ndarray]:
    """
    Confidence-weighted alignment between p(g|prompt) and p(g|track).
    Returns (lambda_eff, alignment[N]).
    """
    if (
        genre_protos is None
        or track_genre_logits is None
        or genre_protos.shape[0] <= 1
    ):
        return 0.0, np.zeros(len(idx), dtype=np.float32)

    prompt_logits = (q_vec @ genre_protos.T).ravel()                  # [G]
    p_g_prompt = _softmax(prompt_logits, temp=proto_temp)             # [G]

    track_logits = track_genre_logits[idx]                            # [N, G]
    p_g_track = _softmax(track_logits, temp=proto_temp, axis=1)       # [N, G]

    align = (p_g_track * p_g_prompt[None, :]).sum(axis=1).astype(np.float32)

    ent = _entropy(p_g_prompt)
    ent_max = np.log(genre_protos.shape[0])
    conf = max(0.0, 1.0 - ent / max(ent_max, 1e-6))  # [0,1]
    lambda_eff = float(genre_align_weight) * conf
    return lambda_eff, align
