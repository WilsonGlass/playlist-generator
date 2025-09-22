import numpy as np
import pandas as pd


def coerce_audio(df: pd.DataFrame, config) -> pd.DataFrame:
    df = df.copy()
    for c in config["audio_columns"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_popularity(s: pd.Series) -> np.ndarray:
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    return s.fillna(0.0).to_numpy()


def l2norm(X: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / denom