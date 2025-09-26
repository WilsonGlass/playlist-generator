from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.recsys.embeddings import TextEncoder

def track_lines(df: pd.DataFrame) -> List[str]:
    """Build per-track text lines for embedding."""
    return [
        f"passage: {r.get('track_name','')} - {r.get('track_artist','')} | "
        f"{r.get('playlist_genre','')} | {r.get('playlist_subgenre','')}"
        for _, r in df.iterrows()
    ]

def build_playlist_supervision(
    df: pd.DataFrame,
    encoder: TextEncoder,
    audio_scaler: StandardScaler,
    audio_cols: List[str],
    min_tracks_per_playlist: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct playlist-level (X_text, y_audio):
      X_text: embedding of (playlist name/genre/subgenre)
      y_audio: mean scaled audio vector per playlist
    """
    counts = df["playlist_id"].value_counts()
    keep_ids = set(counts[counts >= min_tracks_per_playlist].index.tolist())
    pl = df[df["playlist_id"].isin(keep_ids)].copy()
    if pl.empty:
        # fallback: use median-or-2 threshold to guarantee some targets
        keep_ids = set(counts[counts >= max(2, int(counts.quantile(0.5)))].index.tolist())
        pl = df[df["playlist_id"].isin(keep_ids)].copy()

    def playlist_text(g: pd.DataFrame) -> str:
        name = str(g["playlist_name"].iloc[0]) if "playlist_name" in g.columns else ""
        gen = g["playlist_genre"].mode(dropna=True)
        sub = g["playlist_subgenre"].mode(dropna=True)
        gtxt = gen.iloc[0] if len(gen) else ""
        stxt = sub.iloc[0] if len(sub) else ""
        desc = " | ".join([t for t in [name, gtxt, stxt] if str(t).strip()])
        return f"passage: {desc}"

    y_audio = pl.groupby("playlist_id").apply(
        lambda g: audio_scaler.transform(g[audio_cols]).mean(axis=0)
    )
    y_audio = np.stack(y_audio.values, axis=0)  # [P, A]

    pl_texts = pl.groupby("playlist_id").apply(playlist_text).values.tolist()
    X_text = encoder.encode_passages(pl_texts)  # [P, D]
    return X_text, y_audio
