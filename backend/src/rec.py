import pandas as pd
import numpy as np
import yaml
from typing import Optional

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sentence_transformers import SentenceTransformer

from agents.prompt_agent import PromptAgent
from util.util import coerce_audio, normalize_popularity, l2norm


with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


class Prompt2Playlist:
    def __init__(
        self,
        model_name: str = CONFIG["embeddings"]["model"],
        audio_weight: float = CONFIG["weights"]["audio"],
        text_weight: float = CONFIG["weights"]["text"],
        pop_weight: float = CONFIG["weights"]["popularity"],
        min_tracks_per_playlist: int = CONFIG["training"]["min_tracks_per_playlist"],
        ridge_alphas: Optional[list[float]] = CONFIG["training"]["ridge_alphas"],
    ):
        self.st = SentenceTransformer(model_name)
        self.audio_scaler = StandardScaler()
        self.ridge = None
        self.audio_weight = audio_weight
        self.text_weight = text_weight
        self.pop_weight = pop_weight
        self.min_tracks_per_playlist = min_tracks_per_playlist
        self.ridge_alphas = ridge_alphas
        self.df = None
        self.Z_audio = None
        self.song_text_vecs = None
        self.prompt_agent = PromptAgent()
        self.audio_cols = CONFIG["audio_columns"]

    def fit(self):
        spotify_csv = pd.read_csv(CONFIG["data"]["spotify_csv"])
        df = coerce_audio(spotify_csv, CONFIG).dropna(subset=self.audio_cols).copy()
        self.Z_audio = self.audio_scaler.fit_transform(df[self.audio_cols].to_numpy())

        # Song text embeddings
        song_texts = [
            f"passage: {r.get('track_name','')} - {r.get('track_artist','')} | "
            f"{r.get('playlist_genre','')} | {r.get('playlist_subgenre','')}"
            for _, r in df.iterrows()
        ]
        self.song_text_vecs = self.st.encode(song_texts, normalize_embeddings=True)

        # Filter playlists
        counts = df["playlist_id"].value_counts()
        keep_ids = set(counts[counts >= self.min_tracks_per_playlist].index.tolist())
        pl = df[df["playlist_id"].isin(keep_ids)].copy()
        if pl.empty:
            keep_ids = set(counts[counts >= max(2, counts.quantile(0.5))].index.tolist())
            pl = df[df["playlist_id"].isin(keep_ids)].copy()

        # Playlist-level audio target
        y_audio = pl.groupby("playlist_id").apply(
            lambda g: self.audio_scaler.transform(g[self.audio_cols]).mean(axis=0)
        )
        y_audio = np.stack(y_audio.values, axis=0)

        # Playlist-level text embeddings
        def playlist_text(g: pd.DataFrame) -> str:
            name = str(g["playlist_name"].iloc[0]) if "playlist_name" in g.columns else ""
            gen = g["playlist_genre"].mode(dropna=True)
            sub = g["playlist_subgenre"].mode(dropna=True)
            gtxt = gen.iloc[0] if len(gen) else ""
            stxt = sub.iloc[0] if len(sub) else ""
            desc = " | ".join([t for t in [name, gtxt, stxt] if str(t).strip()])
            return f"passage: {desc}"

        pl_texts = pl.groupby("playlist_id").apply(playlist_text).values.tolist()
        X_text = self.st.encode(pl_texts, normalize_embeddings=True)

        # Train ridge regression
        self.ridge = RidgeCV(alphas=self.ridge_alphas, fit_intercept=True)
        self.ridge.fit(X_text, y_audio)
        self.df = df.reset_index(drop=True)

    def recommend(
        self,
        prompt: str,
        k: int = CONFIG["recommend"]["k"],
        per_artist_cap: int = CONFIG["recommend"]["per_artist_cap"],
        min_year: Optional[int] = CONFIG["recommend"]["min_year"],
        min_pop_quantile: float = CONFIG["recommend"]["min_pop_quantile"],
    ) -> pd.DataFrame:
        assert self.ridge is not None, "Call fit() first."

        rewritten = self.prompt_agent.rewrite_prompt(prompt).final_prompt

        # Filters
        df = self.df.copy()
        if min_year is not None and "track_album_release_date" in df.columns:
            years = pd.to_datetime(df["track_album_release_date"], errors="coerce").dt.year
            df = df[years >= min_year].copy()
        if "track_popularity" in df.columns and 0.0 <= min_pop_quantile < 1.0:
            thr = df["track_popularity"].quantile(min_pop_quantile)
            df = df[df["track_popularity"] >= thr].copy()
        if df.empty:
            df = self.df.copy()

        # Build query vector
        idx = df.index.to_numpy()
        Z = self.Z_audio[idx]
        song_vecs = self.song_text_vecs[idx]

        q_vec = self.st.encode([f"query: {rewritten}"], normalize_embeddings=True)
        y_pred = self.ridge.predict(q_vec)

        # Similarities
        audio_sim = (l2norm(Z) @ l2norm(y_pred).T).ravel()
        text_sim = (song_vecs @ q_vec.T).ravel()
        pop = normalize_popularity(df["track_popularity"]) if "track_popularity" in df.columns else np.zeros(len(df))

        score = self.audio_weight * audio_sim + self.text_weight * text_sim + self.pop_weight * pop
        df = df.assign(score=score).sort_values("score", ascending=False)

        # Pick results
        picks, seen = [], {}
        for _, row in df.iterrows():
            a = row.get("track_artist", "")
            if seen.get(a, 0) >= per_artist_cap:
                continue
            picks.append(row)
            seen[a] = seen.get(a, 0) + 1
            if len(picks) >= k:
                break

        cols_out = [
            "track_id", "track_name", "track_artist", "track_popularity",
            "playlist_genre", "playlist_subgenre", "track_album_release_date", "score",
        ]
        return pd.DataFrame(picks)[cols_out].reset_index(drop=True)
