from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from src.agents.prompt_agent import PromptAgent
from src.util.util import coerce_audio, normalize_popularity

from src.recsys.config import load_config
from src.recsys.embeddings import TextEncoder
from src.recsys.training import track_lines, build_playlist_supervision
from src.recsys.prototypes import build_label_prototypes, genre_alignment
from src.recsys.scoring import candidate_similarities, blend_score


CONFIG = load_config()


class Prompt2Playlist:
    """
    Clean, modular recommender orchestrating:
      - Text encoder
      - Text->audio ridge head
      - Data-driven prototype reranker
    """

    def __init__(
        self,
        model_name: str = CONFIG["embeddings"]["model"],
        audio_weight: float = CONFIG["weights"]["audio"],
        text_weight: float = CONFIG["weights"]["text"],
        pop_weight: float = CONFIG["weights"]["popularity"],
        min_tracks_per_playlist: int = CONFIG["training"]["min_tracks_per_playlist"],
        ridge_alphas: Optional[List[float]] = CONFIG["training"]["ridge_alphas"],
    ):
        # Models
        self.encoder = TextEncoder(model_name)
        self.audio_scaler = StandardScaler()
        self.ridge: Optional[RidgeCV] = None

        # Weights
        self.audio_weight = float(audio_weight)
        self.text_weight = float(text_weight)
        self.pop_weight = float(pop_weight)

        # Training knobs
        self.min_tracks_per_playlist = int(min_tracks_per_playlist)
        self.ridge_alphas = ridge_alphas

        # Prototype reranker knobs
        tr_cfg = CONFIG.get("training", {})
        self.label_field: str = tr_cfg.get("label_field", "playlist_subgenre")
        self.min_label_count: int = int(tr_cfg.get("min_label_count", 100))
        self.proto_temp: float = float(tr_cfg.get("prototype_temperature", 0.7))
        self.genre_align_weight: float = float(tr_cfg.get("genre_align_weight", 0.25))

        # Data
        self.df: Optional[pd.DataFrame] = None                # base corpus
        self.Z_audio: Optional[np.ndarray] = None             # [N, A] scaled audio features
        self.song_text_vecs: Optional[np.ndarray] = None      # [N, D] normalized embeddings

        # Prototype reranker artifacts
        self.genre_labels: List[str] = []                     # length G
        self.genre_protos: Optional[np.ndarray] = None        # [G, D]
        self.track_genre_logits: Optional[np.ndarray] = None  # [N, G]

        # Utils
        self.prompt_agent = PromptAgent()
        self.audio_cols: List[str] = CONFIG["audio_columns"]

    # ---------------- Public API ----------------

    def fit(self) -> None:
        """Train models and build all derived artifacts."""
        df = self._load_and_prepare_dataframe()
        self._fit_audio_scaler(df)
        self._embed_tracks(df)
        X_text, y_audio = build_playlist_supervision(
            df=df,
            encoder=self.encoder,
            audio_scaler=self.audio_scaler,
            audio_cols=self.audio_cols,
            min_tracks_per_playlist=self.min_tracks_per_playlist,
        )
        self._fit_ridge(X_text, y_audio)
        self.df = df.reset_index(drop=True)
        self._build_label_prototypes()

    def recommend(
        self,
        prompt: str,
        k: int = CONFIG["recommend"]["k"],
        per_artist_cap: int = CONFIG["recommend"]["per_artist_cap"],
        min_year: Optional[int] = CONFIG["recommend"]["min_year"],
        min_pop_quantile: float = CONFIG["recommend"]["min_pop_quantile"],
    ) -> pd.DataFrame:
        """Recommend top-k tracks for a prompt."""
        self._assert_ready()

        rewritten = self._rewrite_prompt(prompt)
        cand_df = self._filter_candidates(self.df, min_year, min_pop_quantile)
        idx, Z, song_vecs, pop = self._candidate_arrays(cand_df)

        q_vec = self.encoder.encode_query(f"query: {rewritten}")
        y_pred = self.ridge.predict(q_vec)  # [1, A]

        audio_sim, text_sim = candidate_similarities(Z, song_vecs, q_vec, y_pred)
        lambda_eff, g_align = genre_alignment(
            q_vec=q_vec,
            idx=idx,
            genre_protos=self.genre_protos,
            track_genre_logits=self.track_genre_logits,
            proto_temp=self.proto_temp,
            genre_align_weight=self.genre_align_weight,
        )
        score = blend_score(
            audio_sim=audio_sim,
            text_sim=text_sim,
            pop=pop,
            audio_weight=self.audio_weight,
            text_weight=self.text_weight,
            pop_weight=self.pop_weight,
            lambda_eff=lambda_eff,
            genre_align=g_align,
        )

        ranked = (
            cand_df.assign(score=score)
            .sort_values("score", ascending=False)
            .drop_duplicates(subset=["track_id"], keep="first")
        )
        picks = self._select_top(ranked, k=k, per_artist_cap=per_artist_cap)
        return picks[
            [
                "track_id",
                "track_name",
                "track_artist",
                "track_popularity",
                "playlist_genre",
                "playlist_subgenre",
                "track_album_release_date",
                "score",
            ]
        ].reset_index(drop=True)

    # -------------- Private helpers --------------

    def _load_and_prepare_dataframe(self) -> pd.DataFrame:
        spotify_csv = pd.read_csv(CONFIG["data"]["spotify_csv"])
        df = coerce_audio(spotify_csv, CONFIG).dropna(subset=self.audio_cols).copy()
        return df

    def _fit_audio_scaler(self, df: pd.DataFrame) -> None:
        self.Z_audio = self.audio_scaler.fit_transform(df[self.audio_cols].to_numpy())

    def _embed_tracks(self, df: pd.DataFrame) -> None:
        lines = track_lines(df)
        self.song_text_vecs = self.encoder.encode_passages(lines)

    def _fit_ridge(self, X_text: np.ndarray, y_audio: np.ndarray) -> None:
        self.ridge = RidgeCV(alphas=self.ridge_alphas, fit_intercept=True)
        self.ridge.fit(X_text, y_audio)

    def _build_label_prototypes(self) -> None:
        labels, protos, logits = build_label_prototypes(
            df=self.df,
            song_text_vecs=self.song_text_vecs,
            label_field=self.label_field,
            min_label_count=self.min_label_count,
        )
        self.genre_labels = labels
        self.genre_protos = protos
        self.track_genre_logits = logits

    def _assert_ready(self) -> None:
        assert self.ridge is not None, "Call fit() first."
        assert self.df is not None, "Call fit() first."
        assert self.Z_audio is not None and self.song_text_vecs is not None, "Call fit() first."

    def _rewrite_prompt(self, prompt: str) -> str:
        rewritten = self.prompt_agent.rewrite_prompt(prompt).final_prompt
        print("REWRITTEN:", rewritten)
        return rewritten

    def _filter_candidates(
        self, df: pd.DataFrame, min_year: Optional[int], min_pop_q: float
    ) -> pd.DataFrame:
        out = df.copy()
        if min_year is not None and "track_album_release_date" in out.columns:
            years = pd.to_datetime(out["track_album_release_date"], errors="coerce").dt.year
            out = out[years >= int(min_year)].copy()
        if "track_popularity" in out.columns and 0.0 <= float(min_pop_q) < 1.0:
            thr = out["track_popularity"].quantile(float(min_pop_q))
            out = out[out["track_popularity"] >= thr].copy()
        return out if not out.empty else df.copy()

    def _candidate_arrays(
        self, cand_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = cand_df.index.to_numpy()
        Z = self.Z_audio[idx]
        vecs = self.song_text_vecs[idx]
        pop = (
            normalize_popularity(cand_df["track_popularity"]).astype(np.float32)
            if "track_popularity" in cand_df.columns
            else np.zeros(len(cand_df), dtype=np.float32)
        )
        return idx, Z, vecs, pop

    def _select_top(self, ranked: pd.DataFrame, k: int, per_artist_cap: int) -> pd.DataFrame:
        picks, seen = [], {}
        for _, row in ranked.iterrows():
            a = row.get("track_artist", "")
            if seen.get(a, 0) >= int(per_artist_cap):
                continue
            picks.append(row)
            seen[a] = seen.get(a, 0) + 1
            if len(picks) >= int(k):
                break
        return pd.DataFrame(picks) if picks else ranked.head(int(k))
