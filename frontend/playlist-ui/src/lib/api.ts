export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export type RecommendRequest = {
  prompt: string;
  k?: number;
  per_artist_cap?: number;
  min_year?: number | null;
  min_pop_quantile?: number; // [0,1)
  config?: {
    filters?: {
      min_year?: number | null;
      min_pop_quantile?: number;
    };
    scoring?: {
      audio_weight?: number;
      text_weight?: number;
      pop_weight?: number;
    };
    // future fields accepted by backend:
    audio_feature_weights?: Record<string, number>;
    exclude_audio_features?: string[];
  };
};

export type RecommendationRow = {
  track_id: string;
  track_name: string;
  track_artist: string;
  track_popularity?: number;
  playlist_genre?: string;
  playlist_subgenre?: string;
  track_album_release_date?: string;
  score: number;
};

export async function fitModel(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/fit`, { method: "POST" });
  if (!res.ok) throw new Error(`Fit failed: ${res.status}`);
  return res.json();
}

export async function recommend(req: RecommendRequest): Promise<RecommendationRow[]> {
  const res = await fetch(`${API_BASE}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.message ?? `Recommend failed: ${res.status}`);
  }
  return res.json();
}

export async function health(): Promise<{ status: string; model_fitted: boolean }> {
  const res = await fetch(`${API_BASE}/healthz`);
  return res.json();
}
