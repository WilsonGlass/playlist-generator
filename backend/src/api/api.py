from __future__ import annotations

from pathlib import Path

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import yaml

from src.recsys.prompt2playlist import Prompt2Playlist


app = Flask(__name__)
CORS(app)

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

p2p: Prompt2Playlist = Prompt2Playlist(
    model_name=cfg["embeddings"]["model"],
    audio_weight=cfg["weights"]["audio"],
    text_weight=cfg["weights"]["text"],
    pop_weight=cfg["weights"]["popularity"],
    min_tracks_per_playlist=cfg["training"]["min_tracks_per_playlist"],
    ridge_alphas=cfg["training"]["ridge_alphas"],
)


@app.route("/fit", methods=["POST"])
def fit():
    """Train from CONFIG['data']['spotify_csv']."""
    try:
        p2p.fit()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    JSON body:
    {
      "prompt": "string",                // required
      "k": 50,                           // optional
      "per_artist_cap": 2,               // optional
      "min_year": 2016,                  // optional
      "min_pop_quantile": 0.5            // optional in [0,1)
    }
    """
    try:
        payload = request.get_json(force=True) or {}
        prompt = payload["prompt"]  # required
        k = int(payload.get("k", cfg["recommend"]["k"]))
        per_artist_cap = int(payload.get("per_artist_cap", cfg["recommend"]["per_artist_cap"]))
        min_year = payload.get("min_year", cfg["recommend"]["min_year"])
        min_pop_quantile = float(payload.get("min_pop_quantile", cfg["recommend"]["min_pop_quantile"]))

        df = p2p.recommend(
            prompt=prompt,
            k=k,
            per_artist_cap=per_artist_cap,
            min_year=min_year,
            min_pop_quantile=min_pop_quantile,
        )
        return Response(df.to_json(orient="records"),
                        mimetype="application/json", status=200)

    except AssertionError as ae:  # e.g., "Call fit() first."
        return jsonify({"status": "error", "message": str(ae)}), 400
    except KeyError as ke:
        return jsonify({"status": "error", "message": f"Missing field: {ke}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
