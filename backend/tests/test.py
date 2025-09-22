from backend.rec import Prompt2Playlist


def test_rec():
    p2p = Prompt2Playlist()
    p2p.fit()
    test_prompts = [
        "TECHNO",
        "chill acoustic vibes",
        "classic rock 70s"
    ]

    for prompt in test_prompts:
        recs = p2p.recommend(prompt, k=5)
        print(f"\n=== {prompt.upper()} ===")
        print(recs.head(10))


if __name__ == "__main__":
    test_rec()
