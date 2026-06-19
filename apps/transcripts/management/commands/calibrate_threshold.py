# calibrate_threshold.py
import os
from openai import OpenAI
import numpy as np

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # set this in your shell, never paste the key itself

def embed(texts):
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in response.data]

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Pulled from the two HTML dumps you shared — still-colliding pairs after the prompt fix
pairs = [
    {
        "label": "chunk #1 (card-828) — career transition",
        "proposition": "The speaker has worked at SpaceX early in their career, contributed to Founders Fund's success, and is now starting their own business.",
        "claim": "The speaker has transitioned from SpaceX to Founders Fund to starting their own business.",
    },
    {
        "label": "chunk #36 (card-863) — military to commercial",
        "proposition": "Much of the technology developed was originally for military purposes and later became commercial.",
        "claim": "Technology developed for military purposes became commercial.",
    },
    {
        "label": "chunk #74 (card-901) — small emergent market",
        "proposition": "The small market is the emergent market that we can go after that we don't think any incumbent will go after on the same timeframe as us.",
        "claim": "The company targets a small emergent market.",
    },
    # one pair that DID differentiate after the fix, as a control
    {
        "label": "chunk #6 (card-833) — control, post-fix improvement",
        "proposition": "The most successful companies were founder-run all the way to the end.",
        "claim": "The speaker had just come from SpaceX in 2011.",
    },
]

print(f"{'Pair':<55} {'Cosine Sim':>10}")
print("-" * 67)
for pair in pairs:
    emb_prop, emb_claim = embed([pair["proposition"], pair["claim"]])
    sim = cosine_sim(emb_prop, emb_claim)
    print(f"{pair['label']:<55} {sim:>10.4f}")