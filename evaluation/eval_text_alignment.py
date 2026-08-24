import json
import os
import numpy as np
import torch
from itertools import combinations

from src.utils.mclip_wrapper import MClipTextEncoder

DEVICE = torch.device("cuda:0")
LANGS = ["en", "fr", "es", "pt", "ru", "zh", "ar"]
N = 100
SEED = 42

np.random.seed(SEED)

caps = {}
for lang in LANGS:
    with open(f"captions_{lang}.json", encoding="utf-8") as f:
        caps[lang] = json.load(f)

common = set(caps["en"])
for lang in LANGS[1:]:
    common &= set(caps[lang])

motion_dir = "./data/HumanML3D/new_joint_vecs"
ids = sorted(
    cid for cid in common
    if os.path.exists(os.path.join(motion_dir, cid + ".npy"))
)

rng = np.random.default_rng(SEED)
ids = list(rng.choice(ids, size=min(N, len(ids)), replace=False))

print(f"Using {len(ids)} multilingual motion IDs")

print("Loading M-CLIP...")
model = MClipTextEncoder(device=DEVICE)
model.eval()

embeddings = {}

with torch.no_grad():
    for lang in LANGS:
        texts = []

        for cid in ids:
            cap = caps[lang][cid]
            if isinstance(cap, list):
                cap = cap[0]
            texts.append(str(cap))

        chunks = []

        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            z = model.encode_text_raw(batch).float()

            z = z / z.norm(dim=-1, keepdim=True)
            chunks.append(z.cpu().numpy())

        embeddings[lang] = np.concatenate(chunks, axis=0)

        print(f"{lang}: {embeddings[lang].shape}")

print("\n" + "=" * 70)
print("PAIRWISE CROSS-LINGUAL TEXT ALIGNMENT")
print("Mean cosine similarity over 100 shared motion IDs")
print("=" * 70)

pairs = []

for a, b in combinations(LANGS, 2):
    sim = np.sum(
        embeddings[a] * embeddings[b],
        axis=1,
    )

    pairs.append((a, b, float(sim.mean()), float(sim.std())))

    print(
        f"{a:>2} - {b:<2}: "
        f"{sim.mean():.6f} ± {sim.std():.6f}"
    )

print("\n" + "=" * 70)
print("ENGLISH-ANCHORED ALIGNMENT")
print("=" * 70)

for lang in LANGS[1:]:
    sim = np.sum(
        embeddings["en"] * embeddings[lang],
        axis=1,
    )

    print(
        f"EN - {lang:<2}: "
        f"{sim.mean():.6f} ± {sim.std():.6f}"
    )

print("\n" + "=" * 70)
print("AVERAGE ALIGNMENT TO ENGLISH")
print("=" * 70)

en_scores = []

for lang in LANGS[1:]:
    sim = np.sum(
        embeddings["en"] * embeddings[lang],
        axis=1,
    )
    en_scores.append(sim.mean())

print(f"mean EN↔non-EN cosine: {np.mean(en_scores):.6f}")
