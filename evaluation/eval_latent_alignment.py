"""
eval_latent_alignment.py
========================

THE experiment Ismail asked for on 14 August:

    "the question is whether the projections learned to bring the
     languages together in the shared space or not. So you can test that"

What this measures
------------------
For each motion clip we have the SAME semantic content described in 7
languages. We push each language's caption through the text encoder, and we
push the MOTION through the trained VAE encoder to get its latent mean mu.

Then we measure, per language:

    cos( mu_motion , e_text_lang )

If the model is genuinely multilingual, this number should be roughly EQUAL
across all 7 languages. If English is much higher than Arabic/Chinese, the
model is English-centric and the other languages are along for the ride.

This is DIFFERENT from eval_text_alignment.py, which only compared text
embeddings to each other. That measured M-CLIP (a pretrained frozen model).
THIS measures YOUR trained checkpoint. That distinction is the whole point.

Outputs
-------
  latent_alignment_<tag>.json   raw numbers
  a printed table ready to paste into Teams

Usage
-----
  python eval_latent_alignment.py \
      --ckpt exps/vae-multilingual-mclip/checkpoint_0020.pth.tar \
      --tag multilingual

  # control: the English-only model, to prove the multilingual one is better
  python eval_latent_alignment.py \
      --ckpt exps/vae-en-mu-kl1e-3/checkpoint_0020.pth.tar \
      --tag en-only
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# ----------------------------------------------------------------- config

LANGS = ["en", "fr", "es", "pt", "ru", "zh", "ar"]
SEED = 42            # same seed as eval_text_alignment.py -> same clips
N_CLIPS = 300        # 100 was fine for text-only; 300 gives tighter error bars
NUM_FRAMES = 60      # must match training (--num_frames 60)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True,
                   help="short name used in the output filename")
    p.add_argument("--datapath", default="./data/HumanML3D")
    p.add_argument("--caption_dir", default=".")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--text_encoder", default="mclip",
                   choices=["mclip", "perlang", "clip"],
                   help="MUST match how the checkpoint was trained, or you "
                        "compare motion latents against the wrong text space")
    p.add_argument("--n", type=int, default=N_CLIPS)
    p.add_argument("--frames", type=int, default=NUM_FRAMES)
    p.add_argument("--batch", type=int, default=16)
    return p.parse_args()


# ----------------------------------------------------------------- model

def load_model(ckpt, device, datapath, text_encoder="mclip"):
    """Rebuild the model exactly the way eval_guo.py does, then load weights.

    We hijack src.parser.visualize, which reads the opt.yaml sitting next to
    the checkpoint. That is why we do not have to re-specify latent_dim,
    num_layers, etc. -- they come from the file the training run wrote.
    """
    from src.parser.visualize import parser as vparser
    from src.utils.get_model_and_data import get_model_and_data
    from src.utils.misc import load_model_wo_clip

    saved_argv = sys.argv
    sys.argv = ["x", ckpt, "--input_file", "assets/paper_texts.txt"]
    try:
        params, folder, cname, epoch = vparser()
    finally:
        sys.argv = saved_argv

    params.update({
        "device": device,
        "dataset": "humanml3d",     # we load motions ourselves; avoids the
                                    # multilingual dataset's split filtering
        "datapath": datapath,
        "pose_rep": "xyz",
        "text_encoder": text_encoder,
    })

    model, _ = get_model_and_data(params, split="vald")
    load_model_wo_clip(model, torch.load(ckpt, map_location=device))
    model.eval()
    model.to(device)
    return model


# ----------------------------------------------------------------- data

def load_captions(caption_dir):
    caps = {}
    for lg in LANGS:
        path = os.path.join(caption_dir, f"captions_{lg}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing {path} -- run translate_nllb.py --lang {lg}")
        with open(path, encoding="utf-8") as f:
            caps[lg] = json.load(f)
    return caps


def pick_ids(caps, datapath, n):
    """Clips that (a) have a caption in ALL 7 languages and (b) have a motion
    file long enough to take `NUM_FRAMES` frames from."""
    motion_dir = os.path.join(datapath, "new_joint_vecs")

    common = set(caps["en"])
    for lg in LANGS[1:]:
        common &= set(caps[lg])

    ids = sorted(cid for cid in common
                 if os.path.exists(os.path.join(motion_dir, cid + ".npy")))

    rng = np.random.default_rng(SEED)
    ids = list(rng.permutation(ids))
    return ids, motion_dir


def first_caption(c):
    """captions_*.json stores either a string or a list of strings."""
    if isinstance(c, list):
        c = c[0]
    return str(c)


# ----------------------------------------------------------------- main

def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"checkpoint : {args.ckpt}")
    print(f"languages  : {', '.join(LANGS)}")

    caps = load_captions(args.caption_dir)
    ids, motion_dir = pick_ids(caps, args.datapath, args.n)

    mean = np.load(os.path.join(args.datapath, "Mean.npy"))
    std = np.load(os.path.join(args.datapath, "Std.npy"))
    std[std < 1e-6] = 1.0

    print("building model...")
    model = load_model(args.ckpt, device, args.datapath, args.text_encoder)
    print(f"text encoder requested: {args.text_encoder}")

    # ---- collect motion latents mu, and the ids that actually made it ----
    mus, used_ids = [], []
    buf_x, buf_ids = [], []

    def flush():
        if not buf_x:
            return
        x = torch.stack(buf_x).to(device)                  # (B, 263, 1, T)
        lengths = torch.full((x.shape[0],), args.frames,
                             dtype=torch.long, device=device)
        mask = model.lengths_to_mask(lengths)
        y = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.encoder({"x": x, "y": y, "mask": mask,
                                 "lengths": lengths})
        mus.append(out["mu"].float().cpu())
        used_ids.extend(buf_ids)
        buf_x.clear()
        buf_ids.clear()

    print("encoding motions...")
    for cid in ids:
        if len(used_ids) + len(buf_ids) >= args.n:
            break
        motion = np.load(os.path.join(motion_dir, cid + ".npy"))
        if len(motion) < args.frames:
            continue
        motion = motion[:args.frames]
        motion = (motion - mean) / std
        t = torch.from_numpy(motion.astype(np.float32))
        t = t.permute(1, 0)[:, None, :].contiguous()       # (263, 1, T)
        buf_x.append(t)
        buf_ids.append(cid)
        if len(buf_x) >= args.batch:
            flush()
    flush()

    mu = torch.cat(mus, dim=0)
    mu = mu / mu.norm(dim=-1, keepdim=True)
    print(f"  motion latents: {tuple(mu.shape)}  over {len(used_ids)} clips")

    # ---- text embeddings per language, SAME clip order ----
    print("encoding captions...")
    text = {}
    for lg in LANGS:
        texts = [first_caption(caps[lg][cid]) for cid in used_ids]
        chunks = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                e = model.clip_model.encode_text_raw(texts[i:i + 32]).float()
                e = e / e.norm(dim=-1, keepdim=True)
                chunks.append(e.cpu())
        text[lg] = torch.cat(chunks, dim=0)
        print(f"  {lg}: {tuple(text[lg].shape)}")

    # ---- the number that matters ----
    results = {}
    for lg in LANGS:
        cos = (mu * text[lg]).sum(dim=-1).numpy()
        results[lg] = {
            "mean": float(cos.mean()),
            "std": float(cos.std()),
            "n": int(len(cos)),
        }

    # ---- retrieval within each language: does the right caption win? ----
    #      (a stricter test than mean cosine -- a model can have high cosine
    #       everywhere and still not distinguish one motion from another)
    retrieval = {}
    offdiag = {}
    for lg in LANGS:
        sim = mu @ text[lg].T                       # (N, N)
        n = sim.shape[0]
        eye = torch.eye(n, dtype=torch.bool)
        matched = sim[eye].mean().item()
        mismatched = sim[~eye].mean().item()
        offdiag[lg] = {"matched": matched,
                       "mismatched": mismatched,
                       "gap": matched - mismatched}
        rank = sim.argsort(dim=-1, descending=True)
        gold = torch.arange(len(used_ids)).unsqueeze(1)
        pos = (rank == gold).float().argmax(dim=-1)
        retrieval[lg] = {
            "R@1": float((pos < 1).float().mean()),
            "R@3": float((pos < 3).float().mean()),
            "R@10": float((pos < 10).float().mean()),
        }

    # ---- print ----
    print("\n" + "=" * 68)
    print(f"MOTION-TEXT LATENT ALIGNMENT  --  {args.tag}")
    print(f"{len(used_ids)} clips, identical motions, captions in 7 languages")
    print("=" * 68)
    print(f'{"lang":<8}{"cos(mu, text)":>16}{"std":>10}'
          f'{"R@1":>9}{"R@3":>9}{"R@10":>9}')
    print("-" * 68)
    for lg in LANGS:
        r = results[lg]
        q = retrieval[lg]
        print(f'{lg:<8}{r["mean"]:>16.4f}{r["std"]:>10.4f}'
              f'{q["R@1"]:>9.3f}{q["R@3"]:>9.3f}{q["R@10"]:>9.3f}')
    print("-" * 68)

    print()
    print(f'{"lang":<8}{"matched":>12}{"mismatched":>13}{"GAP":>10}')
    print("-" * 68)
    for lg in LANGS:
        o = offdiag[lg]
        print(f'{lg:<8}{o["matched"]:>12.4f}{o["mismatched"]:>13.4f}'
              f'{o["gap"]:>10.4f}')
    print("-" * 68)
    print("GAP is the real signal. Near zero = collapsed space, the mean")
    print("cosine is just a global offset and means nothing.")
    print()

    vals = np.array([results[lg]["mean"] for lg in LANGS])
    en = results["en"]["mean"]
    non_en = np.array([results[lg]["mean"] for lg in LANGS if lg != "en"])
    print(f'english                 : {en:.4f}')
    print(f'mean non-english        : {non_en.mean():.4f}')
    print(f'gap (en - non-en)       : {en - non_en.mean():+.4f}')
    print(f'spread (max - min)      : {vals.max() - vals.min():.4f}')
    print("=" * 68)
    print("Small gap + small spread = the model is genuinely multilingual.")
    print("Large gap = English-centric; other languages ride on M-CLIP alone.")

    out = f"latent_alignment_{args.tag}.json"
    with open(out, "w") as f:
        json.dump({
            "checkpoint": args.ckpt,
            "tag": args.tag,
            "n_clips": len(used_ids),
            "frames": args.frames,
            "seed": SEED,
            "cosine": results,
            "retrieval": retrieval,
            "offdiag": offdiag,
            "en_minus_non_en": float(en - non_en.mean()),
            "spread": float(vals.max() - vals.min()),
        }, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
