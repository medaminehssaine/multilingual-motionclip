"""
plot_training_curves.py
=======================

Ismail asked whether the KL training converged. A number in a Teams message is
weak evidence; a curve is strong evidence. This parses the `training.log` that
every run writes and produces one PNG per run plus a combined comparison.

Usage
-----
  python plot_training_curves.py \
      exps/vae-en-mu-kl1e-3 \
      exps/vae-en-mu-kl0 \
      exps/vae-en-mu-kl1e-3-anneal5 \
      exps/vae-en-mu-kl1e-3-anneal30 \
      exps/vae-multilingual-mclip

Writes: curves_<runname>.png and curves_comparison.png
"""

import ast
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_RE = re.compile(r"Epoch\s+(\d+),\s*train losses:\s*(\{.*\})")
KEYS = ["rc", "vel", "kl", "text_cosine"]


def parse(logpath):
    epochs, series = [], {k: [] for k in KEYS}
    with open(logpath, errors="replace") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            try:
                d = ast.literal_eval(m.group(2))
            except Exception:
                continue
            epochs.append(ep)
            for k in KEYS:
                series[k].append(d.get(k, float("nan")))
    return epochs, series


def plot_one(name, epochs, series, outdir="."):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, k in zip(axes.ravel(), KEYS):
        ax.plot(epochs, series[k], lw=1.8)
        ax.set_title(k)
        ax.set_xlabel("epoch")
        ax.grid(alpha=.3)
    fig.suptitle(f"{name}  --  {len(epochs)} epochs", fontsize=13)
    fig.tight_layout()
    path = os.path.join(outdir, f"curves_{name}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")
    if epochs:
        last = {k: series[k][-1] for k in KEYS}
        print(f"  final epoch {epochs[-1]}: " +
              "  ".join(f"{k}={v:.4f}" for k, v in last.items()))


def main(folders):
    runs = []
    for folder in folders:
        log = os.path.join(folder, "training.log")
        if not os.path.exists(log):
            print(f"skip {folder}: no training.log")
            continue
        name = os.path.basename(folder.rstrip("/"))
        epochs, series = parse(log)
        if not epochs:
            print(f"skip {folder}: no parseable epochs")
            continue
        plot_one(name, epochs, series)
        runs.append((name, epochs, series))

    if len(runs) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, k in zip(axes.ravel(), KEYS):
        for name, epochs, series in runs:
            ax.plot(epochs, series[k], lw=1.6, label=name)
        ax.set_title(k)
        ax.set_xlabel("epoch")
        ax.grid(alpha=.3)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("VAE variants -- training comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig("curves_comparison.png", dpi=130)
    print("wrote curves_comparison.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
