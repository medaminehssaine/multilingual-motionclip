#!/usr/bin/env python
"""Grille de squelettes (version FR) : lignes = prompts, colonnes = instants."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
GIFDIR = os.path.join(HERE, "..", "..", "assets", "gifs", "vae")

rows = [
    ("00_a_person_walks_forward_60f.gif",            "marche en avant"),
    ("01_a_person_is_running_fast_60f.gif",          "course rapide"),
    ("02_a_person_jumps_up_60f.gif",                 "saut"),
    ("03_a_person_kicks_with_the_right_leg_60f.gif", "coup de pied (droite)"),
    ("04_a_person_sits_down_on_a_chair_60f.gif",     "s'assoit"),
]
times = [0, 15, 30, 45, 59]
CROP = (196, 170, 432, 452)
INK  = "#243B43"

fig, axes = plt.subplots(len(rows), len(times),
                         figsize=(len(times)*1.32, len(rows)*1.44))
for r, (fn, label) in enumerate(rows):
    im = Image.open(os.path.join(GIFDIR, fn))
    for c, t in enumerate(times):
        im.seek(t)
        frame = im.convert("RGB").crop(CROP)
        ax = axes[r][c]
        ax.imshow(frame)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#D5DADD"); s.set_linewidth(0.8)
        if r == 0:
            ax.set_title(f"image {t}", fontsize=10, color=INK, pad=4)
        if c == 0:
            ax.set_ylabel(label, fontsize=10, color=INK,
                          rotation=90, va="center", labelpad=6)
fig.subplots_adjust(wspace=0.06, hspace=0.06,
                    left=0.13, right=0.995, top=0.945, bottom=0.008)
fig.savefig(os.path.join(OUT, "fig_skeleton_grid.pdf"), dpi=220,
            bbox_inches="tight", pad_inches=0.02)
print("wrote fig_skeleton_grid.pdf (FR)")
