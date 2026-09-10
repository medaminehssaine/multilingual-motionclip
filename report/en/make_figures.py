#!/usr/bin/env python
"""Generate the data-driven figures for the internship report.

Unified visual system: serif type, a teal/warm categorical palette, light
dashed grids behind the data, only the left and bottom spines, and value
labels where they aid reading. Colours match the report accent (ensamblue
#1B7B94 / ensamorange #EF8B2C).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "results", "alignment")
OUT  = os.path.join(HERE, "figures")

INK   = "#243B43"     # dark slate for text / spines
GRID  = "#C7CDD1"
TEAL  = "#1B7B94"
ORANGE= "#EF8B2C"

rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "text.color": INK,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.alpha": 0.6,
    "grid.linewidth": 0.6,
    "grid.linestyle": (0, (3, 3)),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

LANGS = ["en", "fr", "es", "pt", "ru", "zh", "ar"]
# harmonious, distinguishable, colour-blind-aware
PALETTE = {
    "en": "#1B7B94",  # teal  (accent)
    "fr": "#E8833A",  # orange
    "es": "#4E9F3D",  # green
    "pt": "#C0433E",  # red
    "ru": "#7A5EA6",  # violet
    "zh": "#C79A1E",  # gold
    "ar": "#5B7A8C",  # slate blue
}
LABEL = {"en":"English","fr":"French","es":"Spanish","pt":"Portuguese",
         "ru":"Russian","zh":"Chinese","ar":"Arabic"}

def load(tag):
    with open(os.path.join(DATA, f"latent_alignment_{tag}.json")) as f:
        return json.load(f)

def style(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)

# ---------------------------------------------------------------------------
# Figure 1 : alignment gap vs epoch (100-epoch run), one line per language
# ---------------------------------------------------------------------------
epochs = [20, 40, 60, 80, 100]
d1 = {e: load(f"e{e:04d}") for e in epochs}
fig, ax = plt.subplots(figsize=(7.2, 4.5))
order = ["es","en","fr","pt","ar","ru","zh"]     # legend in ranked order
for lg in order:
    ys = [d1[e]["offdiag"][lg]["gap"] for e in epochs]
    ax.plot(epochs, ys, marker="o", ms=5, lw=2.2,
            color=PALETTE[lg], label=LABEL[lg],
            markerfacecolor="white", markeredgewidth=1.6,
            markeredgecolor=PALETTE[lg], zorder=3)
# direct end-labels for the extremes only, to reinforce the ordering
for lg, dy in [("en", 4), ("zh", -2)]:
    y = d1[100]["offdiag"][lg]["gap"]
    ax.annotate(LABEL[lg], xy=(100, y), xytext=(6, dy), textcoords="offset points",
                color=PALETTE[lg], fontsize=10.5, fontweight="bold", va="center")
style(ax)
ax.set_xlabel("Training epoch")
ax.set_ylabel("Alignment gap  (matched $-$ mismatched cosine)")
ax.set_xticks(epochs)
ax.set_xlim(15, 112)
ax.legend(ncol=2, fontsize=10, frameon=False, loc="upper left",
          handlelength=1.6, columnspacing=1.3, labelspacing=0.35)
fig.savefig(os.path.join(OUT, "fig_gap_vs_epoch.pdf"))
plt.close(fig); print("fig_gap_vs_epoch.pdf")

# ---------------------------------------------------------------------------
# Figure 2 : gap over the long 300-epoch run
# ---------------------------------------------------------------------------
epochs3 = [50, 100, 150, 200, 250, 300]
d3 = {e: load(f"e300_{e:04d}") for e in epochs3}
fig, ax = plt.subplots(figsize=(7.2, 4.5))
for lg in order:
    ys = [d3[e]["offdiag"][lg]["gap"] for e in epochs3]
    ax.plot(epochs3, ys, marker="o", ms=4.5, lw=2.0,
            color=PALETTE[lg], label=LABEL[lg],
            markerfacecolor="white", markeredgewidth=1.4,
            markeredgecolor=PALETTE[lg], zorder=3)
style(ax)
ax.set_xlabel("Training epoch")
ax.set_ylabel("Alignment gap  (matched $-$ mismatched cosine)")
ax.set_xticks(epochs3)
ax.legend(ncol=2, fontsize=10, frameon=False, loc="upper left",
          handlelength=1.6, columnspacing=1.3, labelspacing=0.35)
fig.savefig(os.path.join(OUT, "fig_gap_vs_epoch_300.pdf"))
plt.close(fig); print("fig_gap_vs_epoch_300.pdf")

# ---------------------------------------------------------------------------
# Figure 3 : amplitude decay, two runs, log y, value labels + collapse note
# ---------------------------------------------------------------------------
amp_ep = [50, 100, 150]
amp = {"ce150-combined": [0.01604, 0.01249, 0.01075],
       "ce150-all":      [0.01626, 0.00929, 0.00626]}
fig, ax = plt.subplots(figsize=(6.8, 4.5))
series = [("ce150-combined", TEAL, "o", "CE + cosine"),
          ("ce150-all",      ORANGE, "s", "CE only")]
for key, col, mk, lab in series:
    ax.plot(amp_ep, amp[key], marker=mk, ms=7, lw=2.4, color=col, label=lab,
            markerfacecolor="white", markeredgewidth=1.8, markeredgecolor=col,
            zorder=3)
    up = key == "ce150-combined"
    for x, y in zip(amp_ep, amp[key]):
        if x == 150:
            dx, dy, ha = 9, 4, "left"          # last point: label to the right
        else:
            dx, dy, ha = 0, (10 if up else -15), "center"
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=9, color=col)
ax.set_yscale("log")
style(ax)
ax.set_xlabel("Training epoch")
ax.set_ylabel("Motion amplitude\n(mean absolute frame-to-frame difference)")
ax.set_xticks(amp_ep); ax.set_xlim(40, 172)
ax.grid(True, which="both")
ax.annotate("amplitude collapses\nas training proceeds",
            xy=(128, 0.0079), xytext=(55, 0.0083),
            fontsize=10, color=INK, style="italic", va="center",
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.1,
                            connectionstyle="arc3,rad=-0.15"))
ax.legend(fontsize=10.5, frameon=False, loc="upper right",
          title="150-epoch run", bbox_to_anchor=(1.0, 1.02))
fig.savefig(os.path.join(OUT, "fig_amplitude_decay.pdf"))
plt.close(fig); print("fig_amplitude_decay.pdf")

# ---------------------------------------------------------------------------
# Figure 4 : per-language R@1 across the four conditions, home-set highlighted
# ---------------------------------------------------------------------------
cond = ["EN", "Latin", "FAR", "ALL"]
home = {"EN": {"en"}, "Latin": {"en","fr","es","pt"},
        "FAR": {"en","ru","zh","ar"}, "ALL": set(LANGS)}
r1 = {"EN":    [0.610,0.182,0.188,0.198,0.118,0.066,0.084],
      "Latin": [0.410,0.388,0.430,0.438,0.150,0.100,0.086],
      "FAR":   [0.316,0.150,0.172,0.144,0.288,0.238,0.260],
      "ALL":   [0.316,0.272,0.294,0.298,0.248,0.172,0.184]}
ccol = {"EN":"#1B7B94","Latin":"#E8833A","FAR":"#C0433E","ALL":"#4E9F3D"}
x = np.arange(len(LANGS)); w = 0.20
fig, ax = plt.subplots(figsize=(8.0, 4.4))
for i, c in enumerate(cond):
    xs = x + (i-1.5)*w
    # trained languages solid; untrained languages lighter with hatch
    for xi, lg, val in zip(xs, LANGS, r1[c]):
        trained = lg in home[c]
        ax.bar(xi, val, w, color=ccol[c],
               alpha=1.0 if trained else 0.32,
               edgecolor=ccol[c], linewidth=0.8, zorder=3)
ax.axhline(0.002, ls=(0,(4,3)), lw=1, color=INK, alpha=0.6, zorder=2)
ax.text(6.35, 0.018, "chance (0.002)", fontsize=8.5, color=INK, ha="right", alpha=0.8)
style(ax)
ax.set_xticks(x); ax.set_xticklabels([l.upper() for l in LANGS])
ax.set_xlabel("Evaluation language"); ax.set_ylabel("R@1")
ax.set_ylim(0, 0.66)
# custom legend: colour = condition; solid vs faded = trained vs untrained
from matplotlib.patches import Patch
leg1 = [Patch(facecolor=ccol[c], label=c) for c in cond]
leg2 = [Patch(facecolor="#8a8a8a", label="trained"),
        Patch(facecolor="#8a8a8a", alpha=0.32, label="untrained")]
l1 = ax.legend(handles=leg1, title="Training set", fontsize=9.5,
               frameon=False, loc="upper right", ncol=2, columnspacing=1.0)
ax.add_artist(l1)
ax.legend(handles=leg2, fontsize=9, frameon=False, loc="upper left",
          bbox_to_anchor=(0.13, 1.0), handlelength=1.2)
fig.savefig(os.path.join(OUT, "fig_lang_r1.pdf"))
plt.close(fig); print("fig_lang_r1.pdf")
print("done")
