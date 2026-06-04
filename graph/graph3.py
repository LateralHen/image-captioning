import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Each bar: (label, start, end, delta_label, color)
steps = [
    ("Scenario A\n(baseline)",     0,     8.31,  "8.31",  "#9E9E9E"),
    ("+ Best practice\n(W1)",      8.31,  9.80,  "+1.49\n(+18%)", "#66BB6A"),
    ("+ Flickr30k\n(W2)",          9.80,  18.99, "+9.19\n(+94%)", "#FFA726"),
    ("+ CLIP-L\n(W3)",             18.99, 25.65, "+6.66\n(+35%)", "#42A5F5"),
    ("Totale\n(W3)",               0,     25.65, "25.65\n(+209%)", "#1565C0"),
]

fig, ax = plt.subplots(figsize=(9, 5.5))

for i, (label, start, end, delta, color) in enumerate(steps):
    bottom = min(start, end)
    height = abs(end - start)
    # Full bar for baseline and total
    if i == 0 or i == len(steps) - 1:
        bar = ax.bar(i, end, color=color, width=0.5, edgecolor="white", linewidth=0.8, zorder=3)
    else:
        bar = ax.bar(i, height, bottom=bottom, color=color, width=0.5, edgecolor="white", linewidth=0.8, zorder=3)

    # Connector line to next bar (except last)
    if i < len(steps) - 1 and i > 0:
        ax.plot([i - 0.25, i + 0.25], [end, end], color="#AAAAAA", lw=0.8, linestyle="--", zorder=2)
    if i == 0:
        ax.plot([i + 0.25, i + 0.75], [end, end], color="#AAAAAA", lw=0.8, linestyle="--", zorder=2)

    # Value label inside/above bar
    label_y = end + 0.3 if i in (0, len(steps) - 1) else bottom + height / 2
    va = "bottom" if i in (0, len(steps) - 1) else "center"
    ax.text(i, label_y, delta, ha="center", va=va, fontsize=9,
            fontweight="bold", color="#333333" if i not in (3,) else "white")

ax.set_xticks(range(len(steps)))
ax.set_xticklabels([s[0] for s in steps], fontsize=10)
ax.set_ylabel("BLEU-4", fontsize=12)
ax.set_title("Contributo di ogni intervento al BLEU-4 finale\n(test set Flickr8k)", fontsize=12, fontweight="bold", pad=12)
ax.set_ylim(0, 32)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_handles = [
    mpatches.Patch(color="#9E9E9E", label="Baseline"),
    mpatches.Patch(color="#66BB6A", label="Best practice training (+18%)"),
    mpatches.Patch(color="#FFA726", label="Scale-up dataset Flickr30k (+94%)"),
    mpatches.Patch(color="#42A5F5", label="Encoder CLIP ViT-L/14 (+35%)"),
    mpatches.Patch(color="#1565C0", label="Risultato finale (+209%)"),
]
ax.legend(handles=legend_handles, fontsize=8.5, loc="upper left", framealpha=0.85)

plt.tight_layout()
plt.savefig("plot3_waterfall.pdf", dpi=300, bbox_inches="tight")
plt.savefig("plot3_waterfall.png", dpi=300, bbox_inches="tight")
print("Salvato: plot3_waterfall.pdf / .png")