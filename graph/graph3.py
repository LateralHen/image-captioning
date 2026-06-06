import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

steps = [
    ("baseline",              0,     8.31,  "8.31",           "#9E9E9E"),
    ("+ Best practice",       8.31,  9.80,  "+1.49\n(+18%)",  "#66BB6A"),
    ("+ Passaggio a \n Flickr30k",   9.80,  18.99, "+9.19\n(+94%)",  "#FFA726"),
    ("+ Passaggio a \n CLIP-L",      18.99, 25.65, "+6.66\n(+35%)",  "#42A5F5"),
    ("Totale",                0,     25.65, "25.65\n(+209%)", "#1565C0"),
]

fig, ax = plt.subplots(figsize=(9, 8))
ax.set_ylim(5, 30)

for i, (label, start, end, delta, color) in enumerate(steps):
    bottom = min(start, end)
    height = abs(end - start)

    if i == 0 or i == len(steps) - 1:
        if i == 1:  # verde: hatch + bordo spesso
            bar = ax.bar(i, end-5, bottom=5, color=color, width=0.5, edgecolor="white", linewidth=0.8, zorder=3)
        else:
            bar = ax.bar(i, end-5, bottom=5, color=color, width=0.5, edgecolor="white", linewidth=0.8, zorder=3)
    else:
        bar = ax.bar(i, height, bottom=bottom, color=color, width=0.5, edgecolor="white", linewidth=0.8, zorder=3)

    # Linee orizzontali fino all'ultima barra (xmax fisso = 4.25)
    if i < len(steps) - 1:
        ax.plot([i + 0.25, 4.25], [end, end], color=color, lw=0.8, linestyle="--", zorder=4)

    # Testo sempre sopra
    label_y = end + 0.3
    va = "bottom"

    ax.text(i, label_y, delta, ha="center", va=va, fontsize=9,
            fontweight="bold", color="#333333")

# # Linea grigia baseline
# x_minimo, x_massimo = ax.get_xlim()
# ax.hlines(y=steps[0][2], xmin=-0.25, color=steps[0][4], lw=0.8, linestyle="--", zorder=2)

ax.set_xticks(range(len(steps)))
ax.set_xticklabels([s[0] for s in steps], fontsize=10)
ax.set_ylabel("BLEU-4", fontsize=12)
ax.set_title("Contributo di ogni intervento al BLEU-4 finale\n(test set Flickr8k)", fontsize=12, fontweight="bold", pad=12)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_handles = [
    mpatches.Patch(color="#9E9E9E", label="Baseline"),
    mpatches.Patch(color="#66BB6A", label="Best practice training (+18%)", 
                   hatch="////", edgecolor="#2E7D32", linewidth=1.5),
    mpatches.Patch(color="#FFA726", label="Scale-up dataset Flickr30k (+94%)"),
    mpatches.Patch(color="#42A5F5", label="Encoder CLIP ViT-L/14 (+35%)"),
    mpatches.Patch(color="#1565C0", label="Risultato finale (+209%)"),
]
ax.legend(handles=legend_handles, fontsize=8.5, loc="upper left", framealpha=0.85)

plt.tight_layout()
filename_png = "graph/plot3_waterfall.png"
plt.savefig(filename_png, dpi=300, bbox_inches="tight")
print(f"Salvato: {filename_png}")