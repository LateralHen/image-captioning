import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

labels = ["Scenario A\n(baseline)", "Settimana 1\n(best practice)", "Settimana 2\n(Flickr30k)", "Settimana 3\n(CLIP-L)"]
values = [8.31, 9.80, 18.99, 25.65]
colors = ["#9E9E9E", "#90CAF9", "#42A5F5", "#1565C0"]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="white", linewidth=0.8, zorder=3)

# Value labels on top of each bar
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333333")

# Percentage improvement annotations (arrows between bars)
improvements = [
    (0, 1, "+18%"),
    (1, 2, "+94%"),
    (2, 3, "+35%"),
]
for i, j, label in improvements:
    x1 = i + 0.28
    x2 = j - 0.28
    y = max(values[i], values[j]) + 2.2
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))
    ax.text((x1 + x2) / 2, y + 0.25, label, ha="center", va="bottom",
            fontsize=9.5, color="#555555", fontstyle="italic")

ax.set_ylabel("BLEU-4", fontsize=12)
ax.set_title("Progressione BLEU-4 su Flickr8k test set", fontsize=13, fontweight="bold", pad=14)
ax.set_ylim(0, 32)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("graph1_progression.pdf", dpi=300, bbox_inches="tight")
plt.savefig("graph1_progression.png", dpi=300, bbox_inches="tight")
print("Salvato: graph1_progression.pdf / .png")