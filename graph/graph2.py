import matplotlib.pyplot as plt
import numpy as np

experiments = ["ViT-base + GPT-2", "CLIP-L + GPT-2"]
f8k  = [18.99, 25.65]   # cross-distribution
f30k = [10.49, 11.13]   # in-distribution

x = np.arange(len(experiments))
width = 0.32

fig, ax = plt.subplots(figsize=(7, 5))

bars1 = ax.bar(x - width / 2, f8k,  width, label="Flickr8k (cross-distribution)", color="#1565C0", edgecolor="white", zorder=3)
bars2 = ax.bar(x + width / 2, f30k, width, label="Flickr30k (in-distribution)",   color="#90CAF9", edgecolor="white", zorder=3)
# Delta annotations above each group
deltas_f8k  = [None, f"+{((25.65-18.99)/18.99*100):.0f}%"]  # +35%
deltas_f30k = [None, f"+{((11.13-10.49)/10.49*100):.0f}%"]  # +6%

# Value labels
c=0
for bar in bars1:
    if c==0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#1565C0")
        c+=1
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.2f} ({deltas_f8k[c]})", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#1565C0")
        c=0

for bar in bars2:
    if c == 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#1565C0")
        c+=1
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.2f} ({deltas_f30k[c]})", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#1565C0")
        c=0



ax.set_ylabel("BLEU-4", fontsize=12)
ax.set_title("BLEU-4: cross-distribution vs in-distribution\n(test set Flickr8k e Flickr30k)", fontsize=12, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(experiments, fontsize=10.5)
ax.set_ylim(0, 32)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10, loc="upper left")

plt.tight_layout()
plt.savefig("graph/plot2_cross_vs_indist.png", dpi=300, bbox_inches="tight")
print("Salvato: graph/plot2_cross_vs_indist.pdf / .png")