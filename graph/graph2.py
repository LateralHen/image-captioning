import matplotlib.pyplot as plt
import numpy as np

experiments = ["Settimana 2\n(ViT-base + GPT-2)", "Settimana 3\n(CLIP-L + GPT-2)"]
f8k  = [18.99, 25.65]   # cross-distribution
f30k = [10.49, 11.13]   # in-distribution

x = np.arange(len(experiments))
width = 0.32

fig, ax = plt.subplots(figsize=(7, 5))

bars1 = ax.bar(x - width / 2, f8k,  width, label="Flickr8k (cross-distribution)", color="#1565C0", edgecolor="white", zorder=3)
bars2 = ax.bar(x + width / 2, f30k, width, label="Flickr30k (in-distribution)",   color="#90CAF9", edgecolor="white", zorder=3)

# Value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#1565C0")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#1565C0")

# Delta annotations above each group
deltas_f8k  = [None, f"+{((25.65-18.99)/18.99*100):.0f}%"]  # +35%
deltas_f30k = [None, f"+{((11.13-10.49)/10.49*100):.0f}%"]  # +6%

# for i in range(1, len(experiments)):
#     # F8k delta
#     ax.annotate("", xy=(x[i] - width/2, f8k[i] + 1.5), xytext=(x[i-1] - width/2, f8k[i-1] + 1.5),
#                 arrowprops=dict(arrowstyle="-", color="#1565C0", lw=1, linestyle="dashed"))
#     ax.text((x[i] + x[i-1]) / 2 - width/2, max(f8k) + 2.2, deltas_f8k[i],
#             ha="center", fontsize=9.5, color="#1565C0", fontstyle="italic")
#     # F30k delta
#     ax.annotate("", xy=(x[i] + width/2, f30k[i] + 1.5), xytext=(x[i-1] + width/2, f30k[i-1] + 1.5),
#                 arrowprops=dict(arrowstyle="-", color="#42A5F5", lw=1, linestyle="dashed"))
#     ax.text((x[i] + x[i-1]) / 2 + width/2, max(f30k) + 2.2, deltas_f30k[i],
#             ha="center", fontsize=9.5, color="#42A5F5", fontstyle="italic")

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
plt.savefig("plot2_cross_vs_indist.pdf", dpi=300, bbox_inches="tight")
plt.savefig("plot2_cross_vs_indist.png", dpi=300, bbox_inches="tight")
print("Salvato: plot2_cross_vs_indist.pdf / .png")