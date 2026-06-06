"""
Genera il grafico della traiettoria della validation BLEU-4 (Settimana 3,
CLIP ViT-L/14 su Flickr30k, subset di 200 immagini di validation).

Mostra il picco a epoca 6 (best checkpoint) e la zona di overfitting da epoca 7.

Uso:
    python make_val_bleu4_trajectory_week3.py
Output:
    graph/val_bleu4_trajectory_week3.png  (nella cartella corrente)
"""

import matplotlib
matplotlib.use("Agg")          # backend non interattivo (toglilo se vuoi plt.show())
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --- Dati: traiettoria validation BLEU-4 della Settimana 3 ---
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
bleu4  = [7.87, 9.98, 10.24, 10.22, 10.83, 11.18, 10.36, 10.81, 9.33]
best_epoch, best_val = 6, 11.18          # best checkpoint (early stopping)

# --- Colore principale ---
# Bordeaux Sapienza (default). Per uniformare ai grafici blu della relazione,
# commenta la riga MAROON e usa il blu (riprendi l'esatto hex dalla Figura 1).
MAROON = "#822433"
# MAROON = "#1f5fb4"   # <- alternativa blu
GREY = "#6B6B6B"

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)

# Zona overfitting (dall'epoca 6 in poi, ombreggiatura leggera)
ax.axvspan(6, 9.3, color=MAROON, alpha=0.06, zorder=0)
ax.text(7.65, 7.55, "Overfitting\n(train loss \u2193, val BLEU \u2193)",
        ha="center", va="bottom", fontsize=9, color=GREY, style="italic")

# Linea principale
ax.plot(epochs, bleu4, color=MAROON, linewidth=2.4, marker="o",
        markersize=6, markerfacecolor="white", markeredgecolor=MAROON,
        markeredgewidth=1.8, zorder=3)

# Punto del best checkpoint
ax.plot(best_epoch, best_val, marker="o", markersize=11,
        markerfacecolor=MAROON, markeredgecolor="white",
        markeredgewidth=2, zorder=4)
ax.annotate("Best checkpoint\n(epoca 6 \u2014 11.18)",
            xy=(best_epoch, best_val), xytext=(3.5, 11.35),
            fontsize=9.5, fontweight="bold", color=MAROON, ha="center",
            arrowprops=dict(arrowstyle="->", color=MAROON, lw=1.4))

# Assi e stile
ax.set_xlabel("Epoca", fontsize=11)
ax.set_ylabel("BLEU-4 (subset validation)", fontsize=11)
ax.set_title("Validation BLEU-4 \u2014 Fase 4 (CLIP ViT-L/14 su Flickr30k)",
             fontsize=12.5, fontweight="bold", color=MAROON, pad=12)
ax.set_xlim(0.6, 9.4)
ax.set_ylim(7.3, 11.8)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.grid(axis="y", linestyle="--", alpha=0.35)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GREY)
ax.spines["bottom"].set_color(GREY)
ax.tick_params(colors=GREY)

plt.tight_layout()
plt.savefig("graph/val_bleu4_trajectory_week3.png", bbox_inches="tight", facecolor="white")
print("Salvato: graph/val_bleu4_trajectory_week3.png")