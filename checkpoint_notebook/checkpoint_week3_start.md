# Checkpoint — Fine Settimana 2, inizio Settimana 3

**Data**: maggio 2026
**Progetto**: Image Captioning su Flickr8k/30k

---

## Dove siamo

Settimane 1 e 2 completate. Hai tre esperimenti misurati su test set completi e confrontabili. Il setup per la settimana 3 è verificato e pronto.

Sei in pari con la timeline. Mancano due settimane su quattro.

---

## Risultati misurati (BLEU-4 su test Flickr8k)

| Esperimento | BLEU-4 | Note |
|---|---|---|
| Scenario A (baseline ingenuo) | 8.31 | ViT + GPT-2, 8 epoche, overfit dall'epoca 3 |
| Settimana 1 (Scenario B su Flickr8k) | 9.80 | + augmentation, cosine LR, label smoothing, early stopping |
| Settimana 2 (Scenario B su Flickr30k) | **18.99** | stesso modello, 4x più dati |

Settimana 2 valutata anche su Flickr30k test (in-distribution): BLEU-4 = 10.49. Più basso del Flickr8k è coerente con la letteratura — Flickr30k ha didascalie più ricche.

**Osservazione chiave per il report**: il tuning fine (A → 1) ha dato +18%, il cambio di dataset (1 → 2) ha dato +94%. Quando il bottleneck sono i dati, l'iperparameter tuning conta poco.

---

## Pulizia disco effettuata

- Cancellato `checkpoints/epoch_02/` (duplicato di `best/`)
- Cancellato `checkpoints/scenario_B/last/`
- Cancellato `checkpoints/week2_flickr30k/last/`

Liberati ~2.8 GB. Stato attuale: ~3 GB occupati in checkpoint significativi.

---

## Verifiche effettuate

- **Split Flickr30k**: per immagine, zero intersezione tra train/val/test. No leakage.
- **Tabella comparativa**: generata e salvata in `outputs/comparison_table.md`.
- **`PROJECT_PLAN.md`**: aggiornato con i numeri reali e i task spuntati.

---

## Setup Settimana 3 — verificato

### Modello CLIP-L + GPT-2 + proiezione

| | ViT-base (vecchio) | CLIP-L (nuovo) |
|---|---|---|
| Hidden size | 768 | **1024** |
| Layer | 12 | 24 |
| Patch | 196 (16×16) | 256 (14×14) |
| Parametri encoder | 86M | **303M** |
| **Modello totale** | **239M** | **456.8M** |

Layer di proiezione 1024→768 creato automaticamente da HuggingFace (~0.79M parametri).

CLIP-L scaricato in cache HuggingFace (1.71 GB), non si riscarica più.

### VRAM benchmark misurato

| Step | VRAM |
|---|---|
| Solo modello caricato | 1.85 GB |
| Picco durante forward+backward+optim | **9.31 GB** |
| Disponibili | 12.8 GB (RTX 5070) |
| **Utilizzo** | **73%** |

Margine ~3.5 GB. Sicuro, niente OOM atteso.

### Decisione: batch 16 + gradient accumulation 2 = effective 32

Motivazione:
1. 73% di VRAM è sicuro, ma andare a 95-100% con batch 32 espone a OOM da picchi anomali durante training.
2. Effective batch 32 preserva la confrontabilità con settimana 2.
3. Costo in tempo trascurabile.

### Tempo stimato training

- Per epoca: ~28 minuti
- Budget massimo (25 epoche): 11 ore
- Stima realistica con early stopping: **3-5 ore**

### Iperparametri (identici a settimana 2, eccetto batch size)

- Learning rate: 5e-5 peak, cosine + warmup 1/25
- Weight decay: 0.01
- Label smoothing: 0.1
- Mixed precision: bfloat16
- Gradient clipping: 1.0
- Augmentation: RandomResizedCrop + HorizontalFlip + ColorJitter
- Early stopping: patience 3 su BLEU-4 di subset val (200 img)

Tutto salvato in `outputs/week3_config.json`.

---

## Una cosa da sapere per la cella che faremo dopo

**Normalizzazione immagini diversa**. ViT usa la normalizzazione standard ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). CLIP usa parametri leggermente diversi (`mean≈[0.481, 0.458, 0.408]`, `std≈[0.269, 0.261, 0.276]`). Se sbagli normalizzazione, l'encoder pre-addestrato lavora "fuori distribuzione" e perdi gran parte del vantaggio. Quando scriveremo il `T.Normalize` di torchvision, useremo i valori giusti estratti dal `CLIPImageProcessor`.

---

## Prossimo step

Creare `11_training_week3.ipynb` e procedere a piccole celle. La struttura sarà simile a quella della settimana 2 (`05_training_scenario_B.py`), con tre differenze:

1. Modello CLIP-L + GPT-2 invece di ViT + GPT-2
2. Normalizzazione CLIP invece di ImageNet
3. Gradient accumulation 2 step nel training loop

**Pre-flight check obbligatorio prima di lanciare il training reale**: dry run di 2-3 batch per verificare che tutto giri senza bug stupidi. Se nel pre-flight si rompe qualcosa, lo fixiamo in 2 minuti invece di scoprirlo dopo 2 ore.

---

## File su disco — stato attuale

```
checkpoints/
├── model_initial/           (snapshot vergine ViT+GPT2, 957 MB)
├── best/                    (best Scenario A, epoca 2)
├── scenario_B/
│   ├── best/                (best Settimana 1)
│   └── results_test.json
├── week2_flickr30k/
│   └── best/                (best Settimana 2)

outputs/
├── learning_curves.png
├── test_predictions_baseline.json
├── bleu_results_baseline.json
├── scenario_B_history.json
├── week2_flickr30k_history.json
├── week2_eval_flickr8k.json
├── week2_eval_flickr30k.json
├── comparison_table.md
└── week3_config.json        (NUOVO, configurazione settimana 3)
```

---

## Cosa rimane

- **Settimana 3**: training CLIP-L su Flickr30k, evaluation su F8k + F30k test. Atteso BLEU-4 22-28 su F8k.
- **Settimana 4**: report PDF, slides, pulizia repo finale.