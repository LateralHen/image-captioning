## TITOLO (ipotesi)

**"Progressive Image Captioning with Vision-Language Encoders: A Systematic Ablation Study on Flickr8k and Flickr30k"**

oppure più semplice: **"Encoder Scaling and Data Size in Automatic Image Captioning"**

---

## ABSTRACT (bullet da cui scrivere)

- Task: image captioning automatico (generare descrizione testuale da immagine)
- Architettura base: encoder visivo + decoder linguistico (ViT + GPT-2)
- Approccio: ablation progressiva a variabile singola, 3 settimane
- Dataset: Flickr8k → Flickr30k
- Metrica principale: BLEU-4
- Risultato finale: 8.31 → 25.65 BLEU-4 (+209%) sul test Flickr8k
- Conclusione chiave: i dati (dataset 4×) contribuiscono più del tuning; CLIP-L generalizza meglio cross-distribution

---

## 1. INTRODUZIONE

**Problema**: descrivere automaticamente un'immagine con linguaggio naturale — richiede comprensione visiva + generazione testuale.

**Motivazione**: task fondamentale per accessibility, retrieval, assistenza visiva.

**Approccio scelto**: architettura encoder-decoder con componenti pre-addestrati (ViT per vision, GPT-2 per language). Scelta deliberatamente "non all-in-one" per capire i contributi separati.

**Contributo del paper**:
1. Ablation sistematica: ogni esperimento cambia una sola variabile
2. Analisi del contributo relativo di training tricks, dimensione dataset, qualità encoder
3. Osservazione sull'asimmetria cross-distribution vs in-distribution con CLIP

**Setup hardware**: RTX 5070, PyTorch 2.11+cu128

---

## 2. RELATED WORK (da espandere, ma i punti sono)

- Show and Tell (Vinyals et al., 2015): CNN + LSTM, prima architettura encoder-decoder moderna per captioning
- Attention-based: Show, Attend and Tell (Xu et al., 2015)
- Transformer-based captioning: OSCAR, VinVL, ClipCap
- **ClipCap** (Mokady et al., 2021): il più vicino al nostro approccio — usa CLIP come encoder + GPT-2 come decoder
- BLEU score (Papineni et al., 2002): metrica standard usata
- SOTA 2017-2018: BLEU-4 ~25-30 su Flickr8k; SOTA 2024+: 35-40+ (modelli enormi)

---

## 3. METODO

### 3.1 Architettura generale

```
[Immagine] → [Encoder visivo] → [Projection layer] → [GPT-2 decoder] → [Caption]
```

**Baseline (Scenario A)**:
- Encoder: `google/vit-base-patch16-224-in21k` (ViT-base, pretraining ImageNet-21k, 14M immagini)
- Decoder: GPT-2 small
- Totale parametri: 239M
- Projection: 768→768 (no resize necessario)

**Versione finale (Settimana 3)**:
- Encoder: `openai/clip-vit-large-patch14` (CLIP ViT-L, pretraining 400M coppie web)
- Decoder: GPT-2 small (invariato)
- Projection layer: 1024→768 (aggiunto automaticamente da HuggingFace)
- Normalizzazione immagini: ImageNet stats → CLIP-specific stats

### 3.2 Training setup (Scenario B in poi)

| Componente | Valore |
|---|---|
| Optimizer | AdamW |
| LR scheduler | Cosine con warmup |
| Label smoothing | 0.1 |
| Mixed precision | bf16 |
| Augmentation | RandomHFlip, ColorJitter, RandomCrop |
| Early stopping | su validation BLEU |
| Budget max epoche | 25 |
| Batch size effettivo | 32 |

Per CLIP-L (limiti VRAM): batch=4, gradient accumulation=8 → effective batch 32 invariato.

### 3.3 Evaluation

- Metrica: BLEU-1/2/3/4 (sentence-level + corpus-level)
- Test set Flickr8k: 1091 immagini
- Test set Flickr30k: 1906 immagini
- Validation durante training: subset 200 immagini (per checkpoint selection only, non per stima finale)

---

## 4. ESPERIMENTI & RISULTATI

### 4.1 Dataset

| Dataset | Immagini totali | Train / Val / Test |
|---|---|---|
| Flickr8k | 8091 | 6000 / 1000 / 1091 |
| Flickr30k | ~31000 | 27971 / 1906 / 1906 |

Split per immagine (no caption leakage), zero intersezione verificata tra split.

### 4.2 Tabella principale risultati

| Esperimento | N img test | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Corpus BLEU | Training time |
|---|---|---|---|---|---|---|---|
| Scenario A (baseline) | 1091 | 65.72 | 32.27 | 16.72 | 8.31 | 23.3 | 45 min |
| Week 1 – best practices (F8k) | 1091 | 68.30 | 35.34 | 18.63 | 9.80 | 25.77 | 53 min |
| Week 2 – Flickr30k (→F8k test) | 1091 | 79.57 | 51.31 | 31.76 | 18.99 | 39.61 | 3.7 h |
| Week 2 – Flickr30k (→F30k test) | 1906 | 70.17 | 36.40 | 19.54 | 10.49 | 26.90 | 3.7 h |
| Week 3 – CLIP-L (→F8k test) | 1091 | 83.09 | 57.01 | 38.54 | **25.65** | 46.52 | 7.5 h |
| Week 3 – CLIP-L (→F30k test) | 1906 | 71.13 | 37.33 | 20.36 | 11.13 | 27.85 | 7.5 h |

### 4.3 Contributo relativo di ogni modifica (su test F8k)

| Step | Modifica | BLEU-4 | Δ assoluto | Δ % |
|---|---|---|---|---|
| A | Baseline naive | 8.31 | — | — |
| A→1 | Augmentation + cosine LR + label smoothing + early stop | 9.80 | +1.49 | +18% |
| 1→2 | Dataset 4× (Flickr30k) | 18.99 | +9.19 | +94% |
| 2→3 | Encoder CLIP ViT-L | 25.65 | +6.66 | +35% |
| **Totale** | | **25.65** | **+17.34** | **+209%** |

### 4.4 Osservazione chiave: asimmetria cross-distribution

CLIP-L guadagna +35% su F8k (cross-distribution: trained on F30k, tested on F8k) ma solo +6% su F30k (in-distribution). Interpretazione: CLIP pre-trained su 400M coppie web ha features visive altamente generalizzabili. Quando il test set è out-of-distribution rispetto al training, l'encoder più potente compensa meglio. In-distribution, il bottleneck si sposta sul decoder (GPT-2 fitting al vocabolario specifico del dataset).

### 4.5 Nota metodologica: val subset vs test completo

Il val subset da 200 immagini usato durante training ha sovrastimato leggermente (es. Week 1: val BLEU-4=11.23 vs test=9.80). In Week 3, il val subset (11.18) era quasi identico a Week 2 (11.11) ma il test completo su F8k era 25.65 vs 18.99. Conclusione: il val subset è affidabile solo per selezione checkpoint, non per comparazione tra modelli diversi.

---

## 5. DISCUSSIONE

**Cosa ha funzionato di più**: il cambio dataset (F8k→F30k) — i dati battono il tuning in questo regime.

**Cosa ha sorpreso**: il guadagno asimmetrico di CLIP-L. Ci si aspettava un miglioramento uniforme, invece è concentrato nel caso cross-distribution.

**Limitazioni**:
- Decoder fisso (GPT-2 small): potrebbe essere il collo di bottiglia residuo
- Nessun fine-tuning dell'encoder (frozen weights): potrebbe portare ulteriore guadagno
- BLEU è una metrica imperfetta (non cattura semantica, fluency, ecc.) — CIDEr/METEOR sarebbero complementari
- Budget computazionale limitato (singola GPU consumer)

**Confronto SOTA**:
- SOTA 2017-2018: BLEU-4 ~25-30 → risultato finale (25.65) nel range storico
- SOTA 2024+: 35-40+ ma con modelli enormi (BLIP-2, CoCa, ecc.) fuori scope

---

## 6. CONCLUSIONI

- Ablation a variabile singola consente attribuzione pulita dei miglioramenti
- Il contributo più grande viene dalla scala dei dati (+94%), non dal tuning
- Un encoder pre-addestrato su scala web (CLIP-L, 400M coppie) porta +35% in cross-distribution e generalizza molto meglio out-of-distribution
- Risultato finale nel range SOTA storico con soli ~10 ore di training totale su GPU consumer

---

## FIGURE DA PREPARARE

1. Schema architettura (encoder → projection → GPT-2 decoder)
2. Grafico a barre BLEU-4 per tutti gli esperimenti (su F8k)
3. Tabella comparativa principale (già hai `comparison_table.md`)
4. Curva val BLEU-4 per Settimana 3: `7.87 → 9.98 → 10.24 → 10.22 → 10.83 → 11.18 → 10.36 → 10.81 → 9.33`
5. (Opzionale) Esempi qualitativi: caption generate su immagini di esempio
