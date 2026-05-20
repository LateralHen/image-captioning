# Piano di sviluppo — Image Captioning su Flickr8k/Flickr30k/COCO

**Tempo a disposizione**: 1 mese
**Approccio**: progressivo e analitico, ogni settimana introduce una variabile che migliora il sistema. Confronto sistematico tra esperimenti.

---

## Stato attuale (baseline iniziale)

- **Setup completato**: PyTorch 2.11+cu128 su RTX 5070, ambiente venv, struttura cartelle
- **Modello**: ViT-base + GPT-2 small (239M parametri totali)
- **Dataset**: Flickr8k (8091 immagini, split 6000/1000/1091)
- **Training Scenario A**: 8 epoche, overfitting da epoca 3, best val loss 1.95 a epoca 2
- **BLEU baseline (Scenario A)**: BLEU-4 = 8.31 sul test set

---

## Settimana 1 — Baseline rigoroso (Scenario B)

**Obiettivo**: numeri solidi su Flickr8k con tutte le best practice.

**Modifiche al training**:

- [x] Data augmentation: random horizontal flip, color jitter, random crop
- [x] Cosine learning rate scheduler con warmup
- [x] Early stopping basato su validation BLEU
- [x] 25 epoche di budget con salvataggio del solo best checkpoint
- [x] Label smoothing 0.1 nella cross-entropy
- [x] Mixed precision training (bf16)

**Risultato training**: best epoca 5, BLEU-4 = 11.23 sul subset val (200 immagini).
**Risultato test set completo (1091 immagini)**: BLEU-4 = **9.80**.

Il gap tra subset val (11.23) e test completo (9.80) ha mostrato che il subset val a 200 immagini sovrastima leggermente; va usato solo per la selezione del best checkpoint, non come stima della performance finale.

---

## Settimana 2 — Upgrade dataset (Flickr30k)

**Obiettivo**: ridurre overfitting con 4x più dati di training.

**Modifiche**:

- [x] Download Flickr30k (~31000 immagini, ~4.5 GB)
- [x] Split per immagine (no leakage) — verificato: zero intersezione
- [x] Riuso integrale del codice di training della settimana 1
- [x] Re-training con stessi iperparametri ottimizzati

**Risultato training**: 27971 immagini train, 1906 val, 1906 test. Best epoca 5, ~3.7 ore.

**Risultati evaluation**:

| Test set | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Corpus | avg len gen / ref |
| --- | --- | --- | --- | --- | --- | --- |
| Flickr8k (1091 img) | 79.57 | 51.31 | 31.76 | **18.99** | 39.61 | 11.34 / 11.15 |
| Flickr30k (1906 img) | 70.17 | 36.40 | 19.54 | **10.49** | 26.90 | 12.20 / 11.89 |

Il salto da 9.80 a 18.99 sul test Flickr8k è +94%, attribuibile esclusivamente al cambio di dataset.

---

## Settimana 3 — Upgrade encoder (CLIP ViT-L)

**Obiettivo**: encoder visivo pre-addestrato su 400M coppie immagine-testo (vs ImageNet 14M di ViT-base).

**Modifiche**:

- [x] Encoder: `google/vit-base-patch16-224-in21k` → `openai/clip-vit-large-patch14`
- [x] Layer di proiezione 1024→768 creato automaticamente da HuggingFace
- [x] Normalizzazione immagini cambiata da ImageNet a CLIP-specific
- [x] Migrazione da Windows a Linux Fedora 43 per risolvere bottleneck data loading
- [x] Batch ridotto a 4 con gradient accumulation 8 (effective batch 32 invariato) per limiti VRAM
- [x] Re-training su Flickr30k

**Risultato training**: 9 epoche (early stopping patience 3), best a epoca 6, ~7.5 ore.

**Traiettoria val BLEU-4 (subset 200 img)**: 7.87 → 9.98 → 10.24 → 10.22 → 10.83 → **11.18** → 10.36 → 10.81 → 9.33.
Overfitting visibile dall'epoca 7: train_loss continua a scendere mentre val_loss e BLEU peggiorano.

**Risultati evaluation**:

| Test set | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Corpus | avg len gen / ref |
| --- | --- | --- | --- | --- | --- | --- |
| Flickr8k (1091 img) | 83.09 | 57.01 | 38.54 | **25.65** | 46.52 | 11.27 / 11.09 |
| Flickr30k (1906 img) | 71.13 | 37.33 | 20.36 | **11.13** | 27.85 | 12.43 / 12.13 |

**Osservazione chiave**: il guadagno di CLIP-L è asimmetrico tra i due test set.

- Su Flickr8k (cross-distribution F30k→F8k): **+35% BLEU-4** (18.99 → 25.65)
- Su Flickr30k (in-distribution): **+6% BLEU-4** (10.49 → 11.13)

Interpretazione: CLIP-L pre-addestrato su 400M coppie immagine-testo del web ha rappresentazioni visive molto generali. Su un test diverso dal dataset di training generalizza molto meglio del ViT-base addestrato su ImageNet. Su un test in-distribution il bottleneck è il fitting del decoder GPT-2 a quel dataset specifico, e CLIP-L può aiutare solo marginalmente.

**Lesson learned**: il subset val durante training (11.18) era praticamente identico a quello della Settimana 2 (11.11). Solo l'evaluation di test completo ha mostrato la differenza vera (+35% su F8k). Il subset val è uno strumento per selezionare il best checkpoint, non un predittore affidabile della performance reale.

---

## Settimana 4 — Tuning, scrittura, presentazione

**Obiettivo**: consolidamento e comunicazione.

**Attività**:

- [ ] Scrittura report finale: introduzione, related work, metodo, esperimenti, risultati, discussione, conclusioni
- [ ] Preparazione slide di presentazione con grafici comparativi
- [ ] Pulizia repository: README, requirements.txt verificato, notebook rinumerati e ordinati
- [ ] Repository finale rieseguibile da zero

**Deliverable**: report PDF + slides + repo Git pulito.

---

## Metriche di successo finali

| Esperimento | BLEU-4 atteso | BLEU-4 misurato (test F8k) | Tempo training |
| ------------- | --------------- | ---------------------------- | ----------------- |
| Scenario A (baseline ingenuo) | ~15-20 | **8.31** | 45 min |
| Settimana 1 (B su Flickr8k) | 22-26 | **9.80** | 53 min (early stop) |
| Settimana 2 (B su Flickr30k) | 26-30 | **18.99** | 3.7 h |
| Settimana 3 (CLIP-L + Flickr30k) | 22-28 | **25.65** | 7.5 h (Linux) |

**Stato dell'arte 2017-2018**: BLEU-4 ~25-30 → **risultato finale nel range SOTA storico**.
**Stato dell'arte 2024+**: BLEU-4 35-40+ (richiede modelli enormi, fuori scope).

---

## Quadro completo dei miglioramenti (test Flickr8k)

Da baseline a Settimana 3: **8.31 → 25.65 BLEU-4 = +209%**.

| Passo | Modifica | Contributo |
| --- | --- | --- |
| A → 1 | Augmentation, cosine LR, label smoothing, early stop | +18% |
| 1 → 2 | Dataset 4× più grande (Flickr30k) | +94% |
| 2 → 3 | Encoder pre-addestrato CLIP-L | +35% |

Il salto più grosso resta il cambio di dataset (Settimana 2). Il tuning training (Settimana 1) e il cambio encoder (Settimana 3) hanno contributi simili in proporzione. Conferma: nel range di dataset piccolo-medio (10k-30k immagini), i dati battono il tuning, ma un encoder pre-addestrato su scala web aggiunge un guadagno significativo soprattutto in cross-distribution.

---

## Principi guida

1. **Misura prima di ottimizzare**: ogni esperimento si confronta con un baseline numerico precedente
2. **Cambia una variabile alla volta**: ogni settimana una sola modifica strutturale, per attribuire correttamente i miglioramenti
3. **Documenta gli esperimenti**: tieni un log con iperparametri, BLEU, tempo, esempi qualitativi
4. **Salva i checkpoint best, butta gli intermedi**: spazio disco a controllo
5. **Riproducibilità**: seed fissi, requirements.txt aggiornato, codice nei notebook leggibile