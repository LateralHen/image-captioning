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
- **Esempi qualitativi**: sufficienti su scene canoniche (cane+spiaggia), errori su soggetti rari (uniformi, scene urbane complesse), errori di colore/dettaglio

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
- [x] Split per immagine (no leakage), proporzioni simili a Flickr8k — verificato: zero intersezione tra split
- [x] Riuso integrale del codice di training della settimana 1
- [x] Re-training con stessi iperparametri ottimizzati

**Risultato training**: 27971 immagini train, 1906 val, 1906 test. Best epoca 5, ~3.7 ore di training totale.

**Risultati evaluation**:

| Test set | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Corpus | avg len gen / ref |
|---|---|---|---|---|---|---|
| Flickr8k (1091 img) | 79.57 | 51.31 | 31.76 | **18.99** | 39.61 | 11.34 / 11.15 |
| Flickr30k (1906 img) | 70.17 | 36.40 | 19.54 | **10.49** | 26.90 | 12.20 / 11.89 |

Il salto da 9.80 (Settimana 1) a 18.99 (Settimana 2) sul test Flickr8k è il miglioramento più grande del progetto, +94%, attribuibile esclusivamente al cambio di dataset (4x più dati di training, tutto il resto identico).

Il BLEU-4 in-distribution su Flickr30k (10.49) è inferiore a quello su Flickr8k (18.99). Coerente con la letteratura: Flickr30k ha didascalie più lunghe, vocabolario più ricco e meno ripetizioni rispetto a Flickr8k, quindi è un test più difficile a parità di modello.

---

## Settimana 3 — Upgrade encoder (CLIP ViT-L)

**Obiettivo**: encoder visivo pre-addestrato su 400M coppie immagine-testo (vs ImageNet 14M di ViT-base).

**Modifiche**:
- [ ] Sostituzione encoder: `google/vit-base-patch16-224-in21k` → `openai/clip-vit-large-patch14`
- [ ] Eventuale layer di proiezione tra encoder e decoder se le dimensioni embedding non coincidono (CLIP-L = 1024 dim, GPT-2 small = 768 dim → proiezione necessaria)
- [ ] Verifica compatibilità con `VisionEncoderDecoderModel`
- [ ] Re-training su Flickr30k

**Deliverable**: confronto BLEU Settimana 3 vs Settimana 2. Atteso: BLEU-4 ~22-28 sul test Flickr8k.

**Stretch goal opzionale**: provare anche GPT-2 medium come decoder se la VRAM lo permette.

---

## Settimana 4 — Tuning, scrittura, presentazione

**Obiettivo**: consolidamento e comunicazione.

**Attività**:
- [ ] Iterazione finale sul setup migliore (search di iperparametri se serve)
- [ ] Scrittura report finale: introduzione, related work, metodo, esperimenti, risultati, discussione, conclusioni
- [ ] Preparazione slide di presentazione con grafici comparativi
- [ ] Pulizia repository: README, requirements.txt verificato, notebook rinumerati e ordinati
- [ ] Repository finale rieseguibile da zero

**Deliverable**: report PDF + slides + repo Git pulito.

---

## Metriche di successo finali

| Esperimento | BLEU-4 atteso | BLEU-4 misurato (test F8k) | Tempo training |
|-------------|---------------|----------------------------|-----------------|
| Scenario A (baseline ingenuo) | ~15-20 | **8.31** | 45 min |
| Settimana 1 (B su Flickr8k) | 22-26 | **9.80** | 53 min (early stop) |
| Settimana 2 (B su Flickr30k) | 26-30 | **18.99** | 3.7 h |
| Settimana 3 (CLIP + Flickr30k) | 22-28 | — | 6-10 h |

**Stato dell'arte 2017-2018**: BLEU-4 ~25-30
**Stato dell'arte 2024+**: BLEU-4 35-40+ (richiede modelli enormi, fuori scope)

---

## Principi guida

1. **Misura prima di ottimizzare**: ogni esperimento si confronta con un baseline numerico precedente
2. **Cambia una variabile alla volta**: ogni settimana una sola modifica strutturale, per attribuire correttamente i miglioramenti
3. **Documenta gli esperimenti**: tieni un log con iperparametri, BLEU, tempo, esempi qualitativi
4. **Salva i checkpoint best, butta gli intermedi**: spazio disco a controllo
5. **Riproducibilità**: seed fissi, requirements.txt aggiornato, codice nei notebook leggibile