# 📋 CHECKPOINT — Dove siamo dopo 2 settimane

## Stato del progetto

Hai completato 2 settimane su 4 del piano. Sei in pari con la timeline e i risultati sono migliori delle attese.

## Cosa hai costruito (riassunto tecnico)

### Infrastruttura

- Stack: PyTorch 2.11 + CUDA 12.8 su RTX 5070 Blackwell
- Ambiente: venv Python 3.12 con dipendenze pinnate
- Repository strutturato: `data/`, `notebooks/`, `checkpoints/`, `outputs/`, `script/`
- 9 notebook funzionanti, numerati e ordinati flat in `notebooks/`

### Modello

- Architettura: `VisionEncoderDecoderModel` (Hugging Face)
- Encoder: ViT-base patch16 224 (pretrain ImageNet-21k)
- Decoder: GPT-2 small (124M, 50257 vocab BPE)
- Totale: 239.2M parametri trainable
- Cross-attention inizializzata random, fine-tuned end-to-end

### Pipeline di training

- Data augmentation: `RandomResizedCrop`, `ColorJitter`, `HorizontalFlip`
- Scheduler: cosine learning rate con warmup
- Optimizer: AdamW, weight decay 0.01
- Mixed precision: `bfloat16`
- Gradient clipping
- Label smoothing 0.1
- Early stopping su BLEU-4 (patience 3)
- Doppio checkpoint: `best/` (su BLEU-4) + `last/` (antinfortunistica)

### Pipeline di evaluation

- Generazione con beam search (4 beam)
- Workaround GPT-2 BOS/EOS: `decoder_input_ids=[[BOS, BOS]]`
- Metriche: BLEU-1/2/3/4 + corpus su `sacrebleu`
- Valutazione su test set completi (1091 e 1906 immagini)

## Risultati misurati (test Flickr8k, confrontabili)

| Esperimento | BLEU-4 | Tempo training |
|---|---|---|
| Scenario A (baseline) | 8.31 | 45 min |
| Scenario B (Sett. 1) | 9.80 | 53 min |
| Settimana 2 | 18.99 | 3.7 ore |

Il salto da 9.80 a 18.99 (+94%) tra Settimana 1 e 2 è il miglioramento più grande del progetto ed è attribuibile solo al cambio di dataset (4x più dati). Tutti gli iperparametri erano identici.

## Bug risolti

- Doppio shift labels — fix critico nella cross-entropy custom
- Mascheramento BOS/EOS errato — usato `attention_mask` invece di `pad_token_id`
- `generate` produceva stringa vuota — workaround `decoder_input_ids=[[BOS, BOS]]`

## Lessons learned (per il report)

- Val loss ≠ best model: la val loss minima e il BLEU-4 massimo non coincidono. Selezionare sul BLEU ha guadagnato +1-2 punti su ogni esperimento.
- Val subset è ingannevole: il subset val durante training stimava 11.11 per Sett. 2, mentre il test reale è 18.99 su F8k. Il subset serve solo per "best epoch selection", non per stimare la performance finale.
- Augmentation aiuta poco se il bottleneck sono i dati: +18% da A a B (stesso dataset, miglior training); +94% da B a Sett. 2 (4x più dati). Nel range piccolo-medio, i dati battono il tuning.
- Pre-flight check ha salvato 3 ore: scoprire il bug del doppio shift in un run da 20 minuti invece che da 3 ore.

## File salvati su disco

```text
checkpoints/
├── model_initial/                  (snapshot vergine, 957 MB)
├── best/                           (best Scenario A, epoca 2, 957 MB)
├── epoch_02/                       (duplicato di best/, ridondante)
├── scenario_B/
│   ├── best/                       (best Scenario B, epoca 5, 957 MB)
│   ├── last/                       (ultima epoca scenario B, 957 MB)
│   └── results_test.json
├── week2_flickr30k/
│   ├── best/                       (best Settimana 2, epoca 5, 957 MB)
│   ├── last/                       (ultima epoca settimana 2, 957 MB)

outputs/
├── learning_curves.png
├── test_predictions_baseline.json
├── bleu_results_baseline.json
├── scenario_B_history.json
├── week2_flickr30k_history.json
├── week2_eval_flickr8k.json        (NUOVO)
└── week2_eval_flickr30k.json       (NUOVO)
```

Totale occupato su disco: ~5-6 GB. Gestibile.

## Cosa rimane

### Settimana 3 — Upgrade encoder (CLIP ViT-L)

- Sostituire `google/vit-base-patch16-224-in21k` con `openai/clip-vit-large-patch14`
- Potenzialmente aggiungere proiezione lineare per match dimensionale (CLIP-L: 768 dim, GPT-2: 768 dim)
- Re-training su Flickr30k con stesso protocollo della Sett. 2
- Atteso BLEU-4 su test F8k: 22-28 (sopra Sett. 2, sotto SOTA 2018)
- Tempo stimato: ~5-7 ore (modello più grande)

### Settimana 4 — Report + slides + pulizia

- Scrittura report PDF: introduzione, metodo, esperimenti, risultati, discussione
- Slide di presentazione con grafici comparativi
- Repository finale rieseguibile: `README`, `requirements.txt`
- Pulizia: rimuovere checkpoint ridondanti, ordinare notebook

## Cose da fare prima di passare alla Settimana 3

- Cancellare `checkpoints/epoch_02/` (duplicato di `checkpoints/best/`) — risparmi 957 MB
- Cancellare `checkpoints/scenario_B/last/` (epoca 8 di Scenario B, non più utile) — risparmi 957 MB
- Cancellare `checkpoints/week2_flickr30k/last/` (epoca 8 di Sett. 2, non più utile) — risparmi 957 MB

- Aggiornare `PROJECT_PLAN.md`:
  - Tabella metriche: aggiungere i numeri della Sett. 2 misurati
  - Spuntare i task della Sett. 1 e 2 come completati

- Eseguire la cella 5 del notebook `09` (la tabella comparativa). Mancante.

## Considerazione strategica

Hai due settimane di lavoro davanti a te e un risultato già più che onesto. La Settimana 3 (CLIP) può portarti a numeri da paper accademico (BLEU-4 ~25). Ma anche se per qualche motivo non funzionasse e ti ritrovassi con la stessa performance, hai comunque un progetto eccellente da consegnare:

- 3 esperimenti rigorosi
- Confronto sistematico con baseline
- Pipeline riproducibile
- Bug debuggati onestamente
- Risultati nello stesso ordine di grandezza dello stato dell'arte 2014
