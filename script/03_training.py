"""
03_training.py
==============
Addestramento del modello di image captioning (Scenario A / baseline su Flickr8k).

Lo script:
  1. ricarica lo snapshot iniziale del modello salvato dallo script 02;
  2. ricostruisce i DataLoader di train e validation;
  3. esegue il loop di training per NUM_EPOCHS epoche, con:
       - optimizer AdamW + weight decay,
       - mixed precision (bfloat16) per ridurre memoria e tempo,
       - gradient clipping per la stabilita' dei transformer;
  4. valuta la loss su validation dopo ogni epoca e salva un checkpoint;
  5. traccia le learning curves (train vs val) e individua la "best epoch"
     (quella con la val loss minima), copiandola in checkpoints/best.

Il modello migliore selezionato qui sara' poi valutato dallo script 04.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW              # optimizer Adam con weight decay disaccoppiato
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    GPT2Tokenizer,
    ViTImageProcessor,
)
from tqdm.auto import tqdm                 # barra di avanzamento (versione auto: ok in terminale e notebook)
import time
import json

# ----------------------------------------------------------------------------
# 1. SETUP: RIPRODUCIBILITA' E DEVICE
# ----------------------------------------------------------------------------
# Fissiamo i seed sia di PyTorch sia di NumPy per la riproducibilita'.
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ----------------------------------------------------------------------------
# 2. PATH DEL PROGETTO
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "flickr8k"
IMAGES_DIR = DATA_DIR / "Images"
SPLIT_FILE = DATA_DIR / "captions_with_split.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
INITIAL_MODEL_DIR = CHECKPOINT_DIR / "model_initial"   # snapshot salvato dallo script 02

# ----------------------------------------------------------------------------
# 3. IPERPARAMETRI DI TRAINING (Scenario A)
# ----------------------------------------------------------------------------
NUM_EPOCHS = 8           # budget di epoche
LEARNING_RATE = 5e-5     # learning rate tipico per il fine-tuning di transformer
WEIGHT_DECAY = 0.01      # regolarizzazione L2 (tramite AdamW)
BATCH_SIZE = 32
NUM_WORKERS = 0          # Windows-friendly, come nello script 02
MAX_LENGTH = 30
GRAD_CLIP = 1.0          # soglia per il clipping della norma del gradiente

print(f"\n=== Iperparametri ===")
print(f"Epoche: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max length: {MAX_LENGTH}")
print(f"Gradient clipping: {GRAD_CLIP}")

# ----------------------------------------------------------------------------
# 4. CARICAMENTO DEL MODELLO DALLO SNAPSHOT INIZIALE
# ----------------------------------------------------------------------------
# Ripartiamo esattamente dallo stato salvato dallo script 02: cosi' il training
# inizia sempre dalla stessa configurazione (riproducibilita').
print(f"Caricamento modello da: {INITIAL_MODEL_DIR}")
model = VisionEncoderDecoderModel.from_pretrained(INITIAL_MODEL_DIR)
tokenizer = GPT2Tokenizer.from_pretrained(INITIAL_MODEL_DIR)
image_processor = ViTImageProcessor.from_pretrained(INITIAL_MODEL_DIR)

# Al ricaricamento, la configurazione pad_token va reimpostata a mano
# (non viene serializzata nel tokenizer salvato).
tokenizer.pad_token = tokenizer.eos_token

model = model.to(device)
print(f"Modello caricato su {device}")
print(f"Parametri: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")


# ----------------------------------------------------------------------------
# 5. DATASET (identico allo script 02)
# ----------------------------------------------------------------------------
# Ridefinito qui per rendere lo script autosufficiente. Per il dettaglio dei
# commenti vedere 02_model.py: immagine -> pixel_values normalizzati;
# didascalia -> token con BOS/EOS; padding mascherato a -100 nelle labels.
class Flickr8kCaptionsDataset(Dataset):
    def __init__(self, df_split, images_dir, image_processor, tokenizer, max_length=30):
        self.df = df_split.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Immagine -> tensore normalizzato (3, 224, 224).
        img_path = self.images_dir / row['image']
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)

        # Didascalia -> token con BOS/EOS, padding/troncamento a max_length.
        text = self.tokenizer.bos_token + " " + row['caption'] + " " + self.tokenizer.eos_token
        encoding = self.tokenizer(
            text, padding="max_length", max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Padding mascherato a -100 (ignorato dalla loss). Si usa l'attention_mask
        # e non pad_token_id perche' in GPT-2 pad ed EOS condividono lo stesso ID.
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ----------------------------------------------------------------------------
# 6. DATALOADER DI TRAIN E VALIDATION
# ----------------------------------------------------------------------------
df = pd.read_csv(SPLIT_FILE)
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']

train_dataset = Flickr8kCaptionsDataset(train_df, IMAGES_DIR, image_processor, tokenizer, MAX_LENGTH)
val_dataset = Flickr8kCaptionsDataset(val_df, IMAGES_DIR, image_processor, tokenizer, MAX_LENGTH)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,   # shuffle in training
    num_workers=NUM_WORKERS, pin_memory=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,    # ordine fisso in validation
    num_workers=NUM_WORKERS, pin_memory=True,
)

print(f"\nTrain: {len(train_dataset)} esempi, {len(train_loader)} batch")
print(f"Val:   {len(val_dataset)} esempi, {len(val_loader)} batch")

# ----------------------------------------------------------------------------
# 7. OPTIMIZER E MIXED PRECISION
# ----------------------------------------------------------------------------
# AdamW: variante di Adam con weight decay disaccoppiato, standard per i transformer.
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# Mixed precision in bfloat16: dimezza la memoria delle attivazioni e velocizza
# il calcolo sulle GPU moderne, senza i problemi di overflow di float16 (bf16 ha
# lo stesso range del float32). Se non supportato, restiamo in float32.
use_amp = torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_amp else torch.float32
print(f"Mixed precision (bfloat16): {'attiva' if use_amp else 'non disponibile'}")


# ----------------------------------------------------------------------------
# 8. FUNZIONI DI TRAINING E VALIDAZIONE
# ----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, epoch):
    """Esegue una epoca di training e ritorna la loss media."""
    model.train()                # abilita dropout e altri comportamenti di training
    total_loss = 0.0
    n_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoca {epoch} [train]", leave=False)
    for batch in pbar:
        # Trasferimento sulla GPU (non_blocking sfrutta pin_memory per essere asincrono).
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        optimizer.zero_grad()    # azzera i gradienti accumulati dal passo precedente

        # Forward in mixed precision: le operazioni interne usano bf16 dove conviene.
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss   # cross-entropy gia' calcolata dal modello

        # Backward: calcola i gradienti.
        loss.backward()

        # Gradient clipping: limita la norma del gradiente a GRAD_CLIP per evitare
        # aggiornamenti troppo grandi (instabilita' tipica nei transformer).
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()          # aggiorna i pesi

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})  # mostra la loss corrente sulla barra

    return total_loss / n_batches  # loss media sull'epoca


@torch.no_grad()                  # disabilita il calcolo dei gradienti: piu' veloce, meno memoria
def evaluate(model, loader, device):
    """Calcola la loss media sul set di validazione."""
    model.eval()                  # disabilita dropout: comportamento deterministico
    total_loss = 0.0
    n_batches = len(loader)

    for batch in tqdm(loader, desc="Validazione", leave=False):
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

        total_loss += loss.item()

    return total_loss / n_batches


print("Funzioni di train e validazione definite.")

# ----------------------------------------------------------------------------
# 9. LOOP DI TRAINING
# ----------------------------------------------------------------------------
# Teniamo traccia di loss e tempi per ogni epoca: serviranno per i grafici e
# vengono salvati su disco a ogni epoca (resilienza ai crash).
history = {
    'train_loss': [],
    'val_loss': [],
    'epoch_time_sec': [],
}

print(f"=== Inizio training: {NUM_EPOCHS} epoche ===\n")
overall_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()

    # Una epoca di training.
    train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

    # Validazione a fine epoca.
    val_loss = evaluate(model, val_loader, device)

    epoch_time = time.time() - epoch_start
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['epoch_time_sec'].append(epoch_time)

    print(f"Epoca {epoch:2d}/{NUM_EPOCHS} | "
          f"train loss: {train_loss:.4f} | "
          f"val loss: {val_loss:.4f} | "
          f"tempo: {epoch_time/60:.1f} min")

    # Salviamo un checkpoint per ogni epoca: cosi' potremo recuperare la
    # migliore a posteriori, anche se non e' l'ultima.
    ckpt_dir = CHECKPOINT_DIR / f"epoch_{epoch:02d}"
    model.save_pretrained(ckpt_dir)

    # Salviamo la history a ogni epoca, per non perderla in caso di crash.
    with open(CHECKPOINT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

total_time = time.time() - overall_start
print(f"\n=== Training completato in {total_time/60:.1f} minuti ===")
print(f"Loss finale - train: {history['train_loss'][-1]:.4f}, val: {history['val_loss'][-1]:.4f}")

# ----------------------------------------------------------------------------
# 10. LEARNING CURVES (train vs validation)
# ----------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Rileggiamo la history dal file (anche se e' gia' in memoria) per robustezza.
with open(CHECKPOINT_DIR / "history.json", "r") as f:
    history = json.load(f)

epochs = list(range(1, len(history['train_loss']) + 1))
# La "best epoch" e' quella con val loss minima: argmin + 1 (le epoche partono da 1).
best_epoch = int(np.argmin(history['val_loss'])) + 1
best_val_loss = min(history['val_loss'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Grafico 1: curve di train e validation ---
ax = axes[0]
ax.plot(epochs, history['train_loss'], 'o-', label='Train loss', color='tab:blue')
ax.plot(epochs, history['val_loss'], 's-', label='Val loss', color='tab:orange')
# Linea verticale e punto rosso sulla best epoch.
ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5,
           label=f'Best val loss (epoca {best_epoch})')
ax.scatter([best_epoch], [best_val_loss], color='red', s=120, zorder=5)
ax.set_xlabel('Epoca')
ax.set_ylabel('Loss')
ax.set_title('Learning curves: train vs validation')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Grafico 2: gap (train - val) per visualizzare l'overfitting ---
# Quando il gap diventa molto negativo (train scende, val no) il modello sta
# overfittando: e' il segnale per fermarsi / scegliere un'epoca precedente.
ax = axes[1]
gap = [t - v for t, v in zip(history['train_loss'], history['val_loss'])]
ax.plot(epochs, gap, 'o-', color='tab:purple')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Epoca')
ax.set_ylabel('Train loss - Val loss')
ax.set_title('Gap tra train e validation\n(piu negativo = piu overfitting)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
# Salviamo la figura per inserirla nel report.
plt.savefig(PROJECT_ROOT / "outputs" / "learning_curves.png", dpi=120, bbox_inches='tight')
plt.show()

# Riassunto numerico del training.
print("=== Riassunto training ===")
print(f"Migliore epoca:         {best_epoch} (val loss = {best_val_loss:.4f})")
print(f"Train loss alla best:   {history['train_loss'][best_epoch-1]:.4f}")
print(f"Train loss finale:      {history['train_loss'][-1]:.4f}")
print(f"Val loss finale:        {history['val_loss'][-1]:.4f}")
print(f"Gap train-val finale:   {history['train_loss'][-1] - history['val_loss'][-1]:.4f}  (negativo grande = overfitting)")
print(f"Tempo totale:           {sum(history['epoch_time_sec'])/60:.1f} min")
print(f"\n>>> Per inference useremo il checkpoint dell'epoca {best_epoch} <<<")

# ----------------------------------------------------------------------------
# 11. SELEZIONE E SALVATAGGIO DEL BEST CHECKPOINT
# ----------------------------------------------------------------------------
# Copiamo il checkpoint della best epoch in checkpoints/best, cosi' lo script 04
# carica sempre da un percorso fisso senza dover sapere quale fosse l'epoca migliore.
import shutil

best_epoch = int(np.argmin(history['val_loss'])) + 1
src_dir = CHECKPOINT_DIR / f"epoch_{best_epoch:02d}"   # checkpoint della best epoch
dst_dir = CHECKPOINT_DIR / "best"

# Se esiste gia' una cartella best, la rimuoviamo prima di ricopiare.
if dst_dir.exists():
    shutil.rmtree(dst_dir)
shutil.copytree(src_dir, dst_dir)

# Salviamo anche un file con le informazioni sulla selezione (utile per il report).
best_info = {
    "best_epoch": best_epoch,
    "best_val_loss": float(history['val_loss'][best_epoch - 1]),
    "best_train_loss": float(history['train_loss'][best_epoch - 1]),
    "all_epochs_val_loss": history['val_loss'],
    "all_epochs_train_loss": history['train_loss'],
}
with open(CHECKPOINT_DIR / "best_info.json", "w") as f:
    json.dump(best_info, f, indent=2)

print(f"✅ Best model copiato in: {dst_dir}")
print(f"   Epoca: {best_epoch}")
print(f"   Val loss: {best_info['best_val_loss']:.4f}")