"""
08_training_flickr30k.py
========================
Training della Settimana 2: stessa identica ricetta dello Scenario B (script 05),
ma su Flickr30k invece di Flickr8k.

Idea dell'esperimento: cambiare UNA sola variabile (il dataset, ~4x piu' grande)
mantenendo tutto il resto invariato, per attribuire il miglioramento di BLEU
esclusivamente alla maggiore quantita' di dati di training.

Differenze rispetto allo script 05 (Scenario B):
  - Path dei dati: data/flickr30k, cartella immagini 'flickr30k_images';
  - Checkpoint salvati in checkpoints/week2_flickr30k;
  - NUM_EPOCHS ridotto (10 invece di 25): con 4x piu' dati servono meno epoche;
  - WARMUP_RATIO = 1/10 (coerente col nuovo budget di epoche);
  - VAL_BLEU_SUBSET_SIZE = 300 (il val di Flickr30k e' piu' grande, 1906 immagini);
  - Classe Dataset rinominata Flickr30kCaptionsDataset (logica identica).
Augmentation, scheduler, label smoothing, early stopping su BLEU: tutto invariato.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms as T
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    GPT2Tokenizer,
    ViTImageProcessor,
)
import sacrebleu
from tqdm.auto import tqdm
import time
import json
import math
import shutil
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1. SETUP: RIPRODUCIBILITA' E DEVICE
# ----------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ----------------------------------------------------------------------------
# 2. PATH DEL PROGETTO  <-- DIFFERENZA CHIAVE RISPETTO ALLO SCRIPT 05
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "flickr30k"          # <- Flickr30k invece di Flickr8k
IMAGES_DIR = DATA_DIR / "flickr30k_images"              # <- nome cartella immagini diverso
SPLIT_FILE = DATA_DIR / "captions_with_split.csv"       # prodotto dallo script 07
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
INITIAL_MODEL_DIR = CHECKPOINT_DIR / "model_initial"    # stesso snapshot vergine di A e B
WEEK2_DIR = CHECKPOINT_DIR / "week2_flickr30k"          # <- cartella dedicata ai checkpoint
WEEK2_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# 3. IPERPARAMETRI (quasi identici allo Scenario B, con le poche differenze indicate)
# ----------------------------------------------------------------------------
NUM_EPOCHS = 10                  # <- budget ridotto: con 4x piu' dati servono meno epoche
EARLY_STOP_PATIENCE = 3
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 1/10              # <- coerente col nuovo budget (1 epoca su 10 di warmup)
BATCH_SIZE = 32
NUM_WORKERS = 0
MAX_LENGTH = 30
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
VAL_BLEU_SUBSET_SIZE = 300       # <- val di Flickr30k piu' grande (1906 img): stima piu' stabile

print(f"\n=== Iperparametri Settimana 2 ===")
print(f"Dataset:                Flickr30k")
print(f"Budget epoche:          {NUM_EPOCHS}")
print(f"Early stopping:         {EARLY_STOP_PATIENCE} epoche senza miglioramento BLEU")
print(f"Peak learning rate:     {LEARNING_RATE}")
print(f"Warmup ratio:           {WARMUP_RATIO:.3f}")
print(f"Label smoothing:        {LABEL_SMOOTHING}")
print(f"Batch size:             {BATCH_SIZE}")
print(f"Val BLEU subset size:   {VAL_BLEU_SUBSET_SIZE}")

# ----------------------------------------------------------------------------
# 4. TRASFORMAZIONI IMMAGINI (identiche allo Scenario B)
# ----------------------------------------------------------------------------
# Statistiche di normalizzazione di ImageNet (le stesse su cui e' pre-addestrato ViT).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training con augmentation (riduce overfitting): crop casuale, flip, color jitter.
train_transforms = T.Compose([
    T.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validazione/test deterministica (nessuna augmentation).
eval_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ----------------------------------------------------------------------------
# 5. DATASET (logica identica al Flickr8kCaptionsDataset dello script 05)
# ----------------------------------------------------------------------------
class Flickr30kCaptionsDataset(Dataset):
    """Dataset PyTorch per Flickr30k. Identico al Flickr8kCaptionsDataset dello
    Scenario B: cambia solo il nome (e dall'esterno la cartella delle immagini)."""

    def __init__(self, df_split, images_dir, transforms_fn, tokenizer, max_length=30):
        self.df = df_split.reset_index(drop=True)
        self.images_dir = images_dir
        self.transforms = transforms_fn
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Immagine -> trasformazioni (con augmentation se train).
        img_path = self.images_dir / row['image']
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transforms(image)

        # Didascalia -> token con BOS/EOS, padding/troncamento, padding mascherato a -100.
        caption = row['caption']
        text = self.tokenizer.bos_token + " " + caption + " " + self.tokenizer.eos_token
        encoding = self.tokenizer(
            text, padding="max_length", max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100   # ignora padding nella loss (mask, non pad_token_id)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_filename": row['image'],
            "caption_text": caption,
        }


# ----------------------------------------------------------------------------
# 6. CARICAMENTO MODELLO E DATI
# ----------------------------------------------------------------------------
# Ripartiamo dallo STESSO snapshot vergine usato in A e B: cosi' l'unica variabile
# che cambia rispetto allo Scenario B e' il dataset di training.
print("Caricamento modello iniziale (snapshot vergine, mai addestrato)...")
model = VisionEncoderDecoderModel.from_pretrained(INITIAL_MODEL_DIR)
tokenizer = GPT2Tokenizer.from_pretrained(INITIAL_MODEL_DIR)
image_processor = ViTImageProcessor.from_pretrained(INITIAL_MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token
model = model.to(device)
print(f"Modello su {device}, {sum(p.numel() for p in model.parameters())/1e6:.1f}M parametri")

# Carichiamo lo split di Flickr30k (prodotto dallo script 07).
df = pd.read_csv(SPLIT_FILE)
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']

train_dataset = Flickr30kCaptionsDataset(train_df, IMAGES_DIR, train_transforms, tokenizer, MAX_LENGTH)
val_dataset = Flickr30kCaptionsDataset(val_df, IMAGES_DIR, eval_transforms, tokenizer, MAX_LENGTH)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)

print(f"\nTrain: {len(train_dataset)} esempi, {len(train_loader)} batch (con augmentation)")
print(f"Val:   {len(val_dataset)} esempi, {len(val_loader)} batch (senza augmentation)")

# ----------------------------------------------------------------------------
# 7. OPTIMIZER E LR SCHEDULER (warmup lineare + cosine decay, come script 05)
# ----------------------------------------------------------------------------
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# Numero totale di step e step di warmup (qui ricalcolati sui nuovi NUM_EPOCHS/loader).
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)

def lr_lambda(current_step: int):
    # Warmup lineare 0->1, poi cosine decay 1->0 (stessa forma dello script 05).
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)
    else:
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda)

print(f"Total training steps: {total_steps}")
print(f"Warmup steps:         {warmup_steps} ({warmup_steps/total_steps*100:.1f}% del totale)")

# ----------------------------------------------------------------------------
# 8. LOSS CON LABEL SMOOTHING E MIXED PRECISION
# ----------------------------------------------------------------------------
# Loss definita a mano per applicare il label smoothing (HF non lo fa internamente).
loss_fn = nn.CrossEntropyLoss(
    ignore_index=-100,
    label_smoothing=LABEL_SMOOTHING,
)
print(f"Loss: CrossEntropyLoss con label_smoothing={LABEL_SMOOTHING}")

# Mixed precision bfloat16 (meno memoria, piu' veloce, niente overflow del fp16).
use_amp = torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_amp else torch.float32
print(f"Mixed precision: {'attiva (bfloat16)' if use_amp else 'non disponibile'}")

# Grafico di verifica della curva di learning rate.
fig, ax = plt.subplots(figsize=(10, 4))
test_steps = list(range(0, total_steps + 1, max(1, total_steps // 200)))
test_lrs = [LEARNING_RATE * lr_lambda(s) for s in test_steps]
ax.plot(test_steps, test_lrs)
ax.axvline(warmup_steps, color='red', linestyle='--', alpha=0.5,
           label=f'Fine warmup (step {warmup_steps})')
ax.set_xlabel('Step')
ax.set_ylabel('Learning rate')
ax.set_title(f'LR schedule: warmup + cosine decay (peak={LEARNING_RATE}, total={total_steps} steps)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 9. FUNZIONI DI LOSS, TRAINING E VALIDAZIONE (identiche allo script 05)
# ----------------------------------------------------------------------------
def compute_loss_with_smoothing(outputs, labels, loss_fn):
    """Loss con label smoothing. VisionEncoderDecoderModel shifta gia' i labels
    internamente, quindi confrontiamo logits e labels direttamente, senza shift
    manuale (rifarlo a mano darebbe BLEU vicino a zero)."""
    logits = outputs.logits
    loss = loss_fn(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )
    return loss


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, epoch):
    """Una epoca di training; ritorna la loss media."""
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoca {epoch} [train]", leave=False)
    for batch in pbar:
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = compute_loss_with_smoothing(outputs, labels, loss_fn)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # clipping per stabilita'
        optimizer.step()
        scheduler.step()  # lo scheduler avanza ad ogni step

        total_loss += loss.item()
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'lr': f'{current_lr:.2e}'})

    return total_loss / n_batches


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device):
    """Loss media di validazione (stessa loss con smoothing del training)."""
    model.eval()
    total_loss = 0.0
    n_batches = len(loader)

    for batch in tqdm(loader, desc="Val [loss]", leave=False):
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = compute_loss_with_smoothing(outputs, labels, loss_fn)

        total_loss += loss.item()

    return total_loss / n_batches


@torch.no_grad()
def generate_caption_batch(model, pixel_values_batch, tokenizer, device):
    """Genera didascalie per un batch. Workaround GPT-2 BOS/EOS: decoder_input_ids=[[BOS,BOS]]."""
    batch_size = pixel_values_batch.size(0)
    decoder_input_ids = torch.tensor(
        [[tokenizer.bos_token_id, tokenizer.bos_token_id]] * batch_size,
        device=device,
    )
    output_ids = model.generate(
        pixel_values=pixel_values_batch,
        decoder_input_ids=decoder_input_ids,
        max_length=MAX_LENGTH,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    return [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in output_ids]


@torch.no_grad()
def evaluate_bleu_subset(model, val_df, images_dir, eval_transforms,
                         tokenizer, device, n_samples=300, batch_size=16):
    """BLEU su un sottoinsieme di n_samples immagini uniche del val set
    (stima rapida per selezione del best e early stopping)."""
    model.eval()
    # Campionamento deterministico (seed fisso) di immagini uniche.
    unique_imgs = val_df['image'].unique()
    rng = np.random.default_rng(seed=42)
    sampled_imgs = rng.choice(unique_imgs, size=min(n_samples, len(unique_imgs)), replace=False)

    generated_captions = []
    references_per_image = []

    for batch_start in tqdm(range(0, len(sampled_imgs), batch_size), desc="Val [BLEU]", leave=False):
        batch_imgs = sampled_imgs[batch_start:batch_start + batch_size]
        pixel_values_list = []
        for img_name in batch_imgs:
            image = Image.open(images_dir / img_name).convert("RGB")
            pixel_values_list.append(eval_transforms(image))
        pixel_values_batch = torch.stack(pixel_values_list).to(device)

        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            batch_captions = generate_caption_batch(model, pixel_values_batch, tokenizer, device)

        for img_name, gen_cap in zip(batch_imgs, batch_captions):
            generated_captions.append(gen_cap)
            refs = val_df[val_df['image'] == img_name]['caption'].tolist()
            references_per_image.append(refs)

    # Reference trasposte per sacrebleu (vedi script 04 per il dettaglio del formato).
    references_transposed = [list(refs) for refs in zip(*references_per_image)]
    bleu = sacrebleu.corpus_bleu(generated_captions, references_transposed)

    return {
        'BLEU-1': bleu.precisions[0],
        'BLEU-2': bleu.precisions[1],
        'BLEU-3': bleu.precisions[2],
        'BLEU-4': bleu.precisions[3],
        'BLEU_corpus': bleu.score,
        'n_samples': len(sampled_imgs),
    }


print("Funzioni di train e validation definite.")
print(f"  evaluate_bleu_subset valuta su {VAL_BLEU_SUBSET_SIZE} immagini del val set")

# ----------------------------------------------------------------------------
# 10. LOOP DI TRAINING CON EARLY STOPPING SU BLEU (identico allo script 05)
# ----------------------------------------------------------------------------
history = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'val_bleu_1': [],
    'val_bleu_4': [],
    'val_bleu_corpus': [],
    'lr_at_end_of_epoch': [],
    'epoch_time_sec': [],
}

best_bleu4 = -1.0
best_epoch = 0
epochs_since_improvement = 0
HISTORY_FILE = OUTPUTS_DIR / "week2_flickr30k_history.json"
BEST_DIR = WEEK2_DIR / "best"   # miglior BLEU-4
LAST_DIR = WEEK2_DIR / "last"   # ultimo stato (antinfortunistica)

print(f"=== Avvio training Settimana 2 (Flickr30k) ===")
print(f"Budget: {NUM_EPOCHS} epoche, early stopping con patience {EARLY_STOP_PATIENCE}")
print(f"Stima conservativa tempo totale: 5-6 ore (puo' fermarsi prima per early stop)")
print(f"Best model in: {BEST_DIR}")
print(f"Last model in: {LAST_DIR}")
print()

overall_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()

    # Train + validation loss + validation BLEU.
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, epoch)
    val_loss = evaluate_loss(model, val_loader, loss_fn, device)
    val_bleu = evaluate_bleu_subset(
        model, val_df, IMAGES_DIR, eval_transforms,
        tokenizer, device, n_samples=VAL_BLEU_SUBSET_SIZE, batch_size=16,
    )

    epoch_time = time.time() - epoch_start
    current_lr = scheduler.get_last_lr()[0]

    # Aggiorna e salva la history (resilienza ai crash).
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_bleu_1'].append(val_bleu['BLEU-1'])
    history['val_bleu_4'].append(val_bleu['BLEU-4'])
    history['val_bleu_corpus'].append(val_bleu['BLEU_corpus'])
    history['lr_at_end_of_epoch'].append(current_lr)
    history['epoch_time_sec'].append(epoch_time)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

    # Salva sempre il "last".
    if LAST_DIR.exists():
        shutil.rmtree(LAST_DIR)
    model.save_pretrained(LAST_DIR)

    # Salva il "best" solo se il BLEU-4 migliora; altrimenti incrementa il contatore di stop.
    is_best = val_bleu['BLEU-4'] > best_bleu4
    if is_best:
        best_bleu4 = val_bleu['BLEU-4']
        best_epoch = epoch
        epochs_since_improvement = 0
        if BEST_DIR.exists():
            shutil.rmtree(BEST_DIR)
        model.save_pretrained(BEST_DIR)
        marker = "[BEST!]"
    else:
        epochs_since_improvement += 1
        marker = f"(no improv x{epochs_since_improvement})"

    print(f"Epoca {epoch:2d}/{NUM_EPOCHS} | "
          f"train: {train_loss:.3f} | val: {val_loss:.3f} | "
          f"BLEU-1: {val_bleu['BLEU-1']:.2f} | "
          f"BLEU-4: {val_bleu['BLEU-4']:.2f} | "
          f"lr: {current_lr:.2e} | "
          f"tempo: {epoch_time/60:.1f}min | "
          f"{marker}")

    # Early stopping su BLEU-4.
    if epochs_since_improvement >= EARLY_STOP_PATIENCE:
        print(f"\n>>> Early stopping: BLEU-4 non migliora da {EARLY_STOP_PATIENCE} epoche consecutive")
        break

total_time = time.time() - overall_start
print(f"\n=== Training completato in {total_time/60:.1f} minuti ({total_time/3600:.1f} ore) ===")
print(f"Best epoca:  {best_epoch}")
print(f"Best BLEU-4: {best_bleu4:.2f}")
print(f"Best model salvato in: {BEST_DIR}")
print(f"Last model salvato in: {LAST_DIR}")