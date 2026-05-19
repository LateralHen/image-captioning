import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    GPT2Tokenizer,
    ViTImageProcessor,
)

# Riproducibilita'
torch.manual_seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Path del progetto
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "flickr8k"
IMAGES_DIR = DATA_DIR / "Images"
SPLIT_FILE = DATA_DIR / "captions_with_split.csv"

# Iperparametri (decisi nel notebook 01)
MAX_LENGTH = 30
BATCH_SIZE = 32  # provvisorio, lo ricalibriamo dopo aver visto la VRAM usata

# Modelli pre-addestrati (riferimenti standard)
ENCODER_NAME = "google/vit-base-patch16-224-in21k"
DECODER_NAME = "gpt2"

print(f"\nEncoder: {ENCODER_NAME}")
print(f"Decoder: {DECODER_NAME}")
print(f"Max length: {MAX_LENGTH} token")

# VisionEncoderDecoderModel: combina automaticamente encoder ViT + decoder GPT-2
# aggiungendo i layer di cross-attention nel decoder
print("Caricamento del modello combinato (prima volta: download ~1.3 GB)...")
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    ENCODER_NAME,
    DECODER_NAME,
)

# Tokenizer e image processor
tokenizer = GPT2Tokenizer.from_pretrained(DECODER_NAME)
image_processor = ViTImageProcessor.from_pretrained(ENCODER_NAME)

# Configurazione del decoder GPT-2:
# - GPT-2 non ha pad_token nativo. Usiamo eos_token come pad.
tokenizer.pad_token = tokenizer.eos_token

# Diciamo al modello quali token usare per:
# - decoder_start_token_id: cosa generare per primo (BOS)
# - pad_token_id: token da ignorare nella loss
# - eos_token_id: token che segnala "fine sequenza"
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Iperparametri di generazione (li useremo solo in fase di inference, ma settiamoli ora)
model.config.max_length = MAX_LENGTH
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.num_beams = 4

# Stampa il numero di parametri
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nParametri totali: {total_params/1e6:.1f}M")
print(f"Parametri trainabili: {trainable_params/1e6:.1f}M")

# Sposta su GPU
model = model.to(device)
print(f"Modello su: {next(model.parameters()).device}")

class Flickr8kCaptionsDataset(Dataset):
    """Dataset PyTorch per Flickr8k.
    
    Ogni elemento e' una coppia (immagine, didascalia) gia' processata.
    """
    
    def __init__(self, df_split, images_dir, image_processor, tokenizer, max_length=30):
        """
        Args:
            df_split: DataFrame con colonne 'image' e 'caption' (gia' filtrato per split)
            images_dir: Path della cartella che contiene i .jpg
            image_processor: ViTImageProcessor di HuggingFace
            tokenizer: GPT2Tokenizer di HuggingFace
            max_length: lunghezza massima della sequenza tokenizzata
        """
        self.df = df_split.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Carica e processa l'immagine
        img_path = self.images_dir / row['image']
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.squeeze(0)  # rimuovi la dim batch fittizia
        
        # 2. Tokenizza la didascalia
        # Aggiungiamo BOS all'inizio e EOS alla fine, paddiamo a max_length
        caption = row['caption']
        text = self.tokenizer.bos_token + " " + caption + " " + self.tokenizer.eos_token
        
        encoding = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)        # (max_length,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (max_length,) 1=token reale, 0=pad

        # Mascheriamo SOLO il padding (non tutti gli <|endoftext|>):
        # usiamo attention_mask invece di confrontare con pad_token_id
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


# Carichiamo lo split che abbiamo salvato nel notebook 01
df = pd.read_csv(SPLIT_FILE)
print(f"Dataset totale: {len(df)} esempi")
print(f"Distribuzione split:\n{df['split'].value_counts()}\n")

# Crea i tre dataset
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
test_df = df[df['split'] == 'test']

train_dataset = Flickr8kCaptionsDataset(train_df, IMAGES_DIR, image_processor, tokenizer, MAX_LENGTH)
val_dataset = Flickr8kCaptionsDataset(val_df, IMAGES_DIR, image_processor, tokenizer, MAX_LENGTH)
test_dataset = Flickr8kCaptionsDataset(test_df, IMAGES_DIR, image_processor, tokenizer, MAX_LENGTH)

print(f"Train: {len(train_dataset)} esempi")
print(f"Val:   {len(val_dataset)} esempi")
print(f"Test:  {len(test_dataset)} esempi")

# Estraiamo un singolo esempio e verifichiamo le shape
sample = train_dataset[0]
print("Forma di un singolo esempio:")
print(f"  pixel_values: {sample['pixel_values'].shape}, dtype={sample['pixel_values'].dtype}")
print(f"  labels:       {sample['labels'].shape}, dtype={sample['labels'].dtype}")
print()
print(f"  Range pixel_values: [{sample['pixel_values'].min():.3f}, {sample['pixel_values'].max():.3f}]")
print(f"  Primi 15 token labels: {sample['labels'][:15].tolist()}")
print(f"  Ultimi 15 token labels (-100 = pad ignorato): {sample['labels'][-15:].tolist()}")
print()

# Decodifichiamo i token (sostituendo -100 con pad per leggibilita')
labels_for_decode = sample['labels'].clone()
labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
decoded = tokenizer.decode(labels_for_decode, skip_special_tokens=False)
print(f"  Didascalia decodificata: {decoded!r}")

NUM_WORKERS = 0  # adatta in base al tuo PC; 4 e' sicuro per quasi tutti

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,           # mescola ad ogni epoca
    num_workers=NUM_WORKERS,
    pin_memory=True,
    #persistent_workers=True,  # tieni i worker vivi tra le epoche (no re-spawn ogni volta)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,          # per la validazione l'ordine non conta
    num_workers=NUM_WORKERS,
    pin_memory=True,
    #persistent_workers=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    #persistent_workers=True,
)

print(f"Batch size: {BATCH_SIZE}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
print(f"Test batches:  {len(test_loader)}")

import time

# Estraiamo un batch
print("Caricamento del primo batch (i worker si spawnano ora, puo' richiedere ~10s)...")
t0 = time.time()
batch = next(iter(train_loader))
elapsed = time.time() - t0
print(f"Tempo: {elapsed:.1f}s")
print()
print(f"Forme del batch:")
print(f"  pixel_values: {batch['pixel_values'].shape}")
print(f"  labels:       {batch['labels'].shape}")
print()
print(f"Memoria stimata di un batch: {(batch['pixel_values'].element_size() * batch['pixel_values'].nelement() + batch['labels'].element_size() * batch['labels'].nelement()) / 1e6:.1f} MB")

import time

# Spostiamo il batch sulla GPU
pixel_values = batch['pixel_values'].to(device)
labels = batch['labels'].to(device)

print(f"Input pixel_values shape: {pixel_values.shape}")
print(f"Input labels shape: {labels.shape}")
print()

# Misuriamo VRAM PRIMA del forward
torch.cuda.empty_cache()
mem_before = torch.cuda.memory_allocated() / 1e9
print(f"VRAM occupata prima del forward: {mem_before:.2f} GB")

# === Forward pass ===
print("\nForward pass...")
t0 = time.time()
outputs = model(pixel_values=pixel_values, labels=labels)
torch.cuda.synchronize()  # aspetta che la GPU finisca davvero
forward_time = time.time() - t0

loss = outputs.loss
logits = outputs.logits

print(f"  Tempo forward: {forward_time*1000:.0f} ms")
print(f"  Loss: {loss.item():.4f}")
print(f"  Logits shape: {logits.shape}  (batch, seq_len, vocab_size)")

mem_after_fwd = torch.cuda.memory_allocated() / 1e9
print(f"  VRAM dopo forward: {mem_after_fwd:.2f} GB")

# === Backward pass ===
print("\nBackward pass...")
t0 = time.time()
loss.backward()
torch.cuda.synchronize()
backward_time = time.time() - t0
print(f"  Tempo backward: {backward_time*1000:.0f} ms")

mem_after_bwd = torch.cuda.memory_allocated() / 1e9
print(f"  VRAM dopo backward: {mem_after_bwd:.2f} GB (modello + attivazioni + gradienti)")

# Pulisci i gradienti per non lasciare residui in memoria
model.zero_grad()
torch.cuda.empty_cache()

# === Sanity check ===
import math
expected_loss_random = math.log(50257)
print()
print("=== Sanity check ===")
print(f"  Loss e' un numero finito: {torch.isfinite(loss).item()}")
print(f"  Loss > 0: {loss.item() > 0}")
print(f"  Loss attesa per modello casuale: ~{expected_loss_random:.2f}  (cross-entropy uniforme su 50257 token)")
print(f"  La nostra loss: {loss.item():.4f}")

import json

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Salva il modello (architettura + pesi)
INITIAL_MODEL_DIR = CHECKPOINT_DIR / "model_initial"
print(f"Salvataggio modello in {INITIAL_MODEL_DIR}...")
model.save_pretrained(INITIAL_MODEL_DIR)

# Salva tokenizer e image_processor (utili per inference)
tokenizer.save_pretrained(INITIAL_MODEL_DIR)
image_processor.save_pretrained(INITIAL_MODEL_DIR)

# Salva un piccolo file di metadati con gli iperparametri
metadata = {
    "encoder_name": ENCODER_NAME,
    "decoder_name": DECODER_NAME,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "vocab_size": tokenizer.vocab_size,
    "total_params_M": round(total_params / 1e6, 2),
    "trainable_params_M": round(trainable_params / 1e6, 2),
    "initial_loss": round(loss.item(), 4),
}

with open(CHECKPOINT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nMetadata salvata in {CHECKPOINT_DIR / 'metadata.json'}:")
print(json.dumps(metadata, indent=2))

# Lista dei file salvati
print(f"\nFile salvati in {INITIAL_MODEL_DIR}:")
for f in sorted(INITIAL_MODEL_DIR.iterdir()):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name:40s} {size_mb:>8.1f} MB")