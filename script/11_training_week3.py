import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    VisionEncoderDecoderConfig,
    CLIPVisionModel,
    CLIPImageProcessor,
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
)
import sacrebleu
from tqdm import tqdm
import time
import json
import math
import shutil
import gc

# Riproducibilita'
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Path
PROJECT_ROOT = Path.home() / "Documents/dev/image-cationing"
F30K_DIR = PROJECT_ROOT / "data" / "flickr30k"
F30K_IMAGES_DIR = F30K_DIR / "flickr30k_images"      # o "images", verifichiamo dopo
F30K_SPLIT_FILE = F30K_DIR / "captions_with_split.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
WEEK3_DIR = CHECKPOINT_DIR / "week3_clip"
WEEK3_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Verifica path
print(f"\nPath check:")
print(f"  Flickr30k split CSV: {F30K_SPLIT_FILE.exists()}")
print(f"  Flickr30k images:    {F30K_IMAGES_DIR.exists()}")

# Iperparametri (da week3_config.json)
CLIP_NAME = "openai/clip-vit-large-patch14"
GPT2_NAME = "gpt2"
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM_STEPS
NUM_EPOCHS = 25
EARLY_STOP_PATIENCE = 3
LEARNING_RATE = 5e-5
WARMUP_RATIO = 1/25
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
MAX_LENGTH = 30
NUM_WORKERS = 4
VAL_BLEU_SUBSET_SIZE = 200

print(f"\n=== Iperparametri ===")
print(f"Encoder:          {CLIP_NAME}")
print(f"Decoder:          {GPT2_NAME}")
print(f"Batch (pratico):  {BATCH_SIZE}")
print(f"Grad accumulation:{GRAD_ACCUM_STEPS}")
print(f"Batch (effective):{EFFECTIVE_BATCH}")
print(f"Epoche (budget):  {NUM_EPOCHS}")
print(f"Early stop:       patience {EARLY_STOP_PATIENCE} su BLEU-4")
print(f"Peak LR:          {LEARNING_RATE}")
print(f"Label smoothing:  {LABEL_SMOOTHING}")

print("Contenuto della cartella flickr30k:")
for item in sorted(F30K_DIR.iterdir()):
    if item.is_dir():
        n_files = len(list(item.iterdir()))
        print(f"  [DIR]  {item.name}/  ({n_files} elementi)")
    else:
        size_mb = item.stat().st_size / 1e6
        print(f"  [FILE] {item.name}  ({size_mb:.1f} MB)")

LAST_DIR_CHECK = CHECKPOINT_DIR / "week3_clip" / "last"

if LAST_DIR_CHECK.exists() and any(LAST_DIR_CHECK.iterdir()):
    print(f"Trovato checkpoint last/, riprendo da li'...")
    model = VisionEncoderDecoderModel.from_pretrained(LAST_DIR_CHECK)
    encoder = decoder = None
    print(f"Modello caricato da {LAST_DIR_CHECK}")
else:
    print("Costruzione modello CLIP-L + GPT-2 small da zero...")
    encoder = CLIPVisionModel.from_pretrained(CLIP_NAME)
    decoder_config = GPT2Config.from_pretrained(GPT2_NAME)
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    decoder = GPT2LMHeadModel.from_pretrained(GPT2_NAME, config=decoder_config)
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        encoder.config, decoder.config
    )
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)

# Tokenizer (lo stesso degli scenari precedenti)
tokenizer = GPT2Tokenizer.from_pretrained(GPT2_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Config: token speciali, lunghezza, beam search per inference
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.max_length = MAX_LENGTH
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.num_beams = 4

# Conta parametri
n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nParametri totali:     {n_total/1e6:.1f}M")
print(f"Parametri trainabili: {n_train/1e6:.1f}M")
print(f"Encoder hidden_size:  {model.config.encoder.hidden_size}")
print(f"Decoder hidden_size:  {model.config.decoder.hidden_size}")

# === IMPORTANTE: estraiamo la normalizzazione corretta di CLIP ===
clip_processor = CLIPImageProcessor.from_pretrained(CLIP_NAME)
CLIP_MEAN = clip_processor.image_mean
CLIP_STD = clip_processor.image_std
print(f"\nNormalizzazione CLIP (da usare in torchvision):")
print(f"  mean: {CLIP_MEAN}")
print(f"  std:  {CLIP_STD}")
print(f"\nConfronto con ImageNet standard (vecchio):")
print(f"  mean: [0.485, 0.456, 0.406]")
print(f"  std:  [0.229, 0.224, 0.225]")

# Cleanup riferimenti che non ci servono piu'
if encoder is not None:
    del encoder, decoder
gc.collect()

# === Trasformazioni con normalizzazione di CLIP ===
train_transforms = T.Compose([
    T.RandomResizedCrop(
        size=224,
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1),
    ),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),  # <-- usa la normalizzazione CLIP, non ImageNet
])

eval_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),  # <-- idem
])


# === Dataset class (identica a settimana 2, usa la transform che le passi) ===
class FlickrCaptionsDataset(Dataset):
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
        img_path = self.images_dir / row['image']
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transforms(image)
        
        text = self.tokenizer.bos_token + " " + row['caption'] + " " + self.tokenizer.eos_token
        encoding = self.tokenizer(
            text, padding="max_length", max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_filename": row['image'],
            "caption_text": row['caption'],
        }


# === Carico split Flickr30k ===
df = pd.read_csv(F30K_SPLIT_FILE)
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']

train_dataset = FlickrCaptionsDataset(train_df, F30K_IMAGES_DIR, train_transforms, tokenizer, MAX_LENGTH)
val_dataset = FlickrCaptionsDataset(val_df, F30K_IMAGES_DIR, eval_transforms, tokenizer, MAX_LENGTH)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True,
)

print(f"Train: {len(train_dataset)} esempi, {len(train_loader)} batch (con augmentation)")
print(f"Val:   {len(val_dataset)} esempi, {len(val_loader)} batch (senza augmentation)")

# Sanity check su un esempio
sample = train_dataset[0]
print(f"\nSanity check (un esempio):")
print(f"  pixel_values shape: {sample['pixel_values'].shape}")
print(f"  pixel_values range: [{sample['pixel_values'].min():.2f}, {sample['pixel_values'].max():.2f}]")
print(f"  labels shape:       {sample['labels'].shape}")
print(f"  caption:            {sample['caption_text']!r}")
print(f"  filename:           {sample['image_filename']}")

# === Optimizer ===
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# === LR Scheduler ===
# IMPORTANTE: con gradient accumulation, optimizer.step() viene chiamato
# ogni GRAD_ACCUM_STEPS batch, non ogni batch. Quindi il numero "vero" di
# step di optimizer e' total_batches / GRAD_ACCUM_STEPS.
n_optimizer_steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
total_optimizer_steps = n_optimizer_steps_per_epoch * NUM_EPOCHS
warmup_steps = int(WARMUP_RATIO * total_optimizer_steps)

def lr_lambda(current_step: int):
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)
    progress = (current_step - warmup_steps) / max(1, total_optimizer_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda)

print(f"Optimizer steps per epoca: {n_optimizer_steps_per_epoch}")
print(f"Optimizer steps totali:    {total_optimizer_steps}")
print(f"Warmup steps:              {warmup_steps} ({warmup_steps/total_optimizer_steps*100:.1f}%)")

# === Loss con label smoothing ===
loss_fn = nn.CrossEntropyLoss(
    ignore_index=-100,
    label_smoothing=LABEL_SMOOTHING,
)
print(f"\nLoss: CrossEntropyLoss con label_smoothing={LABEL_SMOOTHING}")

# === Mixed precision ===
use_amp = torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_amp else torch.float32
print(f"Mixed precision: {'bfloat16' if use_amp else 'fp32 (bf16 non disponibile)'}")

# Sposto il modello su GPU
model = model.to(device)
print(f"\nModello su {device}")
print(f"VRAM occupata: {torch.cuda.memory_allocated()/1e9:.2f} GB")

def compute_loss_with_smoothing(outputs, labels, loss_fn):
    """Calcola la loss con label smoothing usando i logits di HF."""
    logits = outputs.logits
    loss = loss_fn(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )
    return loss


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, epoch,
                    grad_accum_steps):
    """Training con gradient accumulation.
    
    A ogni `grad_accum_steps` batch facciamo un singolo optimizer.step(),
    cosi' otteniamo un effective batch = batch_size * grad_accum_steps.
    """
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc=f"Epoca {epoch} [train]", leave=False)
    for step, batch in enumerate(pbar):
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = compute_loss_with_smoothing(outputs, labels, loss_fn)
        
        # Dividiamo la loss per accumulation steps cosi' il gradiente
        # complessivo equivale a quello di un batch grande "vero"
        loss_scaled = loss / grad_accum_steps
        loss_scaled.backward()
        
        # Optimizer.step() solo ogni grad_accum_steps batch
        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()  # logghiamo la loss non scalata
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'lr': f'{current_lr:.2e}'})
    
    # Gestione caso "ultimi batch < grad_accum_steps" (se train_loader non e' multiplo esatto)
    if n_batches % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device):
    """Validation: ritorna la loss media."""
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
    """Generazione batch con beam search (stesso workaround GPT-2 BOS/EOS della settimana 2)."""
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
                         tokenizer, device, n_samples=200, batch_size=16):
    """BLEU su sottoinsieme val per selezione del best checkpoint."""
    model.eval()
    unique_imgs = val_df['image'].unique()
    rng = np.random.default_rng(seed=42)
    sampled_imgs = rng.choice(unique_imgs, size=min(n_samples, len(unique_imgs)), replace=False)
    
    generated_captions = []
    references_per_image = []
    
    pbar = tqdm(range(0, len(sampled_imgs), batch_size), desc="Val [BLEU]", leave=False)
    for batch_start in pbar:
        batch_imgs = sampled_imgs[batch_start:batch_start + batch_size]
        pixel_values_list = [eval_transforms(Image.open(images_dir / img).convert("RGB"))
                             for img in batch_imgs]
        pixel_values_batch = torch.stack(pixel_values_list).to(device)
        
        batch_captions = generate_caption_batch(model, pixel_values_batch, tokenizer, device)
        
        for img_name, gen_cap in zip(batch_imgs, batch_captions):
            generated_captions.append(gen_cap)
            refs = val_df[val_df['image'] == img_name]['caption'].tolist()
            references_per_image.append(refs)
    
    references_transposed = [list(refs) for refs in zip(*references_per_image)]
    bleu = sacrebleu.corpus_bleu(generated_captions, references_transposed)
    
    return {
        'BLEU-1': bleu.precisions[0],
        'BLEU-2': bleu.precisions[1],
        'BLEU-3': bleu.precisions[2],
        'BLEU-4': bleu.precisions[3],
        'BLEU_corpus': bleu.score,
    }


print("Funzioni definite:")
print("  - train_one_epoch (con gradient accumulation)")
print("  - evaluate_loss")
print("  - generate_caption_batch")
print("  - evaluate_bleu_subset")

# Reset peak per misurazioni pulite durante il training
torch.cuda.reset_peak_memory_stats()

# History
HISTORY_FILE = OUTPUTS_DIR / "week3_clip_history.json"

if HISTORY_FILE.exists():
    with open(HISTORY_FILE) as f:
        history = json.load(f)
    start_epoch = max(history['epoch']) + 1
    best_bleu4 = max(history['val_bleu_4'])
    best_epoch = history['epoch'][history['val_bleu_4'].index(best_bleu4)]
    epochs_since_improvement = (max(history['epoch']) - best_epoch)
    # Avanza lo scheduler ai passi gia' completati
    steps_already_done = (start_epoch - 1) * n_optimizer_steps_per_epoch
    for _ in range(steps_already_done):
        scheduler.step()
    print(f"History caricata. Riprendo da epoca {start_epoch}.")
    print(f"Best finora: epoca {best_epoch}, BLEU-4 = {best_bleu4:.2f}")
    print(f"Scheduler avanzato di {steps_already_done} step.")
else:
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'val_bleu_1': [], 'val_bleu_4': [], 'val_bleu_corpus': [],
        'lr_at_end_of_epoch': [], 'epoch_time_sec': [],
    }
    start_epoch = 1
    best_bleu4 = -1.0
    best_epoch = 0
    epochs_since_improvement = 0
BEST_DIR = WEEK3_DIR / "best"
LAST_DIR = WEEK3_DIR / "last"

print(f"=== Avvio training Settimana 3 (CLIP-L + GPT-2 su Flickr30k) ===")
print(f"Budget:           {NUM_EPOCHS} epoche, early stopping patience {EARLY_STOP_PATIENCE}")
print(f"Batch:            {BATCH_SIZE} (effective {EFFECTIVE_BATCH} con grad accum {GRAD_ACCUM_STEPS})")
print(f"Stima:            ~30-35 min/epoca, totale 3-5 ore con early stop")
print(f"Best model:       {BEST_DIR}")
print(f"Last model:       {LAST_DIR}")
print(f"History live:     {HISTORY_FILE}")
print()

overall_start = time.time()

for epoch in range(start_epoch, NUM_EPOCHS + 1):
    epoch_start = time.time()
    
    # 1. Train
    train_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, loss_fn, device,
        epoch=epoch, grad_accum_steps=GRAD_ACCUM_STEPS,
    )
    
    # 2. Val loss
    val_loss = evaluate_loss(model, val_loader, loss_fn, device)
    
    # 3. Val BLEU su subset
    val_bleu = evaluate_bleu_subset(
        model, val_df, F30K_IMAGES_DIR, eval_transforms,
        tokenizer, device, n_samples=VAL_BLEU_SUBSET_SIZE, batch_size=16,
    )
    
    epoch_time = time.time() - epoch_start
    current_lr = scheduler.get_last_lr()[0]
    
    # Aggiorna history
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
    
    # 4a. Salva sempre "last" (antinfortunistica per crash)
    if LAST_DIR.exists():
        shutil.rmtree(LAST_DIR)
    model.save_pretrained(LAST_DIR)
    
    # 4b. Salva "best" solo se BLEU-4 migliora
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
    
    # 5. Early stopping
    if epochs_since_improvement >= EARLY_STOP_PATIENCE:
        print(f"\n>>> Early stopping: BLEU-4 non migliora da {EARLY_STOP_PATIENCE} epoche.")
        break

total_time = time.time() - overall_start
peak_vram = torch.cuda.max_memory_allocated() / 1e9
print(f"\n=== Training completato in {total_time/60:.1f} min ({total_time/3600:.2f} h) ===")
print(f"Best epoca:  {best_epoch}")
print(f"Best BLEU-4: {best_bleu4:.2f}")
print(f"Picco VRAM:  {peak_vram:.2f} GB")
print(f"Salvati:     {BEST_DIR}, {LAST_DIR}")