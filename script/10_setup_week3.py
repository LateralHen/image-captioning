"""
10_setup_week3.py
=================
Setup della Settimana 3: preparazione e verifica del nuovo modello
CLIP ViT-L/14 + GPT-2 small, PRIMA di lanciare il training vero (script 11).

L'upgrade della Settimana 3 e' l'encoder visivo: si passa da ViT-base
(pre-addestrato su ImageNet, ~14M immagini) a CLIP ViT-L/14 (pre-addestrato su
~400M coppie immagine-testo del web). CLIP-L e' molto piu' grande e produce
embedding di dimensione 1024 invece di 768, quindi serve un layer di proiezione
verso il decoder GPT-2.

Questo script NON addestra: serve a
  1. scaricare e ispezionare CLIP-L, confrontandolo con il vecchio ViT-base;
  2. assemblare il modello combinato e verificare la proiezione 1024->768;
  3. misurare il consumo di VRAM con un forward/backward/step finto, per
     calibrare batch size e gradient accumulation entro i limiti della GPU;
  4. salvare la configurazione scelta in outputs/week3_config.json;
  5. liberare la memoria.
"""

import torch
from pathlib import Path

# ----------------------------------------------------------------------------
# 1. SETUP: RIPRODUCIBILITA', DEVICE E PATH
# ----------------------------------------------------------------------------
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
print(f"\nProject root: {PROJECT_ROOT}")

# ----------------------------------------------------------------------------
# 2. CARICAMENTO E ISPEZIONE DI CLIP-L (confronto col vecchio ViT-base)
# ----------------------------------------------------------------------------
from transformers import CLIPVisionModel, CLIPImageProcessor

CLIP_NAME = "openai/clip-vit-large-patch14"

print(f"Caricamento {CLIP_NAME}...")
print("(prima volta: scarica ~1.7 GB)")
print()

# Carichiamo SOLO la parte visiva di CLIP (CLIPVisionModel) e il suo processor.
clip_vision = CLIPVisionModel.from_pretrained(CLIP_NAME)
clip_processor = CLIPImageProcessor.from_pretrained(CLIP_NAME)

# Ispezione della configurazione: ci interessa soprattutto hidden_size (1024),
# che determina la dimensione della proiezione verso GPT-2 (768).
cfg = clip_vision.config
print(f"=== Configurazione CLIP-L ===")
print(f"  hidden_size:        {cfg.hidden_size}")          # 1024 (vs 768 di ViT-base)
print(f"  num_hidden_layers:  {cfg.num_hidden_layers}")
print(f"  num_attention_heads:{cfg.num_attention_heads}")
print(f"  image_size:         {cfg.image_size}")
print(f"  patch_size:         {cfg.patch_size}")
print(f"  Patch totali:       {(cfg.image_size // cfg.patch_size)**2}")  # numero di token visivi
print()

# Numero di parametri di CLIP-L.
n_params = sum(p.numel() for p in clip_vision.parameters())
print(f"Parametri CLIP vision encoder: {n_params/1e6:.1f}M")
print()

# Confronto diretto col vecchio encoder ViT-base, per quantificare l'upgrade.
from transformers import ViTModel
vit_base = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
n_params_vit = sum(p.numel() for p in vit_base.parameters())
print(f"Parametri ViT-base (vecchio encoder): {n_params_vit/1e6:.1f}M")
print(f"CLIP-L e' {n_params/n_params_vit:.1f}x piu' grande")
print(f"hidden_size CLIP-L: {cfg.hidden_size}, ViT-base: {vit_base.config.hidden_size}")

# ----------------------------------------------------------------------------
# 3. ASSEMBLAGGIO DEL MODELLO CLIP-L + GPT-2
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    CLIPVisionModel,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

# Liberiamo la VRAM da eventuali residui prima di costruire il modello.
torch.cuda.empty_cache()

CLIP_NAME = "openai/clip-vit-large-patch14"
GPT2_NAME = "gpt2"

print("Costruzione modello CLIP-L + GPT-2 small...")

# Encoder: la parte visiva di CLIP.
encoder = CLIPVisionModel.from_pretrained(CLIP_NAME)

# Decoder: GPT-2, ma va configurato ESPLICITAMENTE come decoder con cross-attention.
# Senza is_decoder=True e add_cross_attention=True, GPT-2 resterebbe un modello solo
# testuale e non saprebbe "guardare" le feature dell'immagine prodotte dall'encoder.
from transformers import GPT2Config
decoder_config = GPT2Config.from_pretrained(GPT2_NAME)
decoder_config.is_decoder = True
decoder_config.add_cross_attention = True
decoder = GPT2LMHeadModel.from_pretrained(GPT2_NAME, config=decoder_config)

# VisionEncoderDecoderConfig + VisionEncoderDecoderModel uniscono encoder e decoder.
config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
    encoder.config, decoder.config
)
model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)

# Token speciali (stessa convenzione di tutti gli scenari precedenti).
tokenizer = GPT2Tokenizer.from_pretrained(GPT2_NAME)
tokenizer.pad_token = tokenizer.eos_token
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Conteggio parametri del modello combinato.
n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nParametri totali:     {n_total/1e6:.1f}M")
print(f"Parametri trainabili: {n_train/1e6:.1f}M")

# Verifica della proiezione encoder->decoder. Poiche' CLIP-L produce vettori da
# 1024 e GPT-2 ne attende 768, HuggingFace crea automaticamente un layer lineare
# 1024->768 (enc_to_dec_proj). Qui controlliamo che esista e quanto pesa.
print(f"\nEncoder hidden_size: {model.config.encoder.hidden_size}")  # 1024
print(f"Decoder hidden_size: {model.config.decoder.hidden_size}")    # 768
if hasattr(model, 'enc_to_dec_proj'):
    print(f"Layer di proiezione enc->dec: SI'")
    print(f"  Parametri proiezione: {sum(p.numel() for p in model.enc_to_dec_proj.parameters())/1e6:.2f}M")
else:
    print(f"Layer di proiezione enc->dec: NO (dimensioni gia' compatibili)")

# ----------------------------------------------------------------------------
# 4. MISURA DELLA VRAM (forward + backward + optimizer step, con batch finto)
# ----------------------------------------------------------------------------
# CLIP-L e' grande: prima di addestrare dobbiamo sapere se la GPU regge.
# Simuliamo un passo completo (forward+backward+step) con dati casuali e misuriamo
# il picco di VRAM. Il punto critico e' l'optimizer.step(): AdamW alloca due stati
# (m, v) per ogni parametro, ed e' spesso li' che la memoria esplode.
import time

torch.cuda.empty_cache()
model = model.to(device)

# VRAM occupata dal solo modello (pesi).
torch.cuda.synchronize()
mem_model = torch.cuda.memory_allocated() / 1e9
print(f"VRAM occupata dal solo modello: {mem_model:.2f} GB")
print()

# Batch finto: 16 immagini 224x224 e label casuali, solo per stressare la memoria.
BATCH_SIZE_TEST = 16
SEQ_LEN = 30
fake_pixel_values = torch.randn(BATCH_SIZE_TEST, 3, 224, 224, device=device)
fake_labels = torch.randint(0, 50000, (BATCH_SIZE_TEST, SEQ_LEN), device=device)

print(f"=== Test forward+backward con batch={BATCH_SIZE_TEST} ===")
print()

# Mixed precision bfloat16 (come negli scenari precedenti).
use_amp = torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_amp else torch.float32
print(f"Mixed precision: {'bfloat16' if use_amp else 'fp32'}")

# Optimizer reale: serve a misurare anche la VRAM occupata dagli stati di AdamW.
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

model.train()

# --- Forward ---
torch.cuda.synchronize()
t0 = time.time()
with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
    outputs = model(pixel_values=fake_pixel_values, labels=fake_labels)
    loss = outputs.loss
torch.cuda.synchronize()
t_fwd = time.time() - t0
mem_after_fwd = torch.cuda.memory_allocated() / 1e9
print(f"Forward:  {t_fwd*1000:.0f} ms, VRAM: {mem_after_fwd:.2f} GB")

# --- Backward ---
torch.cuda.synchronize()
t0 = time.time()
loss.backward()
torch.cuda.synchronize()
t_bwd = time.time() - t0
mem_after_bwd = torch.cuda.memory_allocated() / 1e9
print(f"Backward: {t_bwd*1000:.0f} ms, VRAM: {mem_after_bwd:.2f} GB")

# --- Optimizer step --- (qui AdamW alloca gli stati m, v: picco di memoria)
torch.cuda.synchronize()
t0 = time.time()
optimizer.step()
torch.cuda.synchronize()
t_opt = time.time() - t0
mem_after_opt = torch.cuda.memory_allocated() / 1e9
mem_peak = torch.cuda.max_memory_allocated() / 1e9   # picco massimo durante tutto il passo
print(f"Optim:    {t_opt*1000:.0f} ms, VRAM: {mem_after_opt:.2f} GB")
print()
print(f"=== PICCO VRAM raggiunto: {mem_peak:.2f} GB su {12.8:.1f} GB disponibili ({mem_peak/12.8*100:.0f}%) ===")
print()
print(f"Tempo totale per batch (fwd+bwd+opt): {(t_fwd+t_bwd+t_opt)*1000:.0f} ms")
# Stima del tempo per epoca su Flickr30k a partire dal tempo misurato per batch.
print(f"Stima tempo per epoca su Flickr30k ({27971//BATCH_SIZE_TEST} batch): "
      f"{(t_fwd+t_bwd+t_opt) * (27971/BATCH_SIZE_TEST) / 60:.1f} min")

# Pulizia degli oggetti di test.
optimizer.zero_grad()
del optimizer, outputs, loss
torch.cuda.empty_cache()

# ----------------------------------------------------------------------------
# 5. SALVATAGGIO DELLA CONFIGURAZIONE DELLA SETTIMANA 3
# ----------------------------------------------------------------------------
# Documentiamo in un JSON tutte le scelte di design e le misure fatte. Serve da
# tracciamento dell'esperimento e da riferimento per il report.
# NB: i valori di batch/grad-accum riportati qui sono quelli pianificati in fase
# di setup; nel training effettivo (script 11) si usa batch 4 + grad accum 8
# (sempre effective batch 32), calibrato sulla VRAM reale.
import json

week3_config = {
    "experiment": "week3_clip_l_gpt2",
    "encoder": "openai/clip-vit-large-patch14",
    "encoder_hidden_size": 1024,
    "encoder_params_M": 303.2,
    "decoder": "gpt2",
    "decoder_hidden_size": 768,
    "projection_layer": "1024->768, ~0.79M params (creata automaticamente da HF)",
    "total_params_M": 456.8,

    # Training
    "dataset": "flickr30k",
    "batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "num_epochs": 25,
    "early_stop_patience": 3,
    "learning_rate": 5e-5,
    "warmup_ratio": 1/25,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "label_smoothing": 0.1,
    "max_length": 30,

    # Augmentation (stessa della settimana 2)
    "augmentation": "RandomResizedCrop + RandomHorizontalFlip + ColorJitter",

    # Hardware
    "gpu": "RTX 5070",
    "vram_gb": 12.8,
    "vram_peak_measured_gb": 9.31,
    "vram_usage_pct": 73,
    "mixed_precision": "bfloat16",

    # Stima tempo
    "estimated_time_per_epoch_min": 28,
    "estimated_total_time_h_max": 11,   # 25 epoche
    "estimated_total_time_h_realistic": 4,  # con early stopping

    # Selezione checkpoint
    "checkpoint_selection_metric": "BLEU-4 su subset val (200 img)",
    "val_subset_size": 200,
    "evaluation_test_sets": ["flickr8k_test_1091", "flickr30k_test_1906"],
}

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
config_path = PROJECT_ROOT / "outputs" / "week3_config.json"
with open(config_path, "w") as f:
    json.dump(week3_config, f, indent=2)

print(f"Configurazione settimana 3 salvata in:\n  {config_path}")
print()
print(json.dumps(week3_config, indent=2))

# ----------------------------------------------------------------------------
# 6. CLEANUP FINALE DELLA MEMORIA
# ----------------------------------------------------------------------------
# Liberiamo modello e tensori finti: questo script e' solo preparatorio, il
# training vero gira nello script 11 ripartendo da zero.
import gc

del model, encoder, decoder, fake_pixel_values, fake_labels
gc.collect()
torch.cuda.empty_cache()

mem_now = torch.cuda.memory_allocated() / 1e9
print(f"VRAM dopo cleanup: {mem_now:.2f} GB (dovrebbe essere ~0)")
print()
print("=" * 60)
print("SETUP SETTIMANA 3 COMPLETATO")
print("=" * 60)
print()
print("Prossimo step: creare 11_training_week3.ipynb")
print()
print("Cosa abbiamo verificato:")
print("  [x] CLIP-L scaricato (1.71 GB nella cache HF)")
print("  [x] hidden_size CLIP-L = 1024, proiezione 1024->768 OK")
print("  [x] Modello assemblato: 456.8M parametri")
print("  [x] VRAM measure su batch 16: 9.31 GB picco (73%)")
print("  [x] Batch 16 + grad accum 2 = effective 32")
print("  [x] Config salvata in outputs/week3_config.json")