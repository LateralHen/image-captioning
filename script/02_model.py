"""
Costruzione e verifica del modello di image captioning (Scenario A / baseline).

Architettura: VisionEncoderDecoderModel di HuggingFace, che combina
  - un ENCODER visivo  : ViT-base (pre-addestrato su ImageNet-21k)
  - un DECODER testuale: GPT-2 small
aggiungendo automaticamente i layer di cross-attention che permettono al
decoder di "guardare" le feature dell'immagine mentre genera la didascalia.

Obiettivo dello script:
  1. assemblare encoder + decoder e configurare i token speciali;
  2. definire il Dataset PyTorch che trasforma (immagine, didascalia) nei
     tensori attesi dal modello (pixel_values, labels);
  3. fare un forward/backward di prova su un batch per verificare che tutto
     funzioni e per misurare tempi e consumo di VRAM;
  4. salvare lo stato iniziale del modello (snapshot pre-training) che verra'
     ricaricato dallo script 03 per l'addestramento.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,   # wrapper encoder-decoder per visione->testo
    GPT2Tokenizer,               # tokenizer del decoder
    ViTImageProcessor,           # pre-processing immagini per l'encoder ViT
)

# ----------------------------------------------------------------------------
# 1. SETUP: RIPRODUCIBILITA' E DEVICE
# ----------------------------------------------------------------------------
# Seed fisso: garantisce che inizializzazioni casuali e shuffle siano
# riproducibili tra esecuzioni diverse.
torch.manual_seed(42)

# Selezione del device: GPU CUDA se disponibile, altrimenti CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    # Stampa nome GPU e VRAM totale: utile per dimensionare il batch size.
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ----------------------------------------------------------------------------
# 2. PATH E IPERPARAMETRI
# ----------------------------------------------------------------------------
# Path robusti (funzionano sia da 'notebooks/' sia dalla root del progetto).
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "flickr8k"
IMAGES_DIR = DATA_DIR / "Images"
SPLIT_FILE = DATA_DIR / "captions_with_split.csv"   # prodotto dallo script 01

# Iperparametri principali (la scelta di MAX_LENGTH=30 deriva dall'analisi
# delle lunghezze fatta nello script 01).
MAX_LENGTH = 30
BATCH_SIZE = 32  # provvisorio: lo ricalibriamo dopo aver misurato la VRAM usata

# Riferimenti standard dei modelli pre-addestrati da combinare.
ENCODER_NAME = "google/vit-base-patch16-224-in21k"
DECODER_NAME = "gpt2"

print(f"\nEncoder: {ENCODER_NAME}")
print(f"Decoder: {DECODER_NAME}")
print(f"Max length: {MAX_LENGTH} token")

# ----------------------------------------------------------------------------
# 3. COSTRUZIONE DEL MODELLO COMBINATO
# ----------------------------------------------------------------------------
# from_encoder_decoder_pretrained carica i due modelli pre-addestrati e li
# unisce, inizializzando da zero i layer di cross-attention del decoder
# (gli unici pesi non pre-addestrati della rete).
print("Caricamento del modello combinato (prima volta: download ~1.3 GB)...")
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    ENCODER_NAME,
    DECODER_NAME,
)

# Tokenizer (testo) e image processor (immagini): devono corrispondere
# rispettivamente al decoder e all'encoder scelti.
tokenizer = GPT2Tokenizer.from_pretrained(DECODER_NAME)
image_processor = ViTImageProcessor.from_pretrained(ENCODER_NAME)

# GPT-2 non ha un pad_token nativo: usiamo l'eos_token come padding.
tokenizer.pad_token = tokenizer.eos_token

# Configuriamo i token speciali nel modello:
# - decoder_start_token_id: token da cui parte la generazione (BOS).
# - pad_token_id          : token di padding, ignorato nel calcolo della loss.
# - eos_token_id          : token che segnala la fine della sequenza.
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
# Allineiamo la vocab_size del wrapper a quella del decoder GPT-2.
model.config.vocab_size = model.config.decoder.vocab_size

# Parametri di generazione (usati in inference). Li impostiamo gia' qui:
# - max_length          : lunghezza massima della didascalia generata;
# - early_stopping      : interrompe il beam search quando trova un EOS;
# - no_repeat_ngram_size: vieta la ripetizione di n-grammi (qui trigrammi),
#                         per evitare loop tipo "a man a man a man";
# - num_beams           : ampiezza del beam search (4 = qualita'/costo bilanciati).
model.config.max_length = MAX_LENGTH
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.num_beams = 4

# Conteggio parametri: utile per il report (dimensione del modello).
# Qui tutti i parametri sono trainabili (nessun layer congelato).
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nParametri totali: {total_params/1e6:.1f}M")
print(f"Parametri trainabili: {trainable_params/1e6:.1f}M")

# Sposta il modello sulla GPU (o CPU).
model = model.to(device)
print(f"Modello su: {next(model.parameters()).device}")


# ----------------------------------------------------------------------------
# 4. DEFINIZIONE DEL DATASET PYTORCH
# ----------------------------------------------------------------------------
class Flickr8kCaptionsDataset(Dataset):
    """Dataset PyTorch per Flickr8k.

    Ogni elemento e' una coppia (immagine, didascalia) gia' processata e
    pronta per il modello: l'immagine come tensore di pixel normalizzati,
    la didascalia come sequenza di token con il padding mascherato.
    """

    def __init__(self, df_split, images_dir, image_processor, tokenizer, max_length=30):
        """
        Args:
            df_split: DataFrame con colonne 'image' e 'caption' (gia' filtrato per split).
            images_dir: Path della cartella che contiene i file .jpg.
            image_processor: ViTImageProcessor di HuggingFace (resize + normalizzazione).
            tokenizer: GPT2Tokenizer di HuggingFace.
            max_length: lunghezza massima della sequenza tokenizzata.
        """
        # reset_index assicura indici 0..N-1 contigui, necessario per iloc.
        self.df = df_split.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # Numero di esempi nel dataset (richiesto da DataLoader).
        return len(self.df)

    def __getitem__(self, idx):
        # Restituisce l'esempio idx-esimo, gia' trasformato in tensori.
        row = self.df.iloc[idx]

        # --- 1. Carica e processa l'immagine ---
        img_path = self.images_dir / row['image']
        image = Image.open(img_path).convert("RGB")  # forza 3 canali (anche per immagini grayscale)
        # L'image processor fa resize a 224x224 e normalizza secondo le statistiche di ImageNet.
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.squeeze(0)        # rimuove la dim batch fittizia: (1,3,224,224)->(3,224,224)

        # --- 2. Tokenizza la didascalia ---
        # Avvolgiamo la frase tra BOS (inizio) ed EOS (fine), come in training.
        caption = row['caption']
        text = self.tokenizer.bos_token + " " + caption + " " + self.tokenizer.eos_token

        # Tokenizza con padding fisso a max_length e troncamento se piu' lunga.
        encoding = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)          # (max_length,) ID dei token
        attention_mask = encoding["attention_mask"].squeeze(0)  # (max_length,) 1=token reale, 0=pad

        # --- 3. Costruisci le labels per la loss ---
        # Le labels sono una copia degli input_ids, ma con il padding messo a -100.
        # PyTorch ignora le posizioni con label = -100 nel calcolo della cross-entropy.
        # IMPORTANTE: mascheriamo SOLO il padding usando l'attention_mask, NON
        # confrontando con pad_token_id. Motivo: pad_token_id == eos_token_id in
        # GPT-2, quindi un confronto diretto cancellerebbe per errore anche gli EOS
        # "veri" che il modello deve invece imparare a produrre.
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Nota: NON restituiamo input_ids/decoder_input_ids. VisionEncoderDecoderModel
        # esegue internamente lo shift delle labels per costruire l'input del decoder;
        # passarli manualmente causerebbe un doppio shift e BLEU vicino a zero.
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


# ----------------------------------------------------------------------------
# 5. CREAZIONE DEI DATASET (TRAIN / VAL / TEST)
# ----------------------------------------------------------------------------
# Carichiamo lo split prodotto dallo script 01.
df = pd.read_csv(SPLIT_FILE)
print(f"Dataset totale: {len(df)} esempi")
print(f"Distribuzione split:\n{df['split'].value_counts()}\n")

# Filtriamo le righe per ciascuno split.
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
test_df = df[df['split'] == 'test']

# Istanziamo i tre dataset con gli stessi processor/tokenizer/max_length.
train_dataset = Flickr8kCaptionsDataset(train_df, IMAGES_DIR, image_processor, tokenizer, MAX_LENGTH)
val_dataset = Flickr8kCaptionsDataset(val_df, IMAGES_DIR, image_processor, tokenizer, MAX_LENGTH)
test_dataset = Flickr8kCaptionsDataset(test_df, IMAGES_DIR, image_processor, tokenizer, MAX_LENGTH)

print(f"Train: {len(train_dataset)} esempi")
print(f"Val:   {len(val_dataset)} esempi")
print(f"Test:  {len(test_dataset)} esempi")

# --- Sanity check su un singolo esempio: verifichiamo shape e contenuto ---
sample = train_dataset[0]
print("Forma di un singolo esempio:")
print(f"  pixel_values: {sample['pixel_values'].shape}, dtype={sample['pixel_values'].dtype}")
print(f"  labels:       {sample['labels'].shape}, dtype={sample['labels'].dtype}")
print()
# Range dei pixel: dopo la normalizzazione sono valori float centrati intorno a 0.
print(f"  Range pixel_values: [{sample['pixel_values'].min():.3f}, {sample['pixel_values'].max():.3f}]")
print(f"  Primi 15 token labels: {sample['labels'][:15].tolist()}")
# Gli ultimi token dovrebbero essere -100 (padding mascherato).
print(f"  Ultimi 15 token labels (-100 = pad ignorato): {sample['labels'][-15:].tolist()}")
print()

# Decodifichiamo i token per leggere la didascalia in chiaro: sostituiamo i -100
# col pad_token solo per poterli decodificare (il -100 non e' un ID valido).
labels_for_decode = sample['labels'].clone()
labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
decoded = tokenizer.decode(labels_for_decode, skip_special_tokens=False)
print(f"  Didascalia decodificata: {decoded!r}")

# ----------------------------------------------------------------------------
# 6. DATALOADER
# ----------------------------------------------------------------------------
# I DataLoader raggruppano gli esempi in batch e gestiscono lo shuffle.
# NUM_WORKERS=0: su Windows i worker (>0) dentro Jupyter possono causare
# deadlock; 0 e' la scelta sicura su questo setup.
NUM_WORKERS = 0  # adatta in base al tuo PC; 4 e' sicuro per quasi tutti

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,            # mescola ad ogni epoca (importante per il training)
    num_workers=NUM_WORKERS,
    pin_memory=True,         # velocizza il trasferimento CPU->GPU
    #persistent_workers=True,  # tiene i worker vivi tra le epoche (no re-spawn)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,           # in validazione l'ordine non conta
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

# ----------------------------------------------------------------------------
# 7. ESTRAZIONE DI UN BATCH DI PROVA
# ----------------------------------------------------------------------------
import time

# Estraiamo il primo batch: la prima chiamata spawna i worker (puo' essere lenta).
print("Caricamento del primo batch (i worker si spawnano ora, puo' richiedere ~10s)...")
t0 = time.time()
batch = next(iter(train_loader))
elapsed = time.time() - t0
print(f"Tempo: {elapsed:.1f}s")
print()
print(f"Forme del batch:")
print(f"  pixel_values: {batch['pixel_values'].shape}")  # (batch, 3, 224, 224)
print(f"  labels:       {batch['labels'].shape}")          # (batch, max_length)
print()
# Stima della memoria occupata dal solo batch (pixel + labels) in MB.
print(f"Memoria stimata di un batch: {(batch['pixel_values'].element_size() * batch['pixel_values'].nelement() + batch['labels'].element_size() * batch['labels'].nelement()) / 1e6:.1f} MB")

# ----------------------------------------------------------------------------
# 8. FORWARD + BACKWARD DI PROVA E MISURA DELLA VRAM
# ----------------------------------------------------------------------------
# Questo blocco serve a (a) verificare che il modello processi un batch reale
# senza errori e (b) misurare quanta VRAM serve, per calibrare il batch size.
import time

# Spostiamo il batch sulla GPU.
pixel_values = batch['pixel_values'].to(device)
labels = batch['labels'].to(device)

print(f"Input pixel_values shape: {pixel_values.shape}")
print(f"Input labels shape: {labels.shape}")
print()

# VRAM occupata PRIMA del forward (solo i pesi del modello + il batch).
torch.cuda.empty_cache()
mem_before = torch.cuda.memory_allocated() / 1e9
print(f"VRAM occupata prima del forward: {mem_before:.2f} GB")

# --- Forward pass ---
# Passando 'labels', il modello calcola direttamente la loss di cross-entropy
# (e fa internamente lo shift delle labels).
print("\nForward pass...")
t0 = time.time()
outputs = model(pixel_values=pixel_values, labels=labels)
torch.cuda.synchronize()   # le op CUDA sono asincrone: aspettiamo la fine per misurare il tempo reale
forward_time = time.time() - t0

loss = outputs.loss        # scalare: la loss media sul batch
logits = outputs.logits    # (batch, seq_len, vocab_size): punteggi per ogni token del vocabolario

print(f"  Tempo forward: {forward_time*1000:.0f} ms")
print(f"  Loss: {loss.item():.4f}")
print(f"  Logits shape: {logits.shape}  (batch, seq_len, vocab_size)")

# VRAM dopo il forward: include le attivazioni intermedie salvate per il backward.
mem_after_fwd = torch.cuda.memory_allocated() / 1e9
print(f"  VRAM dopo forward: {mem_after_fwd:.2f} GB")

# --- Backward pass ---
# Calcola i gradienti rispetto a tutti i parametri (qui solo per misurare,
# senza optimizer.step()).
print("\nBackward pass...")
t0 = time.time()
loss.backward()
torch.cuda.synchronize()
backward_time = time.time() - t0
print(f"  Tempo backward: {backward_time*1000:.0f} ms")

# VRAM dopo il backward: ora include anche i gradienti.
mem_after_bwd = torch.cuda.memory_allocated() / 1e9
print(f"  VRAM dopo backward: {mem_after_bwd:.2f} GB (modello + attivazioni + gradienti)")

# Pulizia: azzeriamo i gradienti e svuotiamo la cache, cosi' la prova non
# lascia residui in memoria.
model.zero_grad()
torch.cuda.empty_cache()

# --- Sanity check sul valore della loss ---
# Per un modello non addestrato, la cross-entropy attesa con predizione
# uniforme sui ~50k token e' log(vocab_size). Se la nostra loss e' nello
# stesso ordine di grandezza, il modello e' inizializzato correttamente.
import math
expected_loss_random = math.log(50257)
print()
print("=== Sanity check ===")
print(f"  Loss e' un numero finito: {torch.isfinite(loss).item()}")
print(f"  Loss > 0: {loss.item() > 0}")
print(f"  Loss attesa per modello casuale: ~{expected_loss_random:.2f}  (cross-entropy uniforme su 50257 token)")
print(f"  La nostra loss: {loss.item():.4f}")

# ----------------------------------------------------------------------------
# 9. SALVATAGGIO DELLO SNAPSHOT INIZIALE DEL MODELLO
# ----------------------------------------------------------------------------
# Salviamo il modello pre-training: lo script 03 ripartira' esattamente da qui,
# garantendo che il training inizi sempre dallo stesso stato.
import json

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Salva architettura + pesi del modello.
INITIAL_MODEL_DIR = CHECKPOINT_DIR / "model_initial"
print(f"Salvataggio modello in {INITIAL_MODEL_DIR}...")
model.save_pretrained(INITIAL_MODEL_DIR)

# save_pretrained NON salva tokenizer e image_processor: vanno salvati a parte,
# altrimenti in inference non sapremmo come tokenizzare/normalizzare gli input.
tokenizer.save_pretrained(INITIAL_MODEL_DIR)
image_processor.save_pretrained(INITIAL_MODEL_DIR)

# Salviamo un piccolo file di metadati con gli iperparametri e le misure,
# utile per il report e per tracciare la configurazione dell'esperimento.
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

# Elenco dei file salvati con relativa dimensione (verifica del salvataggio).
print(f"\nFile salvati in {INITIAL_MODEL_DIR}:")
for f in sorted(INITIAL_MODEL_DIR.iterdir()):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name:40s} {size_mb:>8.1f} MB")