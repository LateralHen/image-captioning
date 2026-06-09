"""
05_training_scenario_B.py
=========================
Training "Scenario B" (Settimana 1): baseline rigoroso su Flickr8k con tutte
le best practice di training.

Differenze rispetto allo Scenario A (script 03):
  - Data augmentation sulle immagini di training (RandomResizedCrop, flip,
    color jitter), tramite torchvision invece dell'image_processor di HF;
  - LR scheduler: warmup lineare + cosine decay;
  - Label smoothing 0.1 nella cross-entropy (loss calcolata "a mano");
  - Early stopping basato sul BLEU di validazione (non sulla loss);
  - Salvataggio separato di "best" (miglior BLEU-4) e "last" (ultimo stato).

L'obiettivo e' ottenere numeri solidi su Flickr8k prima di cambiare dataset
nella Settimana 2.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR        # scheduler personalizzato (warmup+cosine)
import torchvision.transforms as T                   # trasformazioni immagini con augmentation
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    GPT2Tokenizer,
    ViTImageProcessor,
)
import sacrebleu                                      # calcolo del BLEU in validazione
from tqdm.auto import tqdm
import time
import json
import math
import shutil

# ----------------------------------------------------------------------------
# 1. SETUP: RIPRODUCIBILITA' E DEVICE
# ----------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ----------------------------------------------------------------------------
# 2. PATH DEL PROGETTO
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "flickr8k"
IMAGES_DIR = DATA_DIR / "Images"
SPLIT_FILE = DATA_DIR / "captions_with_split.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
INITIAL_MODEL_DIR = CHECKPOINT_DIR / "model_initial"   # snapshot vergine (dallo script 02)
SCENARIO_B_DIR = CHECKPOINT_DIR / "scenario_B"          # cartella dedicata ai checkpoint di questo run
SCENARIO_B_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ----------------------------------------------------------------------------
# 3. IPERPARAMETRI (Scenario B)
# ----------------------------------------------------------------------------
NUM_EPOCHS = 25                  # budget massimo di epoche (l'early stopping ferma prima)
EARLY_STOP_PATIENCE = 3          # epoche consecutive senza miglioramento BLEU prima di fermarsi
LEARNING_RATE = 5e-5             # learning rate di picco, raggiunto a fine warmup
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 1/25              # frazione di training dedicata al warmup (qui ~1 epoca su 25)
BATCH_SIZE = 32
NUM_WORKERS = 0                  # 0 = sicuro su Windows/Jupyter (evita deadlock dei worker)
MAX_LENGTH = 30
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1            # ammorbidisce i target one-hot: riduce overconfidence e overfitting
VAL_BLEU_SUBSET_SIZE = 200       # immagini campionate dal val set per stimare il BLEU rapidamente

print(f"\n=== Iperparametri Scenario B ===")
print(f"Budget epoche:          {NUM_EPOCHS}")
print(f"Early stopping:         {EARLY_STOP_PATIENCE} epoche senza miglioramento BLEU")
print(f"Peak learning rate:     {LEARNING_RATE}")
print(f"Warmup ratio:           {WARMUP_RATIO:.3f} (= {int(WARMUP_RATIO*NUM_EPOCHS)} epoche)")
print(f"Label smoothing:        {LABEL_SMOOTHING}")
print(f"Batch size:             {BATCH_SIZE}")
print(f"Val BLEU subset size:   {VAL_BLEU_SUBSET_SIZE} immagini per validazione rapida")

# ----------------------------------------------------------------------------
# 4. TRASFORMAZIONI IMMAGINI (augmentation in training, pulite in eval)
# ----------------------------------------------------------------------------
# Statistiche di normalizzazione di ImageNet: sono quelle su cui e' stato
# pre-addestrato ViT, quindi vanno usate identiche.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Trasformazioni di TRAINING: includono augmentation casuale. Ad ogni epoca il
# modello vede una versione leggermente diversa della stessa immagine, il che
# riduce l'overfitting (effetto regolarizzante).
train_transforms = T.Compose([
    T.RandomResizedCrop(
        size=224,
        scale=(0.8, 1.0),       # ritaglia un'area tra l'80% e il 100% dell'immagine
        ratio=(0.9, 1.1),       # leggera variazione del rapporto larghezza/altezza
    ),
    T.RandomHorizontalFlip(p=0.5),   # specchia orizzontalmente con probabilita' 50%
    T.ColorJitter(                   # perturba leggermente i colori
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ),
    T.ToTensor(),                    # PIL -> tensore (valori in [0,1])
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # normalizzazione ImageNet
])

# Trasformazioni di VALIDAZIONE/TEST: deterministiche, senza augmentation.
# In valutazione vogliamo input stabili e ripetibili.
eval_transforms = T.Compose([
    T.Resize(256),                # resize del lato corto a 256
    T.CenterCrop(224),            # crop centrale 224x224
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ----------------------------------------------------------------------------
# 5. DATASET (con augmentation tramite torchvision)
# ----------------------------------------------------------------------------
class Flickr8kCaptionsDataset(Dataset):
    """Dataset PyTorch per Flickr8k con augmentation opzionale.

    Rispetto alla versione dello script 03: invece dell'image_processor di
    Hugging Face, usa una pipeline torchvision passata dall'esterno
    (transforms_fn), piu' flessibile per aggiungere augmentation.
    """

    def __init__(self, df_split, images_dir, transforms_fn, tokenizer, max_length=30):
        self.df = df_split.reset_index(drop=True)
        self.images_dir = images_dir
        self.transforms = transforms_fn        # pipeline torchvision (train o eval)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Carica l'immagine e applica le trasformazioni (con augmentation se in training).
        img_path = self.images_dir / row['image']
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transforms(image)

        # 2. Tokenizza la didascalia (stessa logica dello script 03): BOS+caption+EOS,
        #    padding/troncamento a max_length, padding mascherato a -100 nelle labels.
        caption = row['caption']
        text = self.tokenizer.bos_token + " " + caption + " " + self.tokenizer.eos_token
        encoding = self.tokenizer(
            text, padding="max_length", max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100   # ignora il padding nella loss (mask, non pad_token_id)

        # Oltre a pixel_values/labels restituiamo anche il nome file e il testo:
        # serviranno alla validazione con BLEU (che lavora su immagini uniche e reference testuali).
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_filename": row['image'],   # utile per la val con BLEU
            "caption_text": caption,           # utile per la val con BLEU
        }


# ----------------------------------------------------------------------------
# 6. CARICAMENTO MODELLO E DATI
# ----------------------------------------------------------------------------
# Ripartiamo dallo stesso snapshot vergine usato dallo Scenario A: cosi' il
# confronto A vs B isola l'effetto delle sole best practice di training.
print("Caricamento modello iniziale...")
model = VisionEncoderDecoderModel.from_pretrained(INITIAL_MODEL_DIR)
tokenizer = GPT2Tokenizer.from_pretrained(INITIAL_MODEL_DIR)
image_processor = ViTImageProcessor.from_pretrained(INITIAL_MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token   # reimpostazione necessaria dopo il caricamento

model = model.to(device)
print(f"Modello su {device}, {sum(p.numel() for p in model.parameters())/1e6:.1f}M parametri")

# Carichiamo lo split e costruiamo i dataset: train con augmentation, val senza.
df = pd.read_csv(SPLIT_FILE)
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']

train_dataset = Flickr8kCaptionsDataset(train_df, IMAGES_DIR, train_transforms, tokenizer, MAX_LENGTH)
val_dataset = Flickr8kCaptionsDataset(val_df, IMAGES_DIR, eval_transforms, tokenizer, MAX_LENGTH)

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
# 7. VISUALIZZAZIONE DELL'AUGMENTATION (sanity check qualitativo)
# ----------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Funzione inversa della normalizzazione: riporta un tensore normalizzato a
# un'immagine visualizzabile (valori in [0,1], formato HxWxC).
def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# Mostriamo 4 versioni augmentate della STESSA immagine: chiamando piu' volte
# il dataset sullo stesso indice, l'augmentation random produce ogni volta una
# variante diversa. E' la prova visiva che l'augmentation e' attiva.
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Stessa immagine, 4 augmentation diverse (a ogni epoca il modello la vede diversa)", fontsize=12)

sample_idx = 0
for i in range(4):
    sample = train_dataset[sample_idx]
    img_visualizable = denormalize(sample['pixel_values'])
    axes[i].imshow(img_visualizable)
    axes[i].axis('off')
    axes[i].set_title(f"Variante {i+1}")

plt.tight_layout()
plt.show()

# Sanity check numerico sulle dimensioni e sul contenuto.
print(f"\nForma pixel_values: {sample['pixel_values'].shape}")
print(f"Range tensore (normalizzato): [{sample['pixel_values'].min():.2f}, {sample['pixel_values'].max():.2f}]")
print(f"Caption originale: {sample['caption_text']!r}")
print(f"Filename: {sample['image_filename']}")

# ----------------------------------------------------------------------------
# 8. OPTIMIZER E LR SCHEDULER (warmup lineare + cosine decay)
# ----------------------------------------------------------------------------
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# Numero totale di step di ottimizzazione (un passo per batch, per tutte le epoche).
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)


def lr_lambda(current_step: int):
    """Fattore moltiplicativo per LEARNING_RATE in funzione dello step corrente.

    - Warmup: cresce linearmente da 0 a 1 (LR parte basso ed aumenta, evitando
      aggiornamenti distruttivi nelle prime iterazioni con cross-attention random).
    - Decay : scende da 1 a 0 seguendo una curva coseno (raffinamento graduale).
    """
    if current_step < warmup_steps:
        # Warmup lineare: 0 -> 1.
        return current_step / max(1, warmup_steps)
    else:
        # Cosine decay: 1 -> 0 sul resto del training.
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

# LambdaLR applica lr_lambda(step) come moltiplicatore del LR di base.
scheduler = LambdaLR(optimizer, lr_lambda)

print(f"Total training steps: {total_steps}")
print(f"Warmup steps:         {warmup_steps} ({warmup_steps/total_steps*100:.1f}% del totale)")

# ----------------------------------------------------------------------------
# 9. LOSS CON LABEL SMOOTHING
# ----------------------------------------------------------------------------
# VisionEncoderDecoderModel calcola internamente una cross-entropy SENZA label
# smoothing. Per applicarlo definiamo noi la loss e la useremo al posto di
# outputs.loss. ignore_index=-100 salta il padding gia' mascherato nel Dataset.
loss_fn = nn.CrossEntropyLoss(
    ignore_index=-100,
    label_smoothing=LABEL_SMOOTHING,
)
print(f"Loss: CrossEntropyLoss con label_smoothing={LABEL_SMOOTHING}")

# Mixed precision in bfloat16 (come nello script 03): meno memoria, piu' veloce,
# senza i problemi di overflow del float16.
use_amp = torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_amp else torch.float32
print(f"Mixed precision: {'attiva (bfloat16)' if use_amp else 'non disponibile'}")

# Grafico della curva di learning rate: verifica visiva della forma warmup+cosine.
fig, ax = plt.subplots(figsize=(10, 4))
test_steps = list(range(0, total_steps + 1, max(1, total_steps // 200)))
test_lrs = [LEARNING_RATE * lr_lambda(s) for s in test_steps]
ax.plot(test_steps, test_lrs)
ax.axvline(warmup_steps, color='red', linestyle='--', alpha=0.5, label=f'Fine warmup (step {warmup_steps})')
ax.set_xlabel('Step')
ax.set_ylabel('Learning rate')
ax.set_title(f'LR schedule: linear warmup + cosine decay (peak={LEARNING_RATE})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 10. FUNZIONI DI LOSS, TRAINING E VALIDAZIONE
# ----------------------------------------------------------------------------
def compute_loss_with_smoothing(outputs, labels, loss_fn):
    """Calcola la loss con label smoothing a partire dai logits del modello.

    Nota fondamentale: VisionEncoderDecoderModel shifta GIA' i labels
    internamente quando produce outputs.logits. Quindi qui confrontiamo logits
    e labels direttamente, SENZA rifare lo shift a mano (rifarlo darebbe BLEU~0).

    Verifica empirica (nel blocco di debug piu' sotto): con loss_fn senza
    smoothing, questa funzione produce lo stesso valore di outputs.loss.
    """
    logits = outputs.logits  # (batch, seq_len, vocab_size), gia' allineati ai labels
    loss = loss_fn(
        logits.view(-1, logits.size(-1)),  # appiattisce a (batch*seq_len, vocab_size)
        labels.view(-1),                   # appiattisce a (batch*seq_len,)
    )
    return loss


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, epoch):
    """Esegue una epoca di training e ritorna la loss media."""
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
            loss = compute_loss_with_smoothing(outputs, labels, loss_fn)  # loss con smoothing

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # gradient clipping
        optimizer.step()
        scheduler.step()  # IMPORTANTE: lo scheduler avanza ad OGNI step (non ad ogni epoca)

        total_loss += loss.item()
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'lr': f'{current_lr:.2e}'})

    return total_loss / n_batches


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device):
    """Calcola la loss media di validazione (stessa loss con smoothing del training)."""
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
    """Genera didascalie per un intero batch di immagini (versione batch dello script 04).

    Stesso workaround del bug GPT-2 BOS/EOS: forziamo decoder_input_ids=[[BOS,BOS]]
    per ogni esempio del batch, per evitare lo stop immediato sul token ambiguo.
    """
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

    captions = [
        tokenizer.decode(ids, skip_special_tokens=True).strip()
        for ids in output_ids
    ]
    return captions


@torch.no_grad()
def evaluate_bleu_subset(model, val_df, images_dir, eval_transforms,
                         tokenizer, device, n_samples=200, batch_size=16):
    """Calcola il BLEU su un sottoinsieme del val set.

    Il BLEU completo su tutto il val ad ogni epoca sarebbe lento; valutiamo su
    n_samples immagini campionate (deterministicamente, via seed) come stima
    rapida per la selezione del best checkpoint e l'early stopping.

    Note: campioniamo IMMAGINI uniche, non righe del CSV. Per ogni immagine
    confrontiamo la didascalia generata con tutte le 5 reference umane.
    """
    model.eval()

    # Campionamento deterministico di n_samples immagini uniche dal val set.
    unique_imgs = val_df['image'].unique()
    rng = np.random.default_rng(seed=42)
    sampled_imgs = rng.choice(unique_imgs, size=min(n_samples, len(unique_imgs)), replace=False)

    generated_captions = []
    references_per_image = []

    # Generazione in batch per efficienza.
    pbar = tqdm(range(0, len(sampled_imgs), batch_size), desc="Val [BLEU]", leave=False)
    for batch_start in pbar:
        batch_imgs = sampled_imgs[batch_start:batch_start + batch_size]

        # Carica e trasforma (eval_transforms, senza augmentation) le immagini del batch.
        pixel_values_list = []
        for img_name in batch_imgs:
            image = Image.open(images_dir / img_name).convert("RGB")
            pixel_values_list.append(eval_transforms(image))
        pixel_values_batch = torch.stack(pixel_values_list).to(device)

        # Genera le didascalie del batch.
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            batch_captions = generate_caption_batch(model, pixel_values_batch, tokenizer, device)

        # Accumula generate e relative 5 reference.
        for img_name, gen_cap in zip(batch_imgs, batch_captions):
            generated_captions.append(gen_cap)
            refs = val_df[val_df['image'] == img_name]['caption'].tolist()
            references_per_image.append(refs)

    # sacrebleu vuole le reference "trasposte" (vedi script 04 per il dettaglio).
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
# 11. SANITY CHECK: BLEU SUL MODELLO NON ADDESTRATO
# ----------------------------------------------------------------------------
# Sul modello vergine il BLEU sara' molto basso (~3-5), perche' la cross-attention
# e' inizializzata random. Serve solo a verificare che la pipeline di valutazione
# giri senza errori, prima di lanciare il training vero.
print("Test della funzione evaluate_bleu_subset su 50 immagini (sul modello non addestrato)...")
print("Tempo atteso: ~30 secondi")
print()

t0 = time.time()
test_bleu = evaluate_bleu_subset(
    model, val_df, IMAGES_DIR, eval_transforms,
    tokenizer, device, n_samples=50, batch_size=16,
)
elapsed = time.time() - t0

print(f"Tempo: {elapsed:.1f}s")
print(f"BLEU sulla cross-attention NON ADDESTRATA (atteso bassissimo):")
for k, v in test_bleu.items():
    if k.startswith('BLEU'):
        print(f"  {k}: {v:.2f}")

# ----------------------------------------------------------------------------
# 12. DEBUG: VERIFICA DELLA LOSS "A MANO" VS LOSS DI HUGGING FACE
# ----------------------------------------------------------------------------
# Questo blocco diagnostico verifica che la nostra loss "a mano" sia corretta.
# Confronta quattro varianti sullo stesso batch. Conclusione attesa:
#   - "con shift, no smoothing" deve coincidere con la loss interna di HF
#     (conferma che HF NON richiede shift manuale);
#   - "no shift, no smoothing" e' SBAGLIATA (mostra l'effetto del doppio shift);
#   - "con smoothing 0.1" e' quella effettivamente usata in training.
print("Confronto: loss 'a mano' (con label smoothing) vs loss HF (senza smoothing)")
print()

batch = next(iter(train_loader))
pixel_values = batch['pixel_values'].to(device)
labels = batch['labels'].to(device)

model.eval()
with torch.no_grad():
    outputs = model(pixel_values=pixel_values, labels=labels)

    # (a) La nostra loss con smoothing, senza shift manuale (quella usata in training).
    loss_manual_with_shift = compute_loss_with_smoothing(outputs, labels, loss_fn)

    # (b) Loss senza shift e senza smoothing: confronto diretto logits/labels.
    loss_fn_no_smoothing = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.0)
    loss_manual_no_shift = loss_fn_no_smoothing(
        outputs.logits.view(-1, outputs.logits.size(-1)),
        labels.view(-1),
    )

    # (c) Loss CON shift manuale e senza smoothing: serve a mostrare cosa
    #     succederebbe shiftando a mano (logits[:-1] vs labels[1:]).
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_manual_shift_nosmooth = loss_fn_no_smoothing(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    # (d) Loss interna standard di Hugging Face (riferimento).
    loss_hf = outputs.loss

print(f"Loss HF (riferimento, no smoothing):     {loss_hf.item():.4f}")
print(f"Loss a mano CON shift, NO smoothing:     {loss_manual_shift_nosmooth.item():.4f}")
print(f"Loss a mano NO shift, NO smoothing:      {loss_manual_no_shift.item():.4f}")
print(f"Loss a mano CON shift, smoothing 0.1:    {loss_manual_with_shift.item():.4f}  <-- quella usata nel training")

# ----------------------------------------------------------------------------
# 13. LOOP DI TRAINING CON EARLY STOPPING SU BLEU
# ----------------------------------------------------------------------------
# Tracciamo loss, BLEU, LR e tempi per ogni epoca (salvati su disco per i grafici
# e per resilienza ai crash).
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

# Stato per early stopping e per il tracking del best checkpoint.
best_bleu4 = -1.0                  # miglior BLEU-4 visto finora
best_epoch = 0
epochs_since_improvement = 0       # contatore per l'early stopping
HISTORY_FILE = OUTPUTS_DIR / "scenario_B_history.json"
BEST_DIR = SCENARIO_B_DIR / "best"   # modello con il miglior BLEU-4
LAST_DIR = SCENARIO_B_DIR / "last"   # ultimo stato (sovrascritto ogni epoca, antinfortunistica)

print(f"=== Avvio training Scenario B ===")
print(f"Budget: {NUM_EPOCHS} epoche, early stopping con patience {EARLY_STOP_PATIENCE}")
print(f"Stima conservativa tempo totale: 4-7 ore (puo' fermarsi prima per early stop)")
print(f"Best model in:    {BEST_DIR}")
print(f"Last model in:    {LAST_DIR}  (sovrascritto ad ogni epoca, antinfortunistica)")
print()

overall_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()

    # 1. Una epoca di training.
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, epoch)

    # 2. Loss di validazione.
    val_loss = evaluate_loss(model, val_loader, loss_fn, device)

    # 3. BLEU di validazione su sottoinsieme (metrica per la selezione del best).
    val_bleu = evaluate_bleu_subset(
        model, val_df, IMAGES_DIR, eval_transforms,
        tokenizer, device, n_samples=VAL_BLEU_SUBSET_SIZE, batch_size=16,
    )

    epoch_time = time.time() - epoch_start
    current_lr = scheduler.get_last_lr()[0]

    # Aggiorna la history.
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_bleu_1'].append(val_bleu['BLEU-1'])
    history['val_bleu_4'].append(val_bleu['BLEU-4'])
    history['val_bleu_corpus'].append(val_bleu['BLEU_corpus'])
    history['lr_at_end_of_epoch'].append(current_lr)
    history['epoch_time_sec'].append(epoch_time)

    # Salva la history su disco ad ogni epoca (no perdite in caso di crash).
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

    # 4a. Salva SEMPRE il "last": l'ultimo stato del modello (recupero in caso di crash).
    if LAST_DIR.exists():
        shutil.rmtree(LAST_DIR)
    model.save_pretrained(LAST_DIR)

    # 4b. Salva il "best" SOLO se il BLEU-4 di questa epoca supera il record.
    is_best = val_bleu['BLEU-4'] > best_bleu4
    if is_best:
        best_bleu4 = val_bleu['BLEU-4']
        best_epoch = epoch
        epochs_since_improvement = 0   # azzera il contatore di early stopping
        if BEST_DIR.exists():
            shutil.rmtree(BEST_DIR)
        model.save_pretrained(BEST_DIR)
        marker = "[BEST!]"
    else:
        epochs_since_improvement += 1  # un'altra epoca senza miglioramento
        marker = f"(no improv x{epochs_since_improvement})"

    # Riepilogo dell'epoca su una riga.
    print(f"Epoca {epoch:2d}/{NUM_EPOCHS} | "
          f"train: {train_loss:.3f} | val: {val_loss:.3f} | "
          f"BLEU-1: {val_bleu['BLEU-1']:.2f} | "
          f"BLEU-4: {val_bleu['BLEU-4']:.2f} | "
          f"lr: {current_lr:.2e} | "
          f"tempo: {epoch_time/60:.1f}min | "
          f"{marker}")

    # 5. Early stopping: se il BLEU-4 non migliora per PATIENCE epoche, fermati.
    if epochs_since_improvement >= EARLY_STOP_PATIENCE:
        print(f"\n>>> Early stopping: BLEU-4 non migliora da {EARLY_STOP_PATIENCE} epoche consecutive")
        break

total_time = time.time() - overall_start
print(f"\n=== Training completato in {total_time/60:.1f} minuti ({total_time/3600:.1f} ore) ===")
print(f"Best epoca:  {best_epoch}")
print(f"Best BLEU-4: {best_bleu4:.2f}")
print(f"Best model salvato in: {BEST_DIR}")
print(f"Last model salvato in: {LAST_DIR}")