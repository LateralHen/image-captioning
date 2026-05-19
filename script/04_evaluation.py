import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from transformers import (
    VisionEncoderDecoderModel,
    GPT2Tokenizer,
    ViTImageProcessor,
    GenerationConfig,
)
import sacrebleu
from tqdm.auto import tqdm
import json
import random

# Riproducibilita'
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Path
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "flickr8k"
IMAGES_DIR = DATA_DIR / "Images"
SPLIT_FILE = DATA_DIR / "captions_with_split.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_DIR = CHECKPOINT_DIR / "best"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

print(f"Caricamento best model da {BEST_MODEL_DIR}")

# Caricamento
INITIAL_MODEL_DIR = CHECKPOINT_DIR / "model_initial"

model = VisionEncoderDecoderModel.from_pretrained(BEST_MODEL_DIR)
tokenizer = GPT2Tokenizer.from_pretrained(INITIAL_MODEL_DIR)        # <-- da qui
image_processor = ViTImageProcessor.from_pretrained(INITIAL_MODEL_DIR)  # <-- da qui
tokenizer.pad_token = tokenizer.eos_token

# Configurazione di generazione (API moderna -> evitiamo i warning del notebook 02/03)
generation_config = GenerationConfig(
    max_length=30,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.bos_token_id,
)
model.generation_config = generation_config

model = model.to(device)
model.eval()  # IMPORTANTE: disabilita dropout, batch norm in mode eval
print(f"Modello su {device}, in eval mode")


def generate_caption(image_path, model, image_processor, tokenizer, device):
    """Genera una singola didascalia per un'immagine."""
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Forziamo la generazione a partire da [BOS, BOS] cosi' saltiamo il falso EOS iniziale.
    # E' un workaround per il fatto che GPT-2 usa lo stesso ID per BOS/EOS/PAD.
    decoder_input_ids = torch.tensor(
        [[tokenizer.bos_token_id, tokenizer.bos_token_id]],
        device=device,
    )
    
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=30,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
    
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption


# Test su una singola immagine per sanity check
df = pd.read_csv(SPLIT_FILE)
test_df = df[df['split'] == 'test'].copy()
test_images = test_df['image'].unique()
print(f"\nTest set: {len(test_images)} immagini, {len(test_df)} didascalie reference")

# Sanity: genera per la prima immagine test
sample_img = test_images[0]
sample_path = IMAGES_DIR / sample_img
generated = generate_caption(sample_path, model, image_processor, tokenizer, device)
real_captions = test_df[test_df['image'] == sample_img]['caption'].tolist()

print(f"\n=== Sanity check ===")
print(f"Immagine: {sample_img}")
print(f"Generata: {generated!r}")
print(f"Reference (1/5): {real_captions[0]!r}")

# === DEBUG: cosa produce davvero generate()? ===
sample_path = IMAGES_DIR / test_images[0]
image = Image.open(sample_path).convert("RGB")
pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)

print(f"pixel_values shape: {pixel_values.shape}")
print(f"BOS token id: {tokenizer.bos_token_id}")
print(f"EOS token id: {tokenizer.eos_token_id}")
print(f"PAD token id: {tokenizer.pad_token_id}")
print(f"decoder_start_token_id: {model.config.decoder_start_token_id}")
print()

# Test 1: generate "naked" senza nessun parametro
print("--- Test 1: generate() senza parametri ---")
with torch.no_grad():
    out1 = model.generate(pixel_values=pixel_values)
print(f"Output shape: {out1.shape}")
print(f"Output ids: {out1[0].tolist()}")
print(f"Decoded (no skip): {tokenizer.decode(out1[0], skip_special_tokens=False)!r}")
print(f"Decoded (skip):    {tokenizer.decode(out1[0], skip_special_tokens=True)!r}")
print()

# Test 2: greedy senza early stopping, max_length forzato a 30
print("--- Test 2: greedy, max_length=30, no early stop ---")
with torch.no_grad():
    out2 = model.generate(
        pixel_values=pixel_values,
        max_length=30,
        num_beams=1,  # greedy
        do_sample=False,
        early_stopping=False,
        decoder_start_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
print(f"Output shape: {out2.shape}")
print(f"Output ids: {out2[0].tolist()}")
print(f"Decoded (no skip): {tokenizer.decode(out2[0], skip_special_tokens=False)!r}")
print(f"Decoded (skip):    {tokenizer.decode(out2[0], skip_special_tokens=True)!r}")
print()

# Test 3: greedy SENZA fornire eos_token_id (cosi non si ferma)
print("--- Test 3: greedy, eos_token_id=None ---")
with torch.no_grad():
    out3 = model.generate(
        pixel_values=pixel_values,
        max_length=30,
        num_beams=1,
        do_sample=False,
        decoder_start_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=None,
    )
print(f"Output shape: {out3.shape}")
print(f"Output ids: {out3[0].tolist()}")
print(f"Decoded (no skip): {tokenizer.decode(out3[0], skip_special_tokens=False)!r}")
print(f"Decoded (skip):    {tokenizer.decode(out3[0], skip_special_tokens=True)!r}")

def show_image_with_predictions(image_filename, generated_caption, real_captions, images_dir):
    """Mostra immagine + caption generata + 5 caption reali."""
    img = Image.open(images_dir / image_filename)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Generated: \"{generated_caption}\"", 
                 fontsize=11, color='darkblue', wrap=True, pad=10)
    plt.tight_layout()
    plt.show()
    
    print("Reference umane:")
    for i, cap in enumerate(real_captions, 1):
        print(f"  {i}. {cap}")
    print()


# Estrai 8 immagini casuali dal test set
sample_test_images = random.sample(list(test_images), 8)

print("Generazione di 8 didascalie di esempio (dal test set, immagini MAI viste in training)...")
print("=" * 80)

for img_name in sample_test_images:
    img_path = IMAGES_DIR / img_name
    generated = generate_caption(img_path, model, image_processor, tokenizer, device)
    real_caps = test_df[test_df['image'] == img_name]['caption'].tolist()
    show_image_with_predictions(img_name, generated, real_caps, IMAGES_DIR)
    print("-" * 80)
    
import time

print(f"Generazione di {len(test_images)} didascalie sul test set...")
print("(beam search con num_beams=4, ~1 sec per immagine)")
print()

generated_captions = {}   # {image_filename: caption_generata}
real_captions_dict = {}   # {image_filename: [5 reference]}

start = time.time()
for img_name in tqdm(test_images, desc="Generazione"):
    img_path = IMAGES_DIR / img_name
    generated_captions[img_name] = generate_caption(img_path, model, image_processor, tokenizer, device)
    real_captions_dict[img_name] = test_df[test_df['image'] == img_name]['caption'].tolist()

elapsed = time.time() - start
print(f"\nTempo totale: {elapsed/60:.1f} min")
print(f"Throughput: {len(test_images)/elapsed:.2f} immagini/sec")

# Salviamo le didascalie generate per ispezioni successive
output_file = OUTPUTS_DIR / "test_predictions_baseline.json"
output_data = [
    {
        "image": img,
        "generated": generated_captions[img],
        "references": real_captions_dict[img],
    }
    for img in test_images
]
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)
print(f"Predizioni salvate in: {output_file}")

# Anteprima dei primi 3 esempi
print("\nAnteprima:")
for img in test_images[:3]:
    print(f"\n  {img}")
    print(f"    GEN: {generated_captions[img]}")
    print(f"    REF: {real_captions_dict[img][0]}")
    
# sacrebleu vuole input "trasposto":
# - hypotheses: lista di N stringhe (una per immagine)
# - references: lista di K liste di N stringhe (K = numero di reference per immagine = 5)
#   references[k][i] = la k-esima reference per la i-esima immagine

hypotheses = [generated_captions[img] for img in test_images]
references_per_image = [real_captions_dict[img] for img in test_images]  # N x 5

# Tutte le immagini hanno 5 reference: trasponiamo in 5 x N
references_transposed = list(zip(*references_per_image))
references_transposed = [list(refs) for refs in references_transposed]

print(f"Hypotheses: {len(hypotheses)} stringhe")
print(f"References: {len(references_transposed)} liste da {len(references_transposed[0])} stringhe ciascuna")
print()

# Calcolo BLEU
bleu = sacrebleu.corpus_bleu(hypotheses, references_transposed)

print("=" * 55)
print("    BLEU score sul test set di Flickr8k")
print(f"    Modello: best (epoca 2, val loss 1.95)")
print("=" * 55)
print(f"  BLEU-1: {bleu.precisions[0]:.2f}")
print(f"  BLEU-2: {bleu.precisions[1]:.2f}")
print(f"  BLEU-3: {bleu.precisions[2]:.2f}")
print(f"  BLEU-4: {bleu.precisions[3]:.2f}")
print(f"  BLEU score (geometric mean delle 4): {bleu.score:.2f}")
print()
print(f"  Brevity penalty: {bleu.bp:.4f}")
print(f"  Lunghezza media generata:  {bleu.sys_len / len(hypotheses):.1f} parole")
print(f"  Lunghezza media reference: {bleu.ref_len / len(hypotheses):.1f} parole")

# Salviamo i risultati
results = {
    "experiment": "scenario_A_baseline",
    "model": "ViT-base + GPT-2 small",
    "dataset": "Flickr8k",
    "best_epoch": 2,
    "best_val_loss": 1.9470,
    "BLEU_1": round(bleu.precisions[0], 2),
    "BLEU_2": round(bleu.precisions[1], 2),
    "BLEU_3": round(bleu.precisions[2], 2),
    "BLEU_4": round(bleu.precisions[3], 2),
    "BLEU_corpus": round(bleu.score, 2),
    "brevity_penalty": round(bleu.bp, 4),
    "avg_generated_length": round(bleu.sys_len / len(hypotheses), 2),
    "avg_reference_length": round(bleu.ref_len / len(hypotheses), 2),
    "n_test_images": len(test_images),
}
with open(OUTPUTS_DIR / "bleu_results_baseline.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nRisultati salvati in: {OUTPUTS_DIR / 'bleu_results_baseline.json'}")
print("\nQuesto e' il BASELINE da battere nelle settimane 1-3 del piano.")

