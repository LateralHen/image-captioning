import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from transformers import (
    VisionEncoderDecoderModel,
    CLIPImageProcessor,
    GPT2Tokenizer,
)
import sacrebleu
import pandas as pd
import json
from tqdm.auto import tqdm

# Riproducibilita'
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
F8K_DATA_DIR = PROJECT_ROOT / "data" / "flickr8k"
F8K_IMAGES_DIR = F8K_DATA_DIR / "Images"
F8K_SPLIT_FILE = F8K_DATA_DIR / "captions_with_split.csv"

F30K_DATA_DIR = PROJECT_ROOT / "data" / "flickr30k"
F30K_IMAGES_DIR = F30K_DATA_DIR / "flickr30k_images"
F30K_SPLIT_FILE = F30K_DATA_DIR / "captions_with_split.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
WEEK3_BEST_DIR = CHECKPOINT_DIR / "week3_clip" / "best"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MAX_LENGTH = 30
CLIP_NAME = "openai/clip-vit-large-patch14"
GPT2_NAME = "gpt2"

print(f"Device: {device}")
print(f"Best model esiste: {WEEK3_BEST_DIR.exists()}")
print(f"Flickr8k images:  {F8K_IMAGES_DIR.exists()}")
print(f"Flickr30k images: {F30K_IMAGES_DIR.exists()}")

# Carica modello CLIP-L + GPT-2 fine-tunato (Settimana 3)
print(f"\nCaricamento modello da {WEEK3_BEST_DIR}...")
model = VisionEncoderDecoderModel.from_pretrained(WEEK3_BEST_DIR).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(GPT2_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Per la normalizzazione delle immagini, usiamo CLIPImageProcessor
# (NON ImageNet, lo stesso del training della Sett. 3)
clip_processor = CLIPImageProcessor.from_pretrained(CLIP_NAME)
CLIP_MEAN = clip_processor.image_mean
CLIP_STD = clip_processor.image_std

model.eval()
print(f"Modello caricato. Parametri: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"Normalizzazione: mean={CLIP_MEAN}, std={CLIP_STD}")

#---------------------------------------------------------------------------------------------------------------

# Trasformazioni di evaluation (no augmentation, normalizzazione CLIP)
eval_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


@torch.no_grad()
def generate_caption_batch(model, pixel_values_batch, tokenizer, device):
    """Generazione batch con beam search (stesso workaround GPT-2 della Sett. 2)."""
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
def evaluate_on_test_set(model, test_df, images_dir, tokenizer, device,
                         batch_size=16, desc="Test"):
    """Genera didascalie per tutte le immagini uniche del test set, calcola BLEU."""
    model.eval()
    unique_imgs = test_df['image'].unique()
    generated_captions = []
    references_per_image = []
    
    for batch_start in tqdm(range(0, len(unique_imgs), batch_size), desc=desc):
        batch_imgs = unique_imgs[batch_start:batch_start + batch_size]
        pixel_values_list = [
            eval_transforms(Image.open(images_dir / img).convert("RGB"))
            for img in batch_imgs
        ]
        pixel_values_batch = torch.stack(pixel_values_list).to(device)
        batch_captions = generate_caption_batch(model, pixel_values_batch, tokenizer, device)
        
        for img_name, gen_cap in zip(batch_imgs, batch_captions):
            generated_captions.append(gen_cap)
            refs = test_df[test_df['image'] == img_name]['caption'].tolist()
            references_per_image.append(refs)
    
    # BLEU
    references_transposed = [list(refs) for refs in zip(*references_per_image)]
    bleu = sacrebleu.corpus_bleu(generated_captions, references_transposed)
    
    return {
        "n_images": len(generated_captions),
        "BLEU-1": round(bleu.precisions[0], 2),
        "BLEU-2": round(bleu.precisions[1], 2),
        "BLEU-3": round(bleu.precisions[2], 2),
        "BLEU-4": round(bleu.precisions[3], 2),
        "BLEU_corpus": round(bleu.score, 2),
        "avg_generated_length": round(bleu.sys_len / len(generated_captions), 2),
        "avg_reference_length": round(bleu.ref_len / len(generated_captions), 2),
        "generated_captions": generated_captions,
        "references_per_image": references_per_image,
    }


print("Funzioni definite:")
print("  - generate_caption_batch (beam search 4 con workaround GPT-2)")
print("  - evaluate_on_test_set (genera tutte le didascalie + calcola BLEU)")

#---------------------------------------------------------------------------------------------------------------

print("=== Evaluation Settimana 3 su Test FLICKR8K ===")
f8k_df = pd.read_csv(F8K_SPLIT_FILE)
f8k_test_df = f8k_df[f8k_df['split'] == 'test']
print(f"Test Flickr8k: {f8k_test_df['image'].nunique()} immagini uniche, {len(f8k_test_df)} reference")
print()

results_f8k = evaluate_on_test_set(
    model, f8k_test_df, F8K_IMAGES_DIR, tokenizer, device,
    batch_size=16, desc="Test Flickr8k",
)

print("\n=== RISULTATI Settimana 3 su Test FLICKR8K ===")
for k, v in results_f8k.items():
    if k not in ["generated_captions", "references_per_image"]:
        print(f"  {k}: {v}")

# Salvataggio
out_path = OUTPUTS_DIR / "week3_eval_flickr8k.json"
results_save = {k: v for k, v in results_f8k.items() 
                if k not in ["generated_captions", "references_per_image"]}
with open(out_path, "w") as f:
    json.dump(results_save, f, indent=2)
print(f"\nSalvato in: {out_path}")

# Salva anche le didascalie generate (per ispezione qualitativa)
preds_path = OUTPUTS_DIR / "week3_predictions_flickr8k.json"
preds_save = [
    {"image": img, "generated": gen, "references": refs}
    for img, gen, refs in zip(
        f8k_test_df['image'].unique(),
        results_f8k['generated_captions'],
        results_f8k['references_per_image'],
    )
]
with open(preds_path, "w") as f:
    json.dump(preds_save, f, indent=2)
print(f"Predictions salvate in: {preds_path}")

# ---------------------------------------------------------------------------------------------------------------

print("=== Evaluation Settimana 3 su Test FLICKR30K ===")
f30k_df = pd.read_csv(F30K_SPLIT_FILE)
f30k_test_df = f30k_df[f30k_df['split'] == 'test']
print(f"Test Flickr30k: {f30k_test_df['image'].nunique()} immagini uniche, {len(f30k_test_df)} reference")
print()

results_f30k = evaluate_on_test_set(
    model, f30k_test_df, F30K_IMAGES_DIR, tokenizer, device,
    batch_size=16, desc="Test Flickr30k",
)

print("\n=== RISULTATI Settimana 3 su Test FLICKR30K ===")
for k, v in results_f30k.items():
    if k not in ["generated_captions", "references_per_image"]:
        print(f"  {k}: {v}")

# Salvataggio
out_path = OUTPUTS_DIR / "week3_eval_flickr30k.json"
results_save = {k: v for k, v in results_f30k.items() 
                if k not in ["generated_captions", "references_per_image"]}
with open(out_path, "w") as f:
    json.dump(results_save, f, indent=2)
print(f"\nSalvato in: {out_path}")

# Salva anche le didascalie generate
preds_path = OUTPUTS_DIR / "week3_predictions_flickr30k.json"
preds_save = [
    {"image": img, "generated": gen, "references": refs}
    for img, gen, refs in zip(
        f30k_test_df['image'].unique(),
        results_f30k['generated_captions'],
        results_f30k['references_per_image'],
    )
]
with open(preds_path, "w") as f:
    json.dump(preds_save, f, indent=2)
print(f"Predictions salvate in: {preds_path}")

# ---------------------------------------------------------------------------------------------------------------

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Risultati di ogni esperimento
files = {
    "Scenario A (baseline)":            OUTPUTS_DIR / "bleu_results_baseline.json",
    "Settimana 1 (B su Flickr8k)":      CHECKPOINT_DIR / "scenario_B" / "results_test.json",
    "Settimana 2 (su test F8k)":        OUTPUTS_DIR / "week2_eval_flickr8k.json",
    "Settimana 2 (su test F30k)":       OUTPUTS_DIR / "week2_eval_flickr30k.json",
    "Settimana 3 CLIP-L (su test F8k)": OUTPUTS_DIR / "week3_eval_flickr8k.json",
    "Settimana 3 CLIP-L (su test F30k)":OUTPUTS_DIR / "week3_eval_flickr30k.json",
}

rows = []
for label, path in files.items():
    if not path.exists():
        print(f"!! Mancante: {path}")
        continue
    with open(path) as f:
        d = json.load(f)
    def get(d, *keys):
        for k in keys:
            if k in d:
                return d[k]
        return None
    rows.append({
        "Esperimento": label,
        "N img":   get(d, "n_images", "n_test_images"),
        "BLEU-1":  get(d, "BLEU-1", "BLEU_1"),
        "BLEU-2":  get(d, "BLEU-2", "BLEU_2"),
        "BLEU-3":  get(d, "BLEU-3", "BLEU_3"),
        "BLEU-4":  get(d, "BLEU-4", "BLEU_4"),
        "Corpus":  get(d, "BLEU_corpus"),
    })

df_results = pd.DataFrame(rows)
print("=== Tabella comparativa finale ===\n")
print(df_results.to_string(index=False))
print()

md_path = OUTPUTS_DIR / "comparison_table.md"
with open(md_path, "w") as f:
    f.write("# Tabella comparativa esperimenti\n\n")
    f.write(df_results.to_markdown(index=False))
    f.write("\n")
print(f"Tabella aggiornata salvata in: {md_path}")