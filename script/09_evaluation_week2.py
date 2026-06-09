import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from transformers import VisionEncoderDecoderModel, GPT2Tokenizer, ViTImageProcessor
import sacrebleu
import pandas as pd
import json
from tqdm.auto import tqdm

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Path
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
INITIAL_MODEL_DIR = CHECKPOINT_DIR / "model_initial"
BEST_MODEL_DIR = CHECKPOINT_DIR / "week2_flickr30k" / "best"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MAX_LENGTH = 30

# Verifica path
print(f"Best model esiste: {BEST_MODEL_DIR.exists()}")
print(f"Initial model esiste: {INITIAL_MODEL_DIR.exists()}")

# Carica modello
model = VisionEncoderDecoderModel.from_pretrained(str(BEST_MODEL_DIR)).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(str(INITIAL_MODEL_DIR))
image_processor = ViTImageProcessor.from_pretrained(str(INITIAL_MODEL_DIR))
tokenizer.pad_token = tokenizer.eos_token
model.eval()
print(f"Modello caricato: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parametri")

# Trasformazioni per inference (identiche a quelle usate in training/eval)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@torch.no_grad()
def generate_caption_batch(model, pixel_values_batch, tokenizer, device):
    """Genera didascalie per un batch di immagini."""
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


def evaluate_on_test_set(model, test_df, images_dir, tokenizer, device, batch_size=16, desc="Test"):
    """Genera didascalie per tutte le immagini del test set e calcola BLEU."""
    unique_imgs = test_df['image'].unique()
    
    generated_captions = []
    references_per_image = []
    
    for batch_start in tqdm(range(0, len(unique_imgs), batch_size), desc=desc):
        batch_imgs = unique_imgs[batch_start:batch_start + batch_size]
        pixel_values_list = []
        for img_name in batch_imgs:
            image = Image.open(images_dir / img_name).convert("RGB")
            pixel_values_list.append(eval_transforms(image))
        pixel_values_batch = torch.stack(pixel_values_list).to(device)
        batch_captions = generate_caption_batch(model, pixel_values_batch, tokenizer, device)
        for img_name, gen_cap in zip(batch_imgs, batch_captions):
            generated_captions.append(gen_cap)
            refs = test_df[test_df['image'] == img_name]['caption'].tolist()
            references_per_image.append(refs)
    
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


print("Setup completato.")

# -----------------------------------------------------------------------------------------------

# === Evaluation 1: Test Flickr8k ===
F8K_DATA_DIR = PROJECT_ROOT / "data" / "flickr8k"
F8K_IMAGES_DIR = F8K_DATA_DIR / "Images"
F8K_SPLIT_FILE = F8K_DATA_DIR / "captions_with_split.csv"

f8k_df = pd.read_csv(F8K_SPLIT_FILE)
f8k_test_df = f8k_df[f8k_df['split'] == 'test']
print(f"Test Flickr8k: {f8k_test_df['image'].nunique()} immagini uniche")
print()

results_f8k = evaluate_on_test_set(
    model, f8k_test_df, F8K_IMAGES_DIR, tokenizer, device,
    batch_size=16, desc="Test Flickr8k",
)

print("\n=== RISULTATI Settimana 2 su Test FLICKR8K ===")
for k, v in results_f8k.items():
    if k not in ["generated_captions", "references_per_image"]:
        print(f"  {k}: {v}")

# Salvataggio
out_path = OUTPUTS_DIR / "week2_eval_flickr8k.json"
results_save = {k: v for k, v in results_f8k.items() if k not in ["generated_captions", "references_per_image"]}
with open(out_path, "w") as f:
    json.dump(results_save, f, indent=2)
print(f"\nSalvato in: {out_path}")

# -----------------------------------------------------------------------------------------------

# === Evaluation 1: Test Flickr8k ===
F8K_DATA_DIR = PROJECT_ROOT / "data" / "flickr8k"
F8K_IMAGES_DIR = F8K_DATA_DIR / "Images"
F8K_SPLIT_FILE = F8K_DATA_DIR / "captions_with_split.csv"

f8k_df = pd.read_csv(F8K_SPLIT_FILE)
f8k_test_df = f8k_df[f8k_df['split'] == 'test']
print(f"Test Flickr8k: {f8k_test_df['image'].nunique()} immagini uniche")
print()

results_f8k = evaluate_on_test_set(
    model, f8k_test_df, F8K_IMAGES_DIR, tokenizer, device,
    batch_size=16, desc="Test Flickr8k",
)

print("\n=== RISULTATI Settimana 2 su Test FLICKR8K ===")
for k, v in results_f8k.items():
    if k not in ["generated_captions", "references_per_image"]:
        print(f"  {k}: {v}")

# Salvataggio
out_path = OUTPUTS_DIR / "week2_eval_flickr8k.json"
results_save = {k: v for k, v in results_f8k.items() if k not in ["generated_captions", "references_per_image"]}
with open(out_path, "w") as f:
    json.dump(results_save, f, indent=2)
print(f"\nSalvato in: {out_path}")

# -----------------------------------------------------------------------------------------------

# === Evaluation 2: Test Flickr30k ===
F30K_DATA_DIR = PROJECT_ROOT / "data" / "flickr30k"
F30K_IMAGES_DIR = F30K_DATA_DIR / "flickr30k_images"
F30K_SPLIT_FILE = F30K_DATA_DIR / "captions_with_split.csv"

f30k_df = pd.read_csv(F30K_SPLIT_FILE)
f30k_test_df = f30k_df[f30k_df['split'] == 'test']
print(f"Test Flickr30k: {f30k_test_df['image'].nunique()} immagini uniche")
print()

results_f30k = evaluate_on_test_set(
    model, f30k_test_df, F30K_IMAGES_DIR, tokenizer, device,
    batch_size=16, desc="Test Flickr30k",
)

print("\n=== RISULTATI Settimana 2 su Test FLICKR30K ===")
for k, v in results_f30k.items():
    if k not in ["generated_captions", "references_per_image"]:
        print(f"  {k}: {v}")

# Salvataggio
out_path = OUTPUTS_DIR / "week2_eval_flickr30k.json"
results_save = {k: v for k, v in results_f30k.items() if k not in ["generated_captions", "references_per_image"]}
with open(out_path, "w") as f:
    json.dump(results_save, f, indent=2)
print(f"\nSalvato in: {out_path}")

# -----------------------------------------------------------------------------------------------

# === Riassunto comparativo: tutti gli esperimenti fatti finora ===
import json

# Carica i risultati precedenti dei vari esperimenti
baseline_path = OUTPUTS_DIR / "bleu_results_baseline.json"
scenario_b_path = CHECKPOINT_DIR / "scenario_B" / "results_test.json"

experiments = []

if baseline_path.exists():
    with open(baseline_path) as f:
        data = json.load(f)
    experiments.append({
        "Esperimento": "Scenario A (baseline)",
        "Dataset training": "Flickr8k",
        "Dataset test": "Flickr8k",
        "BLEU-1": data.get("BLEU_1", "n/a"),
        "BLEU-4": data.get("BLEU_4", "n/a"),
        "BLEU corpus": data.get("BLEU_corpus", "n/a"),
    })

if scenario_b_path.exists():
    with open(scenario_b_path) as f:
        data = json.load(f)
    experiments.append({
        "Esperimento": "Scenario B (Settimana 1)",
        "Dataset training": "Flickr8k",
        "Dataset test": "Flickr8k",
        "BLEU-1": data.get("BLEU-1", "n/a"),
        "BLEU-4": data.get("BLEU-4", "n/a"),
        "BLEU corpus": data.get("BLEU_corpus", "n/a"),
    })

# Settimana 2 — su entrambi i test set
experiments.append({
    "Esperimento": "Settimana 2",
    "Dataset training": "Flickr30k",
    "Dataset test": "Flickr8k",
    "BLEU-1": results_f8k["BLEU-1"],
    "BLEU-4": results_f8k["BLEU-4"],
    "BLEU corpus": results_f8k["BLEU_corpus"],
})

experiments.append({
    "Esperimento": "Settimana 2",
    "Dataset training": "Flickr30k",
    "Dataset test": "Flickr30k",
    "BLEU-1": results_f30k["BLEU-1"],
    "BLEU-4": results_f30k["BLEU-4"],
    "BLEU corpus": results_f30k["BLEU_corpus"],
})

# Stampa tabella
comparison_df = pd.DataFrame(experiments)
print("=" * 85)
print("TABELLA COMPARATIVA DEGLI ESPERIMENTI")
print("=" * 85)
print(comparison_df.to_string(index=False))
print("=" * 85)

# Salvataggio tabella
comparison_df.to_csv(OUTPUTS_DIR / "comparison_table.csv", index=False)
print(f"\nTabella salvata in: {OUTPUTS_DIR / 'comparison_table.csv'}")

# -----------------------------------------------------------------------------------------------

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
F30K_SPLIT = PROJECT_ROOT / "data" / "flickr30k" / "captions_with_split.csv"

print(f"File esiste: {F30K_SPLIT.exists()}")
print(f"Path: {F30K_SPLIT}")
print()

df30 = pd.read_csv(F30K_SPLIT)
print("Colonne:", df30.columns.tolist())
print("Righe totali:", len(df30))
print("Immagini uniche:", df30['image'].nunique())
print()
print("Distribuzione split (immagini uniche):")
print(df30.groupby('split')['image'].nunique())
print()

train_imgs = set(df30[df30['split'] == 'train']['image'])
val_imgs = set(df30[df30['split'] == 'val']['image'])
test_imgs = set(df30[df30['split'] == 'test']['image'])

print(f"Intersezione train e val:  {len(train_imgs & val_imgs)}")
print(f"Intersezione train e test: {len(train_imgs & test_imgs)}")
print(f"Intersezione val e test:   {len(val_imgs & test_imgs)}")
print()
if (train_imgs & val_imgs) or (train_imgs & test_imgs) or (val_imgs & test_imgs):
    print(">>> ATTENZIONE: c'e' leakage tra split.")
else:
    print(">>> OK: split per immagine, nessun leakage.")

# -----------------------------------------------------------------------------------------------

import shutil
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

to_delete = [
    PROJECT_ROOT / "checkpoints" / "epoch_02",
    PROJECT_ROOT / "checkpoints" / "scenario_B" / "last",
    PROJECT_ROOT / "checkpoints" / "week2_flickr30k" / "last",
]

for path in to_delete:
    if path.exists():
        size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
        shutil.rmtree(path)
        print(f"Cancellato: {path.relative_to(PROJECT_ROOT)}  (liberati {size_mb:.0f} MB)")
    else:
        print(f"Non esiste (gia' cancellato?): {path.relative_to(PROJECT_ROOT)}")

print("\nFatto.")

# -----------------------------------------------------------------------------------------------

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Risultati di ogni esperimento, letti dai file salvati
files = {
    "Scenario A (baseline)":        OUTPUTS_DIR / "bleu_results_baseline.json",
    "Settimana 1 (B su Flickr8k)":  PROJECT_ROOT / "checkpoints" / "scenario_B" / "results_test.json",
    "Settimana 2 (su test F8k)":    OUTPUTS_DIR / "week2_eval_flickr8k.json",
    "Settimana 2 (su test F30k)":   OUTPUTS_DIR / "week2_eval_flickr30k.json",
}

rows = []
for label, path in files.items():
    if not path.exists():
        print(f"!! Mancante: {path}")
        continue
    with open(path) as f:
        d = json.load(f)
    # I JSON hanno nomi di campo leggermente diversi tra loro: gestiamo entrambi
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
print("=== Tabella comparativa esperimenti ===")
print()
print(df_results.to_string(index=False))
print()

# Salvataggio anche in markdown, comodo per il report
md_path = OUTPUTS_DIR / "comparison_table.md"
with open(md_path, "w") as f:
    f.write("# Tabella comparativa esperimenti\n\n")
    f.write(df_results.to_markdown(index=False))
    f.write("\n")
print(f"Tabella salvata anche in: {md_path}")