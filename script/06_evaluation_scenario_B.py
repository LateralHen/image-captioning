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

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "flickr8k"
IMAGES_DIR = DATA_DIR / "Images"
SPLIT_FILE = DATA_DIR / "captions_with_split.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
INITIAL_MODEL_DIR = CHECKPOINT_DIR / "model_initial"
BEST_MODEL_DIR = CHECKPOINT_DIR / "scenario_B" / "best"
MAX_LENGTH = 30

print(f"Device: {device}")
print(f"Best model esiste: {BEST_MODEL_DIR.exists()}")
print(f"Initial model esiste: {INITIAL_MODEL_DIR.exists()}")

# Modello da scenario_B/best, processor e tokenizer dall'initial
model = VisionEncoderDecoderModel.from_pretrained(str(BEST_MODEL_DIR)).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(str(INITIAL_MODEL_DIR))
image_processor = ViTImageProcessor.from_pretrained(str(INITIAL_MODEL_DIR))
tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("Modello caricato.")
print(f"Parametri: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

df = pd.read_csv(SPLIT_FILE)
test_df = df[df['split'] == 'test']
print(f"Test set: {len(test_df)} righe, {test_df['image'].nunique()} immagini uniche")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

df = pd.read_csv(SPLIT_FILE)
test_df = df[df['split'] == 'test']
print(f"Test set: {len(test_df)} righe, {test_df['image'].nunique()} immagini uniche")

@torch.no_grad()
def generate_caption_batch(model, pixel_values_batch, tokenizer, device):
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

unique_imgs = test_df['image'].unique()
generated_captions = []
references_per_image = []
BATCH_SIZE = 16

for batch_start in tqdm(range(0, len(unique_imgs), BATCH_SIZE), desc="Test set"):
    batch_imgs = unique_imgs[batch_start:batch_start + BATCH_SIZE]
    pixel_values_list = []
    for img_name in batch_imgs:
        image = Image.open(IMAGES_DIR / img_name).convert("RGB")
        pixel_values_list.append(eval_transforms(image))
    pixel_values_batch = torch.stack(pixel_values_list).to(device)
    batch_captions = generate_caption_batch(model, pixel_values_batch, tokenizer, device)
    for img_name, gen_cap in zip(batch_imgs, batch_captions):
        generated_captions.append(gen_cap)
        refs = test_df[test_df['image'] == img_name]['caption'].tolist()
        references_per_image.append(refs)

print(f"Generate {len(generated_captions)} didascalie.")

references_transposed = [list(refs) for refs in zip(*references_per_image)]
bleu = sacrebleu.corpus_bleu(generated_captions, references_transposed)

results = {
    "scenario": "B",
    "dataset": "flickr8k_test",
    "n_images": len(generated_captions),
    "BLEU-1": round(bleu.precisions[0], 2),
    "BLEU-2": round(bleu.precisions[1], 2),
    "BLEU-3": round(bleu.precisions[2], 2),
    "BLEU-4": round(bleu.precisions[3], 2),
    "BLEU_corpus": round(bleu.score, 2),
}

print("=== RISULTATI SCENARIO B — TEST SET ===")
for k, v in results.items():
    print(f"  {k}: {v}")

out_path = CHECKPOINT_DIR / "scenario_B" / "results_test.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSalvato in: {out_path}")