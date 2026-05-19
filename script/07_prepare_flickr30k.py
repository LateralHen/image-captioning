import pandas as pd
import numpy as np
from pathlib import Path

# Riproducibilita'
np.random.seed(42)

# Path del progetto
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "flickr30k"
IMAGES_DIR = DATA_DIR / "flickr30k_images"
CAPTIONS_FILE = DATA_DIR / "captions.txt"
SPLIT_FILE = DATA_DIR / "captions_with_split.csv"

print(f"Project root: {PROJECT_ROOT}")
print(f"Images dir esiste: {IMAGES_DIR.exists()}")
print(f"Captions file esiste: {CAPTIONS_FILE.exists()}")

# === 1. Carica captions ===
df = pd.read_csv(CAPTIONS_FILE)
print(f"\nColonne originali: {list(df.columns)}")
print(f"Righe totali: {len(df)}")
print(f"Prime righe:")
print(df.head())

# Rinomina colonne per uniformita' con Flickr8k
df = df.rename(columns={"image_name": "image", "comment": "caption"})
df = df[["image", "caption"]].copy()

print(f"\nImmagini uniche: {df['image'].nunique()}")
print(f"Didascalie per immagine: {len(df) / df['image'].nunique():.2f}")

# === 2. Verifica immagini su disco ===
missing = [img for img in df['image'].unique() if not (IMAGES_DIR / img).exists()]
print(f"\nImmagini mancanti su disco: {len(missing)}")
if missing:
    print(f"  Esempi: {missing[:5]}")

# === 3. Split per immagine (no leakage) ===
# Proporzioni: 88% train, 6% val, 6% test (simili a Flickr8k)
unique_imgs = df['image'].unique()
n_total = len(unique_imgs)
n_test = int(n_total * 0.06)
n_val = int(n_total * 0.06)
n_train = n_total - n_val - n_test

print(f"\nSplit immagini:")
print(f"  Train: {n_train} ({n_train/n_total*100:.1f}%)")
print(f"  Val:   {n_val}   ({n_val/n_total*100:.1f}%)")
print(f"  Test:  {n_test}  ({n_test/n_total*100:.1f}%)")

# Shuffle deterministico (seed fisso)
shuffled_imgs = unique_imgs.copy()
np.random.shuffle(shuffled_imgs)

test_imgs  = set(shuffled_imgs[:n_test])
val_imgs   = set(shuffled_imgs[n_test:n_test + n_val])
train_imgs = set(shuffled_imgs[n_test + n_val:])

# Verifica no overlap
assert len(train_imgs & val_imgs) == 0, "Overlap train/val!"
assert len(train_imgs & test_imgs) == 0, "Overlap train/test!"
assert len(val_imgs & test_imgs) == 0, "Overlap val/test!"
print("\nVerifica no leakage: OK")

# Assegna split
def assign_split(img_name):
    if img_name in train_imgs:
        return "train"
    elif img_name in val_imgs:
        return "val"
    else:
        return "test"

df['split'] = df['image'].apply(assign_split)

# === 4. Verifica distribuzione finale ===
print(f"\nDistribuzione righe per split:")
print(df['split'].value_counts())
print(f"\nDistribuzione immagini per split:")
print(df.groupby('split')['image'].nunique())

# === 5. Statistiche didascalie ===
df['caption_length'] = df['caption'].str.split().str.len()
print(f"\nStatistiche lunghezza didascalie (parole):")
print(df['caption_length'].describe())
print(f"95° percentile: {df['caption_length'].quantile(0.95):.0f} parole")

# === 6. Salva ===
df.to_csv(SPLIT_FILE, index=False)
print(f"\nFile salvato in: {SPLIT_FILE}")
print(f"Dimensione: {SPLIT_FILE.stat().st_size / 1024:.1f} KB")
print(f"Colonne: {list(df.columns)}")

# Sanity check finale: leggi il file e verifica
df_check = pd.read_csv(SPLIT_FILE)
print(f"\nSanity check lettura:")
print(f"  Righe: {len(df_check)}")
print(f"  Colonne: {list(df_check.columns)}")
print(f"  Split unici: {df_check['split'].unique()}")
print(df_check.head())