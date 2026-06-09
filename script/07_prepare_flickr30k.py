"""
07_prepare_flickr30k.py
=======================
Preparazione del dataset Flickr30k per la Settimana 2 (upgrade dataset).

Obiettivo: Flickr30k arriva in un formato leggermente diverso da Flickr8k.
Questo script lo uniforma e crea lo split train/val/test, producendo lo stesso
file 'captions_with_split.csv' che gli script di training/evaluation si aspettano.

Punto critico: lo split e' fatto PER IMMAGINE, non per riga. Ogni immagine ha
5 didascalie: se finissero in split diversi avremmo data leakage (il modello
vedrebbe in training un'immagine poi usata in test). Lo script garantisce e
verifica esplicitamente l'assenza di sovrapposizioni tra gli split.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------------
# SETUP: RIPRODUCIBILITA' E PATH
# ----------------------------------------------------------------------------
# Seed di NumPy: rende deterministico lo shuffle delle immagini (quindi lo split
# e' sempre lo stesso ad ogni esecuzione).
np.random.seed(42)

# Path robusti (funzionano sia da 'notebooks/' sia dalla root del progetto).
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "flickr30k"
IMAGES_DIR = DATA_DIR / "flickr30k_images"          # cartella immagini (nome diverso da Flickr8k)
CAPTIONS_FILE = DATA_DIR / "captions.txt"           # file didascalie originale di Flickr30k
SPLIT_FILE = DATA_DIR / "captions_with_split.csv"   # output: CSV uniformato con la colonna 'split'

print(f"Project root: {PROJECT_ROOT}")
print(f"Images dir esiste: {IMAGES_DIR.exists()}")
print(f"Captions file esiste: {CAPTIONS_FILE.exists()}")

# ----------------------------------------------------------------------------
# 1. CARICAMENTO DELLE DIDASCALIE
# ----------------------------------------------------------------------------
df = pd.read_csv(CAPTIONS_FILE)
print(f"\nColonne originali: {list(df.columns)}")   # in Flickr30k sono 'image_name' e 'comment'
print(f"Righe totali: {len(df)}")
print(f"Prime righe:")
print(df.head())

# Rinominiamo le colonne per uniformarle a Flickr8k ('image', 'caption'): cosi'
# gli script di training/evaluation funzionano senza modifiche su entrambi i dataset.
df = df.rename(columns={"image_name": "image", "comment": "caption"})
df = df[["image", "caption"]].copy()                # teniamo solo le due colonne che ci servono

print(f"\nImmagini uniche: {df['image'].nunique()}")
print(f"Didascalie per immagine: {len(df) / df['image'].nunique():.2f}")

# ----------------------------------------------------------------------------
# 2. VERIFICA CHE LE IMMAGINI ESISTANO SU DISCO
# ----------------------------------------------------------------------------
# Controlliamo che ogni immagine citata nel CSV abbia il file corrispondente.
# Un mismatch qui causerebbe errori in fase di training/eval.
missing = [img for img in df['image'].unique() if not (IMAGES_DIR / img).exists()]
print(f"\nImmagini mancanti su disco: {len(missing)}")
if missing:
    print(f"  Esempi: {missing[:5]}")

# ----------------------------------------------------------------------------
# 3. SPLIT PER IMMAGINE (senza leakage)
# ----------------------------------------------------------------------------
# Proporzioni scelte simili a Flickr8k: 88% train, 6% val, 6% test.
unique_imgs = df['image'].unique()
n_total = len(unique_imgs)
n_test = int(n_total * 0.06)
n_val = int(n_total * 0.06)
n_train = n_total - n_val - n_test   # il resto va in train

print(f"\nSplit immagini:")
print(f"  Train: {n_train} ({n_train/n_total*100:.1f}%)")
print(f"  Val:   {n_val}   ({n_val/n_total*100:.1f}%)")
print(f"  Test:  {n_test}  ({n_test/n_total*100:.1f}%)")

# Shuffle deterministico (grazie al seed fisso): mescoliamo le IMMAGINI uniche
# prima di tagliarle nei tre gruppi.
shuffled_imgs = unique_imgs.copy()
np.random.shuffle(shuffled_imgs)

# Assegnazione: i primi n_test vanno in test, i successivi n_val in val, il resto in train.
# Usiamo dei set per rendere veloce il test di appartenenza piu' avanti.
test_imgs  = set(shuffled_imgs[:n_test])
val_imgs   = set(shuffled_imgs[n_test:n_test + n_val])
train_imgs = set(shuffled_imgs[n_test + n_val:])

# Verifica esplicita di NO leakage: i tre insiemi di immagini devono essere disgiunti.
# Se non lo fossero, il modello verrebbe valutato su immagini gia' viste -> assert.
assert len(train_imgs & val_imgs) == 0, "Overlap train/val!"
assert len(train_imgs & test_imgs) == 0, "Overlap train/test!"
assert len(val_imgs & test_imgs) == 0, "Overlap val/test!"
print("\nVerifica no leakage: OK")

# Funzione che mappa ogni immagine al suo split di appartenenza.
def assign_split(img_name):
    if img_name in train_imgs:
        return "train"
    elif img_name in val_imgs:
        return "val"
    else:
        return "test"

# Applichiamo la funzione a tutte le righe: tutte e 5 le didascalie di una stessa
# immagine ricevono lo stesso split (questo e' cio' che evita il leakage).
df['split'] = df['image'].apply(assign_split)

# ----------------------------------------------------------------------------
# 4. VERIFICA DELLA DISTRIBUZIONE FINALE
# ----------------------------------------------------------------------------
# Numero di RIGHE (didascalie) per split.
print(f"\nDistribuzione righe per split:")
print(df['split'].value_counts())
# Numero di IMMAGINI uniche per split (dovrebbe rispettare le proporzioni 88/6/6).
print(f"\nDistribuzione immagini per split:")
print(df.groupby('split')['image'].nunique())

# ----------------------------------------------------------------------------
# 5. STATISTICHE SULLE DIDASCALIE
# ----------------------------------------------------------------------------
# Lunghezza in parole: utile per confrontare Flickr30k con Flickr8k e confermare
# che MAX_LENGTH=30 resta una scelta valida anche qui.
df['caption_length'] = df['caption'].str.split().str.len()
print(f"\nStatistiche lunghezza didascalie (parole):")
print(df['caption_length'].describe())
print(f"95° percentile: {df['caption_length'].quantile(0.95):.0f} parole")

# ----------------------------------------------------------------------------
# 6. SALVATAGGIO
# ----------------------------------------------------------------------------
df.to_csv(SPLIT_FILE, index=False)
print(f"\nFile salvato in: {SPLIT_FILE}")
print(f"Dimensione: {SPLIT_FILE.stat().st_size / 1024:.1f} KB")
print(f"Colonne: {list(df.columns)}")

# Sanity check finale: rileggiamo il file appena scritto per essere certi che
# sia stato salvato correttamente e leggibile dagli script successivi.
df_check = pd.read_csv(SPLIT_FILE)
print(f"\nSanity check lettura:")
print(f"  Righe: {len(df_check)}")
print(f"  Colonne: {list(df_check.columns)}")
print(f"  Split unici: {df_check['split'].unique()}")
print(df_check.head())