import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Path "robusti" che funzionano da qualsiasi cella
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "flickr8k"
IMAGES_DIR = DATA_DIR / "Images"
CAPTIONS_FILE = DATA_DIR / "captions.txt"

print(f"Project root: {PROJECT_ROOT}")
print(f"Images dir esiste: {IMAGES_DIR.exists()}")
print(f"Captions file esiste: {CAPTIONS_FILE.exists()}")

df = pd.read_csv(CAPTIONS_FILE)
print(f"Righe totali: {len(df)}")
print(f"Immagini uniche: {df['image'].nunique()}")
print(f"Didascalie per immagine (atteso 5): {len(df) / df['image'].nunique():.2f}")
print()
print("Prime righe:")
df.head(10)

df = pd.read_csv(CAPTIONS_FILE)
print(f"Righe totali: {len(df)}")
print(f"Immagini uniche: {df['image'].nunique()}")
print(f"Didascalie per immagine (atteso 5): {len(df) / df['image'].nunique():.2f}")
print()
print("Prime righe:")
df.head(10)

df['caption_length'] = df['caption'].str.split().str.len()

print("Statistiche lunghezza didascalie (in parole):")
print(df['caption_length'].describe())
print()
print(f"95° percentile: {df['caption_length'].quantile(0.95):.0f} parole")
print(f"99° percentile: {df['caption_length'].quantile(0.99):.0f} parole")
print(f"Massimo: {df['caption_length'].max()} parole")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['caption_length'], bins=30, edgecolor='black', alpha=0.7)
ax.set_xlabel('Numero di parole per didascalia')
ax.set_ylabel('Numero di didascalie')
ax.set_title('Distribuzione lunghezza didascalie - Flickr8k')
ax.axvline(df['caption_length'].mean(), color='red', linestyle='--',
           label=f"Media: {df['caption_length'].mean():.1f}")
ax.axvline(df['caption_length'].quantile(0.95), color='orange', linestyle='--',
           label=f"95° perc: {df['caption_length'].quantile(0.95):.0f}")
ax.legend()
plt.tight_layout()
plt.show()

from collections import Counter

all_words = []
for caption in df['caption']:
    words = caption.lower().split()
    all_words.extend(words)

word_counts = Counter(all_words)
print(f"Vocabolario totale (split semplice): {len(word_counts)} parole uniche")
print(f"Token totali: {len(all_words)}")
print()
print("Top 25 parole piu' frequenti:")
for word, count in word_counts.most_common(25):
    print(f"  {word:15s} {count:>6}")
    
# Quante parole "rare" ci sono?
total_unique = len(word_counts)
threshold_levels = [1, 2, 3, 5, 10]

print("Coda lunga del vocabolario:")
print(f"{'Soglia min frequenza':<25} {'# parole':<12} {'% del vocabolario':<20}")
for t in threshold_levels:
    n = sum(1 for w, c in word_counts.items() if c < t + 1)
    pct = 100 * n / total_unique
    print(f"  apparse <= {t} volte{'':<6} {n:<12} {pct:.1f}%")

print()

# Coverage: quante delle occorrenze totali sono coperte dalle top-K parole?
sorted_counts = sorted(word_counts.values(), reverse=True)
total_tokens = sum(sorted_counts)

print("Coverage del vocabolario:")
print(f"{'Top-K parole':<15} {'% token coperti':<20}")
for k in [500, 1000, 2000, 3000, 5000, 8000]:
    if k <= len(sorted_counts):
        coverage = 100 * sum(sorted_counts[:k]) / total_tokens
        print(f"  Top {k:<10} {coverage:.2f}%")
        
# Quante parole "rare" ci sono?
total_unique = len(word_counts)
threshold_levels = [1, 2, 3, 5, 10]

print("Coda lunga del vocabolario:")
print(f"{'Soglia min frequenza':<25} {'# parole':<12} {'% del vocabolario':<20}")
for t in threshold_levels:
    n = sum(1 for w, c in word_counts.items() if c < t + 1)
    pct = 100 * n / total_unique
    print(f"  apparse <= {t} volte{'':<6} {n:<12} {pct:.1f}%")

print()

# Coverage: quante delle occorrenze totali sono coperte dalle top-K parole?
sorted_counts = sorted(word_counts.values(), reverse=True)
total_tokens = sum(sorted_counts)

print("Coverage del vocabolario:")
print(f"{'Top-K parole':<15} {'% token coperti':<20}")
for k in [500, 1000, 2000, 3000, 5000, 8000]:
    if k <= len(sorted_counts):
        coverage = 100 * sum(sorted_counts[:k]) / total_tokens
        print(f"  Top {k:<10} {coverage:.2f}%")

SPLIT_FILE = DATA_DIR / "captions_with_split.csv"
df.to_csv(SPLIT_FILE, index=False)
print(f"Split salvato in: {SPLIT_FILE}")
print(f"Dimensione file: {SPLIT_FILE.stat().st_size / 1024:.1f} KB")

from transformers import GPT2Tokenizer

# Carica il tokenizer pre-addestrato. Prima volta: scarica ~1 MB da HuggingFace
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# GPT-2 di default non ha un padding token: usiamo eos_token come pad
tokenizer.pad_token = tokenizer.eos_token

print(f"Vocabolario GPT-2: {tokenizer.vocab_size} token")
print(f"Token speciali: BOS={tokenizer.bos_token!r}, EOS={tokenizer.eos_token!r}")
print()

# Esempio 1: didascalia normale
example1 = "A little girl climbing into a wooden playhouse ."
tokens1 = tokenizer.tokenize(example1)
ids1 = tokenizer.encode(example1)
print(f"Esempio 1: {example1!r}")
print(f"  Token ({len(tokens1)}): {tokens1}")
print(f"  IDs   ({len(ids1)}): {ids1}")
print()

# Esempio 2: con parola "rara" composta
example2 = "A skateboarder performs a kickflip on a graffitied ramp ."
tokens2 = tokenizer.tokenize(example2)
print(f"Esempio 2: {example2!r}")
print(f"  Token ({len(tokens2)}): {tokens2}")
print()

# Esempio 3: con nome proprio
example3 = "A Yorkshire terrier wearing a Chicago Bulls jersey ."
tokens3 = tokenizer.tokenize(example3)
print(f"Esempio 3: {example3!r}")
print(f"  Token ({len(tokens3)}): {tokens3}")

from transformers import GPT2Tokenizer

# Carica il tokenizer pre-addestrato. Prima volta: scarica ~1 MB da HuggingFace
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# GPT-2 di default non ha un padding token: usiamo eos_token come pad
tokenizer.pad_token = tokenizer.eos_token

print(f"Vocabolario GPT-2: {tokenizer.vocab_size} token")
print(f"Token speciali: BOS={tokenizer.bos_token!r}, EOS={tokenizer.eos_token!r}")
print()

# Esempio 1: didascalia normale
example1 = "A little girl climbing into a wooden playhouse ."
tokens1 = tokenizer.tokenize(example1)
ids1 = tokenizer.encode(example1)
print(f"Esempio 1: {example1!r}")
print(f"  Token ({len(tokens1)}): {tokens1}")
print(f"  IDs   ({len(ids1)}): {ids1}")
print()

# Esempio 2: con parola "rara" composta
example2 = "A skateboarder performs a kickflip on a graffitied ramp ."
tokens2 = tokenizer.tokenize(example2)
print(f"Esempio 2: {example2!r}")
print(f"  Token ({len(tokens2)}): {tokens2}")
print()

# Esempio 3: con nome proprio
example3 = "A Yorkshire terrier wearing a Chicago Bulls jersey ."
tokens3 = tokenizer.tokenize(example3)
print(f"Esempio 3: {example3!r}")
print(f"  Token ({len(tokens3)}): {tokens3}")

from tqdm import tqdm

# Tokenizziamo tutte le didascalie con barra di progresso
print("Tokenizzazione di tutte le didascalie...")
token_lengths = []
for caption in tqdm(df['caption'].tolist()):
    # Aggiungiamo BOS/EOS come faremo in training
    text = tokenizer.bos_token + " " + caption + " " + tokenizer.eos_token
    n_tokens = len(tokenizer.encode(text))
    token_lengths.append(n_tokens)

df['n_tokens'] = token_lengths

print()
print("Statistiche lunghezza didascalie (in token GPT-2, BOS+caption+EOS):")
print(df['n_tokens'].describe())
print()
print(f"95° percentile: {df['n_tokens'].quantile(0.95):.0f} token")
print(f"99° percentile: {df['n_tokens'].quantile(0.99):.0f} token")
print(f"Massimo:        {df['n_tokens'].max()} token")
print()

# Quante didascalie supereranno max_length=30 e verranno troncate?
for ml in [25, 30, 32, 40]:
    n_over = (df['n_tokens'] > ml).sum()
    pct = 100 * n_over / len(df)
    print(f"  max_length = {ml}: {n_over} didascalie troncate ({pct:.2f}%)")
    
SPLIT_FILE = DATA_DIR / "captions_with_split.csv"
df.to_csv(SPLIT_FILE, index=False)
print(f"File aggiornato salvato in: {SPLIT_FILE}")
print(f"Colonne disponibili: {list(df.columns)}")
print(f"Dimensione: {SPLIT_FILE.stat().st_size / 1024:.1f} KB")

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Path "robusti" che funzionano da qualsiasi cella
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "flickr8k"
IMAGES_DIR = DATA_DIR / "Images"
CAPTIONS_FILE = DATA_DIR / "captions.txt"

print(f"Project root: {PROJECT_ROOT}")
print(f"Images dir esiste: {IMAGES_DIR.exists()}")
print(f"Captions file esiste: {CAPTIONS_FILE.exists()}")

df = pd.read_csv(CAPTIONS_FILE)
print(f"Righe totali: {len(df)}")
print(f"Immagini uniche: {df['image'].nunique()}")
print(f"Didascalie per immagine (atteso 5): {len(df) / df['image'].nunique():.2f}")
print()
print("Prime righe:")
df.head(10)

df = pd.read_csv(CAPTIONS_FILE)
print(f"Righe totali: {len(df)}")
print(f"Immagini uniche: {df['image'].nunique()}")
print(f"Didascalie per immagine (atteso 5): {len(df) / df['image'].nunique():.2f}")
print()
print("Prime righe:")
df.head(10)

df['caption_length'] = df['caption'].str.split().str.len()

print("Statistiche lunghezza didascalie (in parole):")
print(df['caption_length'].describe())
print()
print(f"95° percentile: {df['caption_length'].quantile(0.95):.0f} parole")
print(f"99° percentile: {df['caption_length'].quantile(0.99):.0f} parole")
print(f"Massimo: {df['caption_length'].max()} parole")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['caption_length'], bins=30, edgecolor='black', alpha=0.7)
ax.set_xlabel('Numero di parole per didascalia')
ax.set_ylabel('Numero di didascalie')
ax.set_title('Distribuzione lunghezza didascalie - Flickr8k')
ax.axvline(df['caption_length'].mean(), color='red', linestyle='--',
           label=f"Media: {df['caption_length'].mean():.1f}")
ax.axvline(df['caption_length'].quantile(0.95), color='orange', linestyle='--',
           label=f"95° perc: {df['caption_length'].quantile(0.95):.0f}")
ax.legend()
plt.tight_layout()
plt.show()

from collections import Counter

all_words = []
for caption in df['caption']:
    words = caption.lower().split()
    all_words.extend(words)

word_counts = Counter(all_words)
print(f"Vocabolario totale (split semplice): {len(word_counts)} parole uniche")
print(f"Token totali: {len(all_words)}")
print()
print("Top 25 parole piu' frequenti:")
for word, count in word_counts.most_common(25):
    print(f"  {word:15s} {count:>6}")
    
# Quante parole "rare" ci sono?
total_unique = len(word_counts)
threshold_levels = [1, 2, 3, 5, 10]

print("Coda lunga del vocabolario:")
print(f"{'Soglia min frequenza':<25} {'# parole':<12} {'% del vocabolario':<20}")
for t in threshold_levels:
    n = sum(1 for w, c in word_counts.items() if c < t + 1)
    pct = 100 * n / total_unique
    print(f"  apparse <= {t} volte{'':<6} {n:<12} {pct:.1f}%")

print()

# Coverage: quante delle occorrenze totali sono coperte dalle top-K parole?
sorted_counts = sorted(word_counts.values(), reverse=True)
total_tokens = sum(sorted_counts)

print("Coverage del vocabolario:")
print(f"{'Top-K parole':<15} {'% token coperti':<20}")
for k in [500, 1000, 2000, 3000, 5000, 8000]:
    if k <= len(sorted_counts):
        coverage = 100 * sum(sorted_counts[:k]) / total_tokens
        print(f"  Top {k:<10} {coverage:.2f}%")
        
# Quante parole "rare" ci sono?
total_unique = len(word_counts)
threshold_levels = [1, 2, 3, 5, 10]

print("Coda lunga del vocabolario:")
print(f"{'Soglia min frequenza':<25} {'# parole':<12} {'% del vocabolario':<20}")
for t in threshold_levels:
    n = sum(1 for w, c in word_counts.items() if c < t + 1)
    pct = 100 * n / total_unique
    print(f"  apparse <= {t} volte{'':<6} {n:<12} {pct:.1f}%")

print()

# Coverage: quante delle occorrenze totali sono coperte dalle top-K parole?
sorted_counts = sorted(word_counts.values(), reverse=True)
total_tokens = sum(sorted_counts)

print("Coverage del vocabolario:")
print(f"{'Top-K parole':<15} {'% token coperti':<20}")
for k in [500, 1000, 2000, 3000, 5000, 8000]:
    if k <= len(sorted_counts):
        coverage = 100 * sum(sorted_counts[:k]) / total_tokens
        print(f"  Top {k:<10} {coverage:.2f}%")

SPLIT_FILE = DATA_DIR / "captions_with_split.csv"
df.to_csv(SPLIT_FILE, index=False)
print(f"Split salvato in: {SPLIT_FILE}")
print(f"Dimensione file: {SPLIT_FILE.stat().st_size / 1024:.1f} KB")

from transformers import GPT2Tokenizer

# Carica il tokenizer pre-addestrato. Prima volta: scarica ~1 MB da HuggingFace
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# GPT-2 di default non ha un padding token: usiamo eos_token come pad
tokenizer.pad_token = tokenizer.eos_token

print(f"Vocabolario GPT-2: {tokenizer.vocab_size} token")
print(f"Token speciali: BOS={tokenizer.bos_token!r}, EOS={tokenizer.eos_token!r}")
print()

# Esempio 1: didascalia normale
example1 = "A little girl climbing into a wooden playhouse ."
tokens1 = tokenizer.tokenize(example1)
ids1 = tokenizer.encode(example1)
print(f"Esempio 1: {example1!r}")
print(f"  Token ({len(tokens1)}): {tokens1}")
print(f"  IDs   ({len(ids1)}): {ids1}")
print()

# Esempio 2: con parola "rara" composta
example2 = "A skateboarder performs a kickflip on a graffitied ramp ."
tokens2 = tokenizer.tokenize(example2)
print(f"Esempio 2: {example2!r}")
print(f"  Token ({len(tokens2)}): {tokens2}")
print()

# Esempio 3: con nome proprio
example3 = "A Yorkshire terrier wearing a Chicago Bulls jersey ."
tokens3 = tokenizer.tokenize(example3)
print(f"Esempio 3: {example3!r}")
print(f"  Token ({len(tokens3)}): {tokens3}")

from transformers import GPT2Tokenizer

# Carica il tokenizer pre-addestrato. Prima volta: scarica ~1 MB da HuggingFace
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# GPT-2 di default non ha un padding token: usiamo eos_token come pad
tokenizer.pad_token = tokenizer.eos_token

print(f"Vocabolario GPT-2: {tokenizer.vocab_size} token")
print(f"Token speciali: BOS={tokenizer.bos_token!r}, EOS={tokenizer.eos_token!r}")
print()

# Esempio 1: didascalia normale
example1 = "A little girl climbing into a wooden playhouse ."
tokens1 = tokenizer.tokenize(example1)
ids1 = tokenizer.encode(example1)
print(f"Esempio 1: {example1!r}")
print(f"  Token ({len(tokens1)}): {tokens1}")
print(f"  IDs   ({len(ids1)}): {ids1}")
print()

# Esempio 2: con parola "rara" composta
example2 = "A skateboarder performs a kickflip on a graffitied ramp ."
tokens2 = tokenizer.tokenize(example2)
print(f"Esempio 2: {example2!r}")
print(f"  Token ({len(tokens2)}): {tokens2}")
print()

# Esempio 3: con nome proprio
example3 = "A Yorkshire terrier wearing a Chicago Bulls jersey ."
tokens3 = tokenizer.tokenize(example3)
print(f"Esempio 3: {example3!r}")
print(f"  Token ({len(tokens3)}): {tokens3}")

from tqdm import tqdm

# Tokenizziamo tutte le didascalie con barra di progresso
print("Tokenizzazione di tutte le didascalie...")
token_lengths = []
for caption in tqdm(df['caption'].tolist()):
    # Aggiungiamo BOS/EOS come faremo in training
    text = tokenizer.bos_token + " " + caption + " " + tokenizer.eos_token
    n_tokens = len(tokenizer.encode(text))
    token_lengths.append(n_tokens)

df['n_tokens'] = token_lengths

print()
print("Statistiche lunghezza didascalie (in token GPT-2, BOS+caption+EOS):")
print(df['n_tokens'].describe())
print()
print(f"95° percentile: {df['n_tokens'].quantile(0.95):.0f} token")
print(f"99° percentile: {df['n_tokens'].quantile(0.99):.0f} token")
print(f"Massimo:        {df['n_tokens'].max()} token")
print()

# Quante didascalie supereranno max_length=30 e verranno troncate?
for ml in [25, 30, 32, 40]:
    n_over = (df['n_tokens'] > ml).sum()
    pct = 100 * n_over / len(df)
    print(f"  max_length = {ml}: {n_over} didascalie troncate ({pct:.2f}%)")
    
SPLIT_FILE = DATA_DIR / "captions_with_split.csv"
df.to_csv(SPLIT_FILE, index=False)
print(f"File aggiornato salvato in: {SPLIT_FILE}")
print(f"Colonne disponibili: {list(df.columns)}")
print(f"Dimensione: {SPLIT_FILE.stat().st_size / 1024:.1f} KB")