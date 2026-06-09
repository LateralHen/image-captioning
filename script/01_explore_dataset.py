"""
Esplorazione e analisi preliminare del dataset Flickr8k.

Obiettivo dello script: capire la struttura dei dati PRIMA di addestrare
qualsiasi modello. In particolare misuriamo:
  1. quante immagini e quante didascalie abbiamo, e il rapporto fra le due;
  2. la distribuzione della lunghezza delle didascalie (in parole e in token);
  3. la composizione del vocabolario (parole frequenti, parole rare, coverage);
  4. come il tokenizer di GPT-2 spezza le frasi, per scegliere un max_length sensato.

Tutte queste misure servono a giustificare le scelte di iperparametri
fatte negli script successivi (in particolare MAX_LENGTH = 30).

Lo script produce in output il file 'captions_with_split.csv', arricchito
con le colonne di lunghezza, che verra' riusato dagli script 02/03/04.
"""

import pandas as pd                  # gestione del CSV delle didascalie come DataFrame
from pathlib import Path             # costruzione di path indipendente dal sistema operativo
import matplotlib.pyplot as plt      # istogramma della distribuzione delle lunghezze
from PIL import Image                # apertura immagini (importato per coerenza con gli altri script)
from collections import Counter      # conteggio frequenze delle parole del vocabolario
from transformers import GPT2Tokenizer  # tokenizer pre-addestrato di GPT-2
from tqdm import tqdm                # barra di avanzamento per il loop di tokenizzazione

# ----------------------------------------------------------------------------
# 1. PATH DEL PROGETTO
# ----------------------------------------------------------------------------
# Lo script puo' essere eseguito sia dalla cartella 'notebooks/' sia dalla
# root del progetto: questa riga rende i path "robusti" in entrambi i casi.
# Se siamo dentro 'notebooks', risaliamo di un livello; altrimenti usiamo la
# cartella corrente.
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "flickr8k"   # cartella dati del dataset Flickr8k
IMAGES_DIR = DATA_DIR / "Images"                # sottocartella con i file .jpg
CAPTIONS_FILE = DATA_DIR / "captions.txt"       # file con le didascalie (formato CSV: image,caption)

# Verifica che i path esistano davvero prima di proseguire (debug rapido).
print(f"Project root: {PROJECT_ROOT}")
print(f"Images dir esiste: {IMAGES_DIR.exists()}")
print(f"Captions file esiste: {CAPTIONS_FILE.exists()}")

# ----------------------------------------------------------------------------
# 2. CARICAMENTO E STRUTTURA DI BASE DEL DATASET
# ----------------------------------------------------------------------------
df = pd.read_csv(CAPTIONS_FILE)                        # legge il CSV in un DataFrame
print(f"Righe totali: {len(df)}")                      # numero totale di didascalie
print(f"Immagini uniche: {df['image'].nunique()}")     # numero di immagini distinte
# Rapporto didascalie/immagine: in Flickr8k ogni immagine ha 5 didascalie umane.
print(f"Didascalie per immagine (atteso 5): {len(df) / df['image'].nunique():.2f}")
print()
print("Prime righe:")
df.head(10)                                            # anteprima delle prime 10 righe

# ----------------------------------------------------------------------------
# 3. DISTRIBUZIONE DELLA LUNGHEZZA DELLE DIDASCALIE (IN PAROLE)
# ----------------------------------------------------------------------------
# Aggiungiamo una colonna con il numero di parole di ogni didascalia
# (split sugli spazi: conteggio grezzo, non token del modello).
df['caption_length'] = df['caption'].str.split().str.len()

print("Statistiche lunghezza didascalie (in parole):")
print(df['caption_length'].describe())                 # media, std, min, max, quartili
print()
# I percentili alti ci dicono dove "tagliare": se il 95% delle frasi sta sotto
# N parole, un max_length vicino a N copre quasi tutto senza sprechi.
print(f"95° percentile: {df['caption_length'].quantile(0.95):.0f} parole")
print(f"99° percentile: {df['caption_length'].quantile(0.99):.0f} parole")
print(f"Massimo: {df['caption_length'].max()} parole")

# Istogramma della distribuzione: visualizza la forma (tipicamente asimmetrica
# a destra, con poche didascalie molto lunghe).
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['caption_length'], bins=30, edgecolor='black', alpha=0.7)
ax.set_xlabel('Numero di parole per didascalia')
ax.set_ylabel('Numero di didascalie')
ax.set_title('Distribuzione lunghezza didascalie - Flickr8k')
# Linea verticale sulla media.
ax.axvline(df['caption_length'].mean(), color='red', linestyle='--',
           label=f"Media: {df['caption_length'].mean():.1f}")
# Linea verticale sul 95° percentile (soglia pratica per max_length).
ax.axvline(df['caption_length'].quantile(0.95), color='orange', linestyle='--',
           label=f"95° perc: {df['caption_length'].quantile(0.95):.0f}")
ax.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 4. ANALISI DEL VOCABOLARIO
# ----------------------------------------------------------------------------
# Costruiamo la lista di tutte le parole (lowercase, split sugli spazi).
# Nota: e' un conteggio grezzo, NON i token di GPT-2; serve solo a farsi
# un'idea della ricchezza lessicale del dataset.
all_words = []
for caption in df['caption']:
    words = caption.lower().split()
    all_words.extend(words)

word_counts = Counter(all_words)                       # dizionario {parola: frequenza}
print(f"Vocabolario totale (split semplice): {len(word_counts)} parole uniche")
print(f"Token totali: {len(all_words)}")
print()
print("Top 25 parole piu' frequenti:")
# Le parole piu' frequenti sono quasi sempre articoli, preposizioni, "a", "the"...
for word, count in word_counts.most_common(25):
    print(f"  {word:15s} {count:>6}")

# --- Coda lunga del vocabolario (parole rare) ---
# Quante parole compaiono pochissime volte? Una coda lunga molto grande
# significa molte parole rare, difficili da imparare per il modello.
total_unique = len(word_counts)
threshold_levels = [1, 2, 3, 5, 10]

print("Coda lunga del vocabolario:")
print(f"{'Soglia min frequenza':<25} {'# parole':<12} {'% del vocabolario':<20}")
for t in threshold_levels:
    # Conta le parole apparse al massimo t volte (c < t+1  ⇔  c <= t).
    n = sum(1 for w, c in word_counts.items() if c < t + 1)
    pct = 100 * n / total_unique
    print(f"  apparse <= {t} volte{'':<6} {n:<12} {pct:.1f}%")

print()

# --- Coverage del vocabolario ---
# Domanda: prendendo solo le K parole piu' frequenti, quale percentuale di
# TUTTE le occorrenze copriamo? Tipicamente poche migliaia di parole coprono
# la stragrande maggioranza del testo (legge di Zipf).
sorted_counts = sorted(word_counts.values(), reverse=True)  # frequenze in ordine decrescente
total_tokens = sum(sorted_counts)

print("Coverage del vocabolario:")
print(f"{'Top-K parole':<15} {'% token coperti':<20}")
for k in [500, 1000, 2000, 3000, 5000, 8000]:
    if k <= len(sorted_counts):
        # Somma delle frequenze delle prime K parole / totale occorrenze.
        coverage = 100 * sum(sorted_counts[:k]) / total_tokens
        print(f"  Top {k:<10} {coverage:.2f}%")

# ----------------------------------------------------------------------------
# 5. COMPORTAMENTO DEL TOKENIZER GPT-2
# ----------------------------------------------------------------------------
# Carica il tokenizer pre-addestrato di GPT-2 (BPE, ~50k token).
# Prima volta: scarica ~1 MB da HuggingFace.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# GPT-2 non ha un token di padding nativo: riusiamo l'eos_token come pad.
# (Stessa convenzione che adotteremo in training/inference negli altri script.)
tokenizer.pad_token = tokenizer.eos_token

print(f"Vocabolario GPT-2: {tokenizer.vocab_size} token")
print(f"Token speciali: BOS={tokenizer.bos_token!r}, EOS={tokenizer.eos_token!r}")
print()

# Tre esempi mostrano COME il BPE spezza le parole. Importante perche' una
# parola rara puo' diventare piu' token, allungando la sequenza.

# Esempio 1: didascalia normale, parole comuni.
example1 = "A little girl climbing into a wooden playhouse ."
tokens1 = tokenizer.tokenize(example1)      # lista dei sub-token (stringhe)
ids1 = tokenizer.encode(example1)           # lista degli ID numerici corrispondenti
print(f"Esempio 1: {example1!r}")
print(f"  Token ({len(tokens1)}): {tokens1}")
print(f"  IDs   ({len(ids1)}): {ids1}")
print()

# Esempio 2: parola composta/rara ("kickflip", "graffitied"): il BPE la spezza
# in piu' sub-token, quindi piu' token del previsto.
example2 = "A skateboarder performs a kickflip on a graffitied ramp ."
tokens2 = tokenizer.tokenize(example2)
print(f"Esempio 2: {example2!r}")
print(f"  Token ({len(tokens2)}): {tokens2}")
print()

# Esempio 3: nomi propri ("Yorkshire", "Chicago Bulls"): anche questi
# spesso vengono spezzati in piu' sub-token.
example3 = "A Yorkshire terrier wearing a Chicago Bulls jersey ."
tokens3 = tokenizer.tokenize(example3)
print(f"Esempio 3: {example3!r}")
print(f"  Token ({len(tokens3)}): {tokens3}")

# ----------------------------------------------------------------------------
# 6. LUNGHEZZA DELLE DIDASCALIE IN TOKEN GPT-2 (per scegliere MAX_LENGTH)
# ----------------------------------------------------------------------------
# La lunghezza in PAROLE (sezione 3) non basta: il modello lavora sui token.
# Qui ricalcoliamo la lunghezza in token GPT-2, includendo BOS+caption+EOS
# esattamente come faremo in training, per scegliere un max_length corretto.
print("Tokenizzazione di tutte le didascalie...")
token_lengths = []
for caption in tqdm(df['caption'].tolist()):
    # Aggiungiamo i marcatori di inizio/fine sequenza, come in training.
    text = tokenizer.bos_token + " " + caption + " " + tokenizer.eos_token
    n_tokens = len(tokenizer.encode(text))   # numero di token della frase completa
    token_lengths.append(n_tokens)

df['n_tokens'] = token_lengths               # nuova colonna con la lunghezza in token

print()
print("Statistiche lunghezza didascalie (in token GPT-2, BOS+caption+EOS):")
print(df['n_tokens'].describe())
print()
print(f"95° percentile: {df['n_tokens'].quantile(0.95):.0f} token")
print(f"99° percentile: {df['n_tokens'].quantile(0.99):.0f} token")
print(f"Massimo:        {df['n_tokens'].max()} token")
print()

# Verifica diretta dell'effetto di max_length: quante didascalie verrebbero
# troncate per ogni soglia candidata? Questo giustifica la scelta MAX_LENGTH=30
# (troncamento trascurabile, poche frasi perse).
for ml in [25, 30, 32, 40]:
    n_over = (df['n_tokens'] > ml).sum()     # quante superano la soglia ml
    pct = 100 * n_over / len(df)
    print(f"  max_length = {ml}: {n_over} didascalie troncate ({pct:.2f}%)")

# ----------------------------------------------------------------------------
# 7. SALVATAGGIO DEL DATASET ARRICCHITO
# ----------------------------------------------------------------------------
# Salviamo il DataFrame con le colonne aggiuntive ('caption_length', 'n_tokens')
# in un nuovo CSV. Questo file e' l'output di questo script e l'input degli
# script successivi (02/03/04).
SPLIT_FILE = DATA_DIR / "captions_with_split.csv"
df.to_csv(SPLIT_FILE, index=False)
print(f"File aggiornato salvato in: {SPLIT_FILE}")
print(f"Colonne disponibili: {list(df.columns)}")
print(f"Dimensione: {SPLIT_FILE.stat().st_size / 1024:.1f} KB")