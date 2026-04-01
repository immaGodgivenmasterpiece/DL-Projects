import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)                    # 5_LSTM_Sentiment/
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Special tokens — reserved indices at the start of vocabulary
PAD_IDX = 0  # Padding: fills short sequences to fixed length
UNK_IDX = 1  # Unknown: replaces words not in our vocabulary


def get_device():
    """Detect best available device (MPS > CUDA > CPU)."""
    # TODO: fill this in (same pattern as Shakespeare)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def build_vocab(texts, max_size=25000):
    """
    Build word-to-index mapping from training texts.

    Neural networks can't read text — they only understand numbers.
    word_to_idx is the conversion table:  "the" → 2, "movie" → 4, ...
    Later, a review like "the movie was great" becomes [2, 4, 87, 156]
    — a tensor of integers we can feed into nn.Embedding.

    Same idea as Shakespeare's char_to_idx, but at word level (25,000 vs 65).

    Why build from training data only? Same reason we fit a scaler on train only
    in ML — the model shouldn't "see" test vocabulary during training.

    Returns (word_to_idx, idx_to_word).
    """
    # TODO: Step 1 — Count every word across all reviews (use Counter)
    #       Hint: lowercase and split on whitespace
    counter = Counter()

    for text in texts:
        counter.update(text.lower().split())

    # TODO: Step 2 — Keep only the most common words (top max_size)
    most_common_words = counter.most_common(max_size)

    # TODO: Step 3 — Build word_to_idx dict
    #       Reserve index 0 for <pad>, 1 for <unk>
    #       Then assign indices 2, 3, 4... to the most common words
    word_to_idx = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}  # indices 0 and 1 taken

    for word, _ in most_common_words:
        word_to_idx[word] = len(word_to_idx) #from index 2 (len = 2)

    # TODO: Step 4 — Build idx_to_word (reverse mapping)
    idx_to_word = {i: w for w, i in word_to_idx.items()}  # 2 → 'the', 3 → 'a', ...

    # TODO: Print vocab size and return (word_to_idx, idx_to_word)
    print(f"Vocabulary: {len(word_to_idx):,} words (top {max_size} + 2 special tokens)")
    return word_to_idx, idx_to_word

def encode_texts(texts, word_to_idx, max_len=256):
    """
    Tokenize texts and convert to padded index tensors.

    word_to_idx (from build_vocab) creates the lookup table,
    this function uses it to convert raw text → integer tensors.

    Pipeline:
        build_vocab(train_texts)  →  word_to_idx  →  encode_texts(texts, word_to_idx)

    NLP equivalent of image transforms in CNN projects:
    transforms.Resize(224) made all images the same size;
    encode_texts(max_len=256) makes all reviews the same length.

    Each review becomes a fixed-length tensor of word indices:
        "this movie was great" → [45, 892, 23, 156, 0, 0, 0, ...]
                                                      ^^^ padding

    - Words not in vocab → UNK_IDX (1)
    - Sequences longer than max_len → truncated
    - Sequences shorter than max_len → padded with PAD_IDX (0)
    """
    # TODO: For each text:
    #   1. Lowercase and split into tokens
    #   2. Convert each token to its index (use .get() with UNK_IDX as default)
    #   3. Truncate to max_len OR pad with PAD_IDX to max_len
    #   4. Collect all encoded sequences
    encoded = []

    for text in texts:
        tokens = text.lower().split()
        indices = [word_to_idx.get(w, UNK_IDX) for w in tokens]

        # TODO: truncate or pad to max_len
        if len(indices) >= max_len:
            indices = indices[:max_len] #slicing: truncate to max_len 
        else:
            indices = indices + [PAD_IDX] * (max_len - len(indices)) #pad to max_len
        
        encoded.append(indices) #collect indices to encoded

    # TODO: Return as torch.tensor(..., dtype=torch.long)
    return torch.tensor(encoded, dtype=torch.long)


def get_dataloaders(batch_size=64, max_len=256, max_vocab=25000):
    """
    Download IMDB via Hugging Face, build vocab, encode, return DataLoaders.

    Returns (train_loader, val_loader, test_loader, word_to_idx, idx_to_word).
    """

    # TODO: Step 1 — Load IMDB dataset
    #       ds = load_dataset("imdb")
    dataset = load_dataset("imdb")

    # TODO: Step 2 — Extract texts and labels from train and test splits
    # Hint: Hugging Face dataset works like a dictionary of dictionaries: 
    # ds["train"]["text"]  # list of 25,000 review strings
    # ds["train"]["label"] # list of 25,000 labels (0 or 1) 

    train_texts  = dataset["train"]["text"] # list of 25,000 review strings (from train)
    train_labels = dataset["train"]["label"] # list of 25,000 labels (0 or 1) 
    test_texts   = dataset["test"]["text"] 
    test_labels  = dataset["test"]["label"]

    # TODO: Step 3 — Build vocabulary from training texts only
    word_to_idx, idx_to_word = build_vocab(train_texts, max_size=max_vocab)

    # TODO: Step 4 — Encode all texts to padded index tensors
    # use the same word_to_idx (built from training data) for both
    # test data uses the training vocabulary, just like in ML you fit a scaler on train and transform test with it
    train_enc = encode_texts(train_texts, word_to_idx, max_len)
    test_enc  = encode_texts(test_texts, word_to_idx, max_len)

    # TODO: Step 5 — Convert labels to float tensors (BCEWithLogitsLoss needs float)
    train_labels_t = torch.tensor(train_labels, dtype=torch.float32)
    test_labels_t = torch.tensor(test_labels, dtype=torch.float32)

    # TODO: Step 6 — Split training set → 20k train / 5k validation (stratified) -- IMDB dataset has no valid set. 
    #       Use sklearn.model_selection.train_test_split
    #       stratify=train_labels to keep pos/neg ratio balanced (50/50)
    #       random_state=42 for reproducibility

    train_idx, val_idx = train_test_split(
        range(len(train_enc)),      # indices 0..24999
        test_size=5000,             # 5k for validation
        random_state=42,            # reproducible
        stratify=train_labels       # keep pos/neg ratio balanced
    )

    # TODO: Step 7 — Create TensorDatasets and DataLoaders
    #       shuffle=True for train, False for val/test
    train_dataset = TensorDataset(train_enc[train_idx], train_labels_t[train_idx])
    val_dataset   = TensorDataset(train_enc[val_idx],   train_labels_t[val_idx])
    test_dataset  = TensorDataset(test_enc, test_labels_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # TODO: Print sizes for each set and return
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
    return train_loader, val_loader, test_loader, word_to_idx, idx_to_word


# ── Training & Evaluation ────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device,
                    max_grad_norm=1.0) -> tuple[float, float]:
    """
    Run one training epoch. Returns (avg_loss, accuracy).

    Key differences from Shakespeare's train_one_epoch:
    - Returns accuracy instead of perplexity (classification, not generation)
    - Gentler gradient clipping (1.0 vs 5.0) — LSTM gates reduce exploding gradients
    - squeeze(1) on logits to match label shape: (N, 1) → (N,)
    - Accuracy uses sigmoid > 0.5 threshold (not argmax like CNN projects)
    """
    # TODO: fill this in
    pass


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Evaluate model on a dataset. Returns (avg_loss, accuracy)."""
    # TODO: fill this in (same as train_one_epoch but no backward/optimizer/clipping)
    pass


# ── GloVe ────────────────────────────────────────────────────────────────────

def load_glove(glove_path, word_to_idx, embed_dim=100):
    """
    Build an embedding matrix from a GloVe file.

    For each word in our vocabulary, if GloVe has a vector for it,
    use that vector. Otherwise, initialize randomly.

    This is the text equivalent of loading pretrained ResNet weights
    in Project 3 — we start with knowledge someone else learned.

    Returns numpy array of shape (vocab_size, embed_dim).
    """
    # TODO: fill this in (we'll do this when we get to v2)
    pass


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot loss and accuracy curves (same 2-panel style as CNN projects)."""
    # TODO: fill this in
    pass


def plot_confusion_matrix(model, loader, device, class_names=None, save_path=None):
    """Plot 2x2 confusion matrix for binary sentiment classification."""
    # TODO: fill this in
    pass


def plot_wrong_predictions(model, loader, device, idx_to_word, n=10, save_path=None):
    """Show misclassified reviews with true/predicted labels and confidence."""
    # TODO: fill this in
    pass


# ── Smoke Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, test_loader, word_to_idx, idx_to_word = get_dataloaders()
    x, y = next(iter(train_loader))
    print(f"Input shape:  {x.shape}")    # Expected: (64, 256)
    print(f"Labels shape: {y.shape}")    # Expected: (64,)
    print(f"Label values: {y[:10]}")     # Mix of 0.0 and 1.0
    print(f"First 10 words: {[idx_to_word.get(i.item(), '?') for i in x[0][:10]]}")
