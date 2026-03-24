# RNN Shakespeare — Character-Level Text Generation

**Date:** 2026-03-24
**Project:** `4_RNN_Shakespeare`
**Status:** Design approved

---

## 1. Overview & Narrative

CNN projects handled spatial patterns in images. But sequential data — text, time-series, audio — requires understanding order. A CNN sees "to be or not to be" as a bag of characters; an RNN processes them left-to-right, carrying a hidden state that captures context.

This project builds a character-level RNN that learns to write Shakespeare one character at a time from scratch.

**Portfolio position:**
```
MNIST (99%) → CIFAR-10 scratch ceiling (81%) → Transfer Learning (96-98%) → RNN Shakespeare (new domain: sequences)
```

**Key concepts introduced:**
- Sequential processing with hidden states
- Backpropagation Through Time (BPTT)
- Text generation via autoregressive sampling
- Temperature-controlled generation
- Limitations of vanilla RNNs (vanishing gradients → motivation for LSTM)

---

## 2. Dataset

**Tiny Shakespeare** (~1.1MB, ~40,000 lines)
- Source: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- Auto-downloaded in `get_dataloaders()`, same pattern as MNIST/CIFAR-10
- Vocabulary: ~65 unique characters
- Split: 80/10/10 (train/val/test) — contiguous text split, no shuffling across splits

---

## 3. Architecture

**Model: `ShakespeareRNN`**

```
Input char indices → One-hot encoding → nn.RNN (single layer) → nn.Linear → Output logits
```

| Component | Spec | Shape |
|-----------|------|-------|
| Input | Character index sequences | `(N, seq_len)` |
| One-hot | `F.one_hot(x, num_classes=vocab_size).float()` | `(N, seq_len, 65)` |
| RNN | `nn.RNN(input_size=65, hidden_size=512, num_layers=1, batch_first=True)` | `(N, seq_len, 512)` |
| Output | `nn.Linear(512, 65)` | `(N, seq_len, 65)` |

- No softmax in forward (CrossEntropyLoss handles it)
- Hidden size 512: balances capacity and M4 MPS training speed

**Training approach:**
- Slice full text into fixed-length sequences of 100 characters
- Input: characters 0–99, Target: characters 1–100 (next-character prediction)
- CrossEntropyLoss on every timestep, averaged

---

## 4. File Structure

```
4_RNN_Shakespeare/
├── README.md
├── scripts/
│   ├── model.py          # ShakespeareRNN class
│   ├── train.py          # Training loop + early stopping
│   └── utils.py          # Utilities (see below)
├── results/
│   ├── best_model.pth
│   ├── loss_perplexity_curves.png
│   ├── sample_outputs.txt
│   └── training_samples.png
├── notebooks/
└── data/                 # gitignored, auto-downloaded
```

---

## 5. Utilities (utils.py)

| Function | Returns | Notes |
|----------|---------|-------|
| `get_device()` | `torch.device` | MPS > CUDA > CPU (unchanged) |
| `get_dataloaders(batch_size, seq_len)` | `(train_loader, val_loader, test_loader, chars, char_to_idx, idx_to_char)` | Downloads Tiny Shakespeare, builds vocab, creates sequences |
| `train_one_epoch(model, loader, criterion, optimizer, device)` | `(float, float)` | Returns (avg_loss, perplexity) |
| `evaluate(model, loader, criterion, device)` | `(float, float)` | Returns (avg_loss, perplexity) |
| `generate_text(model, device, seed_text, length, temperature, char_to_idx, idx_to_char)` | `str` | Autoregressive character-by-character generation |
| `plot_history(train_losses, val_losses, train_ppls, val_ppls, save_path)` | None | 2-subplot: loss + perplexity curves |
| `plot_training_samples(samples_dict, save_path)` | None | Shows generated text progression across epochs |

**Removed from CNN pattern:** `plot_confusion_matrix()`, `plot_wrong_predictions()` — not applicable to text generation.

---

## 6. Training Plan

| Hyperparameter | Value |
|---------------|-------|
| Sequence length | 100 |
| Batch size | 64 |
| Hidden size | 512 |
| Learning rate | 0.001 (Adam) |
| Max epochs | 50 |
| Early stopping patience | 5 (on val loss) |

**Training loop:**
1. Each epoch: `train_one_epoch()` → `evaluate()` on val set
2. Track loss + perplexity for train and val
3. Save best model on lowest validation loss
4. Early stopping if val loss doesn't improve for 5 epochs
5. Generate a short sample (50 chars) at end of each epoch to visualize learning

**Final evaluation:**
1. Load best model → evaluate on test set → report loss + perplexity
2. Generate text at 3 temperatures (0.5, 1.0, 1.5) → save to `sample_outputs.txt`
3. Save `loss_perplexity_curves.png` and `training_samples.png`

---

## 7. Success Criteria

- Validation loss steadily decreasing over training
- Perplexity dropping from ~65 (random over vocab) toward single digits
- Generated text: recognizable English, Shakespeare-like structure (character names, line breaks, rhythm)
- Imperfect output expected — demonstrates vanilla RNN limitations, motivates LSTM in next project

---

## 8. Teaching Notes

Ryan is a first-time RNN practitioner (read HOML theory, never applied). Every concept must be explained before coding:

- **One-hot encoding:** why we need it for characters, how it differs from image pixels
- **Hidden state:** what it represents, analogy to "memory" while reading
- **BPTT:** connect to manual backprop from DLFS — same chain rule, just unrolled through time
- **Input/target shift:** why input[0:99] maps to target[1:100]
- **Temperature:** logits ÷ T before softmax — low T = conservative, high T = creative
- **Perplexity:** intuitive meaning (how "surprised" the model is) + formula (e^loss)

Pacing: one cell at a time in Colab style. Wait for understanding before moving on.
