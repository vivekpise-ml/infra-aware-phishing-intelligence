"""
train_char_cnn.py

Train a lightweight, CPU-friendly character-level 1D-CNN for URL binary classification
(benign vs. malicious). Designed to work on a laptop with ~600k samples.

Outputs:
 - models/charcnn_model.pt      (model state_dict)
 - models/charcnn_vocab.json    (char -> idx mapping)
 - models/charcnn_config.json   (maxlen, vocab_size, etc.)

Minimal, reproducible, and easy to integrate with Streamlit.
"""

import os
import json
import time
from collections import Counter
from typing import List

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Config (tweak these if needed)
# ---------------------------
DEFAULT_MODEL_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CharCNN hyperparams tuned for CPU
EMBED_DIM = 64
NUM_FILTERS = 128
KERNEL_SIZES = [3, 4, 5]
DROPOUT = 0.3
FC_DIM = 128

BATCH_SIZE = 512        # large batch for CPU RAM; reduce if OOM
EPOCHS = 6
LR = 3e-4
MAXLEN = 200            # max chars kept from each URL


# ---------------------------
# Utilities: tokenizer / vocab
# ---------------------------
def build_char_vocab(texts: List[str], min_freq=1):
    """
    Build a char->idx mapping from given texts. Reserve:
      0 -> PAD
      1 -> UNK
    """
    ctr = Counter()
    for t in texts:
        ctr.update(list(t))

    # keep chars that occur at least min_freq (usually 1)
    chars = sorted([c for c, cnt in ctr.items() if cnt >= min_freq])
    # Build mapping, keep PAD and UNK
    idx = 2
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for c in chars:
        vocab[c] = idx
        idx += 1
    return vocab


def text_to_sequence(text: str, vocab: dict, maxlen=MAXLEN):
    seq = [vocab.get(c, vocab["<UNK>"]) for c in list(text)]
    if len(seq) >= maxlen:
        return seq[:maxlen]
    # pad
    return seq + [vocab["<PAD>"]] * (maxlen - len(seq))


# ---------------------------
# Dataset + DataLoader
# ---------------------------
class URLDataset(Dataset):
    def __init__(self, urls, labels, vocab):
        self.urls = urls
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        u = self.urls[idx]
        seq = text_to_sequence(u, self.vocab)
        y = 1 if self.labels[idx] == "malicious" else 0
        return torch.tensor(seq, dtype=torch.long), torch.tensor(y, dtype=torch.float32)


# ---------------------------
# Model: 1D char-CNN
# ---------------------------
class CharCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, num_filters=NUM_FILTERS,
                 kernel_sizes=KERNEL_SIZES, fc_dim=FC_DIM, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, fc_dim)
        self.out = nn.Linear(fc_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)            # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)           # (batch, embed_dim, seq_len)
        convs = [torch.relu(conv(x)) for conv in self.convs]  # list of (batch, num_filters, L_out)
        pools = [torch.max(c, dim=2)[0] for c in convs]      # global max pool -> (batch, num_filters)
        cat = torch.cat(pools, dim=1)                        # (batch, num_filters * len(kernels))
        x = self.dropout(cat)
        x = torch.relu(self.fc(x))
        x = self.dropout(x)
        logits = self.out(x).squeeze(1)
        return logits


# ---------------------------
# Training loop
# ---------------------------
def train_charcnn(
    data_path,
    url_col="url",
    label_col="type",
    model_dir=DEFAULT_MODEL_DIR,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    maxlen=MAXLEN,
):
    assert os.path.exists(data_path), f"Data path not found: {data_path}"

    print("Loading dataset:", data_path)
    df = pd.read_csv(data_path)
    if url_col not in df.columns:
        raise KeyError(f"url column not found: {url_col}")
    if label_col not in df.columns:
        raise KeyError(f"label column not found: {label_col}")

    # keep only binary labels, normalize
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()
    df = df[df[label_col].isin(["benign", "malicious"])].reset_index(drop=True)

    print("Total samples:", len(df))
    # optionally shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # build vocab on full dataset (fast)
    texts = df[url_col].astype(str).str.lower().tolist()
    vocab = build_char_vocab(texts)
    vocab_size = len(vocab)
    print("Vocab size (chars):", vocab_size)

    # convert texts (lowercase)
    urls = [u.lower()[:maxlen] for u in texts]
    labels = df[label_col].tolist()

    # split
    split = int(len(urls) * 0.8)
    train_urls, val_urls = urls[:split], urls[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    # datasets & loaders
    train_ds = URLDataset(train_urls, train_labels, vocab)
    val_ds = URLDataset(val_urls, val_labels, vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # model
    model = CharCNN(vocab_size=vocab_size, embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # create model_dir
    os.makedirs(model_dir, exist_ok=True)

    # training
    best_val_auc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - train"):
            seqs, ys = batch
            seqs = seqs.to(DEVICE)
            ys = ys.to(DEVICE)

            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * seqs.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_losses = []
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                seqs, ys = batch
                seqs = seqs.to(DEVICE)
                ys = ys.to(DEVICE)
                logits = model(seqs)
                loss = criterion(logits, ys)
                val_losses.append(loss.item() * seqs.size(0))
                all_logits.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(ys.cpu().numpy())

        val_loss = sum(val_losses) / len(val_loader.dataset)
        preds = np.concatenate(all_logits).ravel()
        trues = np.concatenate(all_labels).ravel()

        # compute AUC safely (scikit-learn optional)
        try:
            from sklearn.metrics import roc_auc_score, accuracy_score
            val_auc = roc_auc_score(trues, preds)
            val_acc = accuracy_score(trues, (preds >= 0.5).astype(int))
        except Exception:
            val_auc = None
            val_acc = None

        print(f"Epoch {epoch} finished in {time.time()-t0:.1f}s — train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc}, val_auc={val_auc}")

        # save best
        if val_auc is not None and val_auc > best_val_auc:
            best_val_auc = val_auc
            # SAVE FULL CHECKPOINT — FIXED
            checkpoint = {
                "model_state": model.state_dict(),
                "vocab": vocab,
                "maxlen": maxlen,
                "vocab_size": vocab_size,
            }

            torch.save(checkpoint, os.path.join(model_dir, "charcnn_model.pt"))

            print("Saved full checkpoint (model_state + vocab + config)")
            print("Saved best model — val_auc:", best_val_auc)

    print("Training complete. Best val_auc:", best_val_auc)
    return best_val_auc


# ---------------------------
# CLI entrypoint
# ---------------------------
if __name__ == "__main__":
    import argparse
    from src.config import DATA_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=DATA_PATH)
    parser.add_argument("--url_col", default="url")
    parser.add_argument("--label_col", default="type")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--maxlen", type=int, default=MAXLEN)
    args = parser.parse_args()

    train_charcnn(
        data_path=args.data_path,
        url_col=args.url_col,
        label_col=args.label_col,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
    )
