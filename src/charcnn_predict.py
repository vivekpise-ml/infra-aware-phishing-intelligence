"""
charcnn_predict.py

Correct loader for the SAFE state_dict-only charcnn_model.pt
"""

import os
import json
import torch
import numpy as np
from typing import List

from src.train_char_cnn import CharCNN, text_to_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_charcnn(model_dir="models"):
    """
    Load the CharCNN model saved in the new checkpoint format:
    {
        "model_state": ...,
        "vocab": ...,
        "vocab_size": ...,
        "maxlen": ...
    }
    """

    model_path = os.path.join(model_dir, "charcnn_model.pt")

    print("📁 Loading CharCNN from:", os.path.abspath(model_path))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CharCNN model not found at {model_path}")

    # Load full checkpoint (dict with model_state + metadata)
    ckpt = torch.load(model_path, map_location=DEVICE)

    vocab = ckpt["vocab"]
    maxlen = ckpt["maxlen"]
    vocab_size = ckpt["vocab_size"]
    state_dict = ckpt["model_state"]

    # Build model
    model = CharCNN(vocab_size=vocab_size)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # Pack config
    cfg = {"maxlen": maxlen, "vocab_size": vocab_size}

    return model, vocab, cfg

def encode_url_batch(urls, vocab, maxlen):
    seqs = [
        text_to_sequence(
            u.lower()[:maxlen],
            vocab,
            maxlen=maxlen
        )
        for u in urls
    ]
    return torch.tensor(seqs, dtype=torch.long).to(DEVICE)


def predict_urls(urls: List[str], model, vocab, cfg):
    maxlen = cfg.get("maxlen", 200)

    seqs = [
        text_to_sequence(
            u.lower()[:maxlen],
            vocab,
            maxlen=maxlen
        )
        for u in urls
    ]

    tensor = torch.tensor(seqs, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

    preds = (probs >= 0.5).astype(int)
    return preds, probs


def predict_single(url: str, model_dir="models"):
    model, vocab, cfg = load_charcnn(model_dir)
    preds, probs = predict_urls([url], model, vocab, cfg)

    label = "malicious" if preds[0] == 1 else "benign"
    return label, float(probs[0])
