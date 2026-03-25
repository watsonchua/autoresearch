"""
One-time data preparation for autoresearch experiments.
Reads .md documents from a local data directory and trains a BPE tokenizer.

Usage:
    python prepare.py                        # full prep (tokenizer training)
    python prepare.py --data-dir ./data      # specify data directory

Data is read from ./data (default). Tokenizer is stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import json
import random
import argparse
import pickle

import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 40 * 524288  # number of tokens for val eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VOCAB_SIZE = 8192
VAL_RATIO = 0.1
SPLIT_SEED = 42
CHUNK_SIZE = 10_000
CHUNK_OVERLAP = 1_000

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def list_md_files(data_dir=DEFAULT_DATA_DIR):
    """Return sorted list of all .md file paths under data_dir."""
    md_files = []
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".md"):
                md_files.append(os.path.join(root, f))
    md_files.sort()
    return md_files


def _get_split_groups(data_dir):
    """Group .md files by their immediate parent directory for independent splitting."""
    groups = {}
    for md_path in list_md_files(data_dir):
        parent = os.path.dirname(md_path)
        group_key = os.path.relpath(parent, data_dir).split(os.sep)[0]
        groups.setdefault(group_key, []).append(md_path)
    return groups


def get_train_val_split(data_dir=DEFAULT_DATA_DIR):
    """Return (train_files, val_files). Persists split to JSON for consistency."""
    split_path = os.path.join(data_dir, "train_val_split.json")
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            saved = json.load(f)
        return saved["train"], saved["val"]

    groups = _get_split_groups(data_dir)
    train_files, val_files = [], []
    rng = random.Random(SPLIT_SEED)
    for group_key in sorted(groups):
        files = sorted(groups[group_key])
        rng.shuffle(files)
        n_val = max(1, int(len(files) * VAL_RATIO))
        val_files.extend(files[:n_val])
        train_files.extend(files[n_val:])

    train_files.sort()
    val_files.sort()
    with open(split_path, "w") as f:
        json.dump({"train": train_files, "val": val_files}, f, indent=2)
    print(f"Data: created train/val split ({len(train_files)} train, {len(val_files)} val) at {split_path}")
    return train_files, val_files


def chunk_document(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Yield overlapping chunks from a document."""
    if len(text) <= chunk_size:
        yield text
        return
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        if end >= len(text):
            break
        start = end - overlap


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def text_iterator(data_dir=DEFAULT_DATA_DIR, max_chars=1_000_000_000, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Yield document chunks from training split .md files."""
    train_files, _ = get_train_val_split(data_dir)
    nchars = 0
    for filepath in train_files:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        for chunk in chunk_document(text, chunk_size, overlap):
            nchars += len(chunk)
            yield chunk
            if nchars >= max_chars:
                return


def train_tokenizer(data_dir=DEFAULT_DATA_DIR):
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    train_files, val_files = get_train_val_split(data_dir)
    if len(train_files) == 0 or len(val_files) == 0:
        print("Tokenizer: need .md files in data directory for both train and val splits.")
        sys.exit(1)

    # --- Train with rustbpe ---
    print(f"Tokenizer: training BPE tokenizer on {len(train_files)} train files...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(data_dir), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tokenizer
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # --- Build token_bytes lookup for BPB evaluation ---
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, data_dir=DEFAULT_DATA_DIR, tokenizer_batch_size=128,
                      chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Infinite iterator over document chunk batches from .md files."""
    train_files, val_files = get_train_val_split(data_dir)
    file_paths = train_files if split == "train" else val_files
    assert len(file_paths) > 0, f"No {split} files found. Check data directory."
    epoch = 1
    while True:
        batch = []
        for filepath in file_paths:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            for chunk in chunk_document(text, chunk_size, overlap):
                batch.append(chunk)
                if len(batch) >= tokenizer_batch_size:
                    yield batch, epoch
                    batch = []
        if batch:
            yield batch, epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, data_dir=DEFAULT_DATA_DIR, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split, data_dir=data_dir)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing .md files in subdirectories")
    args = parser.parse_args()

    print(f"Data directory: {args.data_dir}")
    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Create train/val split
    train_files, val_files = get_train_val_split(args.data_dir)
    print(f"Data: {len(train_files)} train files, {len(val_files)} val files")
    print()

    # Step 2: Train tokenizer
    train_tokenizer(args.data_dir)
    print()
    print("Done! Ready to train.")
