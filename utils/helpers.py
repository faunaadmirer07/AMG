"""
Helper utilities — train/val split, corpus iteration.

Usage:
    from utils.helpers import get_train_val_hashes
    train_hashes, val_hashes = get_train_val_hashes(corpus_path, seed=1337, val_frac=0.10)
"""
import os
import random


def get_train_val_hashes(corpus_path, seed=1337, val_frac=0.10):
    """
    Split malware corpus into train/val using consistent seed.
    Matches Malware Makeover split: SEED=1337, VAL_FRAC=0.10.

    Args:
        corpus_path: path to corpus_hashes.txt (one hash per line)
        seed: random seed for reproducible split
        val_frac: fraction for validation

    Returns:
        train_hashes: list of hash strings
        val_hashes: list of hash strings
    """
    with open(corpus_path, 'r') as f:
        all_hashes = sorted([line.strip() for line in f if line.strip()])

    rng = random.Random(seed)
    indices = list(range(len(all_hashes)))
    rng.shuffle(indices)

    val_size = int(len(indices) * val_frac)
    val_indices = set(indices[:val_size])

    train_hashes = [all_hashes[i] for i in range(len(all_hashes)) if i not in val_indices]
    val_hashes = [all_hashes[i] for i in range(len(all_hashes)) if i in val_indices]

    return train_hashes, val_hashes


def hash_to_path(malware_dir, binary_hash):
    """Convert a hash string to full file path."""
    return os.path.join(malware_dir, binary_hash)