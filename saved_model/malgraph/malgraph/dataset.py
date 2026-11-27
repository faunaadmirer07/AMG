from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader


class LazyPtDataset(Dataset):
    def __init__(self, files: List[Path]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        return torch.load(self.files[idx], weights_only=False)


def list_class_files(root: Path, split: str, class_name: str) -> List[Path]:
    return sorted((root / f"{split}_{class_name}").glob("*.pt"))


def build_balanced_split_files(
    root: Path,
    split: str,
    seed: int,
    balance: bool = True,
    limit: int = 0,
) -> Tuple[List[Path], int, int]:
    malware_files = list_class_files(root, split, "malware")
    benign_files = list_class_files(root, split, "benign")
    rng = random.Random(seed)
    malware_files = malware_files[:]
    benign_files = benign_files[:]
    rng.shuffle(malware_files)
    rng.shuffle(benign_files)

    if balance:
        n_per_class = min(len(malware_files), len(benign_files))
        malware_files = malware_files[:n_per_class]
        benign_files = benign_files[:n_per_class]

    if limit > 0:
        per_class_limit = limit // 2
        malware_files = malware_files[:per_class_limit]
        benign_files = benign_files[:per_class_limit]

    files = malware_files + benign_files
    rng.shuffle(files)
    return files, len(malware_files), len(benign_files)


def build_dataloader(
    data_root: Path,
    split: str,
    batch_size: int,
    seed: int = 19920208,
    balance: bool = True,
    limit: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    split_offsets = {"train": 0, "val": 1, "test": 2}
    files, _, _ = build_balanced_split_files(
        data_root,
        split,
        seed + split_offsets.get(split, 0),
        balance=balance,
        limit=limit,
    )
    if not files:
        raise RuntimeError(f"No {split} samples found under {data_root}")
    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "shuffle": split == "train"}
    return DataLoader(LazyPtDataset(files), **loader_kwargs)
