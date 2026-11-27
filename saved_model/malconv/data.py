import csv
import os
from typing import List, Tuple

import pefile
import torch
from torch.utils.data import DataLoader, Dataset


def list_binary_files(input_dir: str, recursive: bool = True) -> List[str]:
    files = []
    if recursive:
        for root, _, names in os.walk(input_dir):
            for name in sorted(names):
                path = os.path.join(root, name)
                if os.path.isfile(path):
                    files.append(path)
    else:
        for name in sorted(os.listdir(input_dir)):
            path = os.path.join(input_dir, name)
            if os.path.isfile(path):
                files.append(path)
    return sorted(files)


def read_binary_as_tokens(
    filepath: str,
    input_len: int,
    pad_token: int = 256,
    exclude_header: bool = False,
) -> List[int]:
    with open(filepath, "rb") as f:
        raw = f.read()

    offset = 0
    if exclude_header:
        try:
            pe = pefile.PE(filepath, fast_load=True)
            if hasattr(pe, "sections") and len(pe.sections) > 0:
                offset = int(pe.sections[0].PointerToRawData)
            else:
                offset = int(pe.OPTIONAL_HEADER.SizeOfHeaders)
            pe.close()
        except Exception:
            offset = 0

    x = list(raw[offset:])

    if len(x) < input_len:
        x = x + [pad_token] * (input_len - len(x))
    else:
        x = x[:input_len]

    return x


class BinaryDirDataset(Dataset):
    def __init__(
        self,
        benign_dir: str,
        malware_dir: str,
        input_len: int,
        pad_token: int = 256,
        exclude_header: bool = False,
        recursive: bool = True,
    ):
        self.input_len = input_len
        self.pad_token = pad_token
        self.exclude_header = exclude_header
        self.samples: List[Tuple[str, int]] = []

        benign_files = list_binary_files(benign_dir, recursive=recursive)
        malware_files = list_binary_files(malware_dir, recursive=recursive)
        self.samples.extend((fp, 0) for fp in benign_files)
        self.samples.extend((fp, 1) for fp in malware_files)

        if not self.samples:
            raise ValueError("No files found in the given benign/malware directories.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        filepath, label = self.samples[idx]
        x = read_binary_as_tokens(
            filepath=filepath,
            input_len=self.input_len,
            pad_token=self.pad_token,
            exclude_header=self.exclude_header,
        )
        return torch.tensor(x, dtype=torch.long), torch.tensor(label, dtype=torch.long), filepath


class SplitLogDataset(Dataset):
    """Load filepaths and labels from a split_log.csv produced during training."""

    def __init__(
        self,
        split_log_csv: str,
        split: str,
        input_len: int,
        pad_token: int = 256,
        exclude_header: bool = False,
    ):
        self.input_len = input_len
        self.pad_token = pad_token
        self.exclude_header = exclude_header
        self.samples: List[Tuple[str, int]] = []

        with open(split_log_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.samples.append((row["filepath"], int(row["label"])))

        if not self.samples:
            raise ValueError(f"No samples found for split={split!r} in {split_log_csv}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        filepath, label = self.samples[idx]
        x = read_binary_as_tokens(
            filepath=filepath,
            input_len=self.input_len,
            pad_token=self.pad_token,
            exclude_header=self.exclude_header,
        )
        return torch.tensor(x, dtype=torch.long), torch.tensor(label, dtype=torch.long), filepath


def collate_batch(batch):
    xs, ys, paths = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), list(paths)


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=pin_memory,
    )
