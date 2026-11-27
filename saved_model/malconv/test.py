#!/usr/bin/env python3
import argparse
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from data import BinaryDirDataset, SplitLogDataset, create_dataloader
from model import load_model


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0
    tp = tn = fp = fn = 0

    for x, y, _ in tqdm(loader, desc="Evaluating"):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size
        correct += (preds == y).sum().item()

        tp += ((preds == 1) & (y == 1)).sum().item()
        tn += ((preds == 0) & (y == 0)).sum().item()
        fp += ((preds == 1) & (y == 0)).sum().item()
        fn += ((preds == 0) & (y == 1)).sum().item()

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "loss": avg_loss,
        "acc": acc,
        "tpr": tpr,
        "tnr": tnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }


def build_dataset(args, config):
    input_len = config.get("input_len", 2097152)
    pad_token = config.get("pad_token", 256)
    exclude_header = bool(config.get("exclude_header", False))

    if args.split_log:
        return SplitLogDataset(
            split_log_csv=args.split_log,
            split=args.split,
            input_len=input_len,
            pad_token=pad_token,
            exclude_header=exclude_header,
        )

    if not args.benign_dir or not args.malware_dir:
        raise ValueError("Provide --benign_dir and --malware_dir, or use --split_log.")

    return BinaryDirDataset(
        benign_dir=args.benign_dir,
        malware_dir=args.malware_dir,
        input_len=input_len,
        pad_token=pad_token,
        exclude_header=exclude_header,
        recursive=bool(args.recursive),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate MalConv checkpoint on benign/malware binaries.")
    default_ckpt = os.path.join(os.path.dirname(__file__), "best_malconv.pt")

    parser.add_argument("--model_pt", type=str, default=default_ckpt)
    parser.add_argument("--benign_dir", type=str, default=None)
    parser.add_argument("--malware_dir", type=str, default=None)
    parser.add_argument("--split_log", type=str, default=None, help="Path to split_log.csv from training")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--recursive", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    model, meta = load_model(args.model_pt, device)
    config = meta["config"]
    print(f"Loaded checkpoint from epoch {meta.get('epoch')} (best_val_acc={meta.get('best_val_acc')})")
    print(f"exclude_header={config.get('exclude_header', False)}")

    dataset = build_dataset(args, config)
    print(f"Evaluating {len(dataset)} samples")

    loader = create_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    metrics = evaluate(model, loader, device, nn.CrossEntropyLoss())

    print("\nResults")
    print(f"loss={metrics['loss']:.4f}")
    print(f"acc={metrics['acc']:.4f}")
    print(f"tpr={metrics['tpr']:.4f}")
    print(f"tnr={metrics['tnr']:.4f}")
    print(f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")


if __name__ == "__main__":
    main()
