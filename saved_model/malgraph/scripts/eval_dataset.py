#!/usr/bin/env python3
"""Evaluate the saved MalGraph checkpoint on a converted .pt dataset split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from malgraph.config import load_config, model_params_from_config, resolved_paths
from malgraph.dataset import build_dataloader
from malgraph.hierarchical_model import HierarchicalGraphNeuralNetwork
from malgraph.real_batch import create_real_batch_data
from malgraph.vocabulary import Vocab


def run_eval(
    model: HierarchicalGraphNeuralNetwork,
    loader,
    device: torch.device,
    pred_threshold: float,
) -> dict:
    criterion = nn.BCELoss()
    model.eval()

    total_loss = 0.0
    total_samples = 0
    tp = tn = fp = fn = 0

    for one_batch in loader:
        real_batch, positions, _, external_list, function_edges, targets = create_real_batch_data(one_batch=one_batch)
        if real_batch is None:
            continue

        real_batch = real_batch.to(device)
        positions = torch.tensor(positions, dtype=torch.long, device=device)
        targets = targets.float().to(device).view(-1)

        with torch.no_grad():
            pred = model(
                real_local_batch=real_batch,
                real_bt_positions=positions,
                bt_external_names=external_list,
                bt_all_function_edges=function_edges,
                local_device=device,
            ).squeeze(-1)

        loss = criterion(pred, targets)
        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        pred_label = (pred >= pred_threshold).float()
        tp += int(((pred_label == 1) & (targets == 1)).sum().item())
        tn += int(((pred_label == 0) & (targets == 0)).sum().item())
        fp += int(((pred_label == 1) & (targets == 0)).sum().item())
        fn += int(((pred_label == 0) & (targets == 1)).sum().item())

    if total_samples == 0:
        return {"samples": 0}

    pos_total = tp + fn
    neg_total = tn + fp
    pos_acc = tp / pos_total if pos_total else 0.0
    neg_acc = tn / neg_total if neg_total else 0.0
    acc = (tp + tn) / total_samples
    return {
        "samples": total_samples,
        "loss": total_loss / total_samples,
        "acc": acc,
        "balanced_acc": (pos_acc + neg_acc) / 2.0,
        "pos_acc": pos_acc,
        "neg_acc": neg_acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MalGraph on a converted dataset split")
    parser.add_argument("--data-root", required=True, type=Path, help="Dataset root with test_malware/ and test_benign/")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    config = load_config()
    paths = resolved_paths(config)
    dataset_cfg = config.get("dataset", {})

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    vocab = Vocab(freq_file=str(paths["vocab_path"]), max_vocab_size=int(config["model"]["max_vocab_size"]))
    model = HierarchicalGraphNeuralNetwork(
        model_params=model_params_from_config(config),
        external_vocab=vocab,
        global_log=None,
        grad_output=False,
    ).to(device)
    model.load_state_dict(torch.load(paths["model_path"], map_location=device, weights_only=False))

    loader = build_dataloader(
        data_root=args.data_root.resolve(),
        split=args.split,
        batch_size=int(dataset_cfg.get("batch_size", 8)),
        seed=int(dataset_cfg.get("seed", 19920208)),
        balance=bool(dataset_cfg.get("balance_classes", True)),
    )
    stats = run_eval(
        model,
        loader,
        device,
        pred_threshold=float(dataset_cfg.get("pred_threshold", 0.5)),
    )
    print(json.dumps({"split": args.split, "data_root": str(args.data_root.resolve()), **stats}, indent=2))


if __name__ == "__main__":
    main()
