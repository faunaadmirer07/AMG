from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MalConvTorch(nn.Module):
    def __init__(
        self,
        input_len: int = 2097152,
        vocab_size: int = 257,
        emb_dim: int = 8,
        channels: int = 128,
        kernel_size: int = 500,
        stride: int = 500,
        fc_dim: int = 128,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_len = input_len
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.conv_A = nn.Conv1d(
            emb_dim, channels, kernel_size=kernel_size, stride=stride, padding=0, bias=True
        )
        self.conv_B = nn.Conv1d(
            emb_dim, channels, kernel_size=kernel_size, stride=stride, padding=0, bias=True
        )
        self.fc = nn.Linear(channels, fc_dim, bias=True)
        self.logits = nn.Linear(fc_dim, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.transpose(1, 2)
        a = self.conv_A(x)
        b = self.conv_B(x)
        x = a * torch.sigmoid(b)
        x = torch.max(x, dim=2).values
        x = F.relu(self.fc(x))
        return self.logits(x)


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[MalConvTorch, Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    model = MalConvTorch(
        input_len=config.get("input_len", 2097152),
        vocab_size=config.get("vocab_size", 257),
        emb_dim=config.get("emb_dim", 8),
        channels=config.get("channels", 128),
        kernel_size=config.get("kernel_size", 500),
        stride=config.get("stride", 500),
        fc_dim=config.get("fc_dim", 128),
        num_classes=config.get("num_classes", 2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    meta = {
        "config": config,
        "epoch": ckpt.get("epoch"),
        "best_val_acc": ckpt.get("best_val_acc"),
    }
    return model, meta
