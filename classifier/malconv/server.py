"""
MalConv server — unified wrapper for PyTorch (.pt) and Keras (.h5) checkpoints.

PyTorch backend  (.pt)  — retrained MalConvTorch:
  bytes are NOT shifted; byte 256 is the pad token.
  truncate/pad to input_len stored in the checkpoint config (default 2_097_152).
  output: 2-class logits → softmax[:, 1] as p(malware).

Keras backend    (.h5)  — legacy checkpoint:
  skip PE header, pad with 256 to 2_097_152, no byte shift.
  output: 2-class softmax[:, 1] as p(malware).
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── PyTorch model ─────────────────────────────────────────────────────────────

class _MalConvTorch(nn.Module):
    def __init__(
        self,
        input_len: int = 2_097_152,
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
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.conv_A = nn.Conv1d(emb_dim, channels, kernel_size=kernel_size, stride=stride, bias=True)
        self.conv_B = nn.Conv1d(emb_dim, channels, kernel_size=kernel_size, stride=stride, bias=True)
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


class _TorchBackend:
    def __init__(self, model_path: str, device: Optional[str]):
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        ckpt = torch.load(model_path, map_location=self.device)
        cfg = ckpt.get('config', {})
        self.input_len    = cfg.get('input_len', 2_097_152)
        self.pad_token    = cfg.get('pad_token', 256)
        self.exclude_header = bool(cfg.get('exclude_header', False))
        self.model = _MalConvTorch(
            input_len   = self.input_len,
            vocab_size  = cfg.get('vocab_size', 257),
            emb_dim     = cfg.get('emb_dim', 8),
            channels    = cfg.get('channels', 128),
            kernel_size = cfg.get('kernel_size', 500),
            stride      = cfg.get('stride', 500),
            fc_dim      = cfg.get('fc_dim', 128),
            num_classes = cfg.get('num_classes', 2),
        ).to(self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

    def score(self, bytez: bytes) -> float:
        raw = list(bytez)
        if self.exclude_header:
            try:
                import pefile
                pe = pefile.PE(data=bytez)
                offset = (
                    int(pe.sections[0].PointerToRawData)
                    if pe.sections
                    else int(pe.OPTIONAL_HEADER.SizeOfHeaders)
                )
                raw = list(bytez[offset:])
            except Exception:
                pass
        if len(raw) < self.input_len:
            raw = raw + [self.pad_token] * (self.input_len - len(raw))
        else:
            raw = raw[:self.input_len]
        x = torch.tensor(raw, dtype=torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            prob = F.softmax(logits, dim=1)[0, 1]
        return float(prob.item())


# ── Keras backend ─────────────────────────────────────────────────────────────

_KR_INPUT_SIZE = 2_097_152
_KR_PAD_VAL    = 256


class _KerasBackend:
    def __init__(self, model_path: str):
        import os as _os
        _os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        import tf_keras as keras
        self.model   = keras.models.load_model(model_path)
        self._tf     = tf
        self._gpu_ok = True

    def score(self, bytez: bytes) -> float:
        import pefile
        import numpy as np
        pe     = pefile.PE(data=bytez)
        offset = int(len(pe.header))
        raw    = list(pe.__data__[offset:])
        if len(raw) > _KR_INPUT_SIZE:
            raw = raw[:_KR_INPUT_SIZE]
        else:
            raw += [_KR_PAD_VAL] * (_KR_INPUT_SIZE - len(raw))
        x = np.array(raw, dtype=np.int32).reshape(1, _KR_INPUT_SIZE)
        if self._gpu_ok:
            try:
                probs = self.model.predict(x, verbose=0)
            except Exception:
                self._gpu_ok = False
                with self._tf.device('/CPU:0'):
                    probs = self.model.predict(x, verbose=0)
        else:
            with self._tf.device('/CPU:0'):
                probs = self.model.predict(x, verbose=0)
        return float(probs[0, 1])


# ── Public server ─────────────────────────────────────────────────────────────

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'saved_model', 'malconv', 'best_malconv.pt',
)


class MalConvServer:
    """
    In-process MalConv scorer.  Backend is chosen by file extension or explicit param.

    Args:
        model_path: path to .pt (PyTorch) or .h5 (Keras) checkpoint
        backend:    'torch' | 'keras' | None (auto-detect from extension)
        device:     'cpu' | 'cuda' | None (auto-detect); PyTorch only
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        backend: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.__name__ = 'MalConv'
        if backend is None:
            backend = 'keras' if model_path.endswith('.h5') else 'torch'
        if backend == 'torch':
            self._backend = _TorchBackend(model_path, device)
        elif backend == 'keras':
            self._backend = _KerasBackend(model_path)
        else:
            raise ValueError("backend must be 'torch' or 'keras', got: {}".format(backend))
        self.backend = backend

    def __call__(self, bytez: bytes) -> float:
        return self.score_pe(bytez)

    def score_pe(self, bytez: bytes) -> float:
        """Return p(malware) in [0, 1]."""
        return self._backend.score(bytez)

    def model_name(self) -> str:
        return self.__name__
