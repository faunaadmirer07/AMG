"""
MalGraph server — wraps MalgraphScorer from saved_model/malgraph.

score_pe() requires IDA Pro to extract a function-call graph from the binary.
score_json_file() / score_json_item() work directly on pre-extracted JSON and
do not require IDA, making them suitable for offline / cached evaluation.

The MALGRAPH_ROOT environment variable is set automatically to point at
saved_model/malgraph so the scorer locates its config, vocab, and checkpoint
without any extra configuration.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

_PID = os.getpid()

_SAVED_MALGRAPH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'saved_model', 'malgraph',
)
# Patched IDA script that writes JSON next to the input binary (writable path)
_DEFAULT_IDA_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'ida_extract_patch.py'
)
if _SAVED_MALGRAPH not in sys.path:
    sys.path.insert(0, _SAVED_MALGRAPH)

from malgraph.scorer import MalgraphScorer  # noqa: E402  (after sys.path patch)


class MalGraphServer:
    """
    In-process MalGraph scorer backed by the retrained malgraph_tuned_v3 checkpoint.

    Args:
        config:    override config dict; if None the bundled malgraph.yaml is used
        device:    'cpu' | 'cuda' | None (auto-detect)
        threshold: classification threshold; if None, taken from config (default 0.5)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
        ida_path: Optional[str] = None,
        ida_script: Optional[str] = None,
        tmp_dir: Optional[str] = None,
    ):
        self.__name__ = 'MalGraph'
        os.environ.setdefault('MALGRAPH_ROOT', _SAVED_MALGRAPH)
        device_str = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Always use the patched IDA script by default so JSON lands next to the binary
        ida_script = ida_script or _DEFAULT_IDA_SCRIPT

        from malgraph.config import load_config
        cfg = config or load_config()
        cfg.setdefault('ida', {})['script'] = ida_script
        if ida_path is not None:
            cfg['ida']['path'] = ida_path
        base_tmp = tmp_dir or os.path.join(tempfile.gettempdir(), 'malgraph_tmp')
        pid_tmp = os.path.join(base_tmp, str(_PID))
        cfg.setdefault('workspace', {})['tmp_sample_root'] = pid_tmp
        config = cfg

        self._scorer = MalgraphScorer(config=config, device=device_str, threshold=threshold)
        self._query_count = 0

    # ── public interface ──────────────────────────────────────────────────────

    def __call__(self, bytez: bytes, data_hash: str = 'sample') -> Optional[float]:
        return self.score_pe(bytez, data_hash=data_hash)

    def score_pe(self, bytez: bytes, data_hash: str = 'sample') -> Optional[float]:
        """
        Score raw PE bytes.  Runs IDA to extract the graph, then scores.
        Returns p(malware) in [0, 1], or None if the graph cannot be built.
        """
        self._query_count += 1
        tmp_root = Path(self._scorer.tmp_sample_root)
        import uuid
        uid = uuid.uuid4().hex[:8]
        sample_name = f"{data_hash}_{uid}"
        sample_path = tmp_root / sample_name
        sample_path.write_bytes(bytez)
        try:
            json_path = self._scorer._extract_json(str(sample_path))
            if not Path(json_path).exists():
                return None
            return self._scorer.score_json_file(json_path)
        finally:
            for suffix in ['', '.json', '.i64', '.idb', '.id0', '.id1', '.id2', '.nam', '.til']:
                p = tmp_root / (sample_name + suffix)
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass

    def score_json_file(self, json_path: Union[str, Path], label: int = 1) -> Optional[float]:
        """Score a pre-extracted JSON graph file (no IDA required)."""
        self._query_count += 1
        return self._scorer.score_json_file(json_path, label=label)

    def score_json_item(self, item: Dict[str, Any], label: int = 1) -> Optional[float]:
        """Score a pre-extracted JSON graph dict (no IDA required)."""
        self._query_count += 1
        return self._scorer.score_json_item(item, label=label)

    def model_name(self) -> str:
        return self.__name__

    # ── query counting (matches Scorer interface) ─────────────────────────────

    @property
    def query_count(self) -> int:
        return self._query_count

    def reset_query_count(self) -> None:
        self._query_count = 0
