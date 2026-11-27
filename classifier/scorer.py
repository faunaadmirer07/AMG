"""
Unified malware scorer — dispatches to the correct model backend.

Supported types
---------------
  malconv   – byte-level CNN  (.pt retrained  or  .h5 Keras legacy)
  malgraph  – hierarchical GNN (.pt retrained, requires IDA for PE bytes)

Config
------
  cfg = {
      'type':   'malconv',
      'model':  '/path/to/best_malconv.pt',   # optional; defaults to saved_model/malconv
      'backend': 'torch',                      # optional; auto-detected from extension
      'device': 'cpu',                         # optional
  }

  cfg = {
      'type':      'malgraph',
      'device':    'cpu',       # optional
      'threshold': 0.5,         # optional; overrides config yaml
  }

Usage
-----
  from classifier.scorer import Scorer
  scorer = Scorer({'type': 'malconv'})
  score  = scorer.score_pe(bytez)   # float in [0, 1]
"""
from __future__ import annotations

from typing import Any, Dict


# ── backends ──────────────────────────────────────────────────────────────────

class _MalConvBackend:
    def __init__(self, cfg: Dict[str, Any]):
        from classifier.malconv.server import MalConvServer, _DEFAULT_MODEL_PATH
        self._server = MalConvServer(
            model_path = cfg.get('model', _DEFAULT_MODEL_PATH),
            backend    = cfg.get('backend'),
            device     = cfg.get('device'),
        )

    def score(self, bytez: bytes) -> float:
        return self._server.score_pe(bytez)


class _MalGraphBackend:
    def __init__(self, cfg: Dict[str, Any]):
        from classifier.malgraph.server import MalGraphServer
        self._server = MalGraphServer(
            device     = cfg.get('device'),
            threshold  = cfg.get('threshold'),
            ida_path   = cfg.get('ida_path'),
            ida_script = cfg.get('ida_script'),
            tmp_dir    = cfg.get('tmp_dir'),
        )

    def score(self, bytez: bytes) -> float:
        result = self._server.score_pe(bytez)
        return result if result is not None else 1.0

    def score_json_item(self, item: Dict[str, Any]) -> float:
        result = self._server.score_json_item(item)
        return result if result is not None else 1.0


class _MalRNNBackend:
    def __init__(self, cfg: Dict[str, Any]):
        from classifier.malrnn.server import MalRNNServer, _DEFAULT_MODEL_PATH
        self._server = MalRNNServer(
            model_path = cfg.get('model', _DEFAULT_MODEL_PATH),
            device     = cfg.get('device'),
        )

    def score(self, bytez: bytes) -> float:
        return self._server.score_pe(bytez)


_BACKENDS = {
    'malconv':  _MalConvBackend,
    'malgraph': _MalGraphBackend,
    'malrnn':   _MalRNNBackend,
}


# ── public interface ──────────────────────────────────────────────────────────

class Scorer:
    """
    Unified malware scorer.

    Interface:
        scorer.score_pe(bytez) -> float in [0, 1]
        scorer.query_count     -> int
        scorer.reset_query_count()
        scorer.model_type      -> str
    """

    def __init__(self, cfg: Dict[str, Any]):
        model_type = cfg.get('type', 'malconv')
        if model_type not in _BACKENDS:
            raise ValueError(
                "Unknown scorer type '{}'. Supported: {}".format(
                    model_type, ', '.join(_BACKENDS)
                )
            )
        self._backend  = _BACKENDS[model_type](cfg)
        self.model_type = model_type
        self.query_count = 0

    def score_pe(self, bytez: bytes) -> float:
        """Return p(malware) in [0, 1] and increment query_count."""
        self.query_count += 1
        return self._backend.score(bytez)

    def score_json_item(self, item: Dict[str, Any]) -> float:
        """Score a pre-extracted graph JSON dict (no IDA). Only for malgraph."""
        self.query_count += 1
        if not hasattr(self._backend, 'score_json_item'):
            raise NotImplementedError(
                "score_json_item not supported for scorer type '{}'".format(self.model_type)
            )
        return self._backend.score_json_item(item)

    def reset_query_count(self):
        self.query_count = 0
