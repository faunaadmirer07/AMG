from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def package_root() -> Path:
    env_root = os.environ.get("MALGRAPH_ROOT")
    if env_root:
        candidate = Path(env_root).resolve()
        if (candidate / "configs" / "malgraph.yaml").exists():
            return candidate
    return PACKAGE_ROOT


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_config(config_path: Path | None = None) -> Dict[str, Any]:
    root = package_root()
    path = config_path or (root / "configs" / "malgraph.yaml")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw


def model_params_from_config(config: Dict[str, Any] | None = None) -> SimpleNamespace:
    root = package_root()
    cfg = config or load_config()
    model_cfg = cfg["model"]
    return SimpleNamespace(
        gnn_type=model_cfg["gnn_type"],
        pool_type=model_cfg["pool_type"],
        acfg_init_dims=model_cfg["acfg_init_dims"],
        cfg_filters=model_cfg["cfg_filters"],
        fcg_filters=model_cfg["fcg_filters"],
        skip_att_heads=model_cfg.get("skip_att_heads", 0),
        dropout_rate=model_cfg["dropout_rate"],
        ablation_models=model_cfg.get("ablation_models", "Full"),
    )


def resolved_paths(config: Dict[str, Any] | None = None) -> Dict[str, Path]:
    root = package_root()
    cfg = config or load_config()
    model_cfg = cfg["model"]
    workspace_cfg = cfg.get("workspace", {})
    ida_cfg = cfg.get("ida", {})
    return {
        "root": root,
        "model_path": _resolve_path(root, model_cfg["path"]),
        "vocab_path": _resolve_path(root, model_cfg["vocab_path"]),
        "ida_path": Path(ida_cfg.get("path", "")).expanduser(),
        "ida_script": _resolve_path(root, ida_cfg.get("script", "ida_extractor/graph_handle_acfg.py")),
        "tmp_sample_root": _resolve_path(root, workspace_cfg.get("tmp_sample_root", "workspace/tmp_samples")),
        "gdl_dir": _resolve_path(root, workspace_cfg.get("gdl_dir", "ida_extractor/gdl")),
    }
