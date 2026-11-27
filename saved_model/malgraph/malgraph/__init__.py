"""Portable MalGraph inference package."""

from malgraph.config import load_config, package_root
from malgraph.scorer import MalgraphScorer

__all__ = ["MalgraphScorer", "load_config", "package_root"]
