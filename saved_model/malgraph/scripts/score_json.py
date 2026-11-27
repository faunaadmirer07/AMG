#!/usr/bin/env python3
"""Score a pre-extracted MalGraph JSON graph file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from malgraph.scorer import MalgraphScorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a MalGraph JSON graph file")
    parser.add_argument("-f", "--file", required=True, type=Path, help="Input graph JSON")
    parser.add_argument("--label", type=int, default=1, help="Label stored in graph Data object (default: 1)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    scorer = MalgraphScorer(device=args.device, threshold=args.threshold)
    score = scorer.score_json_file(args.file, label=args.label)
    if score is None:
        raise RuntimeError("Scoring failed: invalid or empty graph JSON")

    result = {
        "file": str(args.file.resolve()),
        "malware_probability": score,
        "prediction": scorer.classify(score),
        "threshold": scorer.threshold,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
