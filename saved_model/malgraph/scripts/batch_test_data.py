#!/usr/bin/env python3
"""Batch-test MalGraph on PE binaries from an external data directory."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["MALGRAPH_ROOT"] = str(ROOT)

from malgraph.scorer import MalgraphScorer


def collect_files(root: Path, limit: int, seed: int) -> List[Path]:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    rng = random.Random(seed)
    files = files[:]
    rng.shuffle(files)
    return files[:limit]


def evaluate(files: List[Path], expected_label: str, scorer: MalgraphScorer) -> Tuple[dict, list]:
    correct = 0
    failed = 0
    rows = []
    for idx, path in enumerate(files, start=1):
        t0 = time.time()
        try:
            score = scorer.score_pe_file(path)
            if score is None:
                failed += 1
                status = "failed"
                pred = None
                ok = False
            else:
                pred = scorer.classify(score)
                ok = pred == expected_label
                if ok:
                    correct += 1
                status = "ok"
        except Exception as exc:
            failed += 1
            score = None
            pred = None
            ok = False
            status = f"error: {exc}"
        rows.append(
            {
                "file": str(path),
                "expected": expected_label,
                "score": score,
                "prediction": pred,
                "correct": ok,
                "status": status,
                "seconds": round(time.time() - t0, 2),
            }
        )
        print(json.dumps({"n": idx, "total": len(files), "class": expected_label, **rows[-1]}), flush=True)
    scored = len(files) - failed
    return {
        "expected": expected_label,
        "total": len(files),
        "scored": scored,
        "failed": failed,
        "correct": correct,
        "accuracy": (correct / scored) if scored else 0.0,
    }, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch test MalGraph on malware/benign PE binaries")
    parser.add_argument("--malware-root", type=Path, default=Path("./data/malware/sampled_malware2"))
    parser.add_argument("--benign-root", type=Path, default=Path("./data/benign/sampled_benign"))
    parser.add_argument("--limit", type=int, default=100, help="Samples per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", type=Path, default=ROOT / "workspace" / "batch_test_results.json")
    args = parser.parse_args()

    malware_files = collect_files(args.malware_root.resolve(), args.limit, args.seed)
    benign_files = collect_files(args.benign_root.resolve(), args.limit, args.seed + 1)
    if len(malware_files) < args.limit or len(benign_files) < args.limit:
        raise RuntimeError(
            f"Need {args.limit} files per class; found malware={len(malware_files)}, benign={len(benign_files)}"
        )

    scorer = MalgraphScorer(device=args.device)
    mal_summary, mal_rows = evaluate(malware_files, "malware", scorer)
    ben_summary, ben_rows = evaluate(benign_files, "benign", scorer)

    total_scored = mal_summary["scored"] + ben_summary["scored"]
    total_correct = mal_summary["correct"] + ben_summary["correct"]
    payload = {
        "package_root": str(ROOT),
        "malware_root": str(args.malware_root.resolve()),
        "benign_root": str(args.benign_root.resolve()),
        "limit_per_class": args.limit,
        "threshold": scorer.threshold,
        "malware": mal_summary,
        "benign": ben_summary,
        "overall": {
            "scored": total_scored,
            "correct": total_correct,
            "accuracy": (total_correct / total_scored) if total_scored else 0.0,
            "balanced_accuracy": (mal_summary["accuracy"] + ben_summary["accuracy"]) / 2.0,
        },
        "details": {"malware": mal_rows, "benign": ben_rows},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": {k: payload[k] for k in ("malware", "benign", "overall")}, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
