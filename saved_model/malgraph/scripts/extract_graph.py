#!/usr/bin/env python3
"""Extract MalGraph JSON from a PE file using IDA."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from malgraph.config import load_config, package_root, resolved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MalGraph graph JSON from a PE/binary file")
    parser.add_argument("-f", "--file", required=True, type=Path, help="Input PE/binary file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSON path (default: workspace/tmp_samples/<stem>.json)")
    parser.add_argument("--config", type=Path, default=None, help="Path to malgraph.yaml")
    args = parser.parse_args()

    os.environ.setdefault("MALGRAPH_ROOT", str(package_root()))
    config = load_config(args.config) if args.config else load_config()
    paths = resolved_paths(config)

    if not paths["ida_path"].exists():
        raise FileNotFoundError(f"IDA not found: {paths['ida_path']}. Update ida.path in configs/malgraph.yaml")

    input_path = args.file.resolve()
    paths["tmp_sample_root"].mkdir(parents=True, exist_ok=True)
    paths["gdl_dir"].mkdir(parents=True, exist_ok=True)

    sample_copy = paths["tmp_sample_root"] / input_path.name
    shutil.copy2(input_path, sample_copy)

    script_path = paths["ida_script"].resolve()
    env = os.environ.copy()
    env["MALGRAPH_ROOT"] = str(paths["root"])
    env["TVHEADLESS"] = "1"
    cmd = [str(paths["ida_path"]), "-c", "-A", f"-S{script_path}", str(sample_copy)]
    subprocess.run(cmd, check=True, cwd=str(script_path.parent), env=env)

    produced = sample_copy.with_suffix(".json")
    if not produced.exists():
        raise RuntimeError(f"IDA did not produce expected JSON: {produced}")

    output_path = args.output.resolve() if args.output else produced
    if output_path != produced:
        shutil.copy2(produced, output_path)

    print(str(output_path))


if __name__ == "__main__":
    main()
