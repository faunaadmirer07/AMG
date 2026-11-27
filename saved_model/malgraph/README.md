# MalGraph Test

Self-contained MalGraph package for graph extraction (IDA), inference, and batch evaluation on PE binaries. No dependency on the parent `mguise` project.

## Layout

```
malgraph_test/
├── configs/malgraph.yaml      # model, IDA, workspace, and Acfg settings
├── models/
│   ├── malgraph_tuned_v3.pt
│   └── external_function_vocab.jsonl
├── malgraph/                  # Python package (model, vocab, batching, scorer)
├── ida_extractor/             # IDA scripts + vendored shims for IDA Python 2.7
├── workspace/tmp_samples/     # writable temp dir for IDA input/output JSON
└── scripts/
    ├── extract_graph.py       # PE -> JSON
    ├── score_json.py          # JSON -> malware probability
    ├── score_pe.py            # PE -> IDA -> score
    ├── eval_dataset.py        # evaluate on converted .pt dataset split
    └── batch_test_data.py     # batch test on external malware/benign PE folders
```

## Setup

```bash
cd malgraph_test
pip install -r requirements.txt
```

Edit `configs/malgraph.yaml`:

- `ida.path` — path to your IDA Pro executable (e.g. `idal64`)
- Other paths are relative to this folder and usually need no changes

Run scripts with `PYTHONPATH=.` from this directory, or set `MALGRAPH_ROOT` to this folder's absolute path.

## Usage

All examples assume you are inside `malgraph_test`:

```bash
export PYTHONPATH=.
```

### Score a pre-extracted graph JSON

```bash
python scripts/score_json.py -f path/to/sample.json
```

### Extract graph from a PE file (requires IDA)

```bash
python scripts/extract_graph.py -f path/to/sample.exe
```

### Score a PE file end-to-end

```bash
python scripts/score_pe.py -f path/to/sample.exe
```

### Batch test on external PE folders

Scores malware and benign binaries from separate directories (recursively finds files). Optional summary output via `--output`.

```bash
python scripts/batch_test_data.py \
  --malware-root /path/to/data/malware \
  --benign-root /path/to/data/benign \
  --limit 100 \
  --device cuda \
  --output workspace/batch_test_results.json
```

Defaults: `--malware-root ./data/malware/sampled_malware2`, `--benign-root ./data/benign/sampled_benign`, `--limit 100`.

### Evaluate on converted dataset (.pt files)

Expects a dataset root with `test_malware/` and `test_benign/` folders:

```bash
python scripts/eval_dataset.py --data-root /path/to/converted_output --split test
```

## Python API

```python
from malgraph import MalgraphScorer

scorer = MalgraphScorer(device="cuda")
score = scorer.score_json_file("sample.json")
print(score, scorer.classify(score))
```

## Configuration notes

`configs/malgraph.yaml` includes:

| Key | Purpose |
|-----|---------|
| `ida.path` | IDA Pro executable |
| `ida.script` | IDA extractor script (default: `ida_extractor/graph_handle_acfg.py`) |
| `workspace.tmp_sample_root` | Temp PE/JSON output during extraction |
| `workspace.gdl_dir` | IDA call-graph temp files |
| `Acfg.get_half_data` | IDA feature flag (must be `false` for standard extraction) |
| `model.*` | Architecture and checkpoint settings (must match `malgraph_tuned_v3.pt`) |

## Package behavior

- **Self-locating root:** `malgraph/config.py` resolves the package root from the installed location. `MALGRAPH_ROOT` is only used if that path contains `configs/malgraph.yaml` (ignores stale env vars pointing elsewhere).
- **IDA filename handling:** `graph_handle_acfg.py` uses the full basename (without extension) so samples with dots in the name (e.g. `Nuker.Win32.Pnuke.10`) extract correctly.
- **IDA runtime:** extraction runs inside IDA's Python 2.7 using vendored shims in `ida_extractor/`.
- **Inference runtime:** Python 3 with PyTorch + PyTorch Geometric.

## Portability

- Copy this entire folder to another machine (runtime logs and temp files are not part of the package).
- Update `ida.path` in `configs/malgraph.yaml`.
- Ensure `workspace/tmp_samples/` and `ida_extractor/gdl/` are writable.
- Model hyperparameters must match the checkpoint (`dropout_rate: 0.35` for `malgraph_tuned_v3.pt`).
