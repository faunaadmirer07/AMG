# Adversarial Malware Generation — Run Guide

Open-science release for evaluating five attack agents (Random, HillClimb,
MCTS-B40, MCTS-unlimited, PDQN) against MalConv or MalGraph.

This package is intended for a **normal Linux machine**. Configs use relative
paths and `__SET_YOUR_IDA_PATH_HERE__` placeholders.

**Note:** We decided *not* to open source the binary-modification code that
produces evasive PE variants. Demo malware samples and precomputed IDA /
call-site caches are also **not** included in this public repository; they
are available to reviewers on request. Agents, PDQN training code, detectors,
configs, and checkpoints are included. For related questions or the complete
artifact, please reach out via the corresponding author / HotCRP.

## Required environment

- Python 3.8
- Linux
- Packages in `requirements.txt`
- IDA Pro only when you need fresh BB extraction or MalGraph scoring

## Install

```bash
pip install -r requirements.txt
```

## Samples and call-site cache

This public repository does **not** ship malware binaries or BB/call-site
JSON caches. Provide your own PE corpus (isolated analysis environment only)
and set:

| Field | Purpose |
|-------|---------|
| `paths.malware_dir` | Folder containing your malware PE files |
| `paths.call_site_cache` | Directory for `*_bb.json` call-site cache |
| `ida_binary` / `scorer.ida_path` | Your IDA executable (`__SET_YOUR_IDA_PATH_HERE__`) |

| Situation | Need IDA? |
|-----------|-----------|
| You already have `*_bb.json` cache for your samples | **No** (MalConv) |
| New samples without cache | **Yes** — IDA builds `*_bb.json` once per sample |
| MalGraph scoring | **Yes** — IDA runs for every detector query |

See `data/README.md` for the `data/` layout used by paper-scale RQ1 configs.
Reviewers can request the demo 10-sample archive and matching cache.

## IDA on a normal Linux machine

1. Install IDA Pro and accept the license once (creates `~/.idapro`).
2. Replace `__SET_YOUR_IDA_PATH_HERE__` with your `idal64` path in:
   - `configs/rqs/config_demo_10_malgraph.yaml` (MalGraph demo)
   - `saved_model/malgraph/configs/malgraph.yaml`
   - any RQ1 / MalGraph config you use

`IDAUSR` defaults to `~/.idapro` when that directory exists. Override only if
your IDA license/config lives elsewhere.

## Run any agent (MalConv demo)

All commands below assume you are at the **repository root** and have set
`paths.malware_dir` / cache paths in the config.

Use `--corpus folder` to run on **every file** under `paths.malware_dir`. Limit
the run with `--max_samples N`, or pass explicit paths with `--samples`.

```bash
CONFIG=configs/rqs/config_demo_10_malconv.yaml
```

> End-to-end runs that rewrite PEs need the withheld binary-modification code.
> Commands below document the paper setup.

### Random

```bash
python evaluate.py --agent random --corpus folder \
  --config "$CONFIG" --run_tag demo_random
```

### HillClimb

```bash
python evaluate.py --agent hillclimb --corpus folder \
  --config "$CONFIG" --run_tag demo_hillclimb
```

### MCTS-B40 (capped; 41 detector queries with default config)

```bash
python evaluate.py --agent mcts --corpus folder \
  --config "$CONFIG" --run_tag demo_mcts_b40
```

### MCTS-unlimited

```bash
python evaluate.py --agent mcts --mcts_no_budget --corpus folder \
  --config "$CONFIG" --run_tag demo_mcts_unlimited
```

### PDQN

```bash
python evaluate.py --agent pdqn --corpus folder \
  --config "$CONFIG" \
  --checkpoint saved_RL_model/malconv/pdqn_malconv_quartile_v1_ep03573.pt \
  --run_tag demo_pdqn
```

### Run on a subset only

```bash
python evaluate.py --agent random --corpus folder --max_samples 3 \
  --config "$CONFIG" --run_tag demo_random_3
```

Results are written under `logs_demo_10/<run_tag>/` (`episodes.csv`, per-sample
`steps.csv`, modified binaries).

## MalGraph

MalGraph needs a working local IDA install. Use the portable demo config:

```bash
# edit configs/rqs/config_demo_10_malgraph.yaml
# replace __SET_YOUR_IDA_PATH_HERE__ with your idal64 path

CONFIG=configs/rqs/config_demo_10_malgraph.yaml

python evaluate.py --agent pdqn --corpus folder \
  --config "$CONFIG" \
  --checkpoint saved_RL_model/malgraph/pdqn_malgraph_quartile_v2_ep3501.pt \
  --run_tag demo_malgraph_pdqn
```

(Swap `--agent` for `random`, `hillclimb`, or `mcts` as above.)

MalGraph is slower than MalConv because each detector query re-runs IDA.

## Full-paper reproduction (optional)

Paper-scale runs use `configs/rqs/config_rq1_malconv_no_header.yaml` and
`config_rq1_malgraph_pdqn_rerun.yaml` with the `data/` tree populated locally.
Replace placeholders and relative paths as needed. Use `--corpus test` for the
fixed test split, or `--corpus folder` for a custom folder.

Training:

```bash
python train_pointer_dqn.py \
  --config configs/rqs/config_rq1_malconv_no_header.yaml \
  --run_tag pdqn_malconv
```

## What this export contains

```
├── agents/              # Random, HillClimb, PDQN (selection / policy code)
├── core/                # IDA extract / BB cache helpers
├── classifier/          # MalConv and MalGraph scorers
├── saved_model/         # Trained detector weights
├── saved_RL_model/      # PDQN checkpoints
├── configs/rqs/         # Demo and RQ1 configs
├── evaluate.py          # Eval entry (needs withheld PE rewrite to run)
├── train_pointer_dqn.py
└── requirements.txt
```

## Notes

- Checkpoints for MalConv and MalGraph PDQN are included; retraining is optional.
- Demo malware samples and `sample_cache_10/` are available to reviewers on request.
- Without cache, or for MalGraph, set your local `idal64` and accept the IDA license once.
