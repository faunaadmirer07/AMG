"""
Build scored hash lists (train / val / test splits) for a given detector model.

For MalConv  : scores each PE directly.
For MalGraph : uses pre-extracted JSON graphs (no IDA required).
               Pass --json_cache to point at the directory that mirrors
               malware_dir but contains .json files.

Split modes (mutually exclusive):
  --train_n / --val_n   fixed counts; test = all remaining detected samples
  --val_frac / --test_frac  fraction-based (legacy default)

Usage (MalConv, fixed split):
    conda run -n secml_malware_env python3 core/build_scored_hashlist.py \
        --config configs/config_v2_malconv_no_header.yaml \
        --threshold 0.5 \
        --train_n 3500 --val_n 100 \
        --out_prefix configs/scored_hashes_best_malconv_no_header

Usage (MalGraph, fixed split):
    conda run -n secml_malware_env python3 core/build_scored_hashlist.py \
        --config configs/config_v2_malgraph.yaml \
        --threshold 0.5 \
        --json_cache ./data/malgraph_data/extracted_graphs_new/malware/sampled_malware2 \
        --train_n 3500 --val_n 100 \
        --out_prefix configs/scored_hashes_malgraph_tuned_v3

Output:
  {out_prefix}_train.txt
  {out_prefix}_val.txt
  {out_prefix}_test.txt
  {out_prefix}_scores.txt   — all PE files with their scores (tab-separated)
"""
import os
import sys
import random
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _scan_malware_dir(malware_dir):
    """Return sorted list of relative paths (subdir/filename), skipping IDA databases."""
    entries = []
    for subdir in sorted(os.listdir(malware_dir)):
        full_subdir = os.path.join(malware_dir, subdir)
        if not os.path.isdir(full_subdir):
            continue
        for fname in sorted(os.listdir(full_subdir)):
            if fname.endswith('.i64'):
                continue
            entries.append(os.path.join(subdir, fname))
    return entries


def _json_path_for(rel_path, json_cache, json_cache_extra=None):
    """Given a relative malware path, return the matching .json path (or None).
    Checks json_cache first, then json_cache_extra as fallback."""
    stem = os.path.splitext(os.path.basename(rel_path))[0]
    subdir = os.path.dirname(rel_path)
    for cache in filter(None, [json_cache, json_cache_extra]):
        path = os.path.join(cache, subdir, stem + '.json')
        if os.path.exists(path):
            return path
    return None


def _score_malconv(rel_paths, malware_dir, scorer):
    """Score all PEs; return list of (rel_path, score) for successful ones."""
    results = []
    for i, rel in enumerate(rel_paths):
        pe_path = os.path.join(malware_dir, rel)
        try:
            with open(pe_path, 'rb') as f:
                bytez = f.read()
            score = scorer.score_pe(bytez)
        except Exception as e:
            print('  FAIL {}: {}'.format(rel, e))
            continue
        if score is None:
            continue
        if (i + 1) % 500 == 0:
            print('  [{}/{}]'.format(i + 1, len(rel_paths)))
        results.append((rel, score))
    return results


def _score_malgraph(rel_paths, json_cache, server, json_cache_extra=None):
    """Score all PEs via pre-extracted JSONs; return list of (rel_path, score)."""
    results = []
    missing, failed = [], []
    for i, rel in enumerate(rel_paths):
        json_path = _json_path_for(rel, json_cache, json_cache_extra)
        if json_path is None:
            missing.append(rel)
            continue
        try:
            score = server.score_json_file(json_path)
        except Exception as e:
            print('  FAIL {}: {}'.format(rel, e))
            failed.append(rel)
            continue
        if score is None:
            failed.append(rel)
            continue
        if (i + 1) % 500 == 0:
            print('  [{}/{}] last score={:.4f}'.format(i + 1, len(rel_paths), score))
        results.append((rel, score))
    print('  missing JSON: {}  failed: {}'.format(len(missing), len(failed)))
    return results


def _split_fixed(detected, train_n, val_n, seed):
    """Fixed-count split: train=train_n, val=val_n, test=rest."""
    rng = random.Random(seed)
    shuffled = list(detected)
    rng.shuffle(shuffled)
    if train_n + val_n > len(shuffled):
        raise ValueError(
            'train_n ({}) + val_n ({}) = {} > detected ({})'.format(
                train_n, val_n, train_n + val_n, len(shuffled)))
    train = shuffled[:train_n]
    val   = shuffled[train_n:train_n + val_n]
    test  = shuffled[train_n + val_n:]
    return train, val, test


def _split_frac(detected, val_frac, test_frac, seed):
    """Fraction-based split."""
    rng = random.Random(seed)
    shuffled = list(detected)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val  = int(n * val_frac)
    n_test = int(n * test_frac)
    val   = shuffled[:n_val]
    test  = shuffled[n_val:n_val + n_test]
    train = shuffled[n_val + n_test:]
    return train, val, test


def _write(path, items):
    with open(path, 'w') as f:
        for item in items:
            f.write(item + '\n')
    print('  wrote {} → {}'.format(len(items), path))


def _write_scores(path, results):
    """Write all (rel_path, score) pairs, sorted by rel_path."""
    with open(path, 'w') as f:
        f.write('path\tscore\n')
        for rel, score in sorted(results, key=lambda x: x[0]):
            f.write('{}\t{:.6f}\n'.format(rel, score))
    print('  wrote {} scores → {}'.format(len(results), path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--threshold',  type=float, default=0.5)
    parser.add_argument('--out_prefix', required=True)
    parser.add_argument('--valid_list', default=None,
                        help='Restrict scoring/splitting to paths listed in this file '
                             '(one relative path per line, e.g. configs/valid_samples.txt)')
    parser.add_argument('--json_cache', default=None,
                        help='Dir of pre-extracted JSONs (required for MalGraph)')
    parser.add_argument('--json_cache_extra', default=None,
                        help='Supplementary JSON cache (checked if not found in --json_cache)')
    parser.add_argument('--model_path', default=None,
                        help='Override model checkpoint path (malgraph: overrides model.path in yaml)')
    # fixed-count split (preferred)
    parser.add_argument('--train_n', type=int, default=None)
    parser.add_argument('--val_n',   type=int, default=None)
    # fraction-based split (legacy)
    parser.add_argument('--val_frac',  type=float, default=0.10)
    parser.add_argument('--test_frac', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    malware_dir = config['paths']['malware_dir']
    scorer_cfg  = config.get('scorer', {})
    model_type  = scorer_cfg.get('type', 'malconv')

    print('Scanning {}...'.format(malware_dir))
    rel_paths = _scan_malware_dir(malware_dir)
    print('  found {} PE files'.format(len(rel_paths)))

    if args.valid_list:
        with open(args.valid_list) as f:
            valid_set = set(l.strip() for l in f if l.strip())
        before = len(rel_paths)
        rel_paths = [p for p in rel_paths if p in valid_set]
        print('  filtered to {} valid files ({} excluded by --valid_list)'.format(
            len(rel_paths), before - len(rel_paths)))

    print('Loading {} model (threshold={})...'.format(model_type, args.threshold))

    if model_type == 'malgraph':
        if not args.json_cache:
            parser.error('--json_cache is required for MalGraph')
        from classifier.malgraph.server import MalGraphServer
        mg_config = None
        if args.model_path:
            from saved_model.malgraph.malgraph.config import load_config
            mg_config = load_config()
            mg_config['model']['path'] = args.model_path
        server = MalGraphServer(
            config=mg_config,
            device=scorer_cfg.get('device', 'cpu'),
            threshold=args.threshold,
            ida_path=scorer_cfg.get('ida_path'),
            tmp_dir=scorer_cfg.get('tmp_dir'),
        )
        extra = args.json_cache_extra
        print('Scoring with pre-extracted JSONs from {}{}...'.format(
            args.json_cache, ' + {}'.format(extra) if extra else ''))
        all_results = _score_malgraph(rel_paths, args.json_cache, server, extra)
    else:
        from classifier.scorer import Scorer
        scorer = Scorer(scorer_cfg)
        print('Scoring {} binaries...'.format(len(rel_paths)))
        all_results = _score_malconv(rel_paths, malware_dir, scorer)

    detected = [rel for rel, score in all_results if score >= args.threshold]
    print('\nDetected: {}/{}'.format(len(detected), len(rel_paths)))

    if args.train_n is not None and args.val_n is not None:
        train, val, test = _split_fixed(detected, args.train_n, args.val_n, args.seed)
    else:
        train, val, test = _split_frac(detected, args.val_frac, args.test_frac, args.seed)

    print('Split → train:{} val:{} test:{}'.format(len(train), len(val), len(test)))

    t_tag = '_t{}'.format(args.threshold)
    _write(args.out_prefix + t_tag + '_train.txt',  sorted(train))
    _write(args.out_prefix + t_tag + '_val.txt',    sorted(val))
    _write(args.out_prefix + t_tag + '_test.txt',   sorted(test))
    _write_scores(args.out_prefix + '_scores.txt', all_results)
    print('Done.')


if __name__ == '__main__':
    main()
