"""
Batch IDA cache generation for a new dataset.

For each binary in the target directory (or hash list), runs IDA Pro with
get_call_ida_bb_grouped.py and saves a comprehensive _bb.json to the cache dir.

Usage:
    # Generate cache for all binaries in a directory
    python3 generate_cache.py \
        --binary_dir ./data/new_malware \
        --cache_dir  ./data/cache/bb_data \
        --ida        __SET_YOUR_IDA_PATH_HERE__ \
        --script     core/get_call_ida_bb_grouped.py \
        --workers    8

    # Generate cache using a hash list + binary dir
    python3 generate_cache.py \
        --hash_list  configs/train_hashes.txt \
        --binary_dir ./data/dataset_rl/malware5k \
        --cache_dir  ./data/malguise_latest/MalGuise_0320/rl_malware/cache/bb_data \
        --ida        __SET_YOUR_IDA_PATH_HERE__ \
        --script     core/get_call_ida_bb_grouped.py \
        --workers    8

    # Use config file (reads paths.malware_dir, paths.call_site_cache, etc.)
    python3 generate_cache.py \
        --config  configs/config_pointer_dqn_v1.yaml \
        --workers 8

    # Dry run (show how many would be generated, don't run IDA)
    python3 generate_cache.py --config configs/config_pointer_dqn_v1.yaml --dry_run

    # Skip binaries that already have a cache file (default behaviour).
    # To force-regenerate everything, add --overwrite.
    python3 generate_cache.py --config configs/config_pointer_dqn_v1.yaml --overwrite
"""
import os
import sys
import argparse
import time
import json
import yaml
from multiprocessing.pool import ThreadPool


# ── Worker ────────────────────────────────────────────────────────────────

def _run_one(task):
    pe_path, ida_binary, ida_script, cache_dir, overwrite = task
    binary_name = os.path.basename(pe_path)

    # Support subdir layout: cache_dir/<subdir>/<name>_bb.json
    parent_name = os.path.basename(os.path.dirname(pe_path))
    subdir_cache = os.path.join(cache_dir, parent_name)
    if os.path.isdir(subdir_cache):
        effective_cache = subdir_cache
    else:
        effective_cache = cache_dir
    cache_path = os.path.join(effective_cache, binary_name + '_bb.json')

    if not overwrite and os.path.isfile(cache_path):
        return binary_name, 'skip', 0.0

    if not os.path.isfile(pe_path):
        return binary_name, 'missing', 0.0

    os.makedirs(effective_cache, exist_ok=True)
    t0 = time.time()
    log_file = f'/tmp/ida_gen_{binary_name[:12]}.log'
    cmd = (
        f'IDA_CACHE_DIR={effective_cache} TVHEADLESS=1 '
        f'{ida_binary} -A -c -L{log_file} '
        f'-S{ida_script} {pe_path} > /dev/null 2>&1'
    )
    ret = os.system(cmd)
    elapsed = time.time() - t0

    # Clean IDA temp files left next to the binary
    for ext in ('.id0', '.id1', '.id2', '.nam', '.til', '.i64'):
        tmp = pe_path + ext
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

    if os.path.isfile(cache_path):
        return binary_name, 'ok', elapsed

    # Failed: read IDA log for clues
    reason = f'exit={ret}'
    if os.path.isfile(log_file):
        try:
            tail = open(log_file).read()[-300:]
            reason = tail.strip().replace('\n', ' ')
        except Exception:
            pass
    return binary_name, f'failed:{reason}', elapsed


# ── Helpers ────────────────────────────────────────────────────────────────

def _collect_binaries(binary_dir, hash_list=None):
    """Return list of absolute paths to PE binaries."""
    if hash_list:
        hashes = open(hash_list).read().split()
        paths = [os.path.join(binary_dir, h) for h in hashes]
    else:
        paths = []
        for entry in sorted(os.listdir(binary_dir)):
            full = os.path.join(binary_dir, entry)
            if os.path.isfile(full):
                paths.append(full)
            elif os.path.isdir(full):
                # one level of subdirectories (e.g. console_x86/, gui_x86/)
                for f in sorted(os.listdir(full)):
                    fp = os.path.join(full, f)
                    if os.path.isfile(fp):
                        paths.append(fp)
    return paths


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_argument_group('paths (override config)')
    g.add_argument('--binary_dir', default=None,
                   help='Directory containing PE binaries')
    g.add_argument('--hash_list',  default=None,
                   help='Optional text file with one hash per line')
    g.add_argument('--cache_dir',  default=None,
                   help='Output directory for _bb.json files')
    g.add_argument('--ida',        default=None,
                   help='Path to idal64 binary')
    g.add_argument('--script',     default=None,
                   help='Path to get_call_ida_bb_grouped.py')
    p.add_argument('--config',   default=None,
                   help='YAML config file (reads paths section)')
    p.add_argument('--workers',  type=int, default=4,
                   help='Parallel IDA processes (default 4)')
    p.add_argument('--overwrite', action='store_true',
                   help='Regenerate cache even if it already exists')
    p.add_argument('--dry_run',  action='store_true',
                   help='Show what would run, do not execute IDA')
    p.add_argument('--limit',    type=int, default=None,
                   help='Process at most N binaries (for testing)')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────
    binary_dir = args.binary_dir
    hash_list  = args.hash_list
    cache_dir  = args.cache_dir
    ida_binary = args.ida
    ida_script = args.script

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        paths = cfg.get('paths', {})
        binary_dir = binary_dir or paths.get('malware_dir')
        cache_dir  = cache_dir  or paths.get('call_site_cache')
        ida_binary = ida_binary or paths.get('ida_binary')
        ida_script = ida_script or paths.get('ida_script')
        # If no explicit hash_list, use all three splits from config
        if not hash_list and not args.binary_dir:
            all_hashes = []
            for key in ('train_hashes', 'val_hashes', 'test_hashes'):
                hf = paths.get(key)
                if hf and os.path.isfile(hf):
                    all_hashes += open(hf).read().split()
            # Write a combined tmp list
            if all_hashes:
                tmp_list = '/tmp/_generate_cache_hashes.txt'
                with open(tmp_list, 'w') as f:
                    f.write('\n'.join(dict.fromkeys(all_hashes)))
                hash_list = tmp_list

    for name, val in [('binary_dir', binary_dir), ('cache_dir', cache_dir),
                      ('ida_binary', ida_binary), ('ida_script', ida_script)]:
        if not val:
            print(f'ERROR: --{name} not specified and not found in config')
            sys.exit(1)

    os.makedirs(cache_dir, exist_ok=True)

    # ── Collect binaries ───────────────────────────────────────────────────
    all_paths = _collect_binaries(binary_dir, hash_list)
    if args.limit:
        all_paths = all_paths[:args.limit]

    # Pre-scan: how many already have cache
    already   = sum(1 for p in all_paths
                    if os.path.isfile(os.path.join(cache_dir,
                                                   os.path.basename(p) + '_bb.json')))
    to_run    = len(all_paths) - already if not args.overwrite else len(all_paths)
    missing   = sum(1 for p in all_paths if not os.path.isfile(p))

    print(f'Binary dir : {binary_dir}')
    print(f'Cache dir  : {cache_dir}')
    print(f'IDA binary : {ida_binary}')
    print(f'IDA script : {ida_script}')
    print(f'Workers    : {args.workers}')
    print(f'Overwrite  : {args.overwrite}')
    print()
    print(f'Total binaries   : {len(all_paths)}')
    print(f'Already cached   : {already}')
    print(f'Missing on disk  : {missing}')
    print(f'Will run IDA on  : {to_run}')

    if args.dry_run:
        print('\n[dry_run] Exiting without running IDA.')
        return

    if to_run == 0:
        print('\nAll binaries already cached. Use --overwrite to regenerate.')
        return

    # ── Build task list ────────────────────────────────────────────────────
    tasks = [
        (p, ida_binary, ida_script, cache_dir, args.overwrite)
        for p in all_paths
    ]

    # ── Run ───────────────────────────────────────────────────────────────
    print(f'\nStarting {to_run} IDA jobs (skipping {already} cached)...\n')
    t_start = time.time()
    counts  = {'ok': 0, 'skip': 0, 'missing': 0, 'failed': 0}
    failed_list = []

    with ThreadPool(args.workers) as pool:
        for i, (name, status, elapsed) in enumerate(
            pool.imap_unordered(_run_one, tasks), 1
        ):
            if status == 'ok':
                counts['ok'] += 1
            elif status == 'skip':
                counts['skip'] += 1
            elif status == 'missing':
                counts['missing'] += 1
            else:
                counts['failed'] += 1
                failed_list.append((name, status))

            if i % 100 == 0 or status.startswith('failed'):
                elapsed_total = time.time() - t_start
                done = counts['ok'] + counts['failed']
                rate = done / max(elapsed_total, 1)
                eta  = (to_run - done) / max(rate, 0.001) / 60
                print(f'  [{i:>5}/{len(tasks)}]  '
                      f'ok={counts["ok"]}  skip={counts["skip"]}  '
                      f'fail={counts["failed"]}  '
                      f'rate={rate:.1f}/s  eta={eta:.0f}min'
                      + (f'  FAIL: {name} → {status}' if status.startswith('failed') else ''))

    total_time = time.time() - t_start
    print(f'\n{"─"*60}')
    print(f'Done in {total_time/60:.1f} min')
    print(f'  ok={counts["ok"]}  skip={counts["skip"]}  '
          f'missing={counts["missing"]}  failed={counts["failed"]}')

    if failed_list:
        fail_log = 'generate_cache_failed.txt'
        with open(fail_log, 'w') as f:
            for name, reason in failed_list:
                f.write(f'{name}\t{reason}\n')
        print(f'\n  Failed binaries written to {fail_log}')
        print(f'  First 10:')
        for name, reason in failed_list[:10]:
            print(f'    {name}  →  {reason[:80]}')


if __name__ == '__main__':
    main()
