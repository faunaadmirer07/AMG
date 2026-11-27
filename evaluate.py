"""Evaluation entry point."""

import os
import sys
import time
import random
import argparse
import shutil
import signal
import yaml
import numpy as np


class _EpisodeTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _EpisodeTimeout("Episode timed out (SIGALRM)")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier.scorer import Scorer as DetectionHandler
try:
    from core.patcher import Patcher
    from core.env import MalwareEvasionEnv
except ImportError:
    raise SystemExit(
        "Binary-modification code is not included in this public release "
        "(see README.md)."
    )


_POINTER_AGENTS = {'pointer_random', 'pointer_hillclimb', 'pdqn', 'pointer_dqn'}


_worker_state = {}


def _worker_init(config, weights_cpu):
    """Pool initializer: load scorer and Q-network once per worker process."""
    import atexit, tempfile, torch
    from core.patcher import Patcher

    scorer_type = config.get('scorer', {}).get('type', 'malconv')
    tmp_dir = None

    if scorer_type == 'malgraph':
        from classifier.malgraph.server import MalGraphServer
        tmp_dir = tempfile.mkdtemp(prefix='mg_evalw_')
        scorer = MalGraphServer(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            threshold=config['env']['threshold'],
            ida_path=config['scorer'].get('ida_path'),
            tmp_dir=tmp_dir,
        )
        atexit.register(shutil.rmtree, tmp_dir, True)
    else:
        from classifier.scorer import Scorer as _Scorer
        cfg2 = dict(config['scorer'])
        cfg2['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        scorer = _Scorer(cfg2)

    patcher = Patcher(
        selected_semnop_num=config['env']['selected_semnop_num'],
        max_random_number=config['env']['max_random_number'],
    )

    q_net = None
    if weights_cpu is not None:
        from agents.pointer_dqn_network import PointerQNetwork
        cfg_pdqn = config['pointer_dqn']
        q_net = PointerQNetwork(
            cfg_pdqn.get('bb_dim', 30), cfg_pdqn.get('ctx_dim', 9),
            cfg_pdqn.get('hidden_dim', 256), cfg_pdqn.get('use_dueling', True),
        )
        q_net.load_state_dict(weights_cpu)
        q_net.eval()

    _worker_state['scorer']  = scorer
    _worker_state['patcher'] = patcher
    _worker_state['q_net']   = q_net
    _worker_state['config']  = config


def _pdqn_eval_worker_fn(task):
    """
    Subprocess worker for parallel PDQN eval.
    Writes steps.csv and modified.exe directly; returns (sample_hash, status, info).
    status: 'ok' | 'skip_benign' | 'skip_no_calls' | 'error'
    """
    import csv, random as _random, torch
    from datetime import datetime
    from core.pointer_dqn_env import PointerDQNEnv

    sample_hash, pe_path, run_dir, eval_params = task

    allow_rollback = eval_params.get('allow_rollback', True)
    query_budget   = eval_params.get('query_budget', None)
    agent_name     = eval_params.get('agent_name', 'pdqn')

    scorer  = _worker_state['scorer']
    patcher = _worker_state['patcher']
    q_net   = _worker_state['q_net']
    config  = _worker_state['config']

    use_full_agent = agent_name in ('pointer_random', 'pointer_hillclimb')

    env = PointerDQNEnv(scorer, patcher, config,
                        allow_rollback=allow_rollback, eval_mode=True,
                        query_budget=query_budget)

    # --- reset ---
    try:
        bb_feats, context = env.reset(pe_path)
    except Exception as e:
        env.cleanup()
        return sample_hash, 'error', None, str(e)

    if env.done and env.bypassed:
        env.cleanup()
        return sample_hash, 'skip_benign', None, None
    if not env.remaining_bbs:
        env.cleanup()
        return sample_hash, 'skip_no_calls', None, None

    # --- episode loop ---
    try:
        while not env.done:
            if use_full_agent:
                action = _random.randint(0, len(bb_feats) - 1) if len(bb_feats) > 0 else 0
            else:
                with torch.no_grad():
                    N = len(bb_feats)
                    bb_t  = torch.FloatTensor(bb_feats).unsqueeze(0)
                    ctx_t = torch.FloatTensor(context).unsqueeze(0)
                    mask  = torch.ones(1, N, dtype=torch.bool)
                    q     = q_net(bb_t, ctx_t, mask)
                    action = int(q[0].argmax().item())
            (bb_feats, context), _, _, _ = env.step(action)
    except Exception as e:
        env.cleanup()
        return sample_hash, 'error', None, str(e)

    adv_bytes = env.get_adversarial_bytes()
    info      = env._get_info()
    step_log  = list(env.step_log)
    env.cleanup()

    # --- write steps.csv ---
    binary_dir = os.path.join(run_dir, sample_hash)
    os.makedirs(binary_dir, exist_ok=True)
    step_path = os.path.join(binary_dir, 'steps.csv')
    STEP_COLS = ['episode','step','bb_index','bb_start','bb_end','func_addr','action',
                 'num_calls','num_calls_eligible','bb_size','score_before','score_after',
                 'reward','cumulative_delta','budget_used','bypassed','time_elapsed','timestamp']
    write_header = not os.path.exists(step_path)
    with open(step_path, 'a', newline='') as sf:
        w = csv.DictWriter(sf, fieldnames=STEP_COLS)
        if write_header:
            w.writeheader()
        for sd in step_log:
            sd['binary_hash'] = sample_hash
            row = {c: sd.get(c, '') for c in STEP_COLS}
            row['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            w.writerow(row)

    # --- write modified.exe ---
    if info.get('n_mods', 0) > 0 and adv_bytes is not None:
        with open(os.path.join(binary_dir, 'modified.exe'), 'wb') as bf:
            bf.write(adv_bytes)

    return sample_hash, 'ok', info, None


def _build_env(config, allow_rollback=True, agent_name='random', query_budget=None):
    """Build scorer + env. All agents use binary-level modification."""
    scorer_type = config.get('scorer', {}).get('type', 'malconv')
    use_pointer = agent_name in _POINTER_AGENTS

    # --- scorer ---
    if scorer_type == 'malgraph':
        from classifier.malgraph.server import MalGraphServer
        scorer = MalGraphServer(
            device=config['scorer'].get('device', 'cpu'),
            threshold=config['env']['threshold'],
            ida_path=config['scorer'].get('ida_path'),
            tmp_dir=config['scorer'].get('tmp_dir'),
        )
    else:
        scorer = DetectionHandler(config['scorer'])

    # --- pointer agents: always use binary-level PointerDQNEnv ---
    if use_pointer:
        from core.pointer_dqn_env import PointerDQNEnv
        patcher = Patcher(
            selected_semnop_num=config['env']['selected_semnop_num'],
            max_random_number=config['env']['max_random_number'],
        )
        env = PointerDQNEnv(scorer, patcher, config, allow_rollback=allow_rollback, eval_mode=True, query_budget=query_budget)
        return scorer, env

    # --- default env (malconv / malrnn / malgraph binary) ---
    patcher = Patcher(
        selected_semnop_num=config['env']['selected_semnop_num'],
        max_random_number=config['env']['max_random_number'],
    )
    env = MalwareEvasionEnv(scorer, patcher, config, allow_rollback=allow_rollback)
    return scorer, env
from utils.logger import Logger


def get_corpus(config, corpus_type, specific_samples=None):
    """
    Get list of PE paths for evaluation.

    Args:
        config: loaded config dict
        corpus_type: 'val', 'train', or 'all'
        specific_samples: list of specific hashes (overrides corpus_type)

    Returns:
        list of (hash, pe_path) tuples
    """
    malware_dir = config['paths']['malware_dir']

    if specific_samples:
        result = []
        for h in specific_samples:
            path = os.path.join(malware_dir, h)
            if os.path.exists(path):
                result.append((h, path))
            else:
                print("WARNING: sample not found: {}".format(h))
        return result

    # test split uses dedicated hash file — no need to load full corpus
    if corpus_type == 'test':
        test_path = config['paths'].get('test_hashes')
        if not test_path or not os.path.exists(test_path):
            raise ValueError("test_hashes not found in config or on disk: {}".format(test_path))
        with open(test_path) as f:
            selected = [l.strip() for l in f if l.strip()]
        return [(h, os.path.join(malware_dir, h)) for h in selected]

    if corpus_type == 'folder':
        malware_root = os.path.abspath(malware_dir)
        if not os.path.isdir(malware_root):
            raise ValueError("malware_dir not found: {}".format(malware_dir))
        result = []
        for root, _, files in os.walk(malware_root):
            for name in sorted(files):
                full = os.path.join(root, name)
                rel = os.path.relpath(full, malware_root)
                result.append((rel, full))
        if not result:
            raise ValueError("No files found under malware_dir: {}".format(malware_dir))
        return result

    # Get hashes from corpus_hashes.txt (curated list of 3499 binaries)
    corpus_path = config['paths']['corpus_hashes']
    with open(corpus_path, 'r') as f:
        all_hashes = sorted([line.strip() for line in f if line.strip()])

    if corpus_type == 'all':
        return [(h, os.path.join(malware_dir, h)) for h in all_hashes]

    # Train/val split (same as Malware Makeover: SEED=1337, VAL_FRAC=0.10)
    seed = config['training']['seed']
    val_frac = config['training']['val_frac']

    rng = random.Random(seed)
    indices = list(range(len(all_hashes)))
    rng.shuffle(indices)

    val_size = int(len(indices) * val_frac)
    val_indices = set(indices[:val_size])
    train_indices = set(indices[val_size:])

    if corpus_type == 'val':
        selected = [all_hashes[i] for i in sorted(val_indices)]
    elif corpus_type == 'train':
        selected = [all_hashes[i] for i in sorted(train_indices)]
    else:
        raise ValueError("corpus must be 'val', 'train', 'test', or 'all'")

    return [(h, os.path.join(malware_dir, h)) for h in selected]


def build_agent(agent_name, config, args, scorer=None):
    """
    Build an agent by name.

    Args:
        agent_name: 'random', 'hillclimb', 'dqn'
        config: loaded config dict
        args: argparse namespace (for agent-specific params)

    Returns:
        BaseAgent instance or special agent (hillclimb/mcts with run_episode)
    """
    if agent_name == 'random':
        from agents.random_agent import RandomAgent
        prob = args.modify_prob if args.modify_prob else config['baselines']['random']['modify_probability']
        return RandomAgent(modify_prob=prob, seed=args.seed)
    elif agent_name == 'hillclimb':
        from agents.hillclimb_agent import HillClimbAgent
        prob = args.modify_prob if args.modify_prob else config['baselines']['random']['modify_probability']
        return HillClimbAgent(modify_prob=prob, seed=args.seed)
    elif agent_name == 'pointer_random':
        from agents.pointer_random_agent import PointerRandomAgent
        return PointerRandomAgent(seed=args.seed)
    elif agent_name == 'pointer_hillclimb':
        from agents.pointer_hillclimb_agent import PointerHillClimbAgent
        return PointerHillClimbAgent(seed=args.seed)
    elif agent_name == 'dqn':
        from agents.dqn_agent import DQNAgent
        if not args.checkpoint:
            raise ValueError("DQN requires --checkpoint path")
        agent = DQNAgent.load(args.checkpoint, config)
        agent.set_eval_mode()
        return agent
    elif agent_name == 'pdqn':
        from agents.pointer_dqn_agent import PointerDQNAgent
        if not args.checkpoint:
            raise ValueError("pdqn requires --checkpoint path")
        agent = PointerDQNAgent.load(args.checkpoint, config)
        agent.set_eval_mode()
        return agent
    elif agent_name == 'mcts':
        from agents.mcts_agent import MCTSAgent
        no_budget = getattr(args, 'mcts_no_budget', False)
        return MCTSAgent(scorer, config, unlimited=no_budget)
    else:
        raise ValueError("Unknown agent: {}. Available: random, hillclimb, pointer_random, pointer_hillclimb, dqn, pdqn, mcts".format(agent_name))


def evaluate(agent, env, corpus, logger, config, max_samples=None, done_hashes=None,
             num_workers=1, eval_params=None):
    """
    Run agent on corpus and log results.

    Args:
        agent: BaseAgent instance
        env: MalwareEvasionEnv instance
        corpus: list of (hash, pe_path) tuples
        logger: Logger instance
        config: config dict
        max_samples: limit number of samples (None = all)
        done_hashes: set of hashes to skip (for --resume)

    Returns:
        dict with summary statistics
    """
    if max_samples:
        corpus = corpus[:max_samples]
    if done_hashes is None:
        done_hashes = set()

    # Also skip binaries that previously crashed (segfault)
    crash_skip_path = os.path.join(logger.run_dir, 'crash_skip.txt')
    if os.path.exists(crash_skip_path):
        with open(crash_skip_path, 'r') as f:
            crash_hashes = set(line.strip() for line in f if line.strip())
        done_hashes.update(crash_hashes)
        print("SKIP: {} binaries in crash_skip.txt".format(len(crash_hashes)))

    total = 0
    bypassed = 0
    skipped_benign = 0
    skipped_no_calls = 0
    failed = 0
    total_queries = 0
    total_mods = 0
    deltas = []

    # --- parallel path for pointer agents ---
    if num_workers > 1 and agent.name in _POINTER_AGENTS and eval_params is not None:
        import multiprocessing as mp
        import torch

        _done = done_hashes or set()
        filtered = [(h, p) for h, p in (corpus[:max_samples] if max_samples else corpus)
                    if h not in _done and os.path.basename(h) not in _done and os.path.exists(p)]

        weights_cpu = None
        if hasattr(agent, 'q_network'):
            weights_cpu = {k: v.cpu() for k, v in agent.q_network.state_dict().items()}

        tasks = [
            (h, p, logger.run_dir, eval_params)
            for h, p in filtered
        ]

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_workers,
                      initializer=_worker_init,
                      initargs=(config, weights_cpu)) as pool:
            for ep_idx, (sample_hash, status, info, err) in enumerate(
                    pool.imap_unordered(_pdqn_eval_worker_fn, tasks)):
                if status == 'skip_benign':
                    skipped_benign += 1
                    continue
                if status == 'skip_no_calls':
                    skipped_no_calls += 1
                    continue
                if status == 'error':
                    failed += 1
                    print("  ERROR {} — {}".format(sample_hash[:12], err))
                    continue

                total += 1
                ep_time = info.get('wall_time_sec', 0.0)
                logger.start_episode(sample_hash)
                logger.log_episode(ep_idx, info, ep_time)
                logger.flush()

                total_queries += info['query_count']
                total_mods    += info['n_mods']
                deltas.append(info['total_delta'])
                if info['bypassed']:
                    bypassed += 1

                status_str = 'BYPASSED' if info['bypassed'] else 'FAILED'
                print("  [{:>4d}/{}] {} {:.4f}→{:.4f} mods={} q={} {}".format(
                    total, len(filtered), sample_hash[:12],
                    info['start_score'], info['curr_score'],
                    info['n_mods'], info['query_count'], status_str))

        asr = bypassed / total if total > 0 else 0
        return {
            'agent': agent.name,
            'total_attackable': total,
            'bypassed': bypassed,
            'asr': asr,
            'skipped_benign': skipped_benign,
            'skipped_no_calls': skipped_no_calls,
            'failed': failed,
            'avg_queries': total_queries / total if total > 0 else 0,
            'avg_mods': total_mods / total if total > 0 else 0,
            'avg_delta': np.mean(deltas) if deltas else 0,
            'max_delta': max(deltas) if deltas else 0,
        }

    for ep_idx, (sample_hash, pe_path) in enumerate(corpus):
        ep_start = time.time()

        # Skip already-processed binaries (--resume)
        # episodes.csv stores basename; corpus hash may include subdirectory prefix
        if sample_hash in done_hashes or os.path.basename(sample_hash) in done_hashes:
            continue

        # Write current hash to temp file (for crash detection)
        crash_marker = os.path.join(logger.run_dir, '.current_binary')
        with open(crash_marker, 'w') as f:
            f.write(sample_hash)

        # Agents with their own run_episode (hillclimb, mcts)
        if hasattr(agent, 'run_episode'):
            if agent.name == 'mcts':
                # Initial benign/no-calls check via env (consistent with other agents).
                # MCTS scores the original binary, but env scores post-.htext — some
                # binaries drop below threshold just from the section addition. Check
                # here so those aren't counted as attackable.
                #
                # Per-episode hard timeout: catches IDA/lief/pefile hangs that
                # mcts_pause_time cannot (it only guards the inner search loop).
                _ep_timeout = config.get('baselines', {}).get('mcts', {}).get('episode_timeout', 900)
                _old_alarm = signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(_ep_timeout)
                try:
                    try:
                        env.reset(pe_path)
                    except _EpisodeTimeout:
                        raise
                    except Exception as e:
                        print("  SKIP {} — reset failed: {}".format(sample_hash[:12], e))
                        failed += 1
                        env.cleanup()
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, _old_alarm)
                        continue
                    if env.done and env.bypassed:
                        skipped_benign += 1
                        env.cleanup()
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, _old_alarm)
                        continue
                    _bbs = getattr(env, 'remaining_bbs', None) or getattr(env, 'bb_list', [])
                    if len(_bbs) == 0:
                        skipped_no_calls += 1
                        env.cleanup()
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, _old_alarm)
                        continue
                    env.cleanup()

                    info, total_reward, ep_time = agent.run_episode(pe_path)
                except _EpisodeTimeout:
                    print("  TIMEOUT {} — >{:d}s, skipping".format(sample_hash[:12], _ep_timeout))
                    # Add to crash_skip so --resume won't retry
                    crash_skip_path = os.path.join(logger.run_dir, 'crash_skip.txt')
                    with open(crash_skip_path, 'a') as _f:
                        _f.write(sample_hash + '\n')
                    failed += 1
                    env.cleanup()
                    continue
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, _old_alarm)
            else:
                # Hill-climbing uses our env but manages its own loop
                info, total_reward, ep_time = agent.run_episode(env, pe_path, ep_idx)

            if info is None:
                failed += 1
                continue

            total += 1
            logger.start_episode(sample_hash)
            logger.log_episode(ep_idx, info, ep_time)

            # Log steps for hillclimb (env has step_log)
            if agent.name == 'hillclimb' and hasattr(env, 'step_log'):
                for step_data in env.step_log:
                    step_data['binary_hash'] = sample_hash
                    logger.log_step(ep_idx, step_data)

            # Save modified binary for hillclimb (mcts saves via saved_sample_path internally)
            if agent.name == 'hillclimb' and hasattr(agent, 'last_adv_bytes') and agent.last_adv_bytes is not None:
                if info.get('bypassed', False):
                    logger.save_modified_binary(sample_hash, agent.last_adv_bytes)

            # Log steps for mcts (step_log embedded in info)
            if agent.name == 'mcts':
                for step_data in info.get('step_log', []):
                    step_data['binary_hash'] = sample_hash
                    logger.log_step(ep_idx, step_data)

        elif agent.name in _POINTER_AGENTS:
            # Pointer agents: obs = (bb_feats [N, BB_DIM], context [CTX_DIM])
            try:
                bb_feats, context = env.reset(pe_path)
            except Exception as e:
                print("  SKIP {} — reset failed: {}".format(sample_hash[:12], e))
                failed += 1
                continue

            if env.done and env.bypassed:
                skipped_benign += 1
                env.cleanup()
                continue

            if not env.remaining_bbs:
                skipped_no_calls += 1
                env.cleanup()
                continue

            total += 1
            if hasattr(agent, 'on_episode_start'):
                agent.on_episode_start()
            logger.start_episode(sample_hash)

            total_reward = 0.0
            step_error = None
            try:
                while not env.done:
                    if agent.name == 'pointer_dqn':
                        action = agent.select_action(bb_feats, context, training=False)
                    else:
                        action = agent.select_action(bb_feats, context)
                    (bb_feats, context), reward, done, info = env.step(action)
                    total_reward += reward
            except Exception as e:
                step_error = e
                print("  STEP ERROR {} — {}: {}".format(sample_hash[:12], type(e).__name__, e))

            if step_error is not None:
                env.cleanup()
                failed += 1
                continue

            for step_data in env.step_log:
                step_data['binary_hash'] = sample_hash
                logger.log_step(ep_idx, step_data)

            ep_time = time.time() - ep_start
            adv_bytes = env.get_adversarial_bytes()  # updates curr_injected before _get_info
            info = env._get_info()
            logger.log_episode(ep_idx, info, ep_time)
            if hasattr(agent, 'on_episode_end'):
                agent.on_episode_end(info)

            if env.n_mods > 0 and adv_bytes is not None:
                logger.save_modified_binary(sample_hash, adv_bytes)

            env.cleanup()

        else:
            # Standard agents (random, dqn) — use select_action loop
            try:
                obs = env.reset(pe_path)
            except Exception as e:
                print("  SKIP {} — reset failed: {}".format(sample_hash[:12], e))
                failed += 1
                continue

            # Check if already benign
            if env.done and env.bypassed:
                skipped_benign += 1
                env.cleanup()
                continue

            # Check if no BBs with call sites
            _bbs = getattr(env, 'remaining_bbs', None) or getattr(env, 'bb_list', [])
            if len(_bbs) == 0:
                skipped_no_calls += 1
                env.cleanup()
                continue

            total += 1
            if hasattr(agent, 'on_episode_start'):
                agent.on_episode_start()
            logger.start_episode(sample_hash)

            # Run episode
            total_reward = 0.0
            while not env.done:
                action = agent.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                agent.on_step(obs, action, reward, next_obs, done)
                total_reward += reward
                obs = next_obs

            # Log steps (all actions including SKIP)
            for step_data in env.step_log:
                step_data['binary_hash'] = sample_hash
                logger.log_step(ep_idx, step_data)

            # Log episode
            ep_time = time.time() - ep_start
            adv_bytes = env.get_adversarial_bytes()  # updates curr_injected before _get_info
            info = env._get_info()
            logger.log_episode(ep_idx, info, ep_time)
            if hasattr(agent, 'on_episode_end'):
                agent.on_episode_end(info)

            if env.n_mods > 0 and adv_bytes is not None:
                logger.save_modified_binary(sample_hash, adv_bytes)

            env.cleanup()

        # Track stats (common for all agent types)
        total_queries += info['query_count']
        total_mods += info['n_mods']
        deltas.append(info['total_delta'])

        if info['bypassed']:
            bypassed += 1

        # Print progress
        status = 'BYPASSED' if info['bypassed'] else 'FAILED'
        print("  [{:>4d}/{:>4d}] {} {:.6f}→{:.6f} delta={:.4f} mods={} queries={} {:.1f}s {}".format(
            ep_idx + 1, len(corpus),
            sample_hash[:12],
            info['start_score'], info['curr_score'],
            info['total_delta'], info['n_mods'],
            info['query_count'], ep_time, status
        ))

        logger.flush()

    # Summary
    asr = bypassed / total if total > 0 else 0
    avg_queries = total_queries / total if total > 0 else 0
    avg_mods = total_mods / total if total > 0 else 0
    avg_delta = np.mean(deltas) if deltas else 0

    summary = {
        'agent': agent.name,
        'total_attackable': total,
        'bypassed': bypassed,
        'asr': asr,
        'skipped_benign': skipped_benign,
        'skipped_no_calls': skipped_no_calls,
        'failed': failed,
        'avg_queries': avg_queries,
        'avg_mods': avg_mods,
        'avg_delta': avg_delta,
        'max_delta': max(deltas) if deltas else 0,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate an agent on malware corpus')
    parser.add_argument('--agent', type=str, default='random',
                        choices=['random', 'hillclimb', 'pointer_random', 'pointer_hillclimb',
                                 'dqn', 'pdqn', 'mcts'],
                        help='Agent to evaluate')
    parser.add_argument('--corpus', type=str, default='folder',
                        choices=['val', 'train', 'test', 'all', 'folder'],
                        help='Which samples to use (folder = every file under paths.malware_dir)')
    parser.add_argument('--samples', type=str, nargs='+', default=None,
                        help='Specific sample hashes to evaluate (overrides --corpus)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples to evaluate')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Tag for log files (default: agent_corpus)')
    parser.add_argument('--modify_prob', type=float, default=None,
                        help='Modify probability for random/hillclimb agent')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='DQN checkpoint path (required for --agent dqn)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume a previous run — skip already-processed binaries in episodes.csv')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Override log_dir from config (useful for resuming runs in old directories)')
    parser.add_argument('--budget', type=int, default=None,
                        help='Hard cap on scorer calls per binary (query budget)')
    parser.add_argument('--no_rollback', action='store_true',
                        help='Disable env-level rollback during evaluation (pure policy)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Parallel workers for pointer-agent eval (default 1 = serial)')
    parser.add_argument('--mcts_no_budget', action='store_true',
                        help='MCTS: remove query budget (use original unlimited params)')
    parser.add_argument('--config', type=str,
                        default='configs/rqs/config_rq1_malconv_no_header.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Run tag — clean naming, auto-increment if directory exists (unless resuming)
    if args.log_dir:
        config['paths']['log_dir'] = args.log_dir
    base_tag = args.run_tag or '{}_{}'.format(args.agent, args.corpus)
    run_tag = base_tag
    run_dir = os.path.join(config['paths']['log_dir'], run_tag)

    # Resume: reuse existing run directory, skip already-processed binaries
    done_hashes = set()
    if args.resume:
        episodes_path = os.path.join(run_dir, 'episodes.csv')
        if os.path.exists(episodes_path):
            import csv
            with open(episodes_path, 'r') as f:
                for row in csv.DictReader(f):
                    done_hashes.add(row['binary_hash'])
            print("RESUME: found {} already-processed binaries in {}".format(
                len(done_hashes), episodes_path))
        else:
            print("RESUME: no episodes.csv found, starting fresh")
    elif os.path.exists(run_dir):
        run_num = 2
        while os.path.exists(os.path.join(config['paths']['log_dir'],
                                           '{}_{}'.format(base_tag, run_num))):
            run_num += 1
        run_tag = '{}_{}'.format(base_tag, run_num)
        print("NOTE: '{}' exists. Using '{}' instead.".format(base_tag, run_tag))

    print("=" * 70)
    print("EVALUATE: agent={}, corpus={}, run_tag={}".format(
        args.agent,
        args.corpus if not args.samples else 'custom', run_tag))
    print("=" * 70)

    # Build components
    print("Loading model...")
    allow_rollback = args.agent not in ('random', 'pointer_random') and not getattr(args, 'no_rollback', False)
    scorer, env = _build_env(
        config,
        allow_rollback=allow_rollback,
        agent_name=args.agent,
        query_budget=args.budget,
    )
    agent = build_agent(args.agent, config, args, scorer=scorer)

    # Get corpus
    corpus = get_corpus(config, args.corpus, args.samples)
    print("Corpus: {} samples".format(len(corpus)))

    # Logger (creates per-binary folders inside logs/run_tag/)
    logger = Logger(
        log_dir=config['paths']['log_dir'],
        run_tag=run_tag,
        log_steps=config['logging']['step_log'],
        log_episodes=config['logging']['episode_log']
    )

    print("Threshold: {}".format(config['env']['threshold']))
    print("Max modifications: {}".format(config['env']['max_modifications']))
    print()

    # Build eval_params for parallel worker
    _eval_params = {
        'allow_rollback': allow_rollback,
        'query_budget':   args.budget,
        'agent_name':     args.agent,
    }

    # Run evaluation
    summary = evaluate(agent, env, corpus, logger, config,
                       max_samples=args.max_samples, done_hashes=done_hashes,
                       num_workers=args.num_workers, eval_params=_eval_params)

    # Print summary
    print()
    print("=" * 70)
    print("RESULTS: {}".format(run_tag))
    print("=" * 70)
    print("Attackable samples: {}".format(summary['total_attackable']))
    print("Bypassed:           {} ({:.1f}% ASR)".format(
        summary['bypassed'], summary['asr'] * 100))
    print("Skipped (benign):   {}".format(summary['skipped_benign']))
    print("Skipped (no calls): {}".format(summary['skipped_no_calls']))
    print("Failed (errors):    {}".format(summary['failed']))
    print("Avg queries/binary: {:.1f}".format(summary['avg_queries']))
    print("Avg mods/binary:    {:.1f}".format(summary['avg_mods']))
    print("Avg score delta:    {:.4f}".format(summary['avg_delta']))
    print("Max score delta:    {:.4f}".format(summary['max_delta']))
    print()
    log_path = os.path.join(config['paths']['log_dir'], run_tag)
    print("Logs: {}".format(log_path))
    print("  episodes.csv + per-binary folders with steps.csv + modified.exe")

    logger.close()


if __name__ == '__main__':
    main()
