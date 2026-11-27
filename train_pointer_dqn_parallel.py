"""
Parallel Pointer-DQN training — N worker processes + 1 learner process.

Each worker has its own IDA instance and env. Workers collect transitions
with a local CPU copy of the Q-network and push them to the learner via
a shared queue. The learner (main process) owns the replay buffer and
GPU network, trains, and periodically broadcasts updated weights back.

Usage:
    python train_pointer_dqn_parallel.py \
        --config configs/rqs/config_rq1_malgraph_binary.yaml \
        --run_tag pdqn_malgraph_parallel \
        --num_workers 4
"""
import os
import sys
import csv
import yaml
import random
import heapq
import shutil
import time
import argparse
import numpy as np
from datetime import datetime
import multiprocessing as mp

# -----------------------------------------------------------------------
# Helpers shared with train_pointer_dqn.py
# -----------------------------------------------------------------------

def get_train_val_split(config):
    malware_dir = config['paths']['malware_dir']

    def load(path):
        out = []
        with open(path) as f:
            for line in f:
                h = line.strip()
                if not h:
                    continue
                full = os.path.join(malware_dir, h)
                if os.path.exists(full):
                    out.append((h, full))
        return out

    return (
        load(config['paths']['train_hashes']),
        load(config['paths']['val_hashes']),
        load(config['paths']['test_hashes']),
    )


def evaluate_agent(agent, env, eval_paths, max_samples=300, query_budget=None):
    prev_max_bb      = env.max_bb
    prev_qb          = getattr(env, 'query_budget', None)
    env.max_bb       = None
    env.query_budget = query_budget
    paths = eval_paths[-min(max_samples, len(eval_paths)):]
    bypassed = total = 0
    total_delta = 0.0
    total_queries = 0
    for _, pe_path in paths:
        try:
            bb_feats, context = env.reset(pe_path)
        except Exception:
            continue
        if env.done or len(env.remaining_bbs) == 0:
            env.cleanup()
            continue
        total += 1
        while not env.done:
            action = agent.select_action(bb_feats, context, training=False)
            (bb_feats, context), _, _, _ = env.step(action)
        info = env._get_info()
        env.cleanup()
        if info.get('bypassed'):
            bypassed += 1
        total_delta   += info.get('total_delta', 0.0)
        total_queries += info.get('query_count', 0)
    env.max_bb       = prev_max_bb
    env.query_budget = prev_qb
    return {
        'asr':         bypassed / total if total > 0 else 0.0,
        'bypassed':    bypassed,
        'n_tested':    total,
        'avg_delta':   total_delta  / total if total > 0 else 0.0,
        'avg_queries': total_queries / total if total > 0 else 0.0,
    }


def _val_eval_worker_fn(args):
    """Subprocess worker for parallel val eval. Each process builds its own env+scorer on CUDA."""
    import torch, random as _random, numpy as _np
    config, weights_cpu, paths_chunk = args

    scorer_type = config.get('scorer', {}).get('type', 'malconv')
    allow_rollback = config.get('_eval_allow_rollback', True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tmp_dir = None
    try:
        from core.patcher import Patcher
        from core.pointer_dqn_env import PointerDQNEnv
        if scorer_type == 'malgraph':
            import tempfile
            from classifier.malgraph.server import MalGraphServer
            tmp_dir = tempfile.mkdtemp(prefix='mgbin_valeval_')
            scorer = MalGraphServer(
                device=device,
                threshold=config['env']['threshold'],
                ida_path=config['scorer'].get('ida_path'),
                tmp_dir=tmp_dir,
            )
        else:
            from classifier.scorer import Scorer as DetectionHandler
            eval_scorer_cfg = dict(config['scorer'])
            eval_scorer_cfg['device'] = device
            scorer = DetectionHandler(eval_scorer_cfg)
        patcher = Patcher(
            selected_semnop_num=config['env']['selected_semnop_num'],
            max_random_number=config['env']['max_random_number'],
        )
        query_budget = 1 + config['env'].get('max_modifications', 40)
        env = PointerDQNEnv(scorer, patcher, config, eval_mode=True,
                            allow_rollback=allow_rollback, query_budget=query_budget)

        from agents.pointer_dqn_network import PointerQNetwork
        cfg    = config['pointer_dqn']
        q_net  = PointerQNetwork(
            cfg.get('bb_dim', 30), cfg.get('ctx_dim', 9),
            cfg.get('hidden_dim', 256), cfg.get('use_dueling', True),
        )
        q_net.load_state_dict(weights_cpu)
        q_net.eval()

        env.max_bb = None
        bypassed = total = 0
        total_delta = 0.0
        total_queries = 0

        for _, pe_path in paths_chunk:
            try:
                bb_feats, context = env.reset(pe_path)
            except Exception:
                continue
            if env.done or len(env.remaining_bbs) == 0:
                env.cleanup()
                continue
            total += 1
            while not env.done:
                with torch.no_grad():
                    bb_t  = torch.FloatTensor(bb_feats).unsqueeze(0)
                    ctx_t = torch.FloatTensor(context).unsqueeze(0)
                    mask  = torch.ones(1, bb_t.shape[1], dtype=torch.bool)
                    q     = q_net(bb_t, ctx_t, mask)
                    action = int(q[0].argmax().item())
                (bb_feats, context), _, _, _ = env.step(action)
            info = env._get_info()
            env.cleanup()
            if info.get('bypassed'):
                bypassed += 1
            total_delta   += info.get('total_delta', 0.0)
            total_queries += info.get('query_count', 0)

        return bypassed, total, total_delta, total_queries
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def evaluate_agent_parallel(agent, config, eval_paths, max_samples, n_workers, allow_rollback=True):
    """Parallel val eval: split paths across n_workers subprocesses, each with CUDA scorer."""
    import torch
    paths = eval_paths[-min(max_samples, len(eval_paths)):]
    chunk_size = max(1, (len(paths) + n_workers - 1) // n_workers)
    chunks = [paths[i:i + chunk_size] for i in range(0, len(paths), chunk_size)]

    weights_cpu = {k: v.cpu() for k, v in agent.q_network.state_dict().items()}
    eval_config = {**config, '_eval_allow_rollback': allow_rollback}
    task_args   = [(eval_config, weights_cpu, chunk) for chunk in chunks]

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=len(chunks)) as pool:
        results = pool.map(_val_eval_worker_fn, task_args)

    total_bypass  = sum(r[0] for r in results)
    total_n       = sum(r[1] for r in results)
    total_delta   = sum(r[2] for r in results)
    total_queries = sum(r[3] for r in results)
    return {
        'asr':         total_bypass  / total_n if total_n > 0 else 0.0,
        'bypassed':    total_bypass,
        'n_tested':    total_n,
        'avg_delta':   total_delta   / total_n if total_n > 0 else 0.0,
        'avg_queries': total_queries / total_n if total_n > 0 else 0.0,
    }


# -----------------------------------------------------------------------
# Worker process
# -----------------------------------------------------------------------

def _worker(worker_id, config, pe_paths, transition_q, log_q, weight_q, stop_event):
    """
    Collects experience with a local CPU copy of the Q-network.
    Pushes (bb_feats, ctx, action, reward, next_bb_feats, next_ctx, done)
    tuples to transition_q and episode summaries to log_q.
    """
    import random, time, tempfile
    import numpy as np
    import torch

    random.seed(worker_id * 7919 + 1)
    np.random.seed(worker_id * 7919 + 1)

    # Build env
    scorer_type = config.get('scorer', {}).get('type', 'malconv')
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.patcher import Patcher
    from core.pointer_dqn_env import PointerDQNEnv
    if scorer_type == 'malgraph':
        from classifier.malgraph.server import MalGraphServer
        tmp_dir = tempfile.mkdtemp(prefix='mgbin_w{}_{}_'.format(worker_id, os.getpid()))
        scorer = MalGraphServer(
            device='cpu',
            threshold=config['env']['threshold'],
            ida_path=config['scorer'].get('ida_path'),
            tmp_dir=tmp_dir,
        )
    else:
        from classifier.scorer import Scorer as DetectionHandler
        tmp_dir = None
        scorer = DetectionHandler(config['scorer'])
    patcher = Patcher(
        selected_semnop_num=config['env']['selected_semnop_num'],
        max_random_number=config['env']['max_random_number'],
    )
    env = PointerDQNEnv(scorer, patcher, config)

    # Local CPU Q-network (action selection only, no gradients)
    from agents.pointer_dqn_network import PointerQNetwork
    cfg    = config['pointer_dqn']
    bb_dim = cfg.get('bb_dim', 30)
    ctx_dim = cfg.get('ctx_dim', 9)
    hidden  = cfg.get('hidden_dim', 256)
    q_net   = PointerQNetwork(bb_dim, ctx_dim, hidden, cfg.get('use_dueling', True))
    q_net.eval()

    epsilon = cfg.get('epsilon_start', 1.0)
    consecutive_errors = 0

    while not stop_event.is_set():
        # Non-blocking weight sync from learner
        try:
            msg = weight_q.get_nowait()
            q_net.load_state_dict(msg['weights'])
            epsilon = msg['epsilon']
        except Exception:
            pass

        _, pe_path = random.choice(pe_paths)
        ep_start = time.time()

        try:
            bb_feats, context = env.reset(pe_path)
        except Exception as e:
            consecutive_errors += 1
            print('[worker {}] reset error ({}/10): {}'.format(worker_id, consecutive_errors, e), flush=True)
            if consecutive_errors >= 10:
                print('[worker {}] 10 consecutive errors, exiting'.format(worker_id), flush=True)
                break
            continue
        consecutive_errors = 0

        if env.done or len(env.remaining_bbs) == 0:
            env.cleanup()
            continue

        transitions = []
        quartile_frac = config.get('pointer_dqn', {}).get('quartile_explore_frac', 0.0)
        total_bbs = max(len(env._all_bbs), 1)

        while not env.done:
            N = len(bb_feats)
            if N == 0:
                action = 0
            elif random.random() < epsilon:
                # Quartile-stratified exploration: force coverage of all positional ranges.
                # With prob quartile_frac, pick a random quartile then sample uniformly within it.
                visible = env._visible if env._visible is not None else env.remaining_bbs
                if quartile_frac > 0 and random.random() < quartile_frac and len(visible) > 0:
                    q_idx = random.randint(0, 3)
                    lo, hi = q_idx * 0.25, (q_idx + 1) * 0.25
                    cands = [i for i, bb_idx in enumerate(visible)
                             if lo <= bb_idx / total_bbs < hi]
                    action = random.choice(cands) if cands else random.randint(0, N - 1)
                else:
                    action = random.randint(0, N - 1)
            else:
                bb_t  = torch.FloatTensor(bb_feats).unsqueeze(0)
                ctx_t = torch.FloatTensor(context).unsqueeze(0)
                mask  = torch.ones(1, N, dtype=torch.bool)
                with torch.no_grad():
                    q = q_net(bb_t, ctx_t, mask)
                action = int(q[0].argmax().item())

            (next_bb, next_ctx), reward, done, _ = env.step(action)
            transitions.append((bb_feats, context, action, reward, next_bb, next_ctx, done))
            bb_feats, context = next_bb, next_ctx

        info     = env._get_info()
        ep_time  = time.time() - ep_start
        env.cleanup()

        # Push transitions (drop if queue is full to avoid blocking)
        for t in transitions:
            try:
                transition_q.put(t, timeout=1.0)
            except Exception:
                break

        # Push episode summary
        try:
            log_q.put_nowait({
                'worker_id':    worker_id,
                'binary_hash':  env.binary_hash,
                'bypassed':     info.get('bypassed', False),
                'total_delta':  info.get('total_delta', 0.0),
                'n_mods':       info.get('n_mods', 0),
                'queries':      info.get('queries', 0),
                'wall_time_sec': ep_time,
                'start_score':  info.get('start_score', 1.0),
                'final_score':  info.get('final_score', 1.0),
            })
        except Exception:
            pass

    if tmp_dir:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


# -----------------------------------------------------------------------
# Learner (main process)
# -----------------------------------------------------------------------

def main():
    try:
        from core.patcher import Patcher  # noqa: F401
        from core.pointer_dqn_env import PointerDQNEnv  # noqa: F401
    except ImportError:
        raise SystemExit(
            "Binary-modification code is not included in this public release "
            "(see README.md)."
        )

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',       type=str, required=True)
    parser.add_argument('--run_tag',      type=str, default='pdqn_parallel')
    parser.add_argument('--num_workers',  type=int, default=4)
    parser.add_argument('--max_episodes', type=int, default=None)
    parser.add_argument('--resume',       type=str, default=None)
    parser.add_argument('--eval_samples', type=int, default=300)
    parser.add_argument('--top_k',        type=int, default=3)
    parser.add_argument('--weight_sync_freq', type=int, default=5,
                        help='Broadcast weights every N episodes received')
    parser.add_argument('--seed',         type=int, default=42)
    parser.add_argument('--num_eval_workers', type=int, default=2,
                        help='Parallel workers for val eval (1=serial, >1=parallel with CUDA)')
    parser.add_argument('--eval_first', action='store_true',
                        help='Run val eval once before training starts (useful after resume)')
    parser.add_argument('--reset_best_asr', action='store_true',
                        help='Reset best_val_asr to 0 (use when val set changes)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    max_episodes = args.max_episodes or config['training']['max_episodes']
    eval_freq    = config['training']['eval_frequency']
    ckpt_freq    = config['training']['checkpoint_frequency']
    log_dir      = config['paths']['log_dir']
    ckpt_dir     = os.path.join(log_dir, args.run_tag, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print('=' * 70)
    print('PARALLEL POINTER-DQN: run_tag={}, workers={}, max_episodes={}'.format(
        args.run_tag, args.num_workers, max_episodes))
    print('=' * 70)

    # Learner env (for val eval)
    from agents.pointer_dqn_agent import PointerDQNAgent

    scorer_type = config.get('scorer', {}).get('type', 'malconv')
    from core.patcher import Patcher
    from core.pointer_dqn_env import PointerDQNEnv
    if scorer_type == 'malgraph':
        from classifier.malgraph.server import MalGraphServer
        scorer = MalGraphServer(
            device=config['scorer'].get('device', 'cpu'),
            threshold=config['env']['threshold'],
            ida_path=config['scorer'].get('ida_path'),
        )
    else:
        from classifier.scorer import Scorer as DetectionHandler
        scorer = DetectionHandler(config['scorer'])
    patcher = Patcher(
        selected_semnop_num=config['env']['selected_semnop_num'],
        max_random_number=config['env']['max_random_number'],
    )
    eval_env = PointerDQNEnv(scorer, patcher, config, eval_mode=True)

    train_paths, val_paths, _ = get_train_val_split(config)
    print('Train: {}  Val: {}'.format(len(train_paths), len(val_paths)))

    # Learner agent
    if args.resume:
        print('Resuming from: {}'.format(args.resume))
        agent        = PointerDQNAgent.load(args.resume, config)
        start_ep     = getattr(agent, 'resume_episode', 0)
        best_val_asr = 0.0 if args.reset_best_asr else getattr(agent, 'resume_best_val_asr', 0.0)
        if args.reset_best_asr:
            print('best_val_asr reset to 0.0 (new val set)')
    else:
        agent        = PointerDQNAgent(config)
        start_ep     = 0
        best_val_asr = 0.0
    agent.set_train_mode()
    print('Device: {}  epsilon: {:.3f}'.format(agent.device, agent.epsilon))

    # Queues
    transition_q = mp.Queue(maxsize=50000)
    log_q        = mp.Queue(maxsize=10000)
    weight_qs    = [mp.Queue(maxsize=1) for _ in range(args.num_workers)]
    stop_event   = mp.Event()

    # Broadcast initial weights to workers
    def _broadcast_weights():
        msg = {
            'weights': {k: v.cpu() for k, v in agent.q_network.state_dict().items()},
            'epsilon': agent.epsilon,
        }
        for wq in weight_qs:
            try:
                # Replace stale weight message if present
                try:
                    wq.get_nowait()
                except Exception:
                    pass
                wq.put_nowait(msg)
            except Exception:
                pass

    # Start workers
    workers = []
    for i in range(args.num_workers):
        p = mp.Process(
            target=_worker,
            args=(i, config, train_paths, transition_q, log_q, weight_qs[i], stop_event),
            daemon=True,
        )
        p.start()
        workers.append(p)
        print('Worker {} started (PID {})'.format(i, p.pid))

    _broadcast_weights()

    # Episode CSV
    ep_csv_path = os.path.join(log_dir, args.run_tag, 'episodes.csv')
    os.makedirs(os.path.dirname(ep_csv_path), exist_ok=True)
    ep_csv_f = open(ep_csv_path, 'w', newline='')
    ep_writer = csv.writer(ep_csv_f)
    ep_writer.writerow([
        'episode', 'worker_id', 'binary_hash', 'start_score', 'final_score',
        'total_delta', 'n_mods', 'queries', 'bypassed', 'wall_time_sec', 'timestamp',
    ])

    # Val eval CSV
    val_csv_path = os.path.join(log_dir, args.run_tag, 'val_eval.csv')
    val_csv_f = open(val_csv_path, 'w', newline='')
    val_writer = csv.writer(val_csv_f)
    val_writer.writerow([
        'episode', 'asr', 'bypassed', 'total', 'avg_delta', 'avg_queries',
        'epsilon', 'buffer_size', 'avg_loss', 'timestamp',
    ])

    top_checkpoints  = []   # min-heap (asr, ep, path)
    total_ep         = start_ep
    ep_bypassed      = 0
    ep_since_print   = 0
    ep_since_sync    = 0
    ep_since_eval    = 0
    ep_since_ckpt    = 0

    # --- eval_first: run val eval once before training ---
    if args.eval_first:
        print('\n--- eval_first: val eval ({} samples, {} workers) ---'.format(
            args.eval_samples, args.num_eval_workers), flush=True)
        eval_query_budget = 1 + config['env'].get('max_modifications', 40)
        agent.set_eval_mode()
        if args.num_eval_workers > 1:
            res = evaluate_agent_parallel(agent, config, val_paths,
                                          args.eval_samples, args.num_eval_workers,
                                          allow_rollback=False)
        else:
            res = evaluate_agent(agent, eval_env, val_paths, args.eval_samples,
                                 query_budget=eval_query_budget)
        agent.set_train_mode()
        print('Val ASR: {:.1f}% ({}/{}) | avg_delta: {:.4f} | avg_queries: {:.1f} | eps: {:.3f}'.format(
            res['asr'] * 100, res['bypassed'], res['n_tested'],
            res['avg_delta'], res.get('avg_queries', 0.0), agent.epsilon), flush=True)
        val_writer.writerow([
            total_ep, '{:.4f}'.format(res['asr']), res['bypassed'], res['n_tested'],
            '{:.6f}'.format(res['avg_delta']), '{:.2f}'.format(res.get('avg_queries', 0.0)),
            '{:.4f}'.format(agent.epsilon), len(agent.replay_buffer),
            '{:.6f}'.format(agent.get_avg_loss(100)),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        ])
        val_csv_f.flush()
        ep_since_eval = 0

    print('\nLearner running. Waiting for first transitions...', flush=True)

    try:
        while total_ep < max_episodes:
            # --- Drain log queue (count completed episodes) ---
            newly_done = 0
            while not log_q.empty():
                try:
                    info = log_q.get_nowait()
                except Exception:
                    break
                total_ep   += 1
                newly_done += 1
                ep_since_print += 1
                ep_since_sync  += 1
                ep_since_eval  += 1
                ep_since_ckpt  += 1
                if info.get('bypassed'):
                    ep_bypassed += 1
                ep_writer.writerow([
                    total_ep,
                    info['worker_id'],
                    info['binary_hash'],
                    '{:.6f}'.format(info.get('start_score', 1.0)),
                    '{:.6f}'.format(info.get('final_score', 1.0)),
                    '{:.6f}'.format(info.get('total_delta', 0.0)),
                    info.get('n_mods', 0),
                    info.get('queries', 0),
                    info.get('bypassed', False),
                    '{:.2f}'.format(info.get('wall_time_sec', 0.0)),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                ])
                ep_csv_f.flush()

            # --- Drain transition queue and train ---
            # No per-iteration cap: drain everything so the queue never
            # fills to maxsize and workers don't drop bypass transitions.
            drained = 0
            while not transition_q.empty():
                try:
                    t = transition_q.get_nowait()
                except Exception:
                    break
                bb_feats, ctx, action, reward, next_bb, next_ctx, done = t
                agent.replay_buffer.push(bb_feats, ctx, action, reward, next_bb, next_ctx, done)
                agent.total_steps += 1
                agent._decay_epsilon()

                if (len(agent.replay_buffer) >= agent.warmup_steps and
                        agent.replay_buffer.can_sample(agent.batch_size)):
                    for _ in range(agent.updates_per_step):
                        loss = agent._learn()
                        agent.training_losses.append(loss)

                if agent.total_steps % agent.target_update_freq == 0:
                    agent.target_network.load_state_dict(agent.q_network.state_dict())

                drained += 1

            # --- Print progress ---
            if ep_since_print >= 10 and newly_done > 0:
                buf = agent.replay_buffer.counts
                total_buf = sum(buf.values()) or 1
                print(
                    'Ep {:>5d} | loss={:.4f} | eps={:.3f} | '
                    'bypass={}({:.0f}%) pos={}({:.0f}%) rb={}({:.0f}%) | '
                    'bypassed={} | q_trans={} q_log={}'.format(
                        total_ep,
                        agent.get_avg_loss(100),
                        agent.epsilon,
                        buf['bypass'],   buf['bypass']   / total_buf * 100,
                        buf['positive'], buf['positive'] / total_buf * 100,
                        buf['rollback'], buf['rollback'] / total_buf * 100,
                        ep_bypassed,
                        transition_q.qsize(),
                        log_q.qsize(),
                    ), flush=True)
                ep_since_print = 0

            # --- Broadcast weights ---
            if ep_since_sync >= args.weight_sync_freq and newly_done > 0:
                _broadcast_weights()
                ep_since_sync = 0

            # --- Val eval ---
            if ep_since_eval >= eval_freq and newly_done > 0:
                print('\n--- val eval ({} samples, {} workers) ---'.format(
                    args.eval_samples, args.num_eval_workers), flush=True)
                eval_query_budget = 1 + config['env'].get('max_modifications', 40)
                agent.set_eval_mode()
                if args.num_eval_workers > 1:
                    res = evaluate_agent_parallel(agent, config, val_paths,
                                                  args.eval_samples, args.num_eval_workers)
                else:
                    res = evaluate_agent(agent, eval_env, val_paths, args.eval_samples,
                                         query_budget=eval_query_budget)
                agent.set_train_mode()
                print('Val ASR: {:.1f}% ({}/{}) | avg_delta: {:.4f} | avg_queries: {:.1f} | eps: {:.3f}'.format(
                    res['asr'] * 100, res['bypassed'], res['n_tested'],
                    res['avg_delta'], res.get('avg_queries', 0.0), agent.epsilon), flush=True)

                val_writer.writerow([
                    total_ep,
                    '{:.4f}'.format(res['asr']),
                    res['bypassed'],
                    res['n_tested'],
                    '{:.6f}'.format(res['avg_delta']),
                    '{:.2f}'.format(res.get('avg_queries', 0.0)),
                    '{:.4f}'.format(agent.epsilon),
                    len(agent.replay_buffer),
                    '{:.6f}'.format(agent.get_avg_loss(100)),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                ])
                val_csv_f.flush()

                eval_ckpt = os.path.join(ckpt_dir, 'ep{:05d}.pt'.format(total_ep))
                agent.save(eval_ckpt, total_ep, res['asr'])
                heapq.heappush(top_checkpoints, (res['asr'], total_ep, eval_ckpt))
                if len(top_checkpoints) > args.top_k:
                    _, _, evicted = heapq.heappop(top_checkpoints)
                    if os.path.exists(evicted):
                        os.remove(evicted)

                top1_asr, top1_ep, top1_path = max(top_checkpoints, key=lambda x: x[0])
                if top1_asr > best_val_asr:
                    best_val_asr = top1_asr
                    shutil.copy2(top1_path, os.path.join(ckpt_dir, 'best.pt'))
                    print('New best! ep{} ASR={:.1f}%'.format(top1_ep, best_val_asr * 100))
                print()
                ep_since_eval = 0

            # --- Periodic checkpoint ---
            if ep_since_ckpt >= ckpt_freq and newly_done > 0:
                ckpt_path = os.path.join(ckpt_dir, 'ep{}.pt'.format(total_ep))
                agent.save(ckpt_path, total_ep, best_val_asr)
                print('Checkpoint: ep{}'.format(total_ep), flush=True)
                ep_since_ckpt = 0

            # Avoid busy-wait when workers are slow
            if newly_done == 0 and drained == 0:
                time.sleep(0.05)

    except KeyboardInterrupt:
        print('\nInterrupted — saving final checkpoint...', flush=True)

    finally:
        stop_event.set()
        agent.save(os.path.join(ckpt_dir, 'final.pt'), total_ep, best_val_asr)
        ep_csv_f.close()
        val_csv_f.close()
        for p in workers:
            p.join(timeout=30)
        print('Done. Logs: {}'.format(os.path.join(log_dir, args.run_tag)))


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
