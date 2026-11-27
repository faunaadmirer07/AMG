"""
Pointer-DQN Training Loop.

Off-policy DQN where the agent selects WHICH BB to modify at each step.
No STOP action — episodes end naturally (bypass / budget / max_steps).

Usage:
    python3 train_pointer_dqn.py --config configs/config_pointer_dqn_v1.yaml --run_tag pdqn_v1
    python3 train_pointer_dqn.py --run_tag pdqn_v1 --eval_on_test
    python3 train_pointer_dqn.py --resume logs/pdqn_v1/checkpoints/best.pt --run_tag pdqn_v1
"""
import os
import sys
import csv
import time
import random
import heapq
import shutil
import argparse
import yaml
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier.scorer import Scorer as DetectionHandler
try:
    from core.patcher import Patcher
    from core.pointer_dqn_env import PointerDQNEnv
except ImportError:
    raise SystemExit(
        "Binary-modification code is not included in this public release "
        "(see README.md)."
    )
from agents.pointer_dqn_agent import PointerDQNAgent
from utils.logger import Logger


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def get_train_val_split(config):
    malware_dir  = config['paths']['malware_dir']
    train_corpus = config['paths'].get('train_hashes')
    val_corpus   = config['paths'].get('val_hashes')
    test_corpus  = config['paths'].get('test_hashes')

    if train_corpus and val_corpus \
            and os.path.exists(train_corpus) and os.path.exists(val_corpus):
        with open(train_corpus) as f:
            train_hashes = [l.strip() for l in f if l.strip()]
        with open(val_corpus) as f:
            val_hashes = [l.strip() for l in f if l.strip()]
        test_hashes = []
        if test_corpus and os.path.exists(test_corpus):
            with open(test_corpus) as f:
                test_hashes = [l.strip() for l in f if l.strip()]
        print("Filtered corpus: {} train, {} val, {} test".format(
            len(train_hashes), len(val_hashes), len(test_hashes)))
    else:
        corpus_path = config['paths']['corpus_hashes']
        with open(corpus_path) as f:
            all_hashes = sorted([l.strip() for l in f if l.strip()])
        seed     = config['training']['seed']
        val_frac = config['training']['val_frac']
        rng      = random.Random(seed)
        indices  = list(range(len(all_hashes)))
        rng.shuffle(indices)
        val_size    = int(len(indices) * val_frac)
        val_indices = set(indices[:val_size])
        train_hashes = [all_hashes[i] for i in range(len(all_hashes)) if i not in val_indices]
        val_hashes   = [all_hashes[i] for i in range(len(all_hashes)) if i in val_indices]
        test_hashes  = []
        print("Full corpus split: {} train, {} val".format(len(train_hashes), len(val_hashes)))

    train_paths = [(h, os.path.join(malware_dir, h)) for h in train_hashes]
    val_paths   = [(h, os.path.join(malware_dir, h)) for h in val_hashes]
    test_paths  = [(h, os.path.join(malware_dir, h)) for h in test_hashes]
    return train_paths, val_paths, test_paths


# ------------------------------------------------------------------
# Episode runner
# ------------------------------------------------------------------

def run_episode(env, agent, pe_path, training=True, logger=None, episode_num=0,
                bb_log_writer=None):
    """
    Run one full episode.

    Returns:
        info        : dict or None on failure
        total_reward: float
        wall_time   : float
    """
    ep_start = time.time()

    try:
        bb_feats, context = env.reset(pe_path)
    except Exception as e:
        print("[WARN] reset failed ({}): {}".format(type(e).__name__, e), flush=True)
        return None, 0.0, 0.0, True   # is_error=True

    if env.done or len(env.remaining_bbs) == 0:
        env.cleanup()
        return None, 0.0, 0.0, False  # not an error — benign/no-calls/below-threshold

    if logger:
        logger.start_episode(env.binary_hash)

    total_reward = 0.0

    while not env.done:
        action = agent.select_action(bb_feats, context, training=training)

        (next_bb_feats, next_ctx), reward, done, info = env.step(action)

        if training:
            agent.on_step(bb_feats, context, action, reward, next_bb_feats, next_ctx, done)

        total_reward += reward
        bb_feats, context = next_bb_feats, next_ctx

    ep_time = time.time() - ep_start
    info    = env._get_info()

    if bb_log_writer is not None:
        for entry in env.step_log:
            bb_log_writer.writerow([
                episode_num,
                env.binary_hash,
                entry['step'],
                entry['bb_index'],
                entry['bb_start'],
                entry['action'],
                '{:.4f}'.format(entry['score_before']),
                '{:.4f}'.format(entry['score_after']),
                '{:.4f}'.format(entry['reward']),
            ])

    if logger:
        logger.log_episode(episode_num, info, ep_time)

    env.cleanup()
    return info, total_reward, ep_time, False  # is_error=False


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_agent(agent, env, eval_paths, max_samples=300, logger=None):
    agent.set_eval_mode()
    prev_max_bb = env.max_bb
    env.max_bb  = None  # full BB set for fair eval (no training cap)
    sample   = random.sample(eval_paths, min(max_samples, len(eval_paths)))
    bypassed = 0
    total    = 0
    deltas   = []

    for ep_idx, (h, path) in enumerate(sample):
        info, _, _, _ = run_episode(env, agent, path, training=False,
                                    logger=logger, episode_num=ep_idx)
        if info is None:
            continue
        total += 1
        deltas.append(info['total_delta'])
        if info['bypassed']:
            bypassed += 1

    env.max_bb = prev_max_bb
    agent.set_train_mode()
    return {
        'asr':       bypassed / max(total, 1),
        'bypassed':  bypassed,
        'n_tested':  total,
        'avg_delta': float(np.mean(deltas)) if deltas else 0.0,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train Pointer-DQN agent')
    parser.add_argument('--run_tag',      type=str, default='pdqn_train')
    parser.add_argument('--max_episodes', type=int, default=None)
    parser.add_argument('--resume',       type=str, default=None)
    parser.add_argument('--eval_samples', type=int, default=300)
    parser.add_argument('--top_k',        type=int, default=3,
                        help='Keep top-K checkpoints by val ASR')
    parser.add_argument('--eval_on_test', action='store_true')
    parser.add_argument('--config',       type=str,
                        default='configs/config_pointer_dqn_v1.yaml')
    parser.add_argument('--seed',         type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    max_episodes = args.max_episodes if args.max_episodes is not None else config['training']['max_episodes']
    eval_freq    = config['training']['eval_frequency']
    ckpt_freq    = config['training']['checkpoint_frequency']

    run_tag  = args.run_tag
    log_dir  = config['paths']['log_dir']
    ckpt_dir = os.path.join(log_dir, run_tag, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 70)
    print("TRAIN POINTER-DQN: run_tag={}, max_episodes={}".format(run_tag, max_episodes))
    print("=" * 70)

    scorer_type = config.get('scorer', {}).get('type', 'malconv')
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
    patcher = Patcher(
        selected_semnop_num=config['env']['selected_semnop_num'],
        max_random_number=config['env']['max_random_number'],
    )
    env = PointerDQNEnv(scorer, patcher, config)

    top_checkpoints = []  # min-heap of (asr, episode, path)

    if args.resume:
        print("Resuming from: {}".format(args.resume))
        agent        = PointerDQNAgent.load(args.resume, config)
        start_ep     = getattr(agent, 'resume_episode', 0)
        best_val_asr = getattr(agent, 'resume_best_val_asr', 0.0)
        agent.set_train_mode()
    else:
        agent        = PointerDQNAgent(config)
        start_ep     = 0
        best_val_asr = 0.0

    print("Device:  {}".format(agent.device))
    print("Network: bb_dim={}, ctx_dim={}, hidden={}".format(
        agent.bb_dim, agent.ctx_dim, agent.hidden))
    print("Epsilon decay: {} steps".format(agent.epsilon_decay_steps))

    train_paths, val_paths, test_paths = get_train_val_split(config)
    eval_paths = test_paths if args.eval_on_test else val_paths
    eval_label = 'test' if args.eval_on_test else 'val'
    print("Eval corpus: {} ({} samples every {} eps)".format(
        eval_label, args.eval_samples, eval_freq))
    print()

    logger = Logger(
        log_dir=log_dir,
        run_tag=run_tag,
        log_steps=False,
        log_episodes=True,
    )

    bb_log_path   = os.path.join(log_dir, run_tag, 'bb_selections.csv')
    bb_log_f      = open(bb_log_path, 'w', newline='')
    bb_log_writer = csv.writer(bb_log_f)
    bb_log_writer.writerow(['episode', 'binary_hash', 'step', 'bb_index',
                             'bb_start', 'action', 'score_before', 'score_after', 'reward'])

    episode_rewards  = []
    episode_deltas   = []
    episode_bypassed = 0
    consecutive_failures = 0

    for ep in range(start_ep, max_episodes):
        _, pe_path = random.choice(train_paths)

        info, total_reward, ep_time, is_error = run_episode(
            env, agent, pe_path, training=True,
            logger=logger, episode_num=ep,
            bb_log_writer=bb_log_writer,
        )

        if info is None:
            if is_error:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    print("FATAL: 10 consecutive real errors — aborting.", flush=True)
                    sys.exit(1)
            continue
        consecutive_failures = 0

        episode_rewards.append(total_reward)
        episode_deltas.append(info['total_delta'])
        if info['bypassed']:
            episode_bypassed += 1

        if (ep + 1) % 10 == 0:
            buf_counts = agent.replay_buffer.counts
            total_buf  = sum(buf_counts.values()) or 1
            print(
                "Ep {:>5d} | reward={:>7.2f} | delta={:.4f} | loss={:.4f} | eps={:.3f} | "
                "bypass={}({:.0f}%) pos={}({:.0f}%) rb={}({:.0f}%) | "
                "train_bypass={}".format(
                    ep + 1,
                    float(np.mean(episode_rewards[-10:])),
                    float(np.mean(episode_deltas[-10:])),
                    agent.get_avg_loss(100),
                    agent.epsilon,
                    buf_counts['bypass'],   buf_counts['bypass']   / total_buf * 100,
                    buf_counts['positive'], buf_counts['positive'] / total_buf * 100,
                    buf_counts['rollback'], buf_counts['rollback'] / total_buf * 100,
                    episode_bypassed,
                ), flush=True)

        # Periodic evaluation
        if (ep + 1) % eval_freq == 0:
            print("\n--- {} eval ({} samples) ---".format(eval_label, args.eval_samples),
                  flush=True)
            res = evaluate_agent(agent, env, eval_paths, args.eval_samples)
            print("{} ASR: {:.1f}% ({}/{}) | avg_delta: {:.4f} | eps: {:.3f}".format(
                eval_label.capitalize(),
                res['asr'] * 100, res['bypassed'], res['n_tested'],
                res['avg_delta'], agent.epsilon))

            eval_csv    = os.path.join(log_dir, run_tag, '{}_eval.csv'.format(eval_label))
            eval_exists = os.path.exists(eval_csv)
            with open(eval_csv, 'a', newline='') as f:
                w = csv.writer(f)
                if not eval_exists:
                    w.writerow(['episode', 'asr', 'bypassed', 'total', 'avg_delta',
                                'epsilon', 'buffer_size', 'avg_loss', 'timestamp'])
                w.writerow([
                    ep + 1,
                    '{:.4f}'.format(res['asr']),
                    res['bypassed'],
                    res['n_tested'],
                    '{:.6f}'.format(res['avg_delta']),
                    '{:.4f}'.format(agent.epsilon),
                    len(agent.replay_buffer),
                    '{:.6f}'.format(agent.get_avg_loss(100)),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                ])

            # Save checkpoint for this eval point and maintain top-K
            eval_ckpt = os.path.join(ckpt_dir, 'ep{:05d}.pt'.format(ep + 1))
            agent.save(eval_ckpt, ep, res['asr'])
            heapq.heappush(top_checkpoints, (res['asr'], ep + 1, eval_ckpt))
            if len(top_checkpoints) > args.top_k:
                _, _, evicted = heapq.heappop(top_checkpoints)  # remove worst
                if os.path.exists(evicted):
                    os.remove(evicted)

            # Update best.pt if this is the new top-1
            top1_asr, top1_ep, top1_path = max(top_checkpoints, key=lambda x: x[0])
            if top1_asr > best_val_asr:
                best_val_asr = top1_asr
                shutil.copy2(top1_path, os.path.join(ckpt_dir, 'best.pt'))
                print("New best! ep{} ASR={:.1f}% (top-{} pool)".format(
                    top1_ep, best_val_asr * 100, args.top_k))
            print()

        # Periodic checkpoint
        if (ep + 1) % ckpt_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, 'ep{}.pt'.format(ep + 1))
            agent.save(ckpt_path, ep, best_val_asr)
            print("Checkpoint: ep{}".format(ep + 1))

        logger.flush()

    agent.save(os.path.join(ckpt_dir, 'final.pt'), max_episodes, best_val_asr)
    bb_log_f.close()

    # Full val eval on all top-K checkpoints to pick definitive best
    print("\n" + "=" * 70)
    print("POST-TRAINING: full val eval on top-{} checkpoints ({} samples)".format(
        args.top_k, len(eval_paths)))
    final_best_asr  = 0.0
    final_best_path = None
    for ckpt_asr, ckpt_ep, ckpt_path in sorted(top_checkpoints, reverse=True):
        if not os.path.exists(ckpt_path):
            continue
        loaded = PointerDQNAgent.load(ckpt_path, config)
        loaded.set_eval_mode()
        res = evaluate_agent(loaded, env, eval_paths,
                             max_samples=len(eval_paths), logger=logger)
        print("  ep{:05d} (train ASR={:.1f}%) → full {} ASR={:.1f}% ({}/{})".format(
            ckpt_ep, ckpt_asr * 100, eval_label,
            res['asr'] * 100, res['bypassed'], res['n_tested']))
        if res['asr'] > final_best_asr:
            final_best_asr  = res['asr']
            final_best_path = ckpt_path

    if final_best_path:
        shutil.copy2(final_best_path, os.path.join(ckpt_dir, 'best.pt'))
        print("Definitive best: {} → ASR={:.1f}%".format(
            os.path.basename(final_best_path), final_best_asr * 100))

    print("Logs: {}".format(os.path.join(log_dir, run_tag)))
    logger.close()


if __name__ == '__main__':
    main()
