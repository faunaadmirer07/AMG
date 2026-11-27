"""
Pointer-DQN Replay Buffer — 3-queue balanced buffer for variable-size BB sets.

Each transition stores:
    (bb_feats [N, BB_DIM], context [9], action int, reward float,
     next_bb_feats [N', BB_DIM], next_context [9], done bool)

N varies per step (remaining BBs shrink as BBs are popped).
BB capping is handled at the environment level (PointerDQNEnv), so bb_feats
stored here are already bounded by max_bb_train during training.
Padding to batch-max N happens at sample time.

Queues:
    bypass_buf   — reward >= bypass_threshold  (bypass step)
    positive_buf — 0 < reward < threshold      (score improved)
    rollback_buf — reward <= 0                 (rollback, zero_delta, skip_budget)
"""
import random
import numpy as np
from collections import deque


class PointerReplayBuffer:

    def __init__(self, bypass_capacity=1000,
                 positive_capacity=6000,
                 rollback_capacity=3000,
                 bypass_threshold=3.0,
                 bypass_frac=0.25,
                 positive_frac=0.50,
                 rollback_frac=0.10):
        self.bypass_threshold = bypass_threshold
        self.bypass_frac   = bypass_frac
        self.positive_frac = positive_frac
        self.rollback_frac = rollback_frac

        self.bypass_buf   = deque(maxlen=bypass_capacity)
        self.positive_buf = deque(maxlen=positive_capacity)
        self.rollback_buf = deque(maxlen=rollback_capacity)

    def push(self, bb_feats, ctx, action, reward, next_bb_feats, next_ctx, done):
        t = (bb_feats, ctx, int(action), float(reward), next_bb_feats, next_ctx, bool(done))
        if reward >= self.bypass_threshold:
            self.bypass_buf.append(t)
        elif reward > 0:
            self.positive_buf.append(t)
        else:
            self.rollback_buf.append(t)

    @property
    def _uniform_mode(self):
        return (self.bypass_frac == 0.0 and
                self.positive_frac == 0.0 and
                self.rollback_frac == 0.0)

    def can_sample(self, batch_size):
        if self._uniform_mode:
            return len(self) >= batch_size
        return (len(self) >= batch_size and
                (len(self.positive_buf) > 0 or len(self.bypass_buf) > 0))

    def sample(self, batch_size):
        if self._uniform_mode:
            all_buf = (list(self.bypass_buf) + list(self.positive_buf) +
                       list(self.rollback_buf))
            return self._collate(random.sample(all_buf, min(batch_size, len(all_buf))))

        n_bypass   = len(self.bypass_buf)
        n_positive = len(self.positive_buf)
        n_rollback = len(self.rollback_buf)

        want_bypass   = min(round(batch_size * self.bypass_frac),   n_bypass)
        want_positive = min(round(batch_size * self.positive_frac), n_positive)
        want_rollback = min(round(batch_size * self.rollback_frac), n_rollback)
        want_extra    = batch_size - want_bypass - want_positive - want_rollback

        # Fill remaining slots: positive first (more informative), then rollback
        if want_extra > 0:
            extra_pos      = min(want_extra, n_positive - want_positive)
            want_positive += extra_pos
            want_extra    -= extra_pos
        if want_extra > 0:
            want_rollback  = min(want_rollback + want_extra, n_rollback)

        batch = (random.sample(list(self.bypass_buf),   want_bypass)   +
                 random.sample(list(self.positive_buf),  want_positive) +
                 random.sample(list(self.rollback_buf),  want_rollback))
        random.shuffle(batch)
        return self._collate(batch)

    def _collate(self, batch):
        bb_list      = [t[0] for t in batch]
        ctxs         = np.stack([t[1] for t in batch])
        actions      = np.array([t[2] for t in batch], dtype=np.int64)
        rewards      = np.array([t[3] for t in batch], dtype=np.float32)
        next_bb_list = [t[4] for t in batch]
        next_ctxs    = np.stack([t[5] for t in batch])
        dones        = np.array([t[6] for t in batch], dtype=np.float32)

        B      = len(batch)
        bb_dim = bb_list[0].shape[1] if len(bb_list[0].shape) == 2 else 30

        max_n      = max(len(f) for f in bb_list)
        max_n_next = max(len(f) for f in next_bb_list)

        padded      = np.zeros((B, max_n,      bb_dim), dtype=np.float32)
        mask        = np.zeros((B, max_n),              dtype=bool)
        padded_next = np.zeros((B, max_n_next, bb_dim), dtype=np.float32)
        mask_next   = np.zeros((B, max_n_next),         dtype=bool)

        for i, feats in enumerate(bb_list):
            n = len(feats)
            if n > 0:
                padded[i, :n] = feats
                mask[i, :n]   = True

        for i, feats in enumerate(next_bb_list):
            n = len(feats)
            if n > 0:
                padded_next[i, :n] = feats
                mask_next[i, :n]   = True

        return {
            'bb_feats':      padded,
            'contexts':      ctxs,
            'actions':       actions,
            'rewards':       rewards,
            'next_bb_feats': padded_next,
            'next_contexts': next_ctxs,
            'dones':         dones,
            'masks':         mask,
            'next_masks':    mask_next,
        }

    def __len__(self):
        return len(self.bypass_buf) + len(self.positive_buf) + len(self.rollback_buf)

    @property
    def counts(self):
        return {
            'bypass':   len(self.bypass_buf),
            'positive': len(self.positive_buf),
            'rollback': len(self.rollback_buf),
        }
