"""
Uniform replay buffer — single FIFO deque, uniform random sampling.

Used as ablation baseline against PointerReplayBuffer's 4-queue design.
Same _collate logic as PointerReplayBuffer so transitions are interchangeable.
"""
import random
import numpy as np
from collections import deque


class UniformReplayBuffer:

    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, bb_feats, ctx, action, reward, next_bb_feats, next_ctx, done):
        self.buf.append(
            (bb_feats, ctx, int(action), float(reward), next_bb_feats, next_ctx, bool(done))
        )

    def can_sample(self, batch_size):
        return len(self.buf) >= batch_size

    def sample(self, batch_size):
        batch = random.sample(list(self.buf), min(batch_size, len(self.buf)))
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
        return len(self.buf)

    @property
    def counts(self):
        n = len(self.buf)
        return {'bypass': 0, 'positive': n, 'rollback': 0}
