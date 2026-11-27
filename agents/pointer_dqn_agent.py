"""
Pointer-DQN Agent.

Off-policy DQN with Flash-Attention Q-network over variable-size BB sets.

Key design:
  - State = (bb_feats [N, 30], context [9]) — remaining eligible BBs (pop design)
  - Action = index i in [0, N-1] selecting which BB to modify
  - Tried BBs are popped; N decreases each step
  - Double DQN + dueling + 4-queue replay buffer
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.pointer_dqn_network import PointerQNetwork
from utils.pointer_replay_buffer import PointerReplayBuffer
from utils.uniform_replay_buffer import UniformReplayBuffer


class PointerDQNAgent:

    def __init__(self, config):
        cfg = config['pointer_dqn']

        self.bb_dim  = cfg.get('bb_dim', 30)
        self.ctx_dim = cfg.get('ctx_dim', 9)
        self.hidden  = cfg.get('hidden_dim', 256)
        self.gamma   = cfg.get('gamma', 0.99)

        self.batch_size         = cfg.get('batch_size', 32)
        self.target_update_freq = cfg.get('target_update_freq', 500)
        self.warmup_steps       = cfg.get('warmup_steps', 500)
        self.updates_per_step   = cfg.get('updates_per_step', 2)

        self.epsilon             = cfg.get('epsilon_start', 1.0)
        self.epsilon_start       = self.epsilon
        self.epsilon_end         = cfg.get('epsilon_end', 0.10)
        self.epsilon_decay_steps = cfg.get('epsilon_decay_steps', 50000)

        device_str = config.get('training', {}).get('device', 'cpu')
        self.device = torch.device(
            'cuda' if device_str == 'cuda' and torch.cuda.is_available() else 'cpu'
        )

        use_dueling = cfg.get('use_dueling', True)
        self.q_network      = PointerQNetwork(self.bb_dim, self.ctx_dim, self.hidden, use_dueling).to(self.device)
        self.target_network = PointerQNetwork(self.bb_dim, self.ctx_dim, self.hidden, use_dueling).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=cfg.get('learning_rate', 1e-4)
        )

        if config.get('use_uniform_buffer', False):
            total_cap = (cfg.get('bypass_buffer_capacity', 1000) +
                         cfg.get('positive_buffer_capacity', 6000) +
                         cfg.get('rollback_buffer_capacity', 3000))
            self.replay_buffer = UniformReplayBuffer(capacity=total_cap)
        else:
            self.replay_buffer = PointerReplayBuffer(
                bypass_capacity=cfg.get('bypass_buffer_capacity', 1000),
                positive_capacity=cfg.get('positive_buffer_capacity', 6000),
                rollback_capacity=cfg.get('rollback_buffer_capacity', 3000),
                bypass_threshold=cfg.get('bypass_reward_threshold', 3.0),
                bypass_frac=cfg.get('bypass_sample_frac', 0.25),
                positive_frac=cfg.get('positive_sample_frac', 0.50),
                rollback_frac=cfg.get('rollback_sample_frac', 0.10),
            )

        self.total_steps     = 0
        self.training_losses = []
        self.name            = 'pointer_dqn'

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, bb_feats, context, training=True):
        """
        Args:
            bb_feats: ndarray [N, 30]
            context:  ndarray [9]
            training: bool — if True, uses epsilon-greedy

        Returns:
            int in [0, N-1]
        """
        N = len(bb_feats)
        if N == 0:
            return 0

        if training and random.random() < self.epsilon:
            return random.randint(0, N - 1)

        bb_t  = torch.FloatTensor(bb_feats).unsqueeze(0).to(self.device)  # [1, N, 30]
        ctx_t = torch.FloatTensor(context).unsqueeze(0).to(self.device)   # [1, 9]
        mask  = torch.ones(1, N, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            q = self.q_network(bb_t, ctx_t, mask)  # [1, N]

        return int(q[0].argmax().item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def on_step(self, bb_feats, ctx, action, reward, next_bb_feats, next_ctx, done):
        self.replay_buffer.push(bb_feats, ctx, action, reward, next_bb_feats, next_ctx, done)
        self.total_steps += 1
        self._decay_epsilon()

        if (len(self.replay_buffer) >= self.warmup_steps and
                self.replay_buffer.can_sample(self.batch_size)):
            for _ in range(self.updates_per_step):
                loss = self._learn()
                self.training_losses.append(loss)

        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _decay_epsilon(self):
        steps = max(0, self.total_steps - self.warmup_steps)
        if steps < self.epsilon_decay_steps:
            frac = steps / self.epsilon_decay_steps
            self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)
        else:
            self.epsilon = self.epsilon_end

    def _learn(self):
        batch = self.replay_buffer.sample(self.batch_size)

        bb_feats      = torch.FloatTensor(batch['bb_feats']).to(self.device)
        contexts      = torch.FloatTensor(batch['contexts']).to(self.device)
        actions       = torch.LongTensor(batch['actions']).to(self.device)
        rewards       = torch.FloatTensor(batch['rewards']).to(self.device)
        next_bb_feats = torch.FloatTensor(batch['next_bb_feats']).to(self.device)
        next_contexts = torch.FloatTensor(batch['next_contexts']).to(self.device)
        dones         = torch.FloatTensor(batch['dones']).to(self.device)
        masks         = torch.BoolTensor(batch['masks']).to(self.device)
        next_masks    = torch.BoolTensor(batch['next_masks']).to(self.device)

        q_all     = self.q_network(bb_feats, contexts, masks)             # [B, N]
        current_q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)     # [B]

        with torch.no_grad():
            any_next      = next_masks.any(dim=1)
            next_q_online = self.q_network(next_bb_feats, next_contexts, next_masks)
            next_acts     = next_q_online.argmax(dim=1)
            next_q_target = self.target_network(next_bb_feats, next_contexts, next_masks)
            next_q        = next_q_target.gather(1, next_acts.unsqueeze(1)).squeeze(1)

            terminal = dones.bool() | ~any_next
            next_q   = next_q.masked_fill(terminal, 0.0)
            target_q = (rewards + self.gamma * next_q).clamp(-200.0, 200.0)

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def set_train_mode(self):
        self.q_network.train()

    def set_eval_mode(self):
        self.q_network.eval()

    def get_avg_loss(self, n=100):
        if not self.training_losses:
            return 0.0
        return float(np.mean(self.training_losses[-n:]))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path, episode_num=0, best_val_asr=0.0):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            'q_network':       self.q_network.state_dict(),
            'target_network':  self.target_network.state_dict(),
            'optimizer':       self.optimizer.state_dict(),
            'total_steps':     self.total_steps,
            'epsilon':         self.epsilon,
            'training_losses': self.training_losses[-1000:],
            'episode_num':     episode_num,
            'best_val_asr':    best_val_asr,
            'bb_dim':          self.bb_dim,
            'ctx_dim':         self.ctx_dim,
            'hidden_dim':      self.hidden,
        }
        torch.save(ckpt, path)
        print("Saved pointer_dqn agent: {}".format(path))

    @classmethod
    def load(cls, path, config):
        agent = cls(config)
        ckpt  = torch.load(path, map_location=agent.device)
        agent.q_network.load_state_dict(ckpt['q_network'])
        agent.target_network.load_state_dict(ckpt['target_network'])
        agent.optimizer.load_state_dict(ckpt['optimizer'])
        agent.total_steps         = ckpt['total_steps']
        agent.epsilon             = ckpt['epsilon']
        agent.training_losses     = ckpt.get('training_losses', [])
        agent.resume_episode      = ckpt.get('episode_num', 0)
        agent.resume_best_val_asr = ckpt.get('best_val_asr', 0.0)
        return agent
