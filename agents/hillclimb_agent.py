"""
Hill-Climbing Agent — Baseline 2.

Walks through BBs randomly. For each candidate BB:
  1. Snapshot the binary state
  2. MODIFY the BB
  3. If score improved → keep
  4. If score got worse → rollback to snapshot

Unlike the random agent, hill-climbing REJECTS bad modifications.
This requires the env to support snapshot/restore (env.snapshot(), env.restore()).

Usage:
    from agents.hillclimb_agent import HillClimbAgent
    agent = HillClimbAgent()

    # In evaluate.py, use the hillclimb evaluation loop:
    # (standard env.step loop doesn't support rollback)
"""
import os
import sys
import random
import time
import copy
import numpy as np
import pefile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent


class HillClimbAgent(BaseAgent):
    """
    Hill-climbing baseline.

    Cannot use standard env.step() loop because it needs rollback.
    Instead, provides run_episode() that directly manages the env state.
    """

    def __init__(self, modify_prob=0.05, seed=None):
        """
        Args:
            modify_prob: probability of attempting MODIFY per BB
            seed: random seed
        """
        super().__init__(name='hillclimb')
        self.modify_prob = modify_prob
        if seed is not None:
            random.seed(seed)

    def select_action(self, obs):
        """Standard interface — but hill-climbing uses run_episode instead."""
        return 1 if random.random() < self.modify_prob else 0

    def run_episode(self, env, pe_path, episode_num=0):
        """
        Run one hill-climbing episode with rollback on bad modifications.

        Args:
            env: MalwareEvasionEnv instance
            pe_path: path to PE binary
            episode_num: for logging

        Returns:
            info: episode info dict
            total_reward: float
            wall_time: float
        """
        ep_start = time.time()
        self.last_adv_bytes = None

        try:
            obs = env.reset(pe_path)
        except Exception:
            self.last_adv_bytes = None
            return None, 0.0, 0.0

        if env.done or len(env.bb_list) == 0:
            env.cleanup()
            self.last_adv_bytes = None
            return None, 0.0, 0.0

        total_reward = 0.0

        while not env.done:
            action = self.select_action(obs)

            if action == 0:
                # SKIP — standard
                obs, reward, done, info = env.step(0)
            else:
                # MODIFY with rollback capability
                snap = env.snapshot()
                snapshot_score = snap['curr_score']

                # Try the modification
                obs, reward, done, info = env.step(1)

                # Check if score got WORSE (increased)
                if env.curr_score > snapshot_score and not env.bypassed:
                    env.restore(snap)

                    # Fix the step log — mark last entry as REJECTED
                    if env.step_log:
                        env.step_log[-1]['action'] = 'REJECTED'
                        env.step_log[-1]['score_after'] = snapshot_score
                        env.step_log[-1]['reward'] = 0.0
                        env.step_log[-1]['cumulative_delta'] = env.start_score - snapshot_score

                    reward = 0.0  # no reward for rejected modifications

            total_reward += reward

        ep_time = time.time() - ep_start
        adv_bytes = env.get_adversarial_bytes()  # triggers _generate_current_binary, updates curr_injected
        self.last_adv_bytes = adv_bytes
        info = env._get_info()
        env.cleanup()

        return info, total_reward, ep_time