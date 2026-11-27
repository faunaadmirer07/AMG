"""
Random Agent — Baseline 1.

shuffle BB list, then walk through with 50/50 coin flip
at each BB (MODIFY or SKIP). Shuffling spreads modifications across the
entire binary rather than clustering at the start.

Usage:
    from agents.random_agent import RandomAgent
    agent = RandomAgent(modify_prob=0.50)
    action = agent.select_action(obs)
"""
import random
from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Random baseline agent."""

    def __init__(self, modify_prob=0.50, seed=None):
        """
        Args:
            modify_prob: probability of MODIFY per BB (default 0.50, Prof. Lan's design)
            seed: random seed for reproducibility (None = random)
        """
        super().__init__(name='random')
        self.modify_prob = modify_prob
        if seed is not None:
            random.seed(seed)

    def select_action(self, obs):
        """Randomly choose MODIFY (1) or SKIP (0)."""
        return 1 if random.random() < self.modify_prob else 0