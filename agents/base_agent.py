"""
Base Agent — abstract class that all agents inherit.

Every agent must implement select_action(obs) → int (0=SKIP, 1=MODIFY).
This ensures evaluate.py works identically with any agent.
"""
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def select_action(self, obs):
        """
        Choose an action given the current observation.

        Args:
            obs: numpy array (20-dim observation from env)

        Returns:
            int: 0 (SKIP) or 1 (MODIFY)
        """
        pass

    def on_episode_start(self):
        """Called at the start of each episode. Override if needed."""
        pass

    def on_step(self, obs, action, reward, next_obs, done):
        """Called after each step. Override for learning agents (DQN)."""
        pass

    def on_episode_end(self, info):
        """Called at end of episode. Override if needed."""
        pass