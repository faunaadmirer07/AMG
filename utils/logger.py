"""
Logger — per-binary folders with step CSV + modified binary.

Directory structure:
  logs/run_tag/
    episodes.csv                         <- summary of all episodes (flat)
    024b95f5.../
      steps.csv                          <- this binary's step-by-step log
      modified.exe                       <- adversarial binary (if bypassed)
    00745984.../
      steps.csv

During RL training, the same binary appears multiple times.
Each episode appends to that binary's steps.csv with an episode column.

Usage:
    from utils.logger import Logger
    logger = Logger(log_dir='/path/to/logs', run_tag='random_val')
    logger.start_episode(binary_hash)
    logger.log_step(episode, step_data)
    logger.log_episode(episode, episode_data)
    logger.save_modified_binary(binary_hash, bytez)
    logger.close()
"""
import os
import csv
from datetime import datetime


class Logger:
    """CSV logger with per-binary folders."""

    STEP_COLUMNS = [
        'episode', 'step', 'bb_index', 'bb_start', 'bb_end', 'func_addr',
        'action', 'num_calls', 'num_calls_eligible', 'bb_size',
        'score_before', 'score_after',
        'reward', 'cumulative_delta', 'budget_used', 'bypassed',
        'time_elapsed', 'timestamp'
    ]

    EPISODE_COLUMNS = [
        'episode', 'binary_hash', 'start_score', 'final_score',
        'total_delta', 'n_mods', 'n_calls_modified', 'queries',
        'budget_used_pct', 'bypassed', 'total_steps', 'wall_time_sec',
        'timestamp'
    ]

    def __init__(self, log_dir, run_tag, log_steps=True, log_episodes=True):
        """
        Args:
            log_dir: base log directory
            run_tag: identifier for this run
            log_steps: whether to log per-step data
            log_episodes: whether to log per-episode data
        """
        self.log_steps = log_steps
        self.log_episodes = log_episodes
        self.run_dir = os.path.join(log_dir, run_tag)
        os.makedirs(self.run_dir, exist_ok=True)

        # Episode-level CSV (flat, one row per episode)
        self._episode_file = None
        self._episode_writer = None
        if log_episodes:
            episode_path = os.path.join(self.run_dir, 'episodes.csv')
            file_exists = os.path.exists(episode_path) and os.path.getsize(episode_path) > 0
            mode = 'a' if file_exists else 'w'
            self._episode_file = open(episode_path, mode, newline='')
            self._episode_writer = csv.DictWriter(
                self._episode_file, fieldnames=self.EPISODE_COLUMNS
            )
            if not file_exists:
                self._episode_writer.writeheader()

        # Per-binary step CSV writers (opened on demand)
        self._step_writers = {}  # binary_hash -> (file, writer)
        self._current_hash = None

    def _get_binary_dir(self, binary_hash):
        """Get or create per-binary directory."""
        binary_dir = os.path.join(self.run_dir, binary_hash)
        os.makedirs(binary_dir, exist_ok=True)
        return binary_dir

    def _get_step_writer(self, binary_hash):
        """Get or create step CSV writer for a binary."""
        if binary_hash not in self._step_writers:
            binary_dir = self._get_binary_dir(binary_hash)
            step_path = os.path.join(binary_dir, 'steps.csv')
            file_exists = os.path.exists(step_path)
            f = open(step_path, 'a', newline='')
            writer = csv.DictWriter(f, fieldnames=self.STEP_COLUMNS)
            if not file_exists:
                writer.writeheader()
            self._step_writers[binary_hash] = (f, writer)
        return self._step_writers[binary_hash][1]

    def start_episode(self, binary_hash):
        """Call at the start of each episode to set current binary."""
        self._current_hash = binary_hash

    def log_step(self, episode_num, step_data):
        """Log one step to the current binary's steps.csv."""
        if not self.log_steps:
            return

        binary_hash = step_data.get('binary_hash', self._current_hash)
        if not binary_hash:
            return

        writer = self._get_step_writer(binary_hash)
        row = {'episode': episode_num}
        for col in self.STEP_COLUMNS:
            if col == 'episode':
                continue
            val = step_data.get(col, '')
            if isinstance(val, float):
                if col in ('score_before', 'score_after', 'cumulative_delta', 'budget_used'):
                    val = '{:.6f}'.format(val)
                elif col in ('reward',):
                    val = '{:.4f}'.format(val)
                elif col in ('time_elapsed',):
                    val = '{:.3f}'.format(val)
            row[col] = val
        row['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow(row)

    def log_episode(self, episode_num, info, wall_time):
        """Log episode summary to episodes.csv."""
        if not self.log_episodes:
            return

        row = {
            'episode': episode_num,
            'binary_hash': info['binary_hash'],
            'start_score': '{:.6f}'.format(info['start_score']),
            'final_score': '{:.6f}'.format(info['curr_score']),
            'total_delta': '{:.6f}'.format(info['total_delta']),
            'n_mods': info['n_mods'],
            'n_calls_modified': info['total_calls_modified'],
            'queries': info['query_count'],
            'budget_used_pct': '{:.1f}'.format(info['budget_used'] * 100),
            'bypassed': info['bypassed'],
            'total_steps': info['total_steps'],
            'wall_time_sec': '{:.1f}'.format(wall_time),
        }
        row['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._episode_writer.writerow(row)

    def save_modified_binary(self, binary_hash, bytez, episode_num=None):
        """Save adversarial binary into the binary's folder."""
        binary_dir = self._get_binary_dir(binary_hash)
        if episode_num is not None:
            filename = 'modified_ep{}.exe'.format(episode_num)
        else:
            filename = 'modified.exe'
        output_path = os.path.join(binary_dir, filename)
        with open(output_path, 'wb') as f:
            f.write(bytez)

    def flush(self):
        """Flush all buffers to disk."""
        if self._episode_file:
            self._episode_file.flush()
        for f, _ in self._step_writers.values():
            f.flush()

    def close(self):
        """Close all files."""
        if self._episode_file:
            self._episode_file.close()
            self._episode_file = None
        for f, _ in self._step_writers.values():
            f.close()
        self._step_writers = {}

    def __del__(self):
        self.close()