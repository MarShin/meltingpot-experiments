import time
from collections import deque
from typing import Optional

import numpy as np
import gymnasium as gym


class RecordMultiagentEpisodeStatistics(gym.Wrapper):
    """
    Adapted from gym wrapper `RecordEpisodeStatistics`.

    4 social outcome metrics according to 'A multi-agent reinforcement learning model of common-pool resource appropriation' paper.
    1. Unitarian (U) aka Efficiency - sum total of all rewards obtained by all agents - 'Collective Reward' provided by
    1. Equality (E) - using Gini coefficient
    1. Sustainability (S) - average time at which the rewards are collected
    1. Peace (P) - average number of untagged agent steps
    """

    def __init__(self, env, num_agents):
        super().__init__(env)
        self.num_agents = num_agents
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.episode_returns = None
        self.episode_lengths = None
        self.zap_counts = None
        self.sustainability_t_i = None

        self.episode_efficiency = None
        self.episode_equality = None
        self.episode_sustainability = None
        self.episode_peace = None

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=None, options=None)
        self.episode_returns = np.zeros(self.num_agents, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_agents, dtype=np.int32)
        self.sustainability_t_i = 0
        self.zap_counts = 0
        self.episode_efficiency = 0.0
        self.episode_equality = 0.0
        self.episode_sustainability = 0.0
        self.episode_peace = 0.0
        return obs, info

    def _gini_coefficient(self, x):
        """Compute Gini coefficient of array of values"""
        diffsum = 0
        for i, xi in enumerate(x[:-1], 1):
            diffsum += np.sum(np.abs(xi - x[i:]))
        return diffsum / (len(x) ** 2 * np.mean(x))

    # can return 1 episode info dict aggregating all the metrics of each agent
    def step(self, action):
        # (num_agents, )
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)

        # self.episode_returns += rewards
        # self.sustainability_t_i += (rewards > 0.0).sum()
        # self.episode_lengths += 1

        # # only using the last of the stacked frames; any of the vec env same values; remove the padding
        # who_zapped_who = observations[:, :, :, -1][0][:16, :16]
        # self.zap_counts += who_zapped_who.sum()

        # assert (
        #     self.is_vector_env == True
        # ), "this wrapper currently only works with vector env"

        # infos = list(infos)  # Convert infos to mutable type
        # # aggregating each agent

        # # if all agent have finished the episode then report the metrics
        # if dones.all():
        #     for i in range(len(dones)):
        #         infos[i] = infos[i].copy()

        #         self.episode_efficiency = (
        #             self.episode_returns.sum() / self.episode_lengths.max()
        #         )
        #         self.episode_equality = 1 - self._gini_coefficient(self.episode_returns)
        #         self.episode_sustainability = self.sustainability_t_i / self.num_agents
        #         self.episode_peace = (
        #             self.num_agents * self.episode_lengths.max() - self.zap_counts
        #         ) / self.episode_lengths.max()

        #         episode_info = {
        #             "l": self.episode_lengths.max(),
        #             "u": self.episode_efficiency,
        #             "e": self.episode_equality,
        #             "s": self.episode_sustainability,
        #             "p": self.episode_peace,
        #             "t": round(time.perf_counter() - self.t0, 6),
        #         }
        #         infos[i]["ma_episode"] = episode_info
        #     self.episode_count += 1
        #     self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        #     self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        #     self.sustainability_t_i = 0
        #     self.zap_counts = 0
        # if self.is_vector_env:
        #     infos = tuple(infos)

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )
