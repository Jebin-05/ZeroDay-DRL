"""
Replay Buffer implementations for DRL agents.
Includes standard replay buffer and prioritized experience replay (PER).
"""

import numpy as np
from typing import Tuple, List
from collections import deque
import random


class ReplayBuffer:
    """Standard experience replay buffer."""

    def __init__(self, capacity: int):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store a transition in the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough samples.

        Args:
            batch_size: Required batch size

        Returns:
            True if enough samples available
        """
        return len(self.buffer) >= batch_size

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    Samples transitions based on TD-error priority.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store a transition with maximum priority.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        transition = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch based on priorities.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        buffer_len = len(self.buffer)

        # Calculate sampling probabilities
        priorities = self.priorities[:buffer_len]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices based on priorities
        indices = np.random.choice(buffer_len, batch_size, p=probs, replace=False)

        # Calculate importance sampling weights
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        weights = (buffer_len * probs[indices]) ** (-beta)
        weights /= weights.max()

        # Get samples
        batch = [self.buffer[idx] for idx in indices]

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD errors.

        Args:
            indices: Indices of sampled transitions
            td_errors: Absolute TD errors
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6)
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough samples.

        Args:
            batch_size: Required batch size

        Returns:
            True if enough samples available
        """
        return len(self.buffer) >= batch_size

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
