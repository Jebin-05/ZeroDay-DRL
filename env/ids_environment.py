"""
Intrusion Detection System Environment for DRL training.
Provides a Gymnasium-compatible environment for network traffic classification.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any


class IDSEnvironment:
    """
    IDS Environment for training DRL agents.
    Follows the Gymnasium API.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        config: Dict
    ):
        """
        Initialize the IDS environment.

        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Label array (n_samples,)
            config: Configuration dictionary
        """
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.config = config
        self.n_samples = len(labels)
        self.feature_dim = features.shape[1]

        # Environment config
        env_config = config.get('environment', {})
        self.reward_correct = env_config.get('reward_correct', 1.0)
        self.reward_incorrect = env_config.get('reward_incorrect', -1.0)
        self.reward_detect_attack = env_config.get('reward_detect_attack', 2.0)
        self.reward_miss_attack = env_config.get('reward_miss_attack', -3.0)
        self.reward_false_positive = env_config.get('reward_false_positive', -1.5)

        # Episode state
        self.current_idx = 0
        self.episode_predictions = []
        self.episode_labels = []
        self.step_count = 0
        self.max_steps = config.get('training', {}).get('max_steps_per_episode', 200)

        # Shuffle indices for each episode
        self.indices = np.arange(self.n_samples)

    @property
    def observation_space_shape(self) -> Tuple[int]:
        """Return observation space shape."""
        return (self.feature_dim,)

    @property
    def action_space_n(self) -> int:
        """Return number of actions (binary classification)."""
        return 2

    def reset(
        self,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed (optional)

        Returns:
            Tuple of (initial_state, info_dict)
        """
        if seed is not None:
            np.random.seed(seed)

        # Shuffle samples for this episode
        np.random.shuffle(self.indices)
        self.current_idx = 0
        self.step_count = 0

        # Reset tracking
        self.episode_predictions = []
        self.episode_labels = []

        # Get initial state
        state = self._get_state()
        info = self._get_info()

        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Predicted label (0=normal, 1=attack)

        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        # Get current sample info
        true_label = self.labels[self.indices[self.current_idx]]

        # Track prediction
        self.episode_predictions.append(action)
        self.episode_labels.append(true_label)

        # Calculate reward
        reward = self._calculate_reward(action, true_label)

        # Move to next sample
        self.current_idx += 1
        self.step_count += 1

        # Check termination conditions
        terminated = self.current_idx >= self.n_samples
        truncated = self.step_count >= self.max_steps

        # Get next state (or zeros if done)
        if terminated or truncated:
            next_state = np.zeros(self.feature_dim)
        else:
            next_state = self._get_state()

        info = self._get_info()

        return next_state, reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        """Get current state (features of current sample)."""
        idx = self.indices[self.current_idx]
        return self.features[idx].astype(np.float32)

    def _calculate_reward(self, action: int, true_label: int) -> float:
        """
        Calculate reward based on prediction and true label.

        Args:
            action: Predicted label
            true_label: True label

        Returns:
            Reward value
        """
        if action == true_label:
            if true_label == 1:
                # Correctly detected attack
                return self.reward_detect_attack
            else:
                # Correctly identified normal
                return self.reward_correct
        else:
            if true_label == 1:
                # Missed attack (false negative)
                return self.reward_miss_attack
            else:
                # False alarm (false positive)
                return self.reward_false_positive

    def _get_info(self) -> Dict[str, Any]:
        """Get current episode info/metrics."""
        if not self.episode_predictions:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'false_positive_rate': 0.0,
                'step': self.step_count,
                'samples_seen': len(self.episode_predictions)
            }

        predictions = np.array(self.episode_predictions)
        labels = np.array(self.episode_labels)

        # Calculate metrics
        correct = (predictions == labels).sum()
        accuracy = correct / len(predictions)

        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        tn = ((predictions == 0) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': fpr,
            'step': self.step_count,
            'samples_seen': len(self.episode_predictions),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }

    def get_sample_at(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Get a specific sample by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, label)
        """
        return self.features[idx], self.labels[idx]

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.n_samples
