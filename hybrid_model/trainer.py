"""
Hybrid Trainer for the ZeroDay-DRL framework.
Coordinates training of DRL agent and Few-Shot meta-learner.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import time

from .hybrid_detector import HybridDetector
from env.ids_environment import IDSEnvironment
from preprocessing.data_loader import DataLoader


class HybridTrainer:
    """
    Trainer for the hybrid DRL + Few-Shot detection system.
    """

    def __init__(
        self,
        config: Dict,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
            device: Computation device
        """
        self.config = config
        self.device = device

        # Training config
        train_config = config.get('training', {})
        self.num_episodes = train_config.get('num_episodes', 500)
        self.max_steps = train_config.get('max_steps_per_episode', 200)
        self.eval_frequency = train_config.get('eval_frequency', 50)
        self.save_frequency = train_config.get('save_frequency', 100)

        # Few-shot config
        fs_config = config.get('few_shot', {})
        self.fs_episodes = fs_config.get('num_episodes', 1000)

        # Metrics
        self.training_history = {
            'episode': [],
            'reward': [],
            'accuracy': [],
            'drl_loss': [],
            'fs_loss': [],
            'fs_accuracy': [],
            'detection_rate': [],
            'false_positive_rate': []
        }

    def train(
        self,
        detector: HybridDetector,
        data_loader,
        save_path: str = 'checkpoints',
        data: Dict = None
    ) -> Dict:
        """
        Train the hybrid detector.

        Args:
            detector: Hybrid detector instance
            data_loader: Data loader instance
            save_path: Path to save checkpoints
            data: Pre-loaded data dictionary (optional)

        Returns:
            Training results
        """
        print("=" * 60)
        print("Starting Hybrid Training")
        print("=" * 60)

        # Load data if not provided
        if data is None:
            if hasattr(data_loader, 'load_synthetic_data'):
                data = data_loader.load_synthetic_data()
            elif hasattr(data_loader, 'train_features') and data_loader.train_features is not None:
                data = {
                    'train': (data_loader.train_features, data_loader.train_labels),
                    'val': (data_loader.val_features, data_loader.val_labels),
                    'test': (data_loader.test_features, data_loader.test_labels)
                }
            else:
                raise ValueError("No data available. Load data first.")

        train_features, train_labels = data['train']
        val_features, val_labels = data['val']

        print(f"Training samples: {len(train_labels)}")
        print(f"Validation samples: {len(val_labels)}")

        # Phase 1: Pre-train few-shot meta-learner
        print("\n" + "-" * 40)
        print("Phase 1: Pre-training Few-Shot Meta-Learner")
        print("-" * 40)

        fs_results = self._pretrain_few_shot(
            detector, data_loader, train_features, train_labels
        )

        # Initialize few-shot prototypes
        normal_samples = train_features[train_labels == 0]
        botnet_samples = train_features[train_labels == 1]
        detector.initialize_few_shot(normal_samples, botnet_samples)

        # Phase 2: Train DRL agent with few-shot integration
        print("\n" + "-" * 40)
        print("Phase 2: Training DRL Agent (Hybrid Mode)")
        print("-" * 40)

        drl_results = self._train_drl(
            detector, train_features, train_labels,
            val_features, val_labels, save_path
        )

        # Phase 3: Fine-tune hybrid integration
        print("\n" + "-" * 40)
        print("Phase 3: Fine-tuning Hybrid Integration")
        print("-" * 40)

        hybrid_results = self._finetune_hybrid(
            detector, val_features, val_labels
        )

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        return {
            'few_shot': fs_results,
            'drl': drl_results,
            'hybrid': hybrid_results,
            'history': self.training_history
        }

    def _pretrain_few_shot(
        self,
        detector: HybridDetector,
        data_loader,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """Pre-train the few-shot meta-learner."""
        import torch

        # Get few-shot config
        n_way = self.config['few_shot']['n_way']
        k_shot = self.config['few_shot']['k_shot']
        n_query = self.config['few_shot']['n_query']

        # Separate by class
        normal_features = features[labels == 0]
        attack_features = features[labels == 1]

        losses = []
        accuracies = []

        # Limit episodes for efficiency
        num_episodes = min(self.fs_episodes, 200)

        for episode_idx in tqdm(range(num_episodes), desc="Few-Shot Pre-training"):
            # Sample support and query sets
            n_support = min(k_shot, len(normal_features), len(attack_features))
            n_query_actual = min(n_query, len(normal_features) - n_support, len(attack_features) - n_support)

            if n_support < 2 or n_query_actual < 2:
                continue

            # Random sampling
            normal_idx = np.random.permutation(len(normal_features))
            attack_idx = np.random.permutation(len(attack_features))

            support_normal = normal_features[normal_idx[:n_support]]
            support_attack = attack_features[attack_idx[:n_support]]
            query_normal = normal_features[normal_idx[n_support:n_support + n_query_actual]]
            query_attack = attack_features[attack_idx[n_support:n_support + n_query_actual]]

            # Create tensors
            support_features = torch.FloatTensor(np.vstack([support_normal, support_attack]))
            support_labels = torch.LongTensor([0] * n_support + [1] * n_support)
            query_features = torch.FloatTensor(np.vstack([query_normal, query_attack]))
            query_labels = torch.LongTensor([0] * n_query_actual + [1] * n_query_actual)

            # Train episode
            loss, acc = detector.train_few_shot_episode(
                support_features,
                support_labels,
                query_features,
                query_labels
            )

            losses.append(loss)
            accuracies.append(acc)

            if (episode_idx + 1) % 50 == 0:
                avg_loss = np.mean(losses[-50:])
                avg_acc = np.mean(accuracies[-50:])
                print(f"  Episode {episode_idx + 1}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")

        return {
            'final_loss': np.mean(losses[-50:]) if losses else 0,
            'final_accuracy': np.mean(accuracies[-50:]) if accuracies else 0
        }

    def _train_drl(
        self,
        detector: HybridDetector,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        save_path: str
    ) -> Dict:
        """Train the DRL agent."""
        # Create environment
        env = IDSEnvironment(
            train_features, train_labels,
            self.config
        )

        best_reward = float('-inf')
        episode_rewards = []

        for episode in tqdm(range(self.num_episodes), desc="DRL Training"):
            state, info = env.reset()
            episode_reward = 0
            losses = []

            for step in range(self.max_steps):
                # Get hybrid detection
                detection = detector.detect(state, training=True)
                action = detection['drl_action']

                # Environment step
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store transition and train
                loss = detector.train_drl_step(
                    state, action, reward, next_state, done
                )
                if loss is not None:
                    losses.append(loss)

                episode_reward += reward
                state = next_state

                if done:
                    break

            # Decay exploration
            detector.drl_agent.decay_epsilon()

            # Track metrics
            episode_rewards.append(episode_reward)
            self.training_history['episode'].append(episode)
            self.training_history['reward'].append(episode_reward)
            self.training_history['accuracy'].append(info['accuracy'])
            self.training_history['drl_loss'].append(np.mean(losses) if losses else 0)
            self.training_history['detection_rate'].append(info['recall'])
            self.training_history['false_positive_rate'].append(info['false_positive_rate'])

            # Evaluation
            if (episode + 1) % self.eval_frequency == 0:
                val_metrics = self._evaluate(detector, val_features, val_labels)
                avg_reward = np.mean(episode_rewards[-self.eval_frequency:])

                print(f"\n  Episode {episode + 1}:")
                print(f"    Avg Reward: {avg_reward:.2f}")
                print(f"    Val Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"    Val Detection Rate: {val_metrics['detection_rate']:.4f}")
                print(f"    Val FPR: {val_metrics['false_positive_rate']:.4f}")
                print(f"    Epsilon: {detector.drl_agent.epsilon:.4f}")

                if avg_reward > best_reward:
                    best_reward = avg_reward
                    detector.save(f"{save_path}/best_model")

            # Periodic save
            if (episode + 1) % self.save_frequency == 0:
                detector.save(f"{save_path}/checkpoint_{episode + 1}")

        return {
            'final_reward': np.mean(episode_rewards[-50:]),
            'best_reward': best_reward
        }

    def _finetune_hybrid(
        self,
        detector: HybridDetector,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """Fine-tune hybrid integration weights."""
        # Test different ensemble weights
        best_accuracy = 0
        best_weights = (0.6, 0.4)

        weights_to_try = [
            (0.5, 0.5),
            (0.6, 0.4),
            (0.7, 0.3),
            (0.8, 0.2),
            (0.4, 0.6),
            (0.3, 0.7)
        ]

        for drl_w, fs_w in weights_to_try:
            detector.ensemble_weight_drl = drl_w
            detector.ensemble_weight_fewshot = fs_w

            metrics = self._evaluate(detector, features, labels)

            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_weights = (drl_w, fs_w)

        # Set best weights
        detector.ensemble_weight_drl = best_weights[0]
        detector.ensemble_weight_fewshot = best_weights[1]

        print(f"\n  Best ensemble weights: DRL={best_weights[0]}, FS={best_weights[1]}")
        print(f"  Best accuracy: {best_accuracy:.4f}")

        return {
            'best_weights': best_weights,
            'best_accuracy': best_accuracy
        }

    def _evaluate(
        self,
        detector: HybridDetector,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """Evaluate detector on a dataset."""
        predictions = []
        confidences = []

        for i in range(len(features)):
            result = detector.detect(features[i], training=False)
            predictions.append(result['prediction'])
            confidences.append(result['confidence'])

        predictions = np.array(predictions)
        labels = np.array(labels)

        # Calculate metrics
        correct = (predictions == labels).sum()
        accuracy = correct / len(labels)

        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        tn = ((predictions == 0) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()

        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        return {
            'accuracy': accuracy,
            'detection_rate': detection_rate,
            'false_positive_rate': fpr,
            'precision': precision,
            'avg_confidence': np.mean(confidences)
        }

    def evaluate_zero_day(
        self,
        detector: HybridDetector,
        zero_day_features: np.ndarray,
        k_shot_samples: int = 5
    ) -> Dict:
        """
        Evaluate zero-day detection capability.

        Args:
            detector: Trained hybrid detector
            zero_day_features: Features from unknown botnet family
            k_shot_samples: Number of samples for adaptation

        Returns:
            Zero-day detection metrics
        """
        # Initial detection (before adaptation)
        initial_detections = []
        for i in range(len(zero_day_features)):
            result = detector.detect(zero_day_features[i], training=False)
            initial_detections.append(result['prediction'])

        initial_rate = np.mean(initial_detections)

        # Adapt with few samples
        adaptation_samples = zero_day_features[:k_shot_samples]
        for sample in adaptation_samples:
            detector.update_with_feedback(sample, 1)  # All are botnet

        # Force adaptation
        detector.meta_learner.adapt(detector.adaptation_rate)

        # Post-adaptation detection
        post_detections = []
        for i in range(k_shot_samples, len(zero_day_features)):
            result = detector.detect(zero_day_features[i], training=False)
            post_detections.append(result['prediction'])

        post_rate = np.mean(post_detections) if post_detections else 0

        return {
            'initial_detection_rate': initial_rate,
            'post_adaptation_rate': post_rate,
            'improvement': post_rate - initial_rate,
            'k_shot_samples_used': k_shot_samples
        }

    def get_training_summary(self) -> Dict:
        """Get training summary."""
        return {
            'total_episodes': len(self.training_history['episode']),
            'final_reward': self.training_history['reward'][-1] if self.training_history['reward'] else 0,
            'final_accuracy': self.training_history['accuracy'][-1] if self.training_history['accuracy'] else 0,
            'avg_reward': np.mean(self.training_history['reward'][-50:]) if self.training_history['reward'] else 0,
            'avg_accuracy': np.mean(self.training_history['accuracy'][-50:]) if self.training_history['accuracy'] else 0
        }
