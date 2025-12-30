"""
Hybrid Detector: Combines DRL and Few-Shot Meta-Learning.
This is the CORE NOVELTY of the ZeroDay-DRL framework.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from enum import Enum

from drl_agent.dqn_agent import DQNAgent, DoubleDQNAgent
from drl_agent.ppo_agent import PPOAgent
from few_shot.meta_learner import MetaLearner


class DetectionMode(Enum):
    """Detection modes for the hybrid detector."""
    DRL_ONLY = "drl_only"
    FEW_SHOT_ONLY = "few_shot_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class HybridDetector:
    """
    Hybrid Detection System combining:
    1. DRL Agent for adaptive policy-based detection
    2. Few-Shot Meta-Learner for zero-day adaptation

    Detection Flow:
    1. DRL agent makes initial classification
    2. If confidence is low or suspicious, few-shot validates
    3. Few-shot can trigger prototype adaptation for novel threats
    4. Final decision is ensemble of both systems
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict,
        device: torch.device = torch.device('cpu'),
        drl_algorithm: str = 'dqn'
    ):
        """
        Initialize Hybrid Detector.

        Args:
            state_dim: State/feature dimension
            action_dim: Number of actions (2: normal/botnet)
            config: Configuration dictionary
            device: Computation device
            drl_algorithm: DRL algorithm to use ('dqn' or 'ppo')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device

        # Extract hybrid config
        hybrid_config = config.get('hybrid', {})
        self.drl_confidence_threshold = hybrid_config.get('drl_confidence_threshold', 0.7)
        self.few_shot_trigger_threshold = hybrid_config.get('few_shot_trigger_threshold', 0.5)
        self.adaptation_rate = hybrid_config.get('adaptation_rate', 0.1)
        self.ensemble_weight_drl = hybrid_config.get('ensemble_weight_drl', 0.6)
        self.ensemble_weight_fewshot = hybrid_config.get('ensemble_weight_fewshot', 0.4)

        # Initialize DRL agent
        if drl_algorithm == 'dqn':
            self.drl_agent = DoubleDQNAgent(
                state_dim, action_dim, config, device
            )
        elif drl_algorithm == 'ppo':
            self.drl_agent = PPOAgent(
                state_dim, action_dim, config, device
            )
        else:
            raise ValueError(f"Unknown DRL algorithm: {drl_algorithm}")

        self.drl_algorithm = drl_algorithm

        # Initialize Few-Shot Meta-Learner
        self.meta_learner = MetaLearner(
            input_dim=state_dim,
            config=config,
            device=device
        )

        # Detection mode
        self.mode = DetectionMode.HYBRID

        # Metrics tracking
        self.detection_history = []
        self.adaptation_triggers = 0
        self.zero_day_detections = 0

        # Online learning buffer
        self.uncertain_samples = []
        self.confirmed_samples = {0: [], 1: []}

    def set_mode(self, mode: DetectionMode):
        """Set detection mode."""
        self.mode = mode

    def detect(
        self,
        state: np.ndarray,
        training: bool = False
    ) -> Dict:
        """
        Perform hybrid detection on a traffic sample.

        Args:
            state: Feature vector of traffic window
            training: Whether in training mode

        Returns:
            Detection result dictionary
        """
        result = {
            'prediction': 0,
            'confidence': 0.0,
            'drl_action': 0,
            'drl_confidence': 0.0,
            'few_shot_prediction': 0,
            'few_shot_confidence': 0.0,
            'novelty_score': 0.0,
            'adaptation_triggered': False,
            'detection_source': 'none'
        }

        if self.mode == DetectionMode.DRL_ONLY:
            return self._detect_drl_only(state, training, result)
        elif self.mode == DetectionMode.FEW_SHOT_ONLY:
            return self._detect_few_shot_only(state, result)
        elif self.mode == DetectionMode.HYBRID:
            return self._detect_hybrid(state, training, result)
        else:  # ADAPTIVE
            return self._detect_adaptive(state, training, result)

    def _detect_drl_only(
        self,
        state: np.ndarray,
        training: bool,
        result: Dict
    ) -> Dict:
        """DRL-only detection."""
        action, confidence = self.drl_agent.select_action(state, training)

        result['prediction'] = action
        result['confidence'] = confidence
        result['drl_action'] = action
        result['drl_confidence'] = confidence
        result['detection_source'] = 'drl'

        return result

    def _detect_few_shot_only(
        self,
        state: np.ndarray,
        result: Dict
    ) -> Dict:
        """Few-shot only detection."""
        if self.meta_learner.proto_net.prototypes is None:
            result['detection_source'] = 'uninitialized'
            return result

        prediction, confidence = self.meta_learner.classify_single(state)
        novelty = self.meta_learner.compute_novelty_score(state)

        result['prediction'] = prediction
        result['confidence'] = float(confidence)
        result['few_shot_prediction'] = prediction
        result['few_shot_confidence'] = float(confidence)
        result['novelty_score'] = novelty
        result['detection_source'] = 'few_shot'

        return result

    def _detect_hybrid(
        self,
        state: np.ndarray,
        training: bool,
        result: Dict
    ) -> Dict:
        """
        Hybrid detection combining DRL and few-shot.

        Decision Flow:
        1. DRL makes initial decision
        2. If DRL confidence is high, trust DRL
        3. If DRL flags anomaly but low confidence, validate with few-shot
        4. If few-shot detects high novelty, trigger adaptation
        5. Ensemble final decision
        """
        # Step 1: DRL initial detection
        drl_action, drl_confidence = self.drl_agent.select_action(state, training)
        result['drl_action'] = drl_action
        result['drl_confidence'] = drl_confidence

        # Step 2: Check if few-shot validation is needed
        need_few_shot = (
            drl_confidence < self.drl_confidence_threshold or
            drl_action == 1  # Always validate potential threats
        )

        if need_few_shot and self.meta_learner.proto_net.prototypes is not None:
            # Step 3: Few-shot validation
            fs_prediction, fs_confidence = self.meta_learner.classify_single(state)
            novelty = self.meta_learner.compute_novelty_score(state)

            result['few_shot_prediction'] = fs_prediction
            result['few_shot_confidence'] = float(fs_confidence)
            result['novelty_score'] = novelty

            # Step 4: Check for zero-day (high novelty)
            if novelty > self.few_shot_trigger_threshold:
                result['adaptation_triggered'] = True
                self.adaptation_triggers += 1

                # If novel and DRL flagged it, likely zero-day
                if drl_action == 1:
                    self.zero_day_detections += 1
                    result['prediction'] = 1
                    result['confidence'] = max(drl_confidence, fs_confidence)
                    result['detection_source'] = 'zero_day'

                    # Store for adaptation
                    self.meta_learner.add_to_experience(state, 1)
                    return result

            # Step 5: Ensemble decision
            # Weight the confidences
            if drl_action == fs_prediction:
                # Agreement - high confidence
                result['prediction'] = drl_action
                result['confidence'] = (
                    self.ensemble_weight_drl * drl_confidence +
                    self.ensemble_weight_fewshot * fs_confidence
                )
                result['detection_source'] = 'ensemble_agree'
            else:
                # Disagreement - use higher confidence
                if drl_confidence > fs_confidence:
                    result['prediction'] = drl_action
                    result['confidence'] = drl_confidence * 0.8  # Reduce due to disagreement
                    result['detection_source'] = 'drl_override'
                else:
                    result['prediction'] = fs_prediction
                    result['confidence'] = fs_confidence * 0.8
                    result['detection_source'] = 'few_shot_override'

                # Store uncertain sample for later verification
                self.uncertain_samples.append(state)
        else:
            # Trust DRL decision
            result['prediction'] = drl_action
            result['confidence'] = drl_confidence
            result['detection_source'] = 'drl_high_conf'

        # Track detection
        self.detection_history.append({
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'source': result['detection_source']
        })

        return result

    def _detect_adaptive(
        self,
        state: np.ndarray,
        training: bool,
        result: Dict
    ) -> Dict:
        """
        Adaptive mode: Dynamically switches between methods
        based on recent performance.
        """
        # Use hybrid detection
        result = self._detect_hybrid(state, training, result)

        # Periodically adapt prototypes
        if len(self.detection_history) % 50 == 0:
            self.meta_learner.adapt(self.adaptation_rate)

        return result

    def initialize_few_shot(
        self,
        normal_samples: np.ndarray,
        botnet_samples: np.ndarray
    ):
        """
        Initialize few-shot prototypes.

        Args:
            normal_samples: Normal traffic samples
            botnet_samples: Botnet traffic samples
        """
        self.meta_learner.initialize_prototypes(normal_samples, botnet_samples)

    def update_with_feedback(
        self,
        state: np.ndarray,
        true_label: int
    ):
        """
        Update detector with ground truth feedback.

        Args:
            state: Sample features
            true_label: True label
        """
        # Add to confirmed samples
        self.confirmed_samples[true_label].append(state)

        # Add to meta-learner experience
        self.meta_learner.add_to_experience(state, true_label)

        # Periodic prototype adaptation
        if len(self.confirmed_samples[0]) + len(self.confirmed_samples[1]) >= 20:
            self.meta_learner.adapt(self.adaptation_rate)
            self.confirmed_samples = {0: [], 1: []}

    def train_drl_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Optional[float]:
        """
        Perform DRL training step.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag

        Returns:
            Loss if training occurred
        """
        self.drl_agent.store_transition(state, action, reward, next_state, done)
        return self.drl_agent.train_step_fn()

    def train_few_shot_episode(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        query_labels: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Train few-shot on an episode.

        Args:
            support_features: Support set features
            support_labels: Support set labels
            query_features: Query set features
            query_labels: Query set labels

        Returns:
            Tuple of (loss, accuracy)
        """
        self.meta_learner.proto_net.train()

        log_probs, _ = self.meta_learner.proto_net(
            support_features.to(self.device),
            support_labels.to(self.device),
            query_features.to(self.device)
        )

        loss = torch.nn.functional.nll_loss(
            log_probs, query_labels.to(self.device)
        )

        self.meta_learner.optimizer.zero_grad()
        loss.backward()
        self.meta_learner.optimizer.step()

        preds = log_probs.argmax(dim=1)
        accuracy = (preds == query_labels.to(self.device)).float().mean().item()

        return loss.item(), accuracy

    def get_anomaly_score(self, state: np.ndarray) -> float:
        """
        Get composite anomaly score combining DRL and few-shot.

        Args:
            state: Feature vector

        Returns:
            Anomaly score (0-1, higher = more anomalous)
        """
        result = self.detect(state, training=False)

        # Combine prediction confidence and novelty
        if result['prediction'] == 1:
            # Detected as botnet
            score = 0.5 + 0.5 * result['confidence']
        else:
            # Detected as normal but check novelty
            score = 0.3 * result['novelty_score']

        return float(np.clip(score, 0, 1))

    def save(self, path: str):
        """Save hybrid detector state."""
        import os

        if not os.path.exists(path):
            os.makedirs(path)

        # Save DRL agent
        self.drl_agent.save(os.path.join(path, 'drl_agent.pt'))

        # Save meta-learner
        self.meta_learner.save(os.path.join(path, 'meta_learner.pt'))

        # Save hybrid state
        torch.save({
            'mode': self.mode.value,
            'adaptation_triggers': self.adaptation_triggers,
            'zero_day_detections': self.zero_day_detections,
            'config': self.config
        }, os.path.join(path, 'hybrid_state.pt'))

    def load(self, path: str):
        """Load hybrid detector state."""
        import os

        # Load DRL agent
        self.drl_agent.load(os.path.join(path, 'drl_agent.pt'))

        # Load meta-learner
        self.meta_learner.load(os.path.join(path, 'meta_learner.pt'))

        # Load hybrid state
        state = torch.load(os.path.join(path, 'hybrid_state.pt'))
        self.mode = DetectionMode(state['mode'])
        self.adaptation_triggers = state['adaptation_triggers']
        self.zero_day_detections = state['zero_day_detections']

    def get_metrics(self) -> Dict:
        """Get detector metrics."""
        drl_metrics = self.drl_agent.get_metrics()
        meta_metrics = self.meta_learner.get_metrics()

        # Detection statistics
        if self.detection_history:
            recent = self.detection_history[-100:]
            avg_conf = np.mean([d['confidence'] for d in recent])
            detection_rate = np.mean([d['prediction'] for d in recent])
        else:
            avg_conf = 0
            detection_rate = 0

        return {
            'drl': drl_metrics,
            'meta_learner': meta_metrics,
            'mode': self.mode.value,
            'adaptation_triggers': self.adaptation_triggers,
            'zero_day_detections': self.zero_day_detections,
            'avg_confidence': avg_conf,
            'detection_rate': detection_rate,
            'total_detections': len(self.detection_history)
        }
