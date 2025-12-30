"""
Metrics Calculator for ZeroDay-DRL evaluation.
Implements comprehensive metrics for intrusion detection evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import time


@dataclass
class DetectionMetrics:
    """Container for detection metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    detection_latency: float
    adaptation_speed: float

    def to_dict(self) -> Dict:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'false_positive_rate': self.false_positive_rate,
            'false_negative_rate': self.false_negative_rate,
            'detection_latency': self.detection_latency,
            'adaptation_speed': self.adaptation_speed
        }


class MetricsCalculator:
    """
    Comprehensive metrics calculator for IDS evaluation.
    """

    def __init__(self):
        self.history = {
            'predictions': [],
            'labels': [],
            'confidences': [],
            'detection_times': [],
            'adaptation_times': []
        }

    def reset(self):
        """Reset metrics history."""
        for key in self.history:
            self.history[key] = []

    def update(
        self,
        prediction: int,
        label: int,
        confidence: float = 0.0,
        detection_time: float = 0.0
    ):
        """
        Update metrics with a new prediction.

        Args:
            prediction: Predicted label
            label: True label
            confidence: Detection confidence
            detection_time: Time taken for detection
        """
        self.history['predictions'].append(prediction)
        self.history['labels'].append(label)
        self.history['confidences'].append(confidence)
        self.history['detection_times'].append(detection_time)

    def calculate(self) -> DetectionMetrics:
        """
        Calculate all metrics from history.

        Returns:
            DetectionMetrics object
        """
        if len(self.history['predictions']) == 0:
            return DetectionMetrics(0, 0, 0, 0, 0, 0, 0, 0)

        y_true = np.array(self.history['labels'])
        y_pred = np.array(self.history['predictions'])

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix based metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Latency metrics
        detection_latency = np.mean(self.history['detection_times']) * 1000  # ms

        # Adaptation speed (if tracked)
        adaptation_speed = np.mean(self.history['adaptation_times']) if self.history['adaptation_times'] else 0

        return DetectionMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            detection_latency=detection_latency,
            adaptation_speed=adaptation_speed
        )

    def calculate_from_arrays(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate metrics from arrays.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores for ROC/PR curves

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # ROC and PR curves
        if y_scores is not None:
            fpr_curve, tpr_curve, _ = roc_curve(y_true, y_scores)
            metrics['roc_auc'] = auc(fpr_curve, tpr_curve)

            pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_scores)
            metrics['pr_auc'] = auc(pr_recall, pr_precision)

            metrics['roc_curve'] = (fpr_curve, tpr_curve)
            metrics['pr_curve'] = (pr_recall, pr_precision)

        return metrics

    def calculate_early_detection_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        detection_times: np.ndarray,
        attack_start_times: np.ndarray
    ) -> Dict:
        """
        Calculate early detection specific metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            detection_times: Time of each detection
            attack_start_times: Ground truth attack start times

        Returns:
            Early detection metrics
        """
        # Calculate time to detection for true positives
        tp_mask = (y_true == 1) & (y_pred == 1)

        if tp_mask.sum() == 0:
            return {
                'avg_time_to_detection': float('inf'),
                'early_detection_rate': 0,
                'detection_within_1s': 0,
                'detection_within_5s': 0
            }

        detection_delays = detection_times[tp_mask] - attack_start_times[tp_mask]

        return {
            'avg_time_to_detection': np.mean(detection_delays),
            'early_detection_rate': (detection_delays < 5.0).mean(),  # Within 5 seconds
            'detection_within_1s': (detection_delays < 1.0).mean(),
            'detection_within_5s': (detection_delays < 5.0).mean(),
            'min_detection_time': np.min(detection_delays),
            'max_detection_time': np.max(detection_delays)
        }

    def calculate_adaptation_metrics(
        self,
        pre_adaptation_accuracy: float,
        post_adaptation_accuracy: float,
        num_samples: int,
        adaptation_time: float
    ) -> Dict:
        """
        Calculate adaptation speed metrics.

        Args:
            pre_adaptation_accuracy: Accuracy before adaptation
            post_adaptation_accuracy: Accuracy after adaptation
            num_samples: Number of samples used for adaptation
            adaptation_time: Time taken for adaptation

        Returns:
            Adaptation metrics
        """
        improvement = post_adaptation_accuracy - pre_adaptation_accuracy

        return {
            'pre_accuracy': pre_adaptation_accuracy,
            'post_accuracy': post_adaptation_accuracy,
            'improvement': improvement,
            'improvement_rate': improvement / num_samples if num_samples > 0 else 0,
            'samples_used': num_samples,
            'adaptation_time': adaptation_time,
            'efficiency': improvement / (adaptation_time + 1e-6)
        }


class LatencyTracker:
    """
    Track detection latency for real-time performance evaluation.
    """

    def __init__(self, window_size: int = 100):
        self.latencies = []
        self.window_size = window_size

    def start(self) -> float:
        """Start timing."""
        return time.perf_counter()

    def stop(self, start_time: float):
        """Stop timing and record latency."""
        latency = time.perf_counter() - start_time
        self.latencies.append(latency)

        # Keep only recent measurements
        if len(self.latencies) > self.window_size:
            self.latencies = self.latencies[-self.window_size:]

    def get_stats(self) -> Dict:
        """Get latency statistics."""
        if not self.latencies:
            return {'avg': 0, 'min': 0, 'max': 0, 'p95': 0, 'p99': 0}

        latencies_ms = np.array(self.latencies) * 1000  # Convert to ms

        return {
            'avg': np.mean(latencies_ms),
            'min': np.min(latencies_ms),
            'max': np.max(latencies_ms),
            'std': np.std(latencies_ms),
            'p50': np.percentile(latencies_ms, 50),
            'p95': np.percentile(latencies_ms, 95),
            'p99': np.percentile(latencies_ms, 99)
        }
