"""
Baseline Comparator for ZeroDay-DRL.
Compares hybrid model with traditional ML and DL baselines.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import time

from .metrics import MetricsCalculator


class SimpleLSTM(nn.Module):
    """Simple LSTM for baseline comparison."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # Reshape if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class BaselineComparator:
    """
    Compare ZeroDay-DRL with baseline methods.
    """

    def __init__(self, config: Dict):
        """
        Initialize comparator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.baselines = {}
        self.results = {}
        self.metrics_calculator = MetricsCalculator()

    def train_baselines(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ):
        """
        Train all baseline models.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("Training baseline models...")

        # Random Forest
        print("  Training Random Forest...")
        self.baselines['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.baselines['random_forest'].fit(X_train, y_train)

        # SVM
        print("  Training SVM...")
        self.baselines['svm'] = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
        # Use subset for SVM (can be slow on large datasets)
        if len(X_train) > 5000:
            indices = np.random.choice(len(X_train), 5000, replace=False)
            self.baselines['svm'].fit(X_train[indices], y_train[indices])
        else:
            self.baselines['svm'].fit(X_train, y_train)

        # MLP
        print("  Training MLP...")
        self.baselines['mlp'] = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            random_state=42
        )
        self.baselines['mlp'].fit(X_train, y_train)

        # LSTM
        print("  Training LSTM...")
        self._train_lstm(X_train, y_train)

        print("  Baseline training complete!")

    def _train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50
    ):
        """Train LSTM baseline."""
        input_dim = X_train.shape[1]
        self.lstm_model = SimpleLSTM(input_dim)

        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)

        # Training loop
        self.lstm_model.train()
        batch_size = 64

        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train))

            for start in range(0, len(X_train), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                batch_X = X_tensor[batch_idx]
                batch_y = y_tensor[batch_idx]

                optimizer.zero_grad()
                outputs = self.lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        self.baselines['lstm'] = self.lstm_model

    def evaluate_baselines(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate all baselines on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of results per baseline
        """
        results = {}

        for name, model in self.baselines.items():
            print(f"  Evaluating {name}...")

            start_time = time.perf_counter()

            if name == 'lstm':
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test)
                    outputs = model(X_tensor)
                    y_pred = outputs.argmax(dim=1).numpy()
                    y_scores = torch.softmax(outputs, dim=1)[:, 1].numpy()
            else:
                y_pred = model.predict(X_test)
                if hasattr(model, 'predict_proba'):
                    y_scores = model.predict_proba(X_test)[:, 1]
                else:
                    y_scores = y_pred.astype(float)

            inference_time = (time.perf_counter() - start_time) / len(X_test)

            # Calculate metrics
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'inference_time_ms': inference_time * 1000
            }

        self.results = results
        return results

    def compare_with_hybrid(
        self,
        hybrid_results: Dict
    ) -> Dict:
        """
        Compare baseline results with hybrid model.

        Args:
            hybrid_results: Results from hybrid detector

        Returns:
            Comparison summary
        """
        all_results = dict(self.results)
        all_results['hybrid_drl'] = hybrid_results

        # Create comparison table
        comparison = {
            'accuracy': {},
            'precision': {},
            'recall': {},
            'f1': {},
            'inference_time_ms': {}
        }

        for model_name, metrics in all_results.items():
            for metric_name in comparison.keys():
                if metric_name in metrics:
                    comparison[metric_name][model_name] = metrics[metric_name]

        # Calculate rankings
        rankings = {}
        for metric_name, values in comparison.items():
            if metric_name == 'inference_time_ms':
                # Lower is better for inference time
                sorted_models = sorted(values.items(), key=lambda x: x[1])
            else:
                # Higher is better for other metrics
                sorted_models = sorted(values.items(), key=lambda x: x[1], reverse=True)

            rankings[metric_name] = [m[0] for m in sorted_models]

        # Calculate improvement over baselines
        improvements = {}
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            if 'hybrid_drl' in comparison[metric_name]:
                hybrid_value = comparison[metric_name]['hybrid_drl']
                baseline_avg = np.mean([
                    v for k, v in comparison[metric_name].items()
                    if k != 'hybrid_drl'
                ])
                improvements[metric_name] = (hybrid_value - baseline_avg) / baseline_avg * 100

        return {
            'comparison': comparison,
            'rankings': rankings,
            'hybrid_improvements': improvements
        }

    def generate_report(self, comparison_results: Dict) -> str:
        """
        Generate text report of comparison results.

        Args:
            comparison_results: Results from compare_with_hybrid()

        Returns:
            Report string
        """
        report = []
        report.append("=" * 70)
        report.append("ZERODAY-DRL PERFORMANCE COMPARISON REPORT")
        report.append("=" * 70)
        report.append("")

        # Comparison table
        report.append("PERFORMANCE METRICS:")
        report.append("-" * 70)

        metrics = comparison_results['comparison']
        models = list(metrics['accuracy'].keys())

        # Header
        header = f"{'Model':<20}"
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            header += f"{metric.capitalize():<12}"
        report.append(header)
        report.append("-" * 70)

        # Data rows
        for model in models:
            row = f"{model:<20}"
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                value = metrics[metric].get(model, 0)
                row += f"{value:<12.4f}"
            report.append(row)

        report.append("")
        report.append("RANKINGS (Best to Worst):")
        report.append("-" * 70)

        for metric, ranking in comparison_results['rankings'].items():
            report.append(f"  {metric}: {' > '.join(ranking)}")

        report.append("")
        report.append("HYBRID MODEL IMPROVEMENTS OVER BASELINES:")
        report.append("-" * 70)

        for metric, improvement in comparison_results['hybrid_improvements'].items():
            sign = "+" if improvement >= 0 else ""
            report.append(f"  {metric}: {sign}{improvement:.2f}%")

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)
