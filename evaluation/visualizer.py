"""
Result Visualizer for ZeroDay-DRL.
Generates plots for training curves, ROC, PR curves, and comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os


class ResultVisualizer:
    """
    Visualization tools for ZeroDay-DRL results.
    """

    def __init__(self, save_dir: str = 'results/plots'):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = plt.cm.Set2.colors

    def plot_training_curves(
        self,
        history: Dict,
        title: str = "Training Progress",
        filename: str = "training_curves.png"
    ):
        """
        Plot training curves (reward, loss, accuracy).

        Args:
            history: Training history dictionary
            title: Plot title
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Reward curve
        if 'reward' in history and len(history['reward']) > 0:
            ax = axes[0, 0]
            episodes = range(len(history['reward']))
            ax.plot(episodes, history['reward'], alpha=0.3, color=self.colors[0])

            # Moving average
            window = min(50, len(history['reward']) // 5)
            if window > 1:
                ma = np.convolve(history['reward'], np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(history['reward'])), ma,
                       color=self.colors[0], linewidth=2, label='Moving Avg')

            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Episode Reward')
            ax.legend()

        # Loss curve
        if 'drl_loss' in history and len(history['drl_loss']) > 0:
            ax = axes[0, 1]
            ax.plot(history['drl_loss'], color=self.colors[1], alpha=0.5)

            window = min(50, len(history['drl_loss']) // 5)
            if window > 1:
                ma = np.convolve(history['drl_loss'], np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(history['drl_loss'])), ma,
                       color=self.colors[1], linewidth=2)

            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title('DRL Loss')

        # Accuracy curve
        if 'accuracy' in history and len(history['accuracy']) > 0:
            ax = axes[1, 0]
            ax.plot(history['accuracy'], color=self.colors[2], alpha=0.5)

            window = min(50, len(history['accuracy']) // 5)
            if window > 1:
                ma = np.convolve(history['accuracy'], np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(history['accuracy'])), ma,
                       color=self.colors[2], linewidth=2)

            ax.set_xlabel('Episode')
            ax.set_ylabel('Accuracy')
            ax.set_title('Detection Accuracy')
            ax.set_ylim(0, 1)

        # Detection rate & FPR
        ax = axes[1, 1]
        if 'detection_rate' in history and len(history['detection_rate']) > 0:
            ax.plot(history['detection_rate'], color=self.colors[3],
                   alpha=0.7, label='Detection Rate')
        if 'false_positive_rate' in history and len(history['false_positive_rate']) > 0:
            ax.plot(history['false_positive_rate'], color=self.colors[4],
                   alpha=0.7, label='FPR')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Rate')
        ax.set_title('Detection Rate & FPR')
        ax.set_ylim(0, 1)
        ax.legend()

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        title: str = "ROC Curve",
        filename: str = "roc_curve.png"
    ):
        """
        Plot ROC curve.

        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_score: Area under ROC curve
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(fpr, tpr, color=self.colors[0], linewidth=2,
               label=f'ROC Curve (AUC = {auc_score:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

        ax.fill_between(fpr, tpr, alpha=0.3, color=self.colors[0])

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_pr_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        auc_score: float,
        title: str = "Precision-Recall Curve",
        filename: str = "pr_curve.png"
    ):
        """
        Plot Precision-Recall curve.

        Args:
            precision: Precision values
            recall: Recall values
            auc_score: Area under PR curve
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(recall, precision, color=self.colors[1], linewidth=2,
               label=f'PR Curve (AUC = {auc_score:.4f})')

        ax.fill_between(recall, precision, alpha=0.3, color=self.colors[1])

        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str] = ['Normal', 'Botnet'],
        title: str = "Confusion Matrix",
        filename: str = "confusion_matrix.png"
    ):
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix array
            labels: Class labels
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   ax=ax, annot_kws={'size': 14})

        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_comparison_bars(
        self,
        comparison: Dict,
        title: str = "Model Comparison",
        filename: str = "comparison.png"
    ):
        """
        Plot bar chart comparing models.

        Args:
            comparison: Comparison dictionary from BaselineComparator
            title: Plot title
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = ['accuracy', 'precision', 'recall', 'f1']

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            if metric in comparison:
                models = list(comparison[metric].keys())
                values = list(comparison[metric].values())

                # Sort by value
                sorted_idx = np.argsort(values)[::-1]
                models = [models[i] for i in sorted_idx]
                values = [values[i] for i in sorted_idx]

                colors = [self.colors[i % len(self.colors)] for i in range(len(models))]

                bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1)

                # Add value labels
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=10)

                ax.set_ylabel(metric.capitalize(), fontsize=12)
                ax.set_title(f'{metric.capitalize()} Comparison', fontsize=12, fontweight='bold')
                ax.set_ylim(0, 1.1)
                ax.tick_params(axis='x', rotation=45)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_latency_vs_accuracy(
        self,
        results: Dict,
        title: str = "Latency vs Accuracy Trade-off",
        filename: str = "latency_accuracy.png"
    ):
        """
        Plot latency vs accuracy scatter plot.

        Args:
            results: Dictionary with model results
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        for idx, (name, metrics) in enumerate(results.items()):
            accuracy = metrics.get('accuracy', 0)
            latency = metrics.get('inference_time_ms', 0)

            ax.scatter(latency, accuracy, s=200, c=[self.colors[idx % len(self.colors)]],
                      label=name, edgecolors='black', linewidth=1, zorder=5)

            ax.annotate(name, (latency, accuracy), xytext=(10, 5),
                       textcoords='offset points', fontsize=10)

        ax.set_xlabel('Inference Time (ms)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_zero_day_adaptation(
        self,
        adaptation_results: Dict,
        title: str = "Zero-Day Adaptation Performance",
        filename: str = "zero_day_adaptation.png"
    ):
        """
        Plot zero-day detection adaptation curve.

        Args:
            adaptation_results: Adaptation metrics over time
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        samples = list(range(len(adaptation_results.get('detection_rates', []))))
        rates = adaptation_results.get('detection_rates', [])

        ax.plot(samples, rates, color=self.colors[0], linewidth=2, marker='o')
        ax.axhline(y=adaptation_results.get('initial_rate', 0), color='red',
                  linestyle='--', label='Initial Rate')
        ax.axhline(y=adaptation_results.get('final_rate', 0), color='green',
                  linestyle='--', label='Final Rate')

        ax.set_xlabel('Adaptation Samples', fontsize=12)
        ax.set_ylabel('Detection Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def create_summary_dashboard(
        self,
        training_history: Dict,
        test_metrics: Dict,
        comparison_results: Dict,
        filename: str = "dashboard.png"
    ):
        """
        Create comprehensive dashboard with all visualizations.

        Args:
            training_history: Training history
            test_metrics: Test evaluation metrics
            comparison_results: Model comparison results
            filename: Output filename
        """
        fig = plt.figure(figsize=(20, 15))

        # Training reward
        ax1 = fig.add_subplot(3, 3, 1)
        if 'reward' in training_history:
            ax1.plot(training_history['reward'], alpha=0.3, color=self.colors[0])
            window = min(50, len(training_history['reward']) // 5)
            if window > 1:
                ma = np.convolve(training_history['reward'],
                               np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(training_history['reward'])), ma,
                        color=self.colors[0], linewidth=2)
        ax1.set_title('Training Reward', fontweight='bold')
        ax1.set_xlabel('Episode')

        # Training accuracy
        ax2 = fig.add_subplot(3, 3, 2)
        if 'accuracy' in training_history:
            ax2.plot(training_history['accuracy'], color=self.colors[1], alpha=0.7)
        ax2.set_title('Training Accuracy', fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylim(0, 1)

        # Detection metrics
        ax3 = fig.add_subplot(3, 3, 3)
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        values = [test_metrics.get(m, 0) for m in metrics_to_plot]
        bars = ax3.bar(metrics_to_plot, values, color=self.colors[:4])
        ax3.set_title('Test Performance', fontweight='bold')
        ax3.set_ylim(0, 1)
        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=9)

        # ROC curve
        ax4 = fig.add_subplot(3, 3, 4)
        if 'roc_curve' in test_metrics:
            fpr, tpr = test_metrics['roc_curve']
            ax4.plot(fpr, tpr, color=self.colors[0], linewidth=2)
            ax4.plot([0, 1], [0, 1], 'k--')
            ax4.fill_between(fpr, tpr, alpha=0.3)
        ax4.set_title(f'ROC (AUC={test_metrics.get("roc_auc", 0):.3f})', fontweight='bold')
        ax4.set_xlabel('FPR')
        ax4.set_ylabel('TPR')

        # Confusion matrix
        ax5 = fig.add_subplot(3, 3, 5)
        cm = np.array([
            [test_metrics.get('true_negatives', 0), test_metrics.get('false_positives', 0)],
            [test_metrics.get('false_negatives', 0), test_metrics.get('true_positives', 0)]
        ])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                   xticklabels=['Normal', 'Botnet'], yticklabels=['Normal', 'Botnet'])
        ax5.set_title('Confusion Matrix', fontweight='bold')

        # Model comparison
        ax6 = fig.add_subplot(3, 3, 6)
        if 'comparison' in comparison_results and 'accuracy' in comparison_results['comparison']:
            models = list(comparison_results['comparison']['accuracy'].keys())
            accs = list(comparison_results['comparison']['accuracy'].values())
            colors = [self.colors[i % len(self.colors)] for i in range(len(models))]
            ax6.barh(models, accs, color=colors)
            ax6.set_xlim(0, 1)
        ax6.set_title('Model Accuracy Comparison', fontweight='bold')

        # FPR over training
        ax7 = fig.add_subplot(3, 3, 7)
        if 'false_positive_rate' in training_history:
            ax7.plot(training_history['false_positive_rate'], color=self.colors[4])
        ax7.set_title('FPR During Training', fontweight='bold')
        ax7.set_xlabel('Episode')
        ax7.set_ylim(0, 1)

        # Detection rate over training
        ax8 = fig.add_subplot(3, 3, 8)
        if 'detection_rate' in training_history:
            ax8.plot(training_history['detection_rate'], color=self.colors[3])
        ax8.set_title('Detection Rate During Training', fontweight='bold')
        ax8.set_xlabel('Episode')
        ax8.set_ylim(0, 1)

        # Summary text
        ax9 = fig.add_subplot(3, 3, 9)
        ax9.axis('off')
        summary_text = [
            "ZERODAY-DRL SUMMARY",
            "=" * 30,
            f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}",
            f"Detection Rate: {test_metrics.get('recall', 0):.4f}",
            f"False Positive Rate: {test_metrics.get('false_positive_rate', 0):.4f}",
            f"F1 Score: {test_metrics.get('f1', 0):.4f}",
            "",
            "Improvement over baselines:",
        ]
        if 'hybrid_improvements' in comparison_results:
            for metric, imp in comparison_results['hybrid_improvements'].items():
                sign = "+" if imp >= 0 else ""
                summary_text.append(f"  {metric}: {sign}{imp:.2f}%")

        ax9.text(0.1, 0.9, "\n".join(summary_text), transform=ax9.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        plt.suptitle('ZeroDay-DRL Performance Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Dashboard saved to {os.path.join(self.save_dir, filename)}")
