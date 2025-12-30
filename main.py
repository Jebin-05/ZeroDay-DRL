#!/usr/bin/env python3
"""
ZeroDay-DRL: Main Entry Point
Hybrid Deep Reinforcement Learning and Few-Shot Meta-Learning Framework
for Early-Stage IoT Botnet Detection

Usage:
    python main.py --mode train         # Train the model
    python main.py --mode evaluate      # Evaluate trained model
    python main.py --mode demo          # Run demo detection
    python main.py --mode gui           # Launch GUI interface
"""

import argparse
import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config_loader import load_config, get_device
from utils.helpers import set_seed
from preprocessing.data_loader import DataLoader
from hybrid_model.hybrid_detector import HybridDetector, DetectionMode
from hybrid_model.trainer import HybridTrainer
from evaluation.metrics import MetricsCalculator
from evaluation.comparator import BaselineComparator
from evaluation.visualizer import ResultVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ZeroDay-DRL: IoT Botnet Detection Framework'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'evaluate', 'demo', 'gui', 'compare'],
        help='Operation mode'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--drl-algorithm',
        type=str,
        default='dqn',
        choices=['dqn', 'ppo'],
        help='DRL algorithm to use'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )

    parser.add_argument(
        '--num-episodes',
        type=int,
        default=None,
        help='Override number of training episodes'
    )

    parser.add_argument(
        '--data-source',
        type=str,
        default='iot23',
        choices=['iot23', 'synthetic'],
        help='Data source: iot23 (real IoT-23 dataset) or synthetic'
    )

    parser.add_argument(
        '--data-file',
        type=str,
        default='cleaned_data.csv',
        help='Data file name (for iot23 source)'
    )

    return parser.parse_args()


def train(args, config):
    """Train the hybrid detector."""
    print("=" * 60)
    print("ZeroDay-DRL Training")
    print("=" * 60)

    device = get_device(config)
    print(f"Using device: {device}")

    # Override episodes if specified
    if args.num_episodes:
        config['training']['num_episodes'] = args.num_episodes

    # Initialize data loader based on source
    if args.data_source == 'iot23':
        from preprocessing.iot23_loader import IoT23DataLoader
        data_loader = IoT23DataLoader(config, data_dir='data')
        data = data_loader.load_cleaned_data(args.data_file)
        print(f"Loaded IoT-23 dataset from {args.data_file}")
    else:
        data_loader = DataLoader(config)
        data = data_loader.load_synthetic_data()
        print("Using synthetic data")

    feature_dim = data['train'][0].shape[1]

    print(f"Feature dimension: {feature_dim}")
    print(f"Training samples: {len(data['train'][1])}")
    print(f"Validation samples: {len(data['val'][1])}")
    print(f"Test samples: {len(data['test'][1])}")

    # Initialize detector
    detector = HybridDetector(
        state_dim=feature_dim,
        action_dim=2,
        config=config,
        device=device,
        drl_algorithm=args.drl_algorithm
    )

    # Initialize trainer
    trainer = HybridTrainer(config, device)

    # Create output directories
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train
    results = trainer.train(detector, data_loader, checkpoint_dir)

    # Save final model
    detector.save(os.path.join(checkpoint_dir, 'final_model'))

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)

    test_features, test_labels = data['test']
    test_metrics = evaluate_detector(detector, test_features, test_labels)

    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  False Positive Rate: {test_metrics['false_positive_rate']:.4f}")

    # Generate visualizations
    visualizer = ResultVisualizer(os.path.join(args.output_dir, 'plots'))
    visualizer.plot_training_curves(results['history'])

    if 'roc_curve' in test_metrics:
        visualizer.plot_roc_curve(
            test_metrics['roc_curve'][0],
            test_metrics['roc_curve'][1],
            test_metrics['roc_auc']
        )

    print(f"\nResults saved to {args.output_dir}")

    return detector, results


def evaluate(args, config):
    """Evaluate a trained model."""
    print("=" * 60)
    print("ZeroDay-DRL Evaluation")
    print("=" * 60)

    device = get_device(config)

    # Load data based on source
    if args.data_source == 'iot23':
        from preprocessing.iot23_loader import IoT23DataLoader
        data_loader = IoT23DataLoader(config, data_dir='data')
        data = data_loader.load_cleaned_data(args.data_file)
    else:
        data_loader = DataLoader(config)
        data = data_loader.load_synthetic_data()

    feature_dim = data['train'][0].shape[1]

    # Initialize and load detector
    detector = HybridDetector(
        state_dim=feature_dim,
        action_dim=2,
        config=config,
        device=device,
        drl_algorithm=args.drl_algorithm
    )

    if args.checkpoint:
        detector.load(args.checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'final_model')
        if os.path.exists(checkpoint_path):
            detector.load(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print("No checkpoint found. Please train the model first.")
            return

    # Evaluate on test set
    test_features, test_labels = data['test']
    test_metrics = evaluate_detector(detector, test_features, test_labels)

    print(f"\nTest Results:")
    for metric, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")

    # Compare with baselines
    print("\n" + "-" * 40)
    print("Comparing with Baselines")
    print("-" * 40)

    comparator = BaselineComparator(config)
    train_features, train_labels = data['train']
    comparator.train_baselines(train_features, train_labels)
    baseline_results = comparator.evaluate_baselines(test_features, test_labels)

    comparison = comparator.compare_with_hybrid({
        'accuracy': test_metrics['accuracy'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'f1': test_metrics['f1']
    })

    print(comparator.generate_report(comparison))

    # Generate visualizations
    visualizer = ResultVisualizer(os.path.join(args.output_dir, 'plots'))
    visualizer.plot_comparison_bars(comparison['comparison'])

    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    predictions = []
    for i in range(len(test_features)):
        result = detector.detect(test_features[i], training=False)
        predictions.append(result['prediction'])

    cm = confusion_matrix(test_labels, predictions)
    visualizer.plot_confusion_matrix(cm)


def evaluate_detector(detector, features, labels):
    """Helper function to evaluate detector."""
    predictions = []
    confidences = []

    for i in range(len(features)):
        result = detector.detect(features[i], training=False)
        predictions.append(result['prediction'])
        confidences.append(result['confidence'])

    predictions = np.array(predictions)
    confidences = np.array(confidences)

    calculator = MetricsCalculator()
    return calculator.calculate_from_arrays(labels, predictions, confidences)


def demo(args, config):
    """Run demonstration of the detector."""
    print("=" * 60)
    print("ZeroDay-DRL Demo")
    print("=" * 60)

    device = get_device(config)

    # Load data based on source
    if args.data_source == 'iot23':
        from preprocessing.iot23_loader import IoT23DataLoader
        data_loader = IoT23DataLoader(config, data_dir='data')
        data = data_loader.load_cleaned_data(args.data_file)
        print(f"Using IoT-23 dataset: {args.data_file}")
    else:
        data_loader = DataLoader(config)
        data = data_loader.load_synthetic_data()
        print("Using synthetic data")

    feature_dim = data['train'][0].shape[1]

    # Initialize detector
    detector = HybridDetector(
        state_dim=feature_dim,
        action_dim=2,
        config=config,
        device=device,
        drl_algorithm=args.drl_algorithm
    )

    # Try to load checkpoint
    checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'final_model')
    if os.path.exists(checkpoint_path):
        detector.load(checkpoint_path)
        print(f"Loaded trained model from {checkpoint_path}")
    else:
        print("No trained model found. Initializing with few-shot prototypes...")
        normal_samples = data['train'][0][data['train'][1] == 0]
        botnet_samples = data['train'][0][data['train'][1] == 1]
        detector.initialize_few_shot(normal_samples[:10], botnet_samples[:10])

    # Demo detection on test samples
    test_features, test_labels = data['test']

    print("\nDetection Demo:")
    print("-" * 50)

    for i in range(min(20, len(test_features))):
        result = detector.detect(test_features[i], training=False)

        label_str = "BOTNET" if test_labels[i] == 1 else "NORMAL"
        pred_str = "BOTNET" if result['prediction'] == 1 else "NORMAL"
        correct = "✓" if result['prediction'] == test_labels[i] else "✗"

        print(f"Sample {i+1:3d}: True={label_str:7s} | Pred={pred_str:7s} | "
              f"Conf={result['confidence']:.3f} | Source={result['detection_source']:15s} | {correct}")

    # Show metrics
    print("\n" + "-" * 50)
    metrics = detector.get_metrics()
    print(f"Total detections: {metrics['total_detections']}")
    print(f"Zero-day detections: {metrics['zero_day_detections']}")
    print(f"Adaptation triggers: {metrics['adaptation_triggers']}")


def run_gui(args, config):
    """Launch the GUI interface."""
    print("Launching ZeroDay-DRL GUI...")

    try:
        # Use the simple, user-friendly GUI
        from gui.simple_gui import SimpleZeroDayGUI
        app = SimpleZeroDayGUI()
        app.run()
    except ImportError as e:
        print(f"GUI dependencies not available: {e}")
        print("Please install: pip install customtkinter pillow")
        print("")
        print("On Linux, you may also need: sudo apt install python3-tk")


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    set_seed(config['training']['seed'])

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run selected mode
    if args.mode == 'train':
        train(args, config)
    elif args.mode == 'evaluate':
        evaluate(args, config)
    elif args.mode == 'demo':
        demo(args, config)
    elif args.mode == 'gui':
        run_gui(args, config)
    elif args.mode == 'compare':
        evaluate(args, config)  # Compare mode is same as evaluate with baseline comparison


if __name__ == '__main__':
    main()
