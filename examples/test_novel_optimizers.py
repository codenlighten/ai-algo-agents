"""
Example: Testing novel optimizers against baselines
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from optimizers.novel_optimizers import (
    SecondOrderMomentumOptimizer,
    LookAheadWrapper,
    AdaptiveGradientClipping
)
from experiments.experiment_framework import (
    ExperimentRunner,
    ExperimentConfig,
    MinimalBenchmark
)
from utils.device_manager import get_device_manager


def simple_cnn():
    """Simple CNN for MNIST"""
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


def run_optimizer_comparison():
    """Compare novel optimizers against standard baselines"""
    
    print("="*80)
    print("OPTIMIZER COMPARISON EXPERIMENT")
    print("="*80)
    
    # Initialize device manager
    device_mgr = get_device_manager(verbose=True)
    device_mgr.optimize_for_training()
    
    # Get dataset
    dataset = MinimalBenchmark.simple_mnist_task()
    
    # Create experiment runner
    runner = ExperimentRunner(device_manager=device_mgr)
    
    # Define experiments
    experiments = [
        # Baseline: SGD
        ExperimentConfig(
            name="baseline_sgd",
            model_fn=simple_cnn,
            optimizer_fn=lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
            loss_fn=nn.CrossEntropyLoss,
            dataset=dataset,
            batch_size=128,
            num_epochs=10
        ),
        
        # Baseline: Adam
        ExperimentConfig(
            name="baseline_adam",
            model_fn=simple_cnn,
            optimizer_fn=lambda params: torch.optim.Adam(params, lr=0.001),
            loss_fn=nn.CrossEntropyLoss,
            dataset=dataset,
            batch_size=128,
            num_epochs=10
        ),
        
        # Novel: Second-Order Momentum
        ExperimentConfig(
            name="novel_second_order_momentum",
            model_fn=simple_cnn,
            optimizer_fn=lambda params: SecondOrderMomentumOptimizer(
                params, lr=0.001, curvature_momentum=0.9
            ),
            loss_fn=nn.CrossEntropyLoss,
            dataset=dataset,
            batch_size=128,
            num_epochs=10,
            use_mixed_precision=True  # Enable FP16 if supported
        ),
        
        # Novel: LookAhead + Adam
        ExperimentConfig(
            name="novel_lookahead_adam",
            model_fn=simple_cnn,
            optimizer_fn=lambda params: LookAheadWrapper(
                torch.optim.Adam(params, lr=0.001),
                la_steps=5,
                la_alpha=0.5
            ),
            loss_fn=nn.CrossEntropyLoss,
            dataset=dataset,
            batch_size=128,
            num_epochs=10,
            use_mixed_precision=True
        ),
        
        # Novel: Adaptive Gradient Clipping
        ExperimentConfig(
            name="novel_adaptive_clipping",
            model_fn=simple_cnn,
            optimizer_fn=lambda params: AdaptiveGradientClipping(
                params,
                base_optimizer_class=torch.optim.Adam,
                clip_factor=0.01,
                lr=0.001
            ),
            loss_fn=nn.CrossEntropyLoss,
            dataset=dataset,
            batch_size=128,
            num_epochs=10,
            use_mixed_precision=True
        )
    ]
    
    # Run all experiments
    for config in experiments:
        runner.run_experiment(config)
    
    # Print summary
    runner.print_summary()
    
    # Compare to baseline
    print("\n" + "="*80)
    print("COMPARISON TO ADAM BASELINE")
    print("="*80)
    
    comparisons = runner.compare_experiments("baseline_adam")
    
    for name, metrics in comparisons.items():
        print(f"\n{name}:")
        print(f"  Loss improvement: {metrics['loss_improvement']:.2f}%")
        print(f"  Accuracy improvement: {metrics['accuracy_improvement']:.2f}%")
        print(f"  Speedup: {metrics['speedup']:.2f}x")
        print(f"  Memory overhead: {metrics['memory_overhead']:.2f}%")
    
    # Save results
    runner.save_results("optimizer_comparison.json")
    
    return runner


if __name__ == "__main__":
    # Note: This requires PyTorch and torchvision installed
    # Run with: python examples/test_novel_optimizers.py
    
    try:
        results = run_optimizer_comparison()
        print("\nExperiment completed successfully!")
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        print("\nMake sure you have installed all requirements:")
        print("  pip install torch torchvision")
