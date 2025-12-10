"""
Example: Testing novel loss functions
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from loss_functions.novel_losses import (
    ConfidencePenalizedCrossEntropy,
    FocalLoss,
    CurriculumLoss,
    AdaptiveWingLoss
)
from experiments.experiment_framework import (
    ExperimentRunner,
    ExperimentConfig,
    MinimalBenchmark
)
from utils.device_manager import get_device_manager


def run_loss_function_comparison():
    """Compare novel loss functions against standard baselines"""
    
    print("="*80)
    print("LOSS FUNCTION COMPARISON EXPERIMENT")
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
        # Baseline: Cross-Entropy
        ExperimentConfig(
            name="baseline_cross_entropy",
            model_fn=MinimalBenchmark.simple_model,
            optimizer_fn=lambda params: torch.optim.Adam(params, lr=0.001),
            loss_fn=nn.CrossEntropyLoss,
            dataset=dataset,
            batch_size=128,
            num_epochs=10
        ),
        
        # Novel: Confidence-Penalized Cross-Entropy
        ExperimentConfig(
            name="novel_confidence_penalized",
            model_fn=MinimalBenchmark.simple_model,
            optimizer_fn=lambda params: torch.optim.Adam(params, lr=0.001),
            loss_fn=lambda: ConfidencePenalizedCrossEntropy(penalty_weight=0.1),
            dataset=dataset,
            batch_size=128,
            num_epochs=10
        ),
        
        # Novel: Focal Loss
        ExperimentConfig(
            name="novel_focal_loss",
            model_fn=MinimalBenchmark.simple_model,
            optimizer_fn=lambda params: torch.optim.Adam(params, lr=0.001),
            loss_fn=lambda: FocalLoss(gamma=2.0),
            dataset=dataset,
            batch_size=128,
            num_epochs=10
        ),
        
        # Novel: Curriculum Loss
        ExperimentConfig(
            name="novel_curriculum_loss",
            model_fn=MinimalBenchmark.simple_model,
            optimizer_fn=lambda params: torch.optim.Adam(params, lr=0.001),
            loss_fn=lambda: CurriculumLoss(
                nn.CrossEntropyLoss(reduction='none'),
                warmup_steps=500
            ),
            dataset=dataset,
            batch_size=128,
            num_epochs=10
        )
    ]
    
    # Run all experiments
    for config in experiments:
        runner.run_experiment(config)
    
    # Print summary
    runner.print_summary()
    
    # Compare to baseline
    print("\n" + "="*80)
    print("COMPARISON TO CROSS-ENTROPY BASELINE")
    print("="*80)
    
    comparisons = runner.compare_experiments("baseline_cross_entropy")
    
    for name, metrics in comparisons.items():
        print(f"\n{name}:")
        print(f"  Loss improvement: {metrics['loss_improvement']:.2f}%")
        print(f"  Accuracy improvement: {metrics['accuracy_improvement']:.2f}%")
        print(f"  Speedup: {metrics['speedup']:.2f}x")
        print(f"  Memory overhead: {metrics['memory_overhead']:.2f}%")
    
    # Save results
    runner.save_results("loss_function_comparison.json")
    
    return runner


def test_calibration():
    """Test calibration of different loss functions"""
    print("\n" + "="*80)
    print("CALIBRATION ANALYSIS")
    print("="*80)
    
    # This would require implementing ECE (Expected Calibration Error) metric
    # Placeholder for demonstration
    print("\nCalibration metrics would be computed here:")
    print("  - Expected Calibration Error (ECE)")
    print("  - Maximum Calibration Error (MCE)")
    print("  - Brier Score")
    print("  - Negative Log Likelihood")
    
    print("\nExpected results:")
    print("  - Confidence-Penalized CE: Lower ECE, better calibration")
    print("  - Focal Loss: Improved minority class performance")
    print("  - Curriculum Loss: Faster initial convergence")


if __name__ == "__main__":
    try:
        results = run_loss_function_comparison()
        test_calibration()
        print("\n✅ Experiment completed successfully!")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        print("\nMake sure you have installed all requirements:")
        print("  pip install torch torchvision")
