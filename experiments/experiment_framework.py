"""
Experimental validation framework for testing research proposals
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
import time
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.device_manager import DeviceManager, get_device_manager


@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    name: str
    model_fn: Callable
    optimizer_fn: Callable
    loss_fn: Callable
    dataset: Dataset
    batch_size: int = 32
    num_epochs: int = 10
    device: Optional[str] = None  # Auto-detect if None
    log_interval: int = 100
    seed: int = 42
    use_mixed_precision: bool = False  # Enable FP16 training if supported


@dataclass
class ExperimentResult:
    """Results from an experiment"""
    config_name: str
    final_loss: float
    final_accuracy: float
    training_time: float
    avg_epoch_time: float
    losses_per_epoch: List[float]
    accuracies_per_epoch: List[float]
    memory_usage_mb: float
    
    def to_dict(self) -> Dict:
        return {
            'config_name': self.config_name,
            'final_loss': self.final_loss,
            'final_accuracy': self.final_accuracy,
            'training_time': self.training_time,
            'avg_epoch_time': self.avg_epoch_time,
            'losses_per_epoch': self.losses_per_epoch,
            'accuracies_per_epoch': self.accuracies_per_epoch,
            'memory_usage_mb': self.memory_usage_mb
        }


class ExperimentRunner:
    """Run and compare experiments"""
    
    def __init__(self, save_dir: str = "experiments/results", device_manager: Optional[DeviceManager] = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, ExperimentResult] = {}
        
        # Initialize device manager
        self.device_manager = device_manager or get_device_manager(verbose=True)
        self.device = self.device_manager.get_device()
        
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment"""
        print(f"\nRunning experiment: {config.name}")
        
        # Set seed for reproducibility
        torch.manual_seed(config.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(config.seed)
        
        # Setup device (use config device if specified, otherwise use manager's device)
        device = torch.device(config.device) if config.device else self.device
        
        # Initialize model and move to device
        model = config.model_fn()
        model = self.device_manager.to_device(model)
        
        optimizer = config.optimizer_fn(model.parameters())
        criterion = config.loss_fn()
        
        # Setup mixed precision training if enabled and supported
        scaler = None
        if config.use_mixed_precision and device.type == 'cuda':
            if self.device_manager.enable_mixed_precision():
                scaler = torch.cuda.amp.GradScaler()
                print(f"Mixed precision (FP16) training enabled")
            else:
                print(f"Mixed precision not supported on this GPU")
        
        # Data loader
        train_loader = DataLoader(
            config.dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        
        # Training loop
        losses_per_epoch = []
        accuracies_per_epoch = []
        start_time = time.time()
        
        for epoch in range(config.num_epochs):
            epoch_start = time.time()
            model.train()
            
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Mixed precision training
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                    
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard training
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                if batch_idx % config.log_interval == 0:
                    print(f"Epoch {epoch+1}/{config.num_epochs} "
                          f"[{batch_idx}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            accuracy = 100. * correct / total
            losses_per_epoch.append(avg_loss)
            accuracies_per_epoch.append(accuracy)
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s - "
                  f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        total_time = time.time() - start_time
        
        # Memory usage
        if device.type == 'cuda':
            memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            torch.cuda.reset_peak_memory_stats(device)
            
            # Print memory stats
            self.device_manager.print_memory_stats()
        else:
            memory_mb = 0
        
        # Create result
        result = ExperimentResult(
            config_name=config.name,
            final_loss=losses_per_epoch[-1],
            final_accuracy=accuracies_per_epoch[-1],
            training_time=total_time,
            avg_epoch_time=total_time / config.num_epochs,
            losses_per_epoch=losses_per_epoch,
            accuracies_per_epoch=accuracies_per_epoch,
            memory_usage_mb=memory_mb
        )
        
        self.results[config.name] = result
        return result
    
    def compare_experiments(self, baseline_name: str) -> Dict[str, Any]:
        """Compare all experiments to a baseline"""
        if baseline_name not in self.results:
            raise ValueError(f"Baseline {baseline_name} not found in results")
        
        baseline = self.results[baseline_name]
        comparisons = {}
        
        for name, result in self.results.items():
            if name == baseline_name:
                continue
            
            comparisons[name] = {
                'loss_improvement': (baseline.final_loss - result.final_loss) / baseline.final_loss * 100,
                'accuracy_improvement': result.final_accuracy - baseline.final_accuracy,
                'speedup': baseline.training_time / result.training_time,
                'memory_overhead': (result.memory_usage_mb - baseline.memory_usage_mb) / baseline.memory_usage_mb * 100
            }
        
        return comparisons
    
    def save_results(self, filename: str = "results.json"):
        """Save all results to file"""
        filepath = self.save_dir / filename
        results_dict = {
            name: result.to_dict()
            for name, result in self.results.items()
        }
        
        with open(filepath, 'w') as f:
            json.dumps(results_dict, f, indent=2)
        
        print(f"\n💾 Results saved to {filepath}")
        return filepath
    
    def print_summary(self):
        """Print summary of all experiments"""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        for name, result in self.results.items():
            print(f"\n{name}:")
            print(f"  Final Loss: {result.final_loss:.4f}")
            print(f"  Final Accuracy: {result.final_accuracy:.2f}%")
            print(f"  Training Time: {result.training_time:.2f}s")
            print(f"  Avg Epoch Time: {result.avg_epoch_time:.2f}s")
            print(f"  Memory Usage: {result.memory_usage_mb:.2f} MB")


class MinimalBenchmark:
    """Minimal benchmark for quick validation"""
    
    @staticmethod
    def simple_mnist_task():
        """Create a simple MNIST-like task"""
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset = datasets.MNIST(
            'data/mnist',
            train=True,
            download=True,
            transform=transform
        )
        
        # Use subset for quick experiments
        subset_indices = torch.randperm(len(dataset))[:5000]
        subset = torch.utils.data.Subset(dataset, subset_indices)
        
        return subset
    
    @staticmethod
    def simple_model():
        """Simple baseline model"""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    @staticmethod
    def create_baseline_config() -> ExperimentConfig:
        """Create baseline experiment config"""
        return ExperimentConfig(
            name="baseline_sgd",
            model_fn=MinimalBenchmark.simple_model,
            optimizer_fn=lambda params: torch.optim.SGD(params, lr=0.01),
            loss_fn=nn.CrossEntropyLoss,
            dataset=MinimalBenchmark.simple_mnist_task(),
            batch_size=64,
            num_epochs=5
        )
