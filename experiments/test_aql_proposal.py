"""
Test implementation of Adaptive Query-Based Learning (AQL) proposal
Based on: research/sessions/so_much_energy_is_spent_on_creating_ai_models...json

This experiment validates the AQL concept using MNIST dataset.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.device_manager import DeviceManager, get_device_manager


class SimpleNN(nn.Module):
    """Simple neural network for AQL experiments"""
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


def uncertainty_estimate(model, data, device):
    """
    Estimate uncertainty using Monte Carlo Dropout
    Higher variance = higher uncertainty
    """
    model.train()  # Keep dropout active
    n_samples = 10
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(data.to(device))
            probs = torch.softmax(outputs, dim=1)
            predictions.append(probs)
    
    predictions = torch.stack(predictions)
    # Use variance across predictions as uncertainty measure
    uncertainty = torch.var(predictions, dim=0).sum(dim=1)
    return uncertainty


def train_aql(model, train_loader, test_loader, device, n_epochs=10, n_queries=100):
    """Train model using Adaptive Query-Based Learning"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Get all training data for querying
    all_train_data = []
    all_train_labels = []
    for data, labels in train_loader:
        all_train_data.append(data)
        all_train_labels.append(labels)
    all_train_data = torch.cat(all_train_data)
    all_train_labels = torch.cat(all_train_labels)
    
    print(f"\nTraining with AQL (querying {n_queries} samples per epoch)...")
    
    epoch_times = []
    accuracies = []
    
    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        
        # Standard training on full dataset for first pass
        total_loss = 0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Adaptive querying phase
        print(f"  Epoch {epoch+1}: Selecting uncertain samples...", end='')
        uncertainties = uncertainty_estimate(model, all_train_data, device)
        query_indices = torch.argsort(uncertainties, descending=True)[:n_queries].cpu()
        
        # Train on queried (most uncertain) samples
        selected_data = all_train_data[query_indices].to(device)
        selected_labels = all_train_labels[query_indices].to(device)
        
        optimizer.zero_grad()
        outputs = model(selected_data)
        loss = criterion(outputs, selected_labels)
        loss.backward()
        optimizer.step()
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        
        print(f" Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s")
    
    return {
        'epoch_times': epoch_times,
        'accuracies': accuracies,
        'final_accuracy': accuracies[-1],
        'avg_epoch_time': np.mean(epoch_times)
    }


def train_baseline(model, train_loader, test_loader, device, n_epochs=10):
    """Train model using standard SGD/Adam (baseline)"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining baseline (standard training)...")
    
    epoch_times = []
    accuracies = []
    
    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        
        total_loss = 0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        
        print(f"  Epoch {epoch+1}: Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s")
    
    return {
        'epoch_times': epoch_times,
        'accuracies': accuracies,
        'final_accuracy': accuracies[-1],
        'avg_epoch_time': np.mean(epoch_times)
    }


def main():
    print("="*70)
    print("Testing Adaptive Query-Based Learning (AQL) Proposal")
    print("="*70)
    
    # Setup device
    device_manager = get_device_manager(verbose=True)
    device = device_manager.get_device()
    print(f"\nUsing device: {device}")
    
    # Load MNIST dataset
    print("\nLoading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data/mnist',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data/mnist',
        train=False,
        download=True,
        transform=transform
    )
    
    # Use subset for faster experiments
    train_subset = Subset(train_dataset, range(10000))
    test_subset = Subset(test_dataset, range(2000))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=1000, shuffle=False)
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Test samples: {len(test_subset)}")
    
    # Experiment parameters
    n_epochs = 10
    n_queries = 500  # Query 500 most uncertain samples per epoch
    
    # Test 1: Baseline (Standard Training)
    print("\n" + "="*70)
    print("EXPERIMENT 1: Baseline (Standard Adam Optimizer)")
    print("="*70)
    model_baseline = SimpleNN().to(device)
    results_baseline = train_baseline(model_baseline, train_loader, test_loader, device, n_epochs)
    
    # Test 2: AQL Training
    print("\n" + "="*70)
    print("EXPERIMENT 2: Adaptive Query-Based Learning (AQL)")
    print("="*70)
    model_aql = SimpleNN().to(device)
    results_aql = train_aql(model_aql, train_loader, test_loader, device, n_epochs, n_queries)
    
    # Compare results
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    print(f"\nBaseline (Standard Training):")
    print(f"  Final Accuracy: {results_baseline['final_accuracy']:.2f}%")
    print(f"  Avg Epoch Time: {results_baseline['avg_epoch_time']:.2f}s")
    print(f"  Total Time: {sum(results_baseline['epoch_times']):.2f}s")
    
    print(f"\nAQL (Adaptive Query-Based Learning):")
    print(f"  Final Accuracy: {results_aql['final_accuracy']:.2f}%")
    print(f"  Avg Epoch Time: {results_aql['avg_epoch_time']:.2f}s")
    print(f"  Total Time: {sum(results_aql['epoch_times']):.2f}s")
    
    print(f"\nComparison:")
    accuracy_diff = results_aql['final_accuracy'] - results_baseline['final_accuracy']
    time_diff = results_baseline['avg_epoch_time'] - results_aql['avg_epoch_time']
    speedup = results_baseline['avg_epoch_time'] / results_aql['avg_epoch_time']
    
    print(f"  Accuracy Difference: {accuracy_diff:+.2f}%")
    print(f"  Time Difference: {time_diff:+.2f}s per epoch")
    print(f"  Speedup Factor: {speedup:.2f}x")
    
    # Evaluation
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    if accuracy_diff > -1.0:  # Within 1% accuracy
        print("✅ AQL maintains comparable accuracy to baseline")
    else:
        print("⚠️  AQL has lower accuracy than baseline")
    
    if time_diff > 0:
        print(f"✅ AQL is faster (saves {time_diff:.2f}s per epoch)")
    else:
        print(f"⚠️  AQL is slower (adds {abs(time_diff):.2f}s per epoch)")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The AQL approach demonstrates the concept of active learning in deep learning.
The uncertainty-based querying allows the model to focus on hard samples.

Key Observations:
1. AQL adds computational overhead for uncertainty estimation
2. The adaptive querying mechanism targets informative samples
3. Performance depends on uncertainty estimation quality
4. Further optimization needed for production use

Next Steps:
- Optimize uncertainty estimation (reduce overhead)
- Test on larger datasets (CIFAR-10, ImageNet)
- Compare different uncertainty measures
- Implement batch parallelization for queries
""")


if __name__ == "__main__":
    main()
