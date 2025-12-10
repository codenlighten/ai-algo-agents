"""
Streaming AQL Data Selection
Memory-efficient sample selection without loading full dataset
"""
import torch
import torch.nn as nn
from typing import Iterator, Tuple, Optional, List
from dataclasses import dataclass
import numpy as np


@dataclass
class SelectionBuffer:
    """
    Maintains top-k uncertain samples using a min-heap
    Memory: O(k) instead of O(n)
    """
    capacity: int
    uncertainties: List[float]
    indices: List[int]
    samples: List[torch.Tensor]
    targets: List[torch.Tensor]
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.uncertainties = []
        self.indices = []
        self.samples = []
        self.targets = []
    
    def add(self, uncertainty: float, index: int, 
            sample: torch.Tensor, target: torch.Tensor):
        """Add sample to buffer, maintaining top-k by uncertainty"""
        
        if len(self.uncertainties) < self.capacity:
            # Buffer not full, add directly
            self.uncertainties.append(uncertainty)
            self.indices.append(index)
            self.samples.append(sample.cpu())
            self.targets.append(target.cpu())
            
        elif uncertainty > min(self.uncertainties):
            # Replace minimum if new sample is more uncertain
            min_idx = self.uncertainties.index(min(self.uncertainties))
            self.uncertainties[min_idx] = uncertainty
            self.indices[min_idx] = index
            self.samples[min_idx] = sample.cpu()
            self.targets[min_idx] = target.cpu()
    
    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of most uncertain samples"""
        if len(self.samples) == 0:
            return None, None
        
        # Sort by uncertainty (descending)
        sorted_indices = sorted(
            range(len(self.uncertainties)), 
            key=lambda i: self.uncertainties[i], 
            reverse=True
        )
        
        # Take top batch_size samples
        batch_indices = sorted_indices[:min(batch_size, len(sorted_indices))]
        
        samples = torch.stack([self.samples[i] for i in batch_indices])
        targets = torch.stack([self.targets[i] for i in batch_indices])
        
        return samples, targets
    
    def clear(self):
        """Clear buffer"""
        self.uncertainties.clear()
        self.indices.clear()
        self.samples.clear()
        self.targets.clear()
    
    def __len__(self):
        return len(self.uncertainties)


class StreamingAQL:
    """
    Streaming Active Query-Based Learning
    
    Key features:
    - Process dataset in chunks (streaming)
    - Maintain top-k uncertain samples in memory
    - Select batches from buffer
    - Memory efficient: O(k) instead of O(n)
    
    Usage:
        selector = StreamingAQL(
            model=model,
            uncertainty_estimator=laplacian,
            buffer_size=10000,
            selection_ratio=0.1
        )
        
        # Stream through data
        for chunk in dataset_chunks:
            selector.process_chunk(chunk)
        
        # Get selected samples
        for batch in selector.get_selected_batches(batch_size=32):
            train_on_batch(batch)
    """
    
    def __init__(
        self,
        model: nn.Module,
        uncertainty_estimator,
        buffer_size: int = 10000,
        selection_ratio: float = 0.1,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Neural network model
            uncertainty_estimator: Uncertainty estimation module (e.g., LaplacianUncertainty)
            buffer_size: Maximum samples to keep in memory
            selection_ratio: Fraction of data to select
            device: Device for computation
        """
        self.model = model
        self.uncertainty_estimator = uncertainty_estimator
        self.buffer_size = buffer_size
        self.selection_ratio = selection_ratio
        self.device = device
        
        # Selection buffer
        self.buffer = SelectionBuffer(capacity=buffer_size)
        
        # Statistics
        self.total_samples_seen = 0
        self.chunks_processed = 0
    
    def process_chunk(
        self, 
        data: torch.Tensor, 
        targets: torch.Tensor,
        chunk_index: int
    ):
        """
        Process a chunk of data, selecting most uncertain samples
        
        Args:
            data: Input data [chunk_size, ...]
            targets: Target labels [chunk_size]
            chunk_index: Index of current chunk
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move to device in mini-batches to avoid OOM
            batch_size = 256
            uncertainties = []
            
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i+batch_size].to(self.device)
                batch_targets = targets[i:i+batch_size].to(self.device)
                
                # Estimate uncertainty
                unc = self.uncertainty_estimator.estimate(batch_data)
                uncertainties.append(unc.cpu())
            
            uncertainties = torch.cat(uncertainties)
        
        # Add top samples from this chunk to buffer
        chunk_selection_size = int(len(data) * self.selection_ratio)
        top_indices = torch.topk(uncertainties, k=chunk_selection_size).indices
        
        for idx in top_indices:
            idx_val = idx.item()
            global_idx = self.total_samples_seen + idx_val
            
            self.buffer.add(
                uncertainty=uncertainties[idx_val].item(),
                index=global_idx,
                sample=data[idx_val],
                target=targets[idx_val]
            )
        
        self.total_samples_seen += len(data)
        self.chunks_processed += 1
    
    def get_selected_batches(
        self, 
        batch_size: int = 32,
        shuffle: bool = True
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over selected samples in batches
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle selected samples
            
        Yields:
            (data_batch, target_batch)
        """
        if len(self.buffer) == 0:
            return
        
        # Get all samples from buffer
        indices = list(range(len(self.buffer)))
        
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            samples = torch.stack([self.buffer.samples[j] for j in batch_indices])
            targets = torch.stack([self.buffer.targets[j] for j in batch_indices])
            
            yield samples.to(self.device), targets.to(self.device)
    
    def get_statistics(self) -> dict:
        """Get selection statistics"""
        return {
            'total_samples_seen': self.total_samples_seen,
            'chunks_processed': self.chunks_processed,
            'buffer_size': len(self.buffer),
            'selection_rate': len(self.buffer) / max(self.total_samples_seen, 1),
            'avg_uncertainty': np.mean(self.buffer.uncertainties) if self.buffer.uncertainties else 0.0,
            'min_uncertainty': min(self.buffer.uncertainties) if self.buffer.uncertainties else 0.0,
            'max_uncertainty': max(self.buffer.uncertainties) if self.buffer.uncertainties else 0.0,
        }
    
    def reset(self):
        """Reset selector state"""
        self.buffer.clear()
        self.total_samples_seen = 0
        self.chunks_processed = 0


class AdaptiveStreamingAQL(StreamingAQL):
    """
    Adaptive version that adjusts selection ratio based on uncertainty distribution
    
    Key idea: If uncertainty is uniformly high, be more selective
              If uncertainty is uniformly low, be less selective (take more samples)
    """
    
    def __init__(
        self,
        model: nn.Module,
        uncertainty_estimator,
        buffer_size: int = 10000,
        initial_selection_ratio: float = 0.1,
        device: str = 'cuda',
        adaptation_rate: float = 0.01
    ):
        super().__init__(
            model=model,
            uncertainty_estimator=uncertainty_estimator,
            buffer_size=buffer_size,
            selection_ratio=initial_selection_ratio,
            device=device
        )
        self.initial_selection_ratio = initial_selection_ratio
        self.adaptation_rate = adaptation_rate
        self.uncertainty_history = []
    
    def process_chunk(self, data: torch.Tensor, targets: torch.Tensor, chunk_index: int):
        """Process chunk with adaptive selection ratio"""
        
        # First estimate uncertainties for this chunk
        self.model.eval()
        with torch.no_grad():
            batch_size = 256
            uncertainties = []
            
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i+batch_size].to(self.device)
                unc = self.uncertainty_estimator.estimate(batch_data)
                uncertainties.append(unc.cpu())
            
            uncertainties = torch.cat(uncertainties)
        
        # Adapt selection ratio based on uncertainty distribution
        unc_mean = uncertainties.mean().item()
        unc_std = uncertainties.std().item()
        self.uncertainty_history.append((unc_mean, unc_std))
        
        # If uncertainty is high and variable, be more selective
        # If uncertainty is low, take more samples
        if len(self.uncertainty_history) > 5:
            recent_mean = np.mean([h[0] for h in self.uncertainty_history[-5:]])
            recent_std = np.mean([h[1] for h in self.uncertainty_history[-5:]])
            
            # Adaptive adjustment
            if recent_std > unc_mean * 0.3:  # High variability
                # Be more selective
                adjustment = -self.adaptation_rate
            else:  # Low variability
                # Be less selective
                adjustment = self.adaptation_rate
            
            self.selection_ratio = np.clip(
                self.selection_ratio + adjustment,
                0.05,  # Min 5%
                0.3    # Max 30%
            )
        
        # Now call parent's process_chunk with adapted ratio
        super().process_chunk(data, targets, chunk_index)


def test_streaming_aql():
    """Test streaming AQL data selection"""
    print("Testing Streaming AQL Data Selection")
    print("=" * 50)
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Create uncertainty estimator (mock)
    class MockUncertainty:
        def estimate(self, data):
            # Random uncertainties for testing
            return torch.rand(len(data))
    
    uncertainty_estimator = MockUncertainty()
    
    # Create streaming selector
    selector = StreamingAQL(
        model=model,
        uncertainty_estimator=uncertainty_estimator,
        buffer_size=100,
        selection_ratio=0.2,
        device='cpu'
    )
    
    print("\nProcessing data chunks...")
    # Simulate streaming data
    n_chunks = 5
    chunk_size = 200
    
    for chunk_idx in range(n_chunks):
        data = torch.randn(chunk_size, 10)
        targets = torch.randint(0, 5, (chunk_size,))
        
        selector.process_chunk(data, targets, chunk_idx)
        
        stats = selector.get_statistics()
        print(f"  Chunk {chunk_idx + 1}/{n_chunks}:")
        print(f"    Total samples seen: {stats['total_samples_seen']}")
        print(f"    Buffer size: {stats['buffer_size']}")
        print(f"    Selection rate: {stats['selection_rate']:.2%}")
        print(f"    Avg uncertainty: {stats['avg_uncertainty']:.4f}")
    
    print("\nRetrieving selected batches...")
    batch_count = 0
    for batch_data, batch_targets in selector.get_selected_batches(batch_size=32):
        batch_count += 1
        print(f"  Batch {batch_count}: {batch_data.shape}, {batch_targets.shape}")
    
    print(f"\n✅ Total batches retrieved: {batch_count}")
    print("✅ Streaming AQL test passed!")
    
    # Test adaptive version
    print("\n" + "=" * 50)
    print("Testing Adaptive Streaming AQL")
    print("=" * 50)
    
    adaptive_selector = AdaptiveStreamingAQL(
        model=model,
        uncertainty_estimator=uncertainty_estimator,
        buffer_size=100,
        initial_selection_ratio=0.1,
        device='cpu'
    )
    
    print("\nProcessing chunks with adaptive selection...")
    for chunk_idx in range(n_chunks):
        data = torch.randn(chunk_size, 10)
        targets = torch.randint(0, 5, (chunk_size,))
        
        adaptive_selector.process_chunk(data, targets, chunk_idx)
        
        stats = adaptive_selector.get_statistics()
        print(f"  Chunk {chunk_idx + 1}: selection_ratio={adaptive_selector.selection_ratio:.3f}, buffer={stats['buffer_size']}")
    
    print("\n✅ Adaptive streaming AQL test passed!")


if __name__ == "__main__":
    test_streaming_aql()
