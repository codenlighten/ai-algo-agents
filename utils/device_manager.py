"""
Device management utilities for GPU/CPU detection and optimal resource allocation
"""
import torch
import logging
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    """Information about available compute devices"""
    device_type: str  # 'cuda' or 'cpu'
    device_name: str
    device_count: int
    total_memory_gb: Optional[float] = None
    available_memory_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    
    def __str__(self):
        if self.device_type == 'cuda':
            return (f"CUDA Device: {self.device_name}\n"
                   f"  GPU Count: {self.device_count}\n"
                   f"  Total Memory: {self.total_memory_gb:.2f} GB\n"
                   f"  Available Memory: {self.available_memory_gb:.2f} GB\n"
                   f"  CUDA Version: {self.cuda_version}")
        else:
            return f"CPU Device: {self.device_name}"


class DeviceManager:
    """
    Centralized device management for AI research experiments
    
    Features:
    - Automatic GPU detection with CUDA support
    - Intelligent CPU fallback
    - Memory management and monitoring
    - Multi-GPU support
    - Performance optimization recommendations
    """
    
    def __init__(self, preferred_device: Optional[str] = None, verbose: bool = True):
        """
        Initialize device manager
        
        Args:
            preferred_device: Force specific device ('cuda', 'cpu', 'cuda:0', etc.)
            verbose: Print device information
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Detect and configure device
        self.device = self._configure_device(preferred_device)
        self.device_info = self._get_device_info()
        
        if self.verbose:
            self._print_device_info()
    
    def _configure_device(self, preferred_device: Optional[str]) -> torch.device:
        """Configure the compute device"""
        if preferred_device:
            try:
                device = torch.device(preferred_device)
                if device.type == 'cuda' and not torch.cuda.is_available():
                    self.logger.warning(f"CUDA not available, falling back to CPU")
                    return torch.device('cpu')
                return device
            except Exception as e:
                self.logger.error(f"Error setting device {preferred_device}: {e}")
                return self._auto_select_device()
        else:
            return self._auto_select_device()
    
    def _auto_select_device(self) -> torch.device:
        """Automatically select best available device"""
        if torch.cuda.is_available():
            # Select GPU with most free memory
            if torch.cuda.device_count() > 1:
                max_mem_device = self._get_gpu_with_most_memory()
                return torch.device(f'cuda:{max_mem_device}')
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _get_gpu_with_most_memory(self) -> int:
        """Find GPU with most available memory"""
        max_mem = 0
        best_device = 0
        
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                mem_free = torch.cuda.get_device_properties(i).total_memory
                mem_allocated = torch.cuda.memory_allocated(i)
                available = mem_free - mem_allocated
                
                if available > max_mem:
                    max_mem = available
                    best_device = i
            except Exception as e:
                self.logger.warning(f"Error checking GPU {i}: {e}")
        
        return best_device
    
    def _get_device_info(self) -> DeviceInfo:
        """Gather comprehensive device information"""
        if self.device.type == 'cuda':
            device_idx = self.device.index or 0
            props = torch.cuda.get_device_properties(device_idx)
            
            total_mem = props.total_memory / (1024**3)  # Convert to GB
            allocated_mem = torch.cuda.memory_allocated(device_idx) / (1024**3)
            available_mem = total_mem - allocated_mem
            
            return DeviceInfo(
                device_type='cuda',
                device_name=props.name,
                device_count=torch.cuda.device_count(),
                total_memory_gb=total_mem,
                available_memory_gb=available_mem,
                cuda_version=torch.version.cuda
            )
        else:
            import platform
            return DeviceInfo(
                device_type='cpu',
                device_name=platform.processor() or "CPU",
                device_count=1
            )
    
    def _print_device_info(self):
        """Print device information"""
        print("\n" + "="*80)
        print("🔧 DEVICE CONFIGURATION")
        print("="*80)
        print(str(self.device_info))
        
        if self.device.type == 'cuda':
            print(f"\n✅ PyTorch CUDA is available and enabled")
            print(f"   Using device: {self.device}")
            
            # Performance tips
            if torch.cuda.device_count() > 1:
                print(f"\n💡 Multi-GPU detected ({torch.cuda.device_count()} GPUs)")
                print(f"   Consider using DataParallel or DistributedDataParallel")
        else:
            print(f"\n⚠️  Running on CPU (CUDA not available)")
            print(f"   For faster training, ensure NVIDIA drivers and CUDA are installed")
        
        print("="*80 + "\n")
    
    def get_device(self) -> torch.device:
        """Get the configured device"""
        return self.device
    
    def to_device(self, *tensors_or_models):
        """
        Move tensors or models to the configured device
        
        Args:
            *tensors_or_models: Variable number of tensors or models
            
        Returns:
            Tuple of moved objects (or single object if only one provided)
        """
        moved = [obj.to(self.device) for obj in tensors_or_models]
        return moved[0] if len(moved) == 1 else tuple(moved)
    
    def empty_cache(self):
        """Clear GPU cache if using CUDA"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            if self.verbose:
                self.logger.info("GPU cache cleared")
    
    def get_memory_stats(self) -> dict:
        """Get current memory statistics"""
        if self.device.type == 'cuda':
            device_idx = self.device.index or 0
            return {
                'allocated_gb': torch.cuda.memory_allocated(device_idx) / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved(device_idx) / (1024**3),
                'max_allocated_gb': torch.cuda.max_memory_allocated(device_idx) / (1024**3),
            }
        return {}
    
    def print_memory_stats(self):
        """Print current memory usage"""
        if self.device.type == 'cuda':
            stats = self.get_memory_stats()
            print(f"\n📊 GPU Memory Usage:")
            print(f"   Allocated: {stats['allocated_gb']:.2f} GB")
            print(f"   Reserved:  {stats['reserved_gb']:.2f} GB")
            print(f"   Peak:      {stats['max_allocated_gb']:.2f} GB\n")
    
    def optimize_for_inference(self):
        """Apply optimizations for inference"""
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            if self.verbose:
                self.logger.info("cuDNN optimizations enabled for inference")
    
    def optimize_for_training(self):
        """Apply optimizations for training"""
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            if self.verbose:
                self.logger.info("cuDNN optimizations enabled for training")
    
    def enable_mixed_precision(self) -> bool:
        """
        Check if mixed precision (FP16) training is available
        
        Returns:
            True if mixed precision is supported
        """
        if self.device.type == 'cuda':
            # Check for Tensor Cores (Volta, Turing, Ampere GPUs)
            compute_capability = torch.cuda.get_device_capability(self.device)
            supports_fp16 = compute_capability[0] >= 7  # Volta and newer
            
            if self.verbose and supports_fp16:
                print(f"✅ Mixed Precision (FP16) training supported")
                print(f"   GPU Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
            
            return supports_fp16
        return False
    
    def get_optimal_batch_size(self, model: torch.nn.Module, 
                              sample_input_shape: tuple,
                              max_memory_fraction: float = 0.8) -> int:
        """
        Estimate optimal batch size for given model and input
        
        Args:
            model: PyTorch model
            sample_input_shape: Shape of single input sample (e.g., (3, 224, 224))
            max_memory_fraction: Maximum fraction of GPU memory to use
            
        Returns:
            Recommended batch size
        """
        if self.device.type != 'cuda':
            return 32  # Default for CPU
        
        try:
            model = model.to(self.device)
            model.eval()
            
            # Get available memory
            device_idx = self.device.index or 0
            props = torch.cuda.get_device_properties(device_idx)
            available_mem = props.total_memory * max_memory_fraction
            
            # Test with batch size 1
            sample_input = torch.randn(1, *sample_input_shape).to(self.device)
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                _ = model(sample_input)
            
            mem_per_sample = torch.cuda.max_memory_allocated(device_idx)
            
            # Estimate batch size (conservative estimate)
            estimated_batch_size = int((available_mem / mem_per_sample) * 0.7)
            
            # Clean up
            del sample_input
            self.empty_cache()
            
            # Round to power of 2 for optimal GPU utilization
            batch_size = 2 ** int(torch.log2(torch.tensor(estimated_batch_size)))
            
            if self.verbose:
                print(f"💡 Recommended batch size: {batch_size}")
            
            return max(1, batch_size)
            
        except Exception as e:
            self.logger.warning(f"Error estimating batch size: {e}")
            return 32


# Singleton instance for global access
_global_device_manager: Optional[DeviceManager] = None


def get_device_manager(preferred_device: Optional[str] = None, 
                       verbose: bool = True,
                       force_new: bool = False) -> DeviceManager:
    """
    Get or create global device manager instance
    
    Args:
        preferred_device: Force specific device
        verbose: Print device info
        force_new: Force creation of new instance
        
    Returns:
        DeviceManager instance
    """
    global _global_device_manager
    
    if _global_device_manager is None or force_new:
        _global_device_manager = DeviceManager(preferred_device, verbose)
    
    return _global_device_manager


# Convenience functions
def get_device(preferred_device: Optional[str] = None) -> torch.device:
    """Get the configured device"""
    return get_device_manager(preferred_device, verbose=False).get_device()


def to_device(*tensors_or_models, device: Optional[torch.device] = None):
    """Move tensors/models to device"""
    if device is None:
        device = get_device()
    
    moved = [obj.to(device) for obj in tensors_or_models]
    return moved[0] if len(moved) == 1 else tuple(moved)
