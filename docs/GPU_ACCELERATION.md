# GPU Acceleration Guide

## Overview

The AI Algorithm Research Agent system automatically detects and utilizes NVIDIA CUDA GPUs when available, with seamless CPU fallback. This ensures optimal performance for your research lab experiments.

## Features

### ✅ Automatic GPU Detection
- Detects NVIDIA CUDA GPUs automatically
- Selects GPU with most available memory (multi-GPU systems)
- Falls back to CPU if no GPU available
- No manual configuration required

### ⚡ Performance Optimizations
- **Mixed Precision (FP16) Training**: 1.5-2x speedup on modern GPUs (Volta/Turing/Ampere)
- **Automatic Batch Size Optimization**: Maximizes GPU memory utilization
- **cuDNN Acceleration**: Optimized deep learning primitives
- **Memory Management**: Automatic cache clearing and monitoring

### 📊 Device Information
- Real-time GPU memory usage tracking
- Device capabilities and CUDA version
- Performance recommendations

## Quick Start

### Check GPU Availability

```python
from utils.device_manager import get_device_manager

# Initialize device manager (prints device info)
device_mgr = get_device_manager(verbose=True)

# Get device
device = device_mgr.get_device()
print(f"Using device: {device}")  # cuda:0 or cpu
```

### Run GPU Performance Test

```bash
# Comprehensive GPU benchmark
python examples/gpu_performance_test.py
```

This will:
1. Detect available GPUs
2. Benchmark GPU vs CPU performance
3. Test mixed precision speedup (if supported)
4. Recommend optimal batch size
5. Provide configuration recommendations

## Usage in Experiments

### Automatic GPU Usage

All experiments automatically use GPU when available:

```python
from experiments.experiment_framework import ExperimentRunner, ExperimentConfig

# Create runner (automatically uses GPU)
runner = ExperimentRunner()

# Configure experiment
config = ExperimentConfig(
    name="my_experiment",
    model_fn=lambda: MyModel(),
    optimizer_fn=lambda params: torch.optim.Adam(params),
    loss_fn=nn.CrossEntropyLoss,
    dataset=my_dataset,
    batch_size=128,
    num_epochs=10,
    use_mixed_precision=True  # Enable FP16 if supported
)

# Run (uses GPU automatically)
result = runner.run_experiment(config)
```

### Manual Device Management

```python
from utils.device_manager import DeviceManager

# Create device manager
device_mgr = DeviceManager(verbose=True)

# Move model to device
model = MyModel()
model = device_mgr.to_device(model)

# Move multiple objects
model, optimizer, criterion = device_mgr.to_device(
    model, optimizer, criterion
)

# Get memory stats
device_mgr.print_memory_stats()

# Clear cache
device_mgr.empty_cache()
```

## Mixed Precision Training (FP16)

Mixed precision uses 16-bit floats for faster training with reduced memory usage.

### Requirements
- NVIDIA GPU with compute capability ≥ 7.0 (Volta, Turing, Ampere)
- Examples: RTX 20/30/40 series, Tesla V100, A100

### Enable Mixed Precision

```python
config = ExperimentConfig(
    name="mixed_precision_experiment",
    model_fn=lambda: MyModel(),
    optimizer_fn=lambda params: torch.optim.Adam(params),
    loss_fn=nn.CrossEntropyLoss,
    dataset=my_dataset,
    use_mixed_precision=True  # ← Enable here
)
```

### Benefits
- **1.5-2x speedup** on compatible GPUs
- **30-50% memory savings**
- **Minimal accuracy impact** (< 0.1% typical)

### Automatic Detection

The system automatically:
1. Checks GPU compute capability
2. Enables FP16 if supported
3. Falls back to FP32 if not supported

## Optimal Batch Size

Automatically calculate optimal batch size for your GPU:

```python
from utils.device_manager import get_device_manager

device_mgr = get_device_manager()

# Get optimal batch size for your model
optimal_batch = device_mgr.get_optimal_batch_size(
    model=my_model,
    sample_input_shape=(3, 224, 224),  # e.g., ImageNet size
    max_memory_fraction=0.8  # Use 80% of GPU memory
)

print(f"Recommended batch size: {optimal_batch}")
```

## Multi-GPU Support

### Detect Multiple GPUs

```python
device_mgr = get_device_manager()

if device_mgr.device_info.device_count > 1:
    print(f"Found {device_mgr.device_info.device_count} GPUs")
```

### Use Specific GPU

```python
# Force specific GPU
device_mgr = DeviceManager(preferred_device="cuda:1")

# Or use environment variable
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1
```

### DataParallel (Coming Soon)

For multi-GPU training, use PyTorch DataParallel:

```python
import torch.nn as nn

model = MyModel()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = device_mgr.to_device(model)
```

## Performance Tips

### 1. **Use Mixed Precision** (if supported)
```python
use_mixed_precision=True
```
Speedup: 1.5-2x, Memory: -30-50%

### 2. **Optimize Batch Size**
```python
batch_size = device_mgr.get_optimal_batch_size(model, input_shape)
```
Maximizes GPU utilization

### 3. **Enable cuDNN Optimizations**
```python
device_mgr.optimize_for_training()
```
Automatically enabled by ExperimentRunner

### 4. **Clear GPU Cache Between Experiments**
```python
device_mgr.empty_cache()
```
Prevents memory fragmentation

### 5. **Monitor Memory Usage**
```python
device_mgr.print_memory_stats()
```
Track peak memory usage

## Troubleshooting

### No GPU Detected

**Check CUDA availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
```

**If CUDA not available:**
1. Install NVIDIA GPU drivers: [NVIDIA Drivers](https://www.nvidia.com/download/index.aspx)
2. Install CUDA Toolkit (compatible with PyTorch)
3. Verify installation: `nvidia-smi`
4. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### Out of Memory (OOM) Errors

**Solutions:**
1. **Reduce batch size:**
   ```python
   batch_size = batch_size // 2
   ```

2. **Enable mixed precision:**
   ```python
   use_mixed_precision=True
   ```

3. **Clear cache:**
   ```python
   device_mgr.empty_cache()
   ```

4. **Use gradient accumulation:**
   ```python
   # Accumulate gradients over multiple batches
   accumulation_steps = 4
   for i, batch in enumerate(dataloader):
       loss = model(batch) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

### Slow Training on GPU

**Check:**
1. **Data loading bottleneck:**
   ```python
   dataloader = DataLoader(dataset, num_workers=4, pin_memory=True)
   ```

2. **Small batch size:**
   - Use optimal batch size calculation
   - Larger batches = better GPU utilization

3. **CPU-GPU transfer overhead:**
   - Use `pin_memory=True` in DataLoader
   - Keep data on GPU when possible

## System Requirements

### Minimum Requirements
- **CPU**: Any modern CPU (automatic fallback)
- **RAM**: 8 GB minimum, 16 GB recommended
- **GPU** (optional): NVIDIA GPU with CUDA support

### Recommended for Production
- **GPU**: NVIDIA RTX 3060 or better
- **VRAM**: 8 GB minimum, 12+ GB recommended
- **CUDA**: 11.8 or 12.1
- **RAM**: 32 GB for large-scale experiments

### Supported GPUs
- **Consumer**: RTX 20/30/40 series, GTX 1660+
- **Workstation**: RTX A4000/A5000/A6000
- **Data Center**: Tesla V100, A100, H100

## Example: Complete GPU Setup

```python
from utils.device_manager import get_device_manager
from experiments.experiment_framework import ExperimentRunner, ExperimentConfig
import torch.nn as nn

# 1. Initialize device manager
device_mgr = get_device_manager(verbose=True)
device_mgr.optimize_for_training()

# 2. Check capabilities
print(f"Device: {device_mgr.device}")
print(f"Mixed precision supported: {device_mgr.enable_mixed_precision()}")

# 3. Get optimal batch size
model = MyResearchModel()
optimal_batch = device_mgr.get_optimal_batch_size(
    model=model,
    sample_input_shape=(3, 224, 224)
)

# 4. Configure experiment
config = ExperimentConfig(
    name="gpu_optimized_experiment",
    model_fn=lambda: MyResearchModel(),
    optimizer_fn=lambda params: torch.optim.AdamW(params, lr=1e-4),
    loss_fn=nn.CrossEntropyLoss,
    dataset=my_dataset,
    batch_size=optimal_batch,
    num_epochs=100,
    use_mixed_precision=True  # Auto-disabled if not supported
)

# 5. Run experiment (uses GPU automatically)
runner = ExperimentRunner(device_manager=device_mgr)
result = runner.run_experiment(config)

# 6. Check memory usage
device_mgr.print_memory_stats()
```

## Integration with Research Workflow

All existing scripts automatically use GPU:

```bash
# Optimizer comparison (uses GPU)
python examples/test_novel_optimizers.py

# Loss function comparison (uses GPU)
python examples/test_novel_losses.py

# Architecture testing (uses GPU)
python examples/test_novel_architectures.py

# Main research interface (uses GPU)
python main.py
```

No code changes required - GPU acceleration is transparent!

## Best Practices for Research Lab

1. **Always run GPU performance test first:**
   ```bash
   python examples/gpu_performance_test.py
   ```

2. **Monitor GPU usage during experiments:**
   ```bash
   watch -n 1 nvidia-smi  # Linux/WSL
   ```

3. **Use mixed precision for large models:**
   - Enables larger batch sizes
   - Faster training
   - Minimal accuracy impact

4. **Optimize batch size for each model:**
   - Use `get_optimal_batch_size()`
   - Test different sizes empirically

5. **Clear cache between experiments:**
   - Prevents memory leaks
   - Ensures consistent performance

6. **Log device info in experiment results:**
   - Reproducibility
   - Performance tracking

## Summary

✅ **Automatic GPU detection** - No manual configuration  
✅ **Seamless CPU fallback** - Works everywhere  
✅ **Mixed precision (FP16)** - 2x faster on modern GPUs  
✅ **Batch size optimization** - Maximize GPU utilization  
✅ **Memory management** - Track and optimize usage  
✅ **Multi-GPU ready** - Scale to multiple GPUs  

Your research lab is now equipped with production-grade GPU acceleration! 🚀
