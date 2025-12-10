"""
GPU Performance Testing and Benchmarking
Demonstrates GPU acceleration with CUDA and CPU fallback
"""
import torch
import torch.nn as nn
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.device_manager import get_device_manager, DeviceManager
from experiments.experiment_framework import ExperimentRunner, ExperimentConfig, MinimalBenchmark


def benchmark_gpu_vs_cpu():
    """
    Benchmark training performance on GPU vs CPU
    Automatically uses GPU if available, falls back to CPU
    """
    
    print("\n" + "="*80)
    print("🚀 GPU vs CPU PERFORMANCE BENCHMARK")
    print("="*80)
    
    # Test different batch sizes to show GPU advantage
    batch_sizes = [32, 64, 128, 256]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n📊 Testing with batch size: {batch_size}")
        print("-" * 80)
        
        # GPU Test
        device_mgr = get_device_manager(verbose=False)
        device = device_mgr.get_device()
        
        print(f"\n✓ Running on: {device}")
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        model = device_mgr.to_device(model)
        
        # Create dummy data
        num_batches = 100
        total_samples = batch_size * num_batches
        
        # Warm-up
        dummy_input = torch.randn(batch_size, 784, device=device)
        dummy_target = torch.randint(0, 10, (batch_size,), device=device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        for _ in range(5):
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Actual benchmark
        start_time = time.time()
        
        for i in range(num_batches):
            data = torch.randn(batch_size, 784, device=device)
            target = torch.randint(0, 10, (batch_size,), device=device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        samples_per_sec = total_samples / elapsed_time
        
        results[batch_size] = {
            'device': str(device),
            'time': elapsed_time,
            'samples_per_sec': samples_per_sec
        }
        
        print(f"⏱️  Time: {elapsed_time:.3f}s")
        print(f"📈 Throughput: {samples_per_sec:.1f} samples/sec")
        
        if device.type == 'cuda':
            device_mgr.print_memory_stats()
            device_mgr.empty_cache()
    
    # Print summary
    print("\n" + "="*80)
    print("📊 PERFORMANCE SUMMARY")
    print("="*80)
    print(f"\n{'Batch Size':<12} {'Device':<10} {'Time (s)':<12} {'Samples/sec':<15}")
    print("-" * 80)
    
    for batch_size, result in results.items():
        print(f"{batch_size:<12} {result['device']:<10} {result['time']:<12.3f} {result['samples_per_sec']:<15.1f}")
    
    return results


def test_mixed_precision_speedup():
    """
    Test mixed precision (FP16) training speedup
    Only works on CUDA GPUs with compute capability >= 7.0 (Volta+)
    """
    
    print("\n" + "="*80)
    print("⚡ MIXED PRECISION (FP16) SPEEDUP TEST")
    print("="*80)
    
    device_mgr = get_device_manager(verbose=True)
    
    if device_mgr.device.type != 'cuda':
        print("\n⚠️  Mixed precision requires CUDA GPU. Skipping test.")
        return None
    
    # Check if FP16 is supported
    if not device_mgr.enable_mixed_precision():
        print("\n⚠️  Mixed precision not supported on this GPU (requires compute capability >= 7.0)")
        print("    Current GPU is suitable for FP32 training only.")
        return None
    
    dataset = MinimalBenchmark.simple_mnist_task()
    
    # Test FP32 (standard)
    print("\n🔹 Testing FP32 (Standard Precision)...")
    runner_fp32 = ExperimentRunner(device_manager=device_mgr)
    config_fp32 = ExperimentConfig(
        name="fp32_baseline",
        model_fn=lambda: nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ),
        optimizer_fn=lambda params: torch.optim.Adam(params, lr=0.001),
        loss_fn=nn.CrossEntropyLoss,
        dataset=dataset,
        batch_size=256,
        num_epochs=5,
        use_mixed_precision=False
    )
    result_fp32 = runner_fp32.run_experiment(config_fp32)
    
    # Test FP16 (mixed precision)
    print("\n🔸 Testing FP16 (Mixed Precision)...")
    device_mgr.empty_cache()
    runner_fp16 = ExperimentRunner(device_manager=device_mgr)
    config_fp16 = ExperimentConfig(
        name="fp16_mixed_precision",
        model_fn=lambda: nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ),
        optimizer_fn=lambda params: torch.optim.Adam(params, lr=0.001),
        loss_fn=nn.CrossEntropyLoss,
        dataset=dataset,
        batch_size=256,
        num_epochs=5,
        use_mixed_precision=True
    )
    result_fp16 = runner_fp16.run_experiment(config_fp16)
    
    # Compare results
    speedup = result_fp32.training_time / result_fp16.training_time
    memory_savings = ((result_fp32.memory_usage_mb - result_fp16.memory_usage_mb) / 
                     result_fp32.memory_usage_mb * 100)
    
    print("\n" + "="*80)
    print("📊 MIXED PRECISION RESULTS")
    print("="*80)
    print(f"\nFP32 Training Time: {result_fp32.training_time:.2f}s")
    print(f"FP16 Training Time: {result_fp16.training_time:.2f}s")
    print(f"⚡ Speedup: {speedup:.2f}x")
    print(f"\nFP32 Memory Usage: {result_fp32.memory_usage_mb:.1f} MB")
    print(f"FP16 Memory Usage: {result_fp16.memory_usage_mb:.1f} MB")
    print(f"💾 Memory Savings: {memory_savings:.1f}%")
    print(f"\nFP32 Final Accuracy: {result_fp32.final_accuracy:.2f}%")
    print(f"FP16 Final Accuracy: {result_fp16.final_accuracy:.2f}%")
    print(f"Accuracy Difference: {abs(result_fp32.final_accuracy - result_fp16.final_accuracy):.2f}%")
    
    return {
        'fp32': result_fp32,
        'fp16': result_fp16,
        'speedup': speedup,
        'memory_savings': memory_savings
    }


def demonstrate_optimal_batch_size():
    """
    Demonstrate automatic batch size optimization based on GPU memory
    """
    
    print("\n" + "="*80)
    print("🎯 OPTIMAL BATCH SIZE ESTIMATION")
    print("="*80)
    
    device_mgr = get_device_manager(verbose=True)
    
    if device_mgr.device.type != 'cuda':
        print("\n⚠️  Batch size optimization is most useful for GPU training.")
        print("    Using default batch size of 32 for CPU.")
        return 32
    
    # Create test model
    test_model = nn.Sequential(
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    # Estimate optimal batch size
    print("\n🔍 Analyzing GPU memory and model size...")
    optimal_batch = device_mgr.get_optimal_batch_size(
        model=test_model,
        sample_input_shape=(784,),
        max_memory_fraction=0.8
    )
    
    print(f"\n✅ Recommended batch size: {optimal_batch}")
    print(f"   This utilizes ~80% of available GPU memory")
    print(f"   Adjust based on your specific needs")
    
    return optimal_batch


def full_gpu_demonstration():
    """
    Complete demonstration of GPU acceleration features
    """
    
    print("\n" + "="*100)
    print(" "*30 + "🚀 GPU ACCELERATION DEMONSTRATION")
    print("="*100)
    
    # 1. Device detection and info
    print("\n[1/4] DEVICE DETECTION")
    device_mgr = get_device_manager(verbose=True)
    
    # 2. Batch size optimization
    print("\n[2/4] BATCH SIZE OPTIMIZATION")
    optimal_batch = demonstrate_optimal_batch_size()
    
    # 3. Performance benchmark
    print("\n[3/4] PERFORMANCE BENCHMARK")
    perf_results = benchmark_gpu_vs_cpu()
    
    # 4. Mixed precision test (if available)
    print("\n[4/4] MIXED PRECISION TEST")
    mp_results = test_mixed_precision_speedup()
    
    print("\n" + "="*100)
    print("✅ GPU DEMONSTRATION COMPLETE")
    print("="*100)
    
    # Final recommendations
    print("\n💡 RECOMMENDATIONS FOR YOUR RESEARCH LAB:")
    print("-" * 100)
    
    if device_mgr.device.type == 'cuda':
        print("✅ CUDA GPU detected and configured")
        print(f"   Device: {device_mgr.device_info.device_name}")
        print(f"   Memory: {device_mgr.device_info.total_memory_gb:.1f} GB")
        print(f"\n📊 Performance:")
        print(f"   - Use batch size: {optimal_batch} (or adjust based on model size)")
        
        if mp_results:
            print(f"   - Mixed precision (FP16) speedup: {mp_results['speedup']:.2f}x")
            print(f"   - Memory savings with FP16: {mp_results['memory_savings']:.1f}%")
            print(f"\n   ⚡ RECOMMENDATION: Enable mixed precision for faster training!")
        else:
            print(f"\n   ℹ️  Mixed precision not available (requires Volta+ GPU)")
            print(f"   ✓  FP32 training will work efficiently")
        
        print(f"\n🔧 Configuration:")
        print(f"   - All experiments will automatically use GPU")
        print(f"   - Memory is monitored and optimized")
        print(f"   - Multi-GPU support available (if you have multiple GPUs)")
        
    else:
        print("⚠️  No CUDA GPU detected - running on CPU")
        print("\nℹ️  For production AI research lab:")
        print("   1. Install NVIDIA GPU drivers")
        print("   2. Install CUDA Toolkit (compatible with PyTorch)")
        print("   3. Verify with: nvidia-smi")
        print("   4. Reinstall PyTorch with CUDA support:")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("\n   CPU training will still work but will be significantly slower for large models.")
    
    print("-" * 100)
    
    return {
        'device_manager': device_mgr,
        'optimal_batch_size': optimal_batch,
        'performance_results': perf_results,
        'mixed_precision_results': mp_results
    }


if __name__ == "__main__":
    # Run complete demonstration
    results = full_gpu_demonstration()
    
    print("\n✅ All tests completed successfully!")
    print("\n💡 Next steps:")
    print("   - Run: python examples/test_novel_optimizers.py (uses GPU automatically)")
    print("   - Run: python examples/test_novel_losses.py (uses GPU automatically)")
    print("   - All training will leverage GPU when available")
