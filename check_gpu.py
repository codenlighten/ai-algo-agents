"""
Quick GPU Detection Test
Run this to verify GPU/CUDA setup
"""
import torch

print("="*60)
print("GPU DETECTION TEST")
print("="*60)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("\n✅ GPU is ready for use!")
else:
    print("\n⚠️  No CUDA GPU detected - will use CPU")
    print("\nTo enable GPU:")
    print("1. Install NVIDIA drivers")
    print("2. Install CUDA Toolkit")
    print("3. Install PyTorch with CUDA:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

print("="*60)
