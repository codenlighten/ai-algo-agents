"""
Quick System Verification Test
Tests that GPU detection and basic training loop work correctly
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.device_manager import get_device_manager

def test_device_detection():
    """Test 1: Device detection"""
    print("\n" + "="*60)
    print("TEST 1: Device Detection")
    print("="*60)
    
    device_mgr = get_device_manager(verbose=False)
    device = device_mgr.get_device()
    
    print(f"Device detected: {device}")
    print(f"Device type: {device.type}")
    
    if device.type == 'cuda':
        print(f"GPU Name: {device_mgr.device_info.device_name}")
        print(f"GPU Memory: {device_mgr.device_info.total_memory_gb:.2f} GB")
        print("Status: PASS - GPU detected")
    else:
        print("Status: PASS - CPU fallback working")
    
    return True


def test_simple_training():
    """Test 2: Simple training loop"""
    print("\n" + "="*60)
    print("TEST 2: Simple Training Loop")
    print("="*60)
    
    device_mgr = get_device_manager(verbose=False)
    device = device_mgr.get_device()
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )
    model = device_mgr.to_device(model)
    
    # Create optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Run 10 training steps
    print(f"Running 10 training steps on {device}...")
    for i in range(10):
        # Generate random data
        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 2, (4,), device=device)
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if i % 3 == 0:
            print(f"  Step {i+1}/10 - Loss: {loss.item():.4f}")
    
    print("Status: PASS - Training loop works")
    return True


def test_model_transfer():
    """Test 3: Model device transfer"""
    print("\n" + "="*60)
    print("TEST 3: Model Device Transfer")
    print("="*60)
    
    device_mgr = get_device_manager(verbose=False)
    
    # Create model on CPU
    model = nn.Linear(10, 5)
    print(f"Model created on: CPU")
    
    # Transfer to device
    model = device_mgr.to_device(model)
    device_str = next(model.parameters()).device
    print(f"Model transferred to: {device_str}")
    
    # Test forward pass
    x = torch.randn(2, 10)
    x = device_mgr.to_device(x)
    output = model(x)
    
    print(f"Forward pass successful")
    print(f"Output shape: {output.shape}")
    print("Status: PASS - Device transfer works")
    
    return True


def run_all_tests():
    """Run all verification tests"""
    print("\n" + "="*60)
    print("AI RESEARCH AGENT SYSTEM - VERIFICATION TESTS")
    print("="*60)
    
    tests = [
        ("Device Detection", test_device_detection),
        ("Simple Training Loop", test_simple_training),
        ("Model Device Transfer", test_model_transfer),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\nERROR in {test_name}: {e}")
            results.append((test_name, "FAIL"))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, status in results:
        status_mark = "[OK]" if status == "PASS" else "[FAIL]"
        print(f"{status_mark} {test_name}")
    
    all_passed = all(status == "PASS" for _, status in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED - System is ready!")
    else:
        print("SOME TESTS FAILED - Please check errors above")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
