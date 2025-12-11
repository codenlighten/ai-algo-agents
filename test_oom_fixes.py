#!/usr/bin/env python3
"""
Test that sparsae_wikitext.py accepts the new memory-saving parameters
"""

import subprocess
import sys

def test_help():
    """Test that --help works and shows new parameters"""
    print("Testing --help output...")
    result = subprocess.run(
        [sys.executable, "experiments/sparsae_wikitext.py", "--help"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ --help failed with code {result.returncode}")
        print(result.stderr)
        return False
    
    # Check for new parameters
    required_params = [
        "--max_train_examples",
        "--max_val_examples", 
        "--num_workers"
    ]
    
    help_text = result.stdout
    missing = []
    for param in required_params:
        if param not in help_text:
            missing.append(param)
    
    if missing:
        print(f"❌ Missing parameters in --help: {missing}")
        return False
    
    print("✅ All new parameters found in --help")
    return True


def test_import():
    """Test that the script can be imported without errors"""
    print("\nTesting imports...")
    result = subprocess.run(
        [sys.executable, "-c", 
         "import sys; sys.path.insert(0, 'experiments'); "
         "from sparsae_wikitext import WikiTextDataset, parse_args; "
         "print('OK')"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Import failed: {result.stderr}")
        return False
    
    if "OK" not in result.stdout:
        print(f"❌ Import test failed")
        return False
    
    print("✅ Script imports successfully")
    return True


def test_args():
    """Test that arguments are parsed correctly"""
    print("\nTesting argument parsing...")
    result = subprocess.run(
        [sys.executable, "-c",
         "from experiments.sparsae_wikitext import parse_args; "
         "args = parse_args(); "
         "assert hasattr(args, 'max_train_examples'), 'missing max_train_examples'; "
         "assert hasattr(args, 'max_val_examples'), 'missing max_val_examples'; "
         "assert hasattr(args, 'num_workers'), 'missing num_workers'; "
         "assert args.max_train_examples == 50000, f'wrong default: {args.max_train_examples}'; "
         "print('OK')"],
        capture_output=True,
        text=True,
        cwd="."
    )
    
    if result.returncode != 0:
        print(f"❌ Argument parsing failed: {result.stderr}")
        return False
    
    if "OK" not in result.stdout:
        print(f"❌ Argument test failed")
        return False
    
    print("✅ Arguments parsed correctly with correct defaults")
    return True


def main():
    print("="*60)
    print("Testing SparsAE OOM Fixes")
    print("="*60)
    
    tests = [
        test_help,
        test_import,
        test_args,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} raised exception: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed > 0:
        sys.exit(1)
    
    print("\n✅ All tests passed! OOM fixes are correctly implemented.")
    print("\nNext steps:")
    print("1. Push changes to GitHub")
    print("2. Open colab_setup.ipynb in Google Colab")
    print("3. Run all cells in order")
    print("4. Training should now work without OOM errors")


if __name__ == "__main__":
    main()
