"""
Simple verification - checks file structure
Run: python verify_structure.py
"""
import os
from pathlib import Path


def check_file_exists(path, description):
    """Check if a file exists"""
    if os.path.exists(path):
        print(f"  ✓ {description}")
        return True
    else:
        print(f"  ✗ {description} - NOT FOUND")
        return False


def main():
    print("="*60)
    print("AI Research Agent System - Structure Verification")
    print("="*60)
    
    all_ok = True
    
    # Core files
    print("\n📁 Core Files:")
    all_ok &= check_file_exists("README.md", "README.md")
    all_ok &= check_file_exists("QUICKSTART.md", "QUICKSTART.md")
    all_ok &= check_file_exists("SYSTEM_OVERVIEW.md", "SYSTEM_OVERVIEW.md")
    all_ok &= check_file_exists("RESEARCH_IDEAS.md", "RESEARCH_IDEAS.md")
    all_ok &= check_file_exists("COMPLETE_SUMMARY.md", "COMPLETE_SUMMARY.md")
    all_ok &= check_file_exists("requirements.txt", "requirements.txt")
    all_ok &= check_file_exists(".env", ".env")
    all_ok &= check_file_exists("main.py", "main.py")
    
    # Agent system
    print("\n🤖 Agent System:")
    all_ok &= check_file_exists("agents/__init__.py", "agents/__init__.py")
    all_ok &= check_file_exists("agents/base_agent.py", "agents/base_agent.py")
    
    # Optimizers
    print("\n⚡ Novel Optimizers:")
    all_ok &= check_file_exists("optimizers/__init__.py", "optimizers/__init__.py")
    all_ok &= check_file_exists("optimizers/novel_optimizers.py", "optimizers/novel_optimizers.py")
    
    # Loss functions
    print("\n🎯 Novel Loss Functions:")
    all_ok &= check_file_exists("loss_functions/__init__.py", "loss_functions/__init__.py")
    all_ok &= check_file_exists("loss_functions/novel_losses.py", "loss_functions/novel_losses.py")
    
    # Architectures
    print("\n🏗️  Novel Architectures:")
    all_ok &= check_file_exists("models/__init__.py", "models/__init__.py")
    all_ok &= check_file_exists("models/novel_architectures.py", "models/novel_architectures.py")
    
    # Research system
    print("\n🔬 Research System:")
    all_ok &= check_file_exists("research/__init__.py", "research/__init__.py")
    all_ok &= check_file_exists("research/proposal_system.py", "research/proposal_system.py")
    all_ok &= check_file_exists("research/README.md", "research/README.md")
    
    # Experiments
    print("\n🧪 Experimental Framework:")
    all_ok &= check_file_exists("experiments/__init__.py", "experiments/__init__.py")
    all_ok &= check_file_exists("experiments/experiment_framework.py", "experiments/experiment_framework.py")
    
    # Examples
    print("\n📚 Examples:")
    all_ok &= check_file_exists("examples/example_proposals.py", "examples/example_proposals.py")
    all_ok &= check_file_exists("examples/test_novel_optimizers.py", "examples/test_novel_optimizers.py")
    all_ok &= check_file_exists("examples/test_novel_losses.py", "examples/test_novel_losses.py")
    
    # Tests
    print("\n✅ Tests:")
    all_ok &= check_file_exists("tests/__init__.py", "tests/__init__.py")
    all_ok &= check_file_exists("tests/test_system.py", "tests/test_system.py")
    
    # Utils
    print("\n🛠️  Utilities:")
    all_ok &= check_file_exists("utils/research_prompts.md", "utils/research_prompts.md")
    
    # Summary
    print("\n" + "="*60)
    if all_ok:
        print("✅ STRUCTURE VERIFICATION SUCCESSFUL")
        print("="*60)
        print("\nAll files are in place!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run full verification: python verify_system.py")
        print("  3. Start using: python main.py")
    else:
        print("❌ STRUCTURE VERIFICATION FAILED")
        print("="*60)
        print("\nSome files are missing. Please check the errors above.")
    
    return all_ok


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
