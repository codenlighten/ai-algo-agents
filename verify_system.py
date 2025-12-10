"""
Verification script to check system is working correctly
Run: python verify_system.py
"""
import sys
import os

def check_imports():
    """Verify all required packages are installed"""
    print("🔍 Checking imports...")
    
    errors = []
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        errors.append(f"  ✗ PyTorch: {e}")
    
    try:
        import openai
        print(f"  ✓ OpenAI SDK")
    except ImportError as e:
        errors.append(f"  ✗ OpenAI: {e}")
    
    try:
        from rich import print as rprint
        print(f"  ✓ Rich")
    except ImportError as e:
        errors.append(f"  ✗ Rich: {e}")
    
    try:
        from dotenv import load_dotenv
        print(f"  ✓ python-dotenv")
    except ImportError as e:
        errors.append(f"  ✗ python-dotenv: {e}")
    
    return errors


def check_modules():
    """Verify all custom modules can be imported"""
    print("\n🔍 Checking custom modules...")
    
    errors = []
    
    try:
        from agents.base_agent import AgentTeam, AgentRole
        print(f"  ✓ agents.base_agent")
    except Exception as e:
        errors.append(f"  ✗ agents.base_agent: {e}")
    
    try:
        from optimizers.novel_optimizers import SecondOrderMomentumOptimizer
        print(f"  ✓ optimizers.novel_optimizers")
    except Exception as e:
        errors.append(f"  ✗ optimizers.novel_optimizers: {e}")
    
    try:
        from loss_functions.novel_losses import ConfidencePenalizedCrossEntropy
        print(f"  ✓ loss_functions.novel_losses")
    except Exception as e:
        errors.append(f"  ✗ loss_functions.novel_losses: {e}")
    
    try:
        from models.novel_architectures import DynamicDepthNetwork
        print(f"  ✓ models.novel_architectures")
    except Exception as e:
        errors.append(f"  ✗ models.novel_architectures: {e}")
    
    try:
        from research.proposal_system import ResearchProposalBuilder
        print(f"  ✓ research.proposal_system")
    except Exception as e:
        errors.append(f"  ✗ research.proposal_system: {e}")
    
    try:
        from experiments.experiment_framework import ExperimentRunner
        print(f"  ✓ experiments.experiment_framework")
    except Exception as e:
        errors.append(f"  ✗ experiments.experiment_framework: {e}")
    
    return errors


def check_env():
    """Check .env configuration"""
    print("\n🔍 Checking .env configuration...")
    
    errors = []
    
    if not os.path.exists('.env'):
        errors.append("  ✗ .env file not found")
        return errors
    
    print("  ✓ .env file exists")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"  ✓ OPENAI_API_KEY is set")
    else:
        errors.append("  ✗ OPENAI_API_KEY not set in .env")
    
    return errors


def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\n🔍 Testing basic functionality...")
    
    errors = []
    
    # Test optimizer
    try:
        import torch
        import torch.nn as nn
        from optimizers.novel_optimizers import SecondOrderMomentumOptimizer
        
        model = nn.Linear(10, 5)
        optimizer = SecondOrderMomentumOptimizer(model.parameters(), lr=0.01)
        
        x = torch.randn(8, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        
        print("  ✓ SecondOrderMomentumOptimizer works")
    except Exception as e:
        errors.append(f"  ✗ Optimizer test failed: {e}")
    
    # Test loss function
    try:
        import torch
        from loss_functions.novel_losses import ConfidencePenalizedCrossEntropy
        
        criterion = ConfidencePenalizedCrossEntropy()
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        loss = criterion(logits, targets)
        
        print("  ✓ ConfidencePenalizedCrossEntropy works")
    except Exception as e:
        errors.append(f"  ✗ Loss function test failed: {e}")
    
    # Test architecture
    try:
        import torch
        from models.novel_architectures import DynamicDepthNetwork
        
        # Fix: hidden_dim should match or be compatible with input_dim for residual connections
        model = DynamicDepthNetwork(input_dim=10, hidden_dim=10, output_dim=5, max_layers=3)
        x = torch.randn(8, 10)
        output = model(x)
        
        print("  ✓ DynamicDepthNetwork works")
    except Exception as e:
        errors.append(f"  ✗ Architecture test failed: {e}")
    
    # Test proposal system
    try:
        from research.proposal_system import ResearchProposalBuilder, ExperimentalSetup
        
        setup = ExperimentalSetup(
            datasets=["test"],
            metrics=["accuracy"],
            baselines=["baseline"],
            expected_improvements={"accuracy": "10%"},
            minimal_compute_requirements="1 GPU"
        )
        
        builder = ResearchProposalBuilder()
        builder.set_title("Test")
        builder.set_author("Test")
        builder.set_core_concept("Concept")
        builder.set_summary("Summary")
        builder.set_related_work("Work")
        builder.set_novelty("Novel")
        builder.set_implementation("PyTorch", "code")
        builder.set_experimental_setup(setup)
        builder.set_scalability("Good")
        builder.set_engineering_constraints("None")
        builder.set_reasoning_path("Path")
        proposal = builder.build()
        
        print("  ✓ ResearchProposalBuilder works")
    except Exception as e:
        errors.append(f"  ✗ Proposal system test failed: {e}")
    
    return errors


def main():
    """Run all verification checks"""
    print("="*60)
    print("AI Research Agent System - Verification")
    print("="*60)
    
    all_errors = []
    
    # Check imports
    errors = check_imports()
    all_errors.extend(errors)
    
    # Check modules
    errors = check_modules()
    all_errors.extend(errors)
    
    # Check .env
    errors = check_env()
    all_errors.extend(errors)
    
    # Test functionality
    errors = test_basic_functionality()
    all_errors.extend(errors)
    
    # Summary
    print("\n" + "="*60)
    if all_errors:
        print("❌ VERIFICATION FAILED")
        print("="*60)
        print("\nErrors found:")
        for error in all_errors:
            print(error)
        print("\nPlease fix the errors above before using the system.")
        sys.exit(1)
    else:
        print("✅ VERIFICATION SUCCESSFUL")
        print("="*60)
        print("\nAll checks passed! The system is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python main.py")
        print("  2. Or try: python main.py --example")
        print("  3. Or run: python examples/example_proposals.py")
        sys.exit(0)


if __name__ == "__main__":
    main()
