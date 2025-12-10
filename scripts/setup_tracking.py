"""
Weights & Biases Setup for Experiment Tracking
"""
import wandb
import os
from pathlib import Path

def setup_wandb():
    """Initialize Weights & Biases for experiment tracking"""
    
    print("="*70)
    print("Weights & Biases Setup")
    print("="*70)
    
    # Check if already logged in
    try:
        wandb.login(relogin=True)
        print("\n✅ W&B authentication successful!")
    except Exception as e:
        print(f"\n⚠️  W&B login failed: {e}")
        print("\nTo set up W&B:")
        print("1. Go to https://wandb.ai/authorize")
        print("2. Copy your API key")
        print("3. Run: wandb login")
        return False
    
    # Initialize test project
    print("\n📊 Initializing project...")
    
    try:
        run = wandb.init(
            project="efficient-llm-training",
            name="test-setup",
            tags=["setup", "test"],
            config={
                "purpose": "Initial setup verification",
                "lab": "AI Research Lab - Efficiency Initiative"
            }
        )
        
        # Log test metrics
        wandb.log({
            "test_metric": 1.0,
            "setup_complete": True
        })
        
        wandb.finish()
        
        print("✅ W&B project initialized successfully!")
        print(f"   Project: efficient-llm-training")
        print(f"   Dashboard: {wandb.run.url if hasattr(wandb, 'run') else 'https://wandb.ai'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing project: {e}")
        return False

def create_config_templates():
    """Create experiment config templates"""
    
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    # AQL v2.0 config
    aql_config = {
        "experiment": "aql_v2",
        "model": {
            "type": "gpt2",
            "n_layers": 12,
            "n_heads": 12,
            "d_model": 768,
            "vocab_size": 50257
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 3e-4,
            "warmup_steps": 500,
            "max_steps": 50000,
            "gradient_accumulation": 4
        },
        "aql": {
            "enabled": True,
            "query_ratio": 0.2,
            "uncertainty_method": "laplace",
            "update_frequency": 1000
        },
        "logging": {
            "log_every": 100,
            "eval_every": 1000,
            "save_every": 5000
        }
    }
    
    import json
    with open(config_dir / "aql_v2_config.json", 'w') as f:
        json.dump(aql_config, f, indent=2)
    
    print(f"\n✅ Config templates created in: {config_dir}/")
    
    return config_dir

if __name__ == "__main__":
    print("\n🚀 Setting up experiment tracking infrastructure...\n")
    
    # Setup W&B
    success = setup_wandb()
    
    if success:
        # Create config templates
        config_dir = create_config_templates()
        
        print("\n" + "="*70)
        print("SETUP COMPLETE")
        print("="*70)
        print("\nYou're ready to track experiments!")
        print("\nNext steps:")
        print("  1. Review configs in: configs/")
        print("  2. Run experiments with W&B logging")
        print("  3. View results at: https://wandb.ai/")
    else:
        print("\n⚠️  Manual W&B setup required")
        print("   Run: wandb login")
