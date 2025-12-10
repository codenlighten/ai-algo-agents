"""
Quick Start Example: Training with AQL v2.0
Demonstrates the complete workflow from data loading to training
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.aql_v2.aql_v2_trainer import AQLv2Trainer
from models.novel_architectures import SimpleTransformer


def create_simple_transformer(vocab_size=5000, d_model=256, nhead=8, num_layers=4):
    """Create a simple transformer for language modeling"""
    model = nn.Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=d_model * 4,
        dropout=0.1
    )
    
    # Add embedding and output layers
    embedding = nn.Embedding(vocab_size, d_model)
    output_layer = nn.Linear(d_model, vocab_size)
    
    class TransformerLM(nn.Module):
        def __init__(self, transformer, embedding, output):
            super().__init__()
            self.embedding = embedding
            self.transformer = transformer
            self.output = output
            self.d_model = d_model
        
        def forward(self, x):
            # x: [batch, seq_len]
            x = self.embedding(x) * (self.d_model ** 0.5)
            x = x.transpose(0, 1)  # [seq_len, batch, d_model]
            
            # Transformer expects src and tgt
            out = self.transformer.encoder(x)
            out = out.transpose(0, 1)  # [batch, seq_len, d_model]
            out = out.mean(dim=1)  # Simple pooling
            
            return self.output(out)
    
    return TransformerLM(model, embedding, output_layer)


def load_wikitext_dataset(data_dir='data/wikitext', max_samples=10000):
    """
    Load WikiText-103 dataset (simplified version for quick testing)
    For full training, use the datasets library
    """
    from datasets import load_from_disk
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        print("Please run: python scripts/download_wikitext.py")
        return None, None
    
    try:
        dataset = load_from_disk(str(data_path))
        
        # Simple tokenization (for demonstration - use proper tokenizer in production)
        def simple_tokenize(examples):
            # This is just a placeholder - use proper tokenizer
            return {'input_ids': [[ord(c) % 5000 for c in text[:50]] for text in examples['text']]}
        
        train_dataset = dataset['train'].select(range(min(max_samples, len(dataset['train']))))
        val_dataset = dataset['validation'].select(range(min(max_samples // 10, len(dataset['validation']))))
        
        # Note: This is simplified - proper implementation needs tokenization
        print(f"✅ Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


def create_toy_dataset(n_train=1000, n_val=200, vocab_size=5000, seq_len=50):
    """Create a toy dataset for quick testing"""
    class ToyDataset(torch.utils.data.Dataset):
        def __init__(self, n_samples, vocab_size, seq_len):
            self.data = torch.randint(0, vocab_size, (n_samples, seq_len))
            # Create targets (next token prediction)
            self.targets = torch.randint(0, vocab_size, (n_samples,))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    train_dataset = ToyDataset(n_train, vocab_size, seq_len)
    val_dataset = ToyDataset(n_val, vocab_size, seq_len)
    
    print(f"✅ Created toy dataset: {n_train} train, {n_val} val samples")
    return train_dataset, val_dataset


def main():
    """Main training workflow"""
    print("=" * 70)
    print("AQL v2.0 Quick Start Example")
    print("=" * 70)
    
    # Configuration
    config = {
        # Model
        'vocab_size': 5000,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        
        # Optimization
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        
        # AQL v2.0
        'use_curriculum_selection': True,
        'use_streaming': False,
        'uncertainty_ema': 0.95,
        'uncertainty_update_freq': 10,
        
        # Curriculum
        'total_steps': 1000,
        'warmup_ratio': 0.2,
        'pacing_function': 'root',
        
        # Training
        'batch_size': 32,
        'epochs': 5,
        
        # Checkpointing
        'save_path': 'experiments/checkpoints/aql_v2_quickstart/'
    }
    
    print("\n1️⃣ Creating model...")
    model = create_simple_transformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    )
    print(f"   Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    print("\n2️⃣ Loading dataset...")
    # Try to load real dataset, fall back to toy dataset
    train_dataset, val_dataset = load_wikitext_dataset(max_samples=1000)
    
    if train_dataset is None:
        print("   Using toy dataset for demonstration...")
        train_dataset, val_dataset = create_toy_dataset(
            n_train=1000,
            n_val=200,
            vocab_size=config['vocab_size']
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print("\n3️⃣ Setting up AQL v2.0 trainer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    trainer = AQLv2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    print("\n4️⃣ Starting training...")
    results = trainer.train(epochs=config['epochs'])
    
    print("\n" + "=" * 70)
    print("Training Complete! 🎉")
    print("=" * 70)
    print(f"Final Train Accuracy: {results['final_train_acc']:.4f}")
    print(f"Final Val Accuracy:   {results['final_val_acc']:.4f}")
    print(f"Best Val Accuracy:    {results['best_val_acc']:.4f}")
    
    print("\n5️⃣ Saving metrics...")
    metrics_path = Path(config['save_path']) / 'metrics.json'
    trainer.save_metrics(metrics_path)
    
    print("\n✅ All done! Check the checkpoint directory for saved models.")
    print(f"   Location: {config['save_path']}")


if __name__ == "__main__":
    main()
