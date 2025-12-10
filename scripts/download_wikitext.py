"""
Download and prepare WikiText-103 dataset for experiments
"""
from datasets import load_dataset
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import json

def download_wikitext():
    """Download WikiText-103 dataset"""
    print("Downloading WikiText-103 dataset...")
    
    # Download train/validation/test splits
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    
    print("\nDataset info:")
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Validation samples: {len(dataset['validation'])}")
    print(f"  Test samples: {len(dataset['test'])}")
    
    # Save statistics
    stats = {
        'train_size': len(dataset['train']),
        'val_size': len(dataset['validation']),
        'test_size': len(dataset['test']),
        'total_tokens_train': sum(len(item['text'].split()) for item in dataset['train']),
        'total_tokens_val': sum(len(item['text'].split()) for item in dataset['validation']),
        'total_tokens_test': sum(len(item['text'].split()) for item in dataset['test'])
    }
    
    # Save to disk for quick reference
    stats_dir = Path("data/wikitext")
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    with open(stats_dir / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n✅ Dataset downloaded and cached!")
    print(f"   Total train tokens: {stats['total_tokens_train']:,}")
    print(f"   Total val tokens: {stats['total_tokens_val']:,}")
    print(f"   Stats saved to: {stats_dir / 'dataset_stats.json'}")
    
    return dataset, stats

if __name__ == "__main__":
    dataset, stats = download_wikitext()
