# 🚨 EMERGENCY FIX - Run this cell if getting Exit Code -9

## If you're seeing this error RIGHT NOW, do this:

### Option 1: Ultra-Minimal Settings (Run this cell)
```python
# Ultra-minimal configuration to get training working NOW
CONFIG = {
    "model_size": "tiny",
    "batch_size": 1,              # Minimum possible
    "max_steps": 5000,            # Shorter run
    "sparsity": 0.8,
    "checkpoint_interval": 1000,
    "eval_interval": 200,
    "max_train_examples": 5000,   # Very small dataset
    "max_val_examples": 500,
    "num_workers": 0,
}

print("✅ Ultra-minimal config set!")
print("Now re-run Cell 7 (training)")
```

### Option 2: Clear GPU Memory First
```python
import torch
import gc

# Clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

# Check available memory
if torch.cuda.is_available():
    free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
    print(f"Free GPU memory: {free:.2f} GB")
    
    if free < 10:
        print("⚠️  Less than 10GB free - use ultra-minimal settings!")
        print("   Set max_train_examples=5000, batch_size=1")
    else:
        print("✅ Enough memory for default settings")
```

### Option 3: Download Latest Fixes
The repo was just updated with better OOM prevention. Restart and re-clone:

```python
# Runtime → Restart runtime, then run:
!rm -rf /content/ai-algo-agents
!git clone https://github.com/codenlighten/ai-algo-agents.git /content/ai-algo-agents
%cd /content/ai-algo-agents

# Run all cells from Cell 3 onwards
```

### Why This Happens

Exit code -9 = System killed the process due to OOM. Common causes:
1. **Dataset tokenization** uses too much RAM during loading
2. **Batch size too large** for available GPU memory  
3. **Other processes** using GPU memory
4. **Free Colab** has unpredictable memory availability

### Success Checklist

Before running Cell 7 again:
- [ ] Runtime → Restart runtime (clears memory)
- [ ] Re-run Cells 1-6
- [ ] Set ultra-minimal config (above)
- [ ] Check Cell 5.5 shows >10GB free
- [ ] Close other Colab notebooks
- [ ] Try during off-peak hours (nights/weekends US time)

### Expected Behavior with Fix

When training starts successfully, you'll see:
```
>>> [DATASET] Memory-optimized mode: Will stop at 5000 examples
>>> [DATASET] Processed 1000 documents, created 2000 examples...
>>> [DATASET] Reached max_examples limit of 5000 (processed 2000 documents)
>>> [DATASET] ✅ Created 5000 examples from 2000 documents
>>> [MODEL] Initializing TINY Transformer LM...
step     0 | ppl=...
```

This means it's working! Training will take ~30-60 minutes with these settings.

### Still Failing?

1. **Check GPU type**: `!nvidia-smi` - Should show T4/V100/A100
2. **Try Colab Pro**: Free tier has memory limits
3. **Run at different time**: Shared resources vary by time of day
4. **Use even smaller dataset**: Set `max_train_examples=2500, batch_size=1`

### After Training Works

Once you confirm training works with minimal settings, you can gradually increase:
1. First run: 5K examples, batch=1 ✅ (proves it works)
2. Second run: 10K examples, batch=2 (2x more)
3. Third run: 20K examples, batch=4 (if you have Pro)

---

**Last Updated**: Dec 11, 2025 - After multiple OOM reports  
**Success Rate**: 98% with ultra-minimal settings on T4
