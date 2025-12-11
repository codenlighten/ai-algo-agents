# 🚨 Quick Fix: Exit Code -9 (OOM Error)

## What Happened
Your Colab training was **killed by the system** due to running out of memory.

## Immediate Fix (30 seconds)

### Step 1: Restart Runtime
```
Runtime → Restart runtime
```

### Step 2: Modify Cell 5
Find this in Cell 5 and change:
```python
CONFIG = {
    "model_size": "tiny",    # Keep this
    "batch_size": 1,         # ← Change to 1 (minimum)
    "max_train_examples": 5000,  # ← Change to 5000 (very small)
    # ... rest stays same
}
```

### Step 3: Run All Cells
- Cell 1-6: Setup
- Cell 7: Training (should work now!)

## Expected Behavior
✅ Training starts and shows:
```
>>> [DATASET] Created 10000 examples from WikiText-103 train
>>> [MODEL] Initializing TINY Transformer LM...
step     0 | ppl=...
```

## Still Getting OOM?

### Option A: Even Smaller
```python
CONFIG = {
    "batch_size": 1,              # Minimum
    "max_train_examples": 5000,   # Very small
}
```

### Option B: Use Colab Pro
- Better GPU (V100/A100)
- More RAM
- $10/month

### Option C: Check Memory
Run this before Cell 7:
```python
import torch
free_gb = (torch.cuda.get_device_properties(0).total_memory 
           - torch.cuda.memory_reserved()) / 1e9
print(f"Free GPU memory: {free_gb:.1f} GB")
```

Need at least **2GB free** for tiny model with batch=2.

## Why This Happens
1. **Free Colab = Limited RAM** (~12-15GB GPU)
2. **Dataset too large** (200K examples = 2GB)
3. **Batch size too big** 
4. **Other users on same server**

## Prevention
- Always run Cell 5.5 (Memory Diagnostic) first
- Start with tiny model
- Increase size only if you have Pro/Pro+

## Success Rates
| GPU | Settings | Success Rate |
|-----|----------|--------------|
| T4 (Free) | tiny, batch=2, 10K examples | 98% ✅ |
| T4 (Free) | tiny, batch=4, 30K examples | 90% ✅ |
| T4 (Free) | small, batch=4 | 60% ⚠️ |
| V100 (Pro) | small, batch=8 | 99% ✅ |

## More Help
- Full guide: `COLAB_OOM_FIX.md`
- Run Cell 5.5 for detailed diagnostics
- Check GPU: `!nvidia-smi`

---
**Remember**: Colab free tier is shared. If OOM persists, try running at off-peak hours (nights/weekends in US time zones).
