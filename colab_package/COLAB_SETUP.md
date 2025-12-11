# 🚀 Google Colab Setup Guide for SparsAE

## Quick Start (5 minutes)

### Option 1: Direct Colab Link (Easiest)
1. Click here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/codenlighten/ai-algo-agents/blob/main/colab_setup.ipynb)
2. Run all cells (Runtime → Run all)
3. Monitor training progress

### Option 2: Manual Upload
1. Go to [Google Colab](https://colab.research.google.com)
2. Upload `colab_setup.ipynb` from this repo
3. Run all cells

---

## 📊 GPU Tiers & Recommendations

| Tier | GPU | VRAM | Cost | Best For | Training Time (125M) |
|------|-----|------|------|----------|---------------------|
| **Free** | T4 | 15GB | $0 | Testing, 49-125M models | ~3 hrs |
| **Pro** | T4/V100 | 16GB | $10/mo | Regular dev, 125M | ~2 hrs |
| **Pro+** | V100/A100 | 40GB | $50/mo | 350M+ models | ~1 hr |

### How to Select GPU:
1. In Colab: **Runtime** → **Change runtime type**
2. Select **Hardware accelerator: GPU**
3. For Pro+: Select **High-RAM** if available

---

## 🎯 Recommended Workflow

### Week 1-2: Use Free Tier
**Goal:** Validate SparsAE on 125M model

```python
# In colab_setup.ipynb, use these settings:
CONFIG = {
    "model_size": "small",  # 125M
    "batch_size": 8,
    "max_steps": 10000,
}
```

**Cost:** $0  
**Time:** ~3 hours per run  
**Runs per week:** 20-30 (with session management)

### Week 3-4: Upgrade to Pro+ for 350M
**Goal:** Scale to larger models

```python
CONFIG = {
    "model_size": "medium",  # 350M
    "batch_size": 32,
    "max_steps": 20000,
}
```

**Cost:** $50/month  
**Time:** ~12 hours per run  
**Benefit:** 4× faster than local RTX 3070

---

## 💾 Data Persistence Strategy

### Option 1: Google Drive (Recommended)
```python
# In Colab notebook:
from google.colab import drive
drive.mount('/content/drive')

# Checkpoints saved to:
# /content/drive/MyDrive/sparsae_checkpoints/
```

**Pros:**
- ✅ Survives session disconnects
- ✅ Easy to download
- ✅ 15GB free storage

**Cons:**
- ⚠️ Slightly slower I/O

### Option 2: Local Checkpoints
```python
# Checkpoints in /content/checkpoints/
# Download before session ends!
```

**Pros:**
- ✅ Fastest I/O
- ✅ No Drive quotas

**Cons:**
- ❌ Lost on disconnect
- ❌ Must download manually

---

## 🔄 Resuming from Checkpoint

If your session disconnects:

```python
# In training cell, add:
!python experiments/sparsae_wikitext.py \
    --resume_from /content/drive/MyDrive/sparsae_checkpoints/checkpoint_step_5000.pt \
    --model_size small \
    --batch_size 8
```

---

## 📈 Monitoring Training

### Real-time GPU Monitoring
```python
# Run this in separate cell while training:
import time
from IPython.display import clear_output

for i in range(60):  # Monitor for 10 minutes
    clear_output(wait=True)
    !nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
    time.sleep(10)
```

### Expected Metrics (125M model):
- **GPU Util:** 95-100% ✅
- **Temp:** 60-75°C ✅ (much cooler than local!)
- **Memory:** 6-8GB / 15GB ✅
- **Steps/sec:** 2-3 ✅

---

## 🐛 Troubleshooting

### Issue 1: "Runtime disconnected"
**Cause:** Session timeout (12hrs free, 24hrs Pro)  
**Solution:**
```python
# Add keep-alive in first cell:
from IPython.display import Javascript
display(Javascript('''
    function ClickConnect(){
        console.log("Clicking connect");
        document.querySelector("colab-connect-button").shadowRoot.getElementById("connect").click()
    }
    setInterval(ClickConnect, 60000)
'''))
```

### Issue 2: "Out of memory"
**Solutions:**
```python
# Option A: Reduce batch size
CONFIG["batch_size"] = 6  # Instead of 8

# Option B: Reduce sequence length
CONFIG["seq_len"] = 128  # Instead of 256

# Option C: Use smaller model
CONFIG["model_size"] = "tiny"  # 49M instead of 125M
```

### Issue 3: "GPU not available"
**Solutions:**
1. Check runtime: Runtime → Change runtime type → GPU
2. Try reconnecting: Runtime → Restart runtime
3. Free tier limits reached (wait 12-24 hours)

### Issue 4: Slow training
**Checks:**
```python
# Verify GPU is being used:
!nvidia-smi

# Check data loading:
import torch
print(f"Num workers: {train_loader.num_workers}")  # Should be 2+
print(f"Pin memory: {train_loader.pin_memory}")    # Should be True
```

---

## 💰 Cost Management

### Free Tier Strategy:
- Use during off-peak hours (better GPU availability)
- Run shorter experiments (< 5 hours)
- Save checkpoints every 1000 steps
- Download results immediately

### Pro Strategy ($10/mo):
- Longer sessions (24hrs)
- Priority GPU access
- Background execution
- Worth it for 125M experiments

### Pro+ Strategy ($50/mo):
- A100 access (essential for 350M+)
- 40GB VRAM
- 4× faster training
- **Recommended for Week 3-4 of roadmap**

---

## 📊 Cost Comparison: Local vs Colab

### 125M Model (10K steps):
```
Local RTX 3070:
  - Time: 48 hours (thermal throttling)
  - Cost: $0 (electricity: ~$2)
  - Issues: Overheating, slow

Colab Free (T4):
  - Time: 3 hours
  - Cost: $0
  - Issues: Session limits

Colab Pro (V100):
  - Time: 1.5 hours
  - Cost: $10/mo (unlimited runs)
  - Issues: None

Verdict: Colab Pro wins! ✅
```

### 350M Model (20K steps):
```
Local RTX 3070:
  - Status: ❌ Won't fit (OOM)
  - Time: N/A
  - Cost: N/A

Colab Pro (V100):
  - Status: ⚠️ Barely fits
  - Time: 18 hours
  - Cost: $10/mo

Colab Pro+ (A100):
  - Status: ✅ Fits comfortably
  - Time: 12 hours
  - Cost: $50/mo

Verdict: Colab Pro+ essential ✅
```

---

## 🎓 Best Practices

### 1. Checkpoint Often
```python
CONFIG["checkpoint_interval"] = 500  # Every 500 steps
```

### 2. Log Everything
```python
import pandas as pd

metrics = []
# In training loop:
metrics.append({
    'step': step,
    'loss': loss.item(),
    'ppl': ppl,
    'gpu_mem': torch.cuda.memory_allocated() / 1e9
})

# Save periodically:
pd.DataFrame(metrics).to_csv('metrics.csv', index=False)
```

### 3. Use Wandb (Optional)
```python
!pip install wandb
import wandb

wandb.init(project="sparsae", config=CONFIG)
# Log metrics: wandb.log({"loss": loss, "step": step})
```

### 4. Download Results
```python
# At end of training:
from google.colab import files

!tar -czf results.tar.gz checkpoints/ *.csv *.png
files.download('results.tar.gz')
```

---

## 🚀 Advanced: Multi-Run Experiments

### Hyperparameter Sweep:
```python
configs = [
    {"model_size": "small", "sparsity": 0.7, "batch_size": 8},
    {"model_size": "small", "sparsity": 0.8, "batch_size": 8},
    {"model_size": "small", "sparsity": 0.9, "batch_size": 8},
]

for i, config in enumerate(configs):
    print(f"\n{'='*60}")
    print(f"Run {i+1}/{len(configs)}: {config}")
    print(f"{'='*60}\n")
    
    !python experiments/sparsae_wikitext.py \
        --model_size {config['model_size']} \
        --sparsity {config['sparsity']} \
        --batch_size {config['batch_size']} \
        --checkpoint_dir /content/drive/MyDrive/sparsae_sweep/run_{i}
```

---

## 📞 Getting Help

1. **GitHub Issues:** https://github.com/codenlighten/ai-algo-agents/issues
2. **Colab Docs:** https://colab.research.google.com/notebooks/
3. **Stack Overflow:** Tag with `google-colab` and `pytorch`

---

## ✅ Checklist: First Colab Run

- [ ] GPU selected (Runtime → Change runtime type → GPU)
- [ ] Repository cloned successfully
- [ ] Dependencies installed (torch, transformers, datasets)
- [ ] Google Drive mounted (if using checkpoints)
- [ ] Config set (model_size, batch_size, max_steps)
- [ ] Training started (check GPU utilization)
- [ ] Checkpoints saving correctly
- [ ] Monitoring GPU temperature (<80°C)

---

**Ready to train?** Open `colab_setup.ipynb` and run all cells! 🚀

**Estimated setup time:** 5 minutes  
**Estimated training time (125M):** 1.5-3 hours  
**Estimated cost (Week 1-2):** $0 (Free tier)

**Questions?** Open an issue on GitHub!
