# 🚀 Quick Start: Your First Colab Run

## Step 1: Open Colab (2 minutes)

1. Click this link: https://colab.research.google.com/github/codenlighten/ai-algo-agents/blob/main/colab_setup.ipynb

2. Sign in with Google account

3. Check GPU: **Runtime** → **Change runtime type** → **GPU** → **Save**

---

## Step 2: Run Setup Cells (3 minutes)

Click the "play" button on these cells in order:

1. ✅ **Check GPU** - Should show T4/V100/A100
2. ✅ **Clone repo** - Downloads your code
3. ✅ **Install deps** - Installs PyTorch, transformers
4. ✅ **Verify** - Confirms everything works
5. ✅ **Config** - Auto-detects optimal settings
6. ⏭️ **Mount Drive** - Optional, for saving checkpoints

---

## Step 3: Start Training (1 click)

1. Run the **Training** cell
2. Watch the output for progress
3. Expected time: 1.5-3 hours for 125M model

---

## Step 4: Monitor (Optional)

Open new cell and run:
```python
!nvidia-smi
```

Should show:
- GPU Util: 95-100% ✅
- Temp: 60-75°C ✅ (much cooler than your local!)
- Memory: 6-8GB used ✅

---

## 🎯 What You'll Get

After 1.5-3 hours:
- ✅ Trained 125M model
- ✅ Validation perplexity ~30
- ✅ Checkpoints saved
- ✅ Training curves plotted
- ✅ No thermal throttling!

---

## 💰 Cost

- **This first run:** $0 (free tier)
- **Week 1-2:** $0 (free tier sufficient)
- **Week 3-4:** $50 (Pro+ for 350M models)

---

## 🆘 If Something Goes Wrong

### "Runtime disconnected"
→ Just rerun the training cell, it will resume from checkpoint

### "Out of memory"
→ In Config cell, change:
```python
CONFIG["batch_size"] = 6  # Was 8
```

### "No GPU available"
→ Runtime → Restart runtime → Change runtime type → GPU

---

## 📊 Expected Results (125M model)

```
Step 0:    ppl=52,489  (random init)
Step 1000: ppl=971     (90× better!)
Step 2000: ppl=609     (36% improvement)
Step 10000: ppl=93     (Final, excellent!)
```

---

## ✅ Success Checklist

After your first run, you should have:

- [ ] Training completed to 10K steps
- [ ] Final validation perplexity < 100
- [ ] Checkpoints saved (5-10 files)
- [ ] Training curves plotted
- [ ] No OOM errors
- [ ] No thermal issues

---

## 🎉 Next Steps

After this works:

1. **Compare to dense baseline** (add `--sparsity 0.0`)
2. **Try different sparsity levels** (0.7, 0.8, 0.9)
3. **Scale to 350M** (upgrade to Pro+, use `model_size=medium`)
4. **Run ablations** (remove distillation, ES, etc.)

---

## 🔗 Links

- **Colab Notebook:** https://colab.research.google.com/github/codenlighten/ai-algo-agents/blob/main/colab_setup.ipynb
- **Full Guide:** See COLAB_SETUP.md
- **GitHub Repo:** https://github.com/codenlighten/ai-algo-agents

---

**Ready?** Click the Colab link above and run all cells! 🚀

**Estimated total time:** 5 min setup + 2 hours training = **2 hours 5 minutes**  
**Cost:** **$0**  
**Benefit:** **No more thermal throttling!** 🎉
