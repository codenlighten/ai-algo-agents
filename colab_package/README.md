# 🚀 Google Colab Training Package

This folder contains everything needed to run SparsAE training on Google Colab.

## 📦 Contents

### 1. **colab_setup.ipynb** - Main Jupyter Notebook
- Complete training workflow for Google Colab
- Auto-detects GPU type (T4/V100/A100) and configures settings
- Includes GPU monitoring, checkpoint management, and result viewing
- **Quick Start**: Upload to Colab → Runtime → Run all

### 2. **COLAB_QUICKSTART.md** - 5-Minute Quick Start
- Fastest way to get started
- Expected results and timing
- Success checklist
- **Use this first** if you just want to run training immediately

### 3. **COLAB_SETUP.md** - Comprehensive Guide
- Detailed documentation (20+ sections)
- GPU tier comparison and cost analysis
- Data persistence strategies (Google Drive)
- Monitoring, troubleshooting, and best practices
- **Use this** for understanding the full system

### 4. **COLAB_PRIVATE_REPO.md** - Private Repository Guide
- Solutions for using private GitHub repos
- Personal Access Token setup
- Alternative upload methods
- **Note**: Our repo is now public, so this is optional

### 5. **BUFFERING_FIX.md** - Technical Details ⭐ NEW
- Explains the "silent hang" issue and how we fixed it
- Python `-u` flag and `flush=True` usage
- Useful for understanding output buffering in Colab

### 6. **TROUBLESHOOTING_CELLS.md** - Diagnostic Code ⭐ NEW
- Ready-to-use cells for testing and debugging
- Quick test, GPU monitoring, checkpoint viewing
- Resume from checkpoint, force stop training
- Copy-paste into your notebook as needed

---

## 🎯 How to Use

### For First-Time Users:
1. Open **COLAB_QUICKSTART.md** - read the 5-minute guide
2. Go to [Google Colab](https://colab.research.google.com)
3. Upload **colab_setup.ipynb** (File → Upload notebook)
4. Runtime → Change runtime type → **GPU**
5. Runtime → Run all (Ctrl+F9)
6. Wait 1.5-2 hours for training to complete

### For Detailed Setup:
1. Read **COLAB_SETUP.md** for full understanding
2. Follow the workflow recommendations by week
3. Set up Google Drive for checkpoint persistence
4. Monitor GPU with the built-in cells

---

## 💰 Cost Summary

| Tier | GPU | VRAM | Cost/Month | Best For |
|------|-----|------|------------|----------|
| **Free** | T4 | 15GB | $0 | Week 1-2 (49M/125M models) |
| **Pro** | V100 | 16GB | $10 | Week 1-2 (faster training) |
| **Pro+** | A100 | 40GB | $50 | Week 3-4 (350M model) |

**Recommendation**: Start with **Free tier** for initial experiments. Upgrade to Pro+ only when scaling to 350M model in Week 3.

---

## 📊 Expected Results

### Tiny Model (49M params, 80% sparse):
- Training time: ~2 hours on T4
- Initial perplexity: ~52,000
- Final perplexity: ~93 (training), ~609 (validation)
- Memory usage: ~4GB VRAM

### Small Model (125M params, 80% sparse):
- Training time: ~2-3 hours on T4
- Initial perplexity: ~52,000
- Final perplexity: ~93 (training), ~609 (validation)
- Memory usage: ~8GB VRAM

### Medium Model (350M params, 80% sparse):
- Training time: ~12 hours on A100
- Requires Pro+ tier ($50/month)
- Memory usage: ~25GB VRAM

---

## 🔧 Troubleshooting

### Error: "CUDA out of memory"
**Solution**: Reduce batch size in Cell 5:
```python
CONFIG["batch_size"] = 4  # Reduce from 8
```

### Error: "Repository not found"
**Solution**: Repository is now public, should work automatically. If not, see **COLAB_PRIVATE_REPO.md**

### Error: "Session disconnected"
**Solution**: 
1. Mount Google Drive (Cell 6) to save checkpoints
2. Rerun Cell 7 with `--resume_from` flag:
```python
cmd.extend(["--resume_from", "/content/drive/MyDrive/sparsae_checkpoints/checkpoint_step_5000.pt"])
```

---

## 📁 File Structure

```
colab_package/
├── README.md                    # This file
├── colab_setup.ipynb           # Main notebook (upload to Colab)
├── COLAB_QUICKSTART.md         # 5-min quick start guide
├── COLAB_SETUP.md              # Full documentation
└── COLAB_PRIVATE_REPO.md       # Private repo setup (optional)
```

---

## 🎓 For the Team

### Week 1-2 Goals (Free Tier):
- ✅ Validate SparsAE on WikiText-103
- ✅ Run dense baseline comparison
- ✅ Test static pruning baseline
- ✅ Implement RigL baseline
- ✅ Generate initial results for Paper 1

### Week 3-4 Goals (Pro+ Tier):
- Scale to 350M model
- Integrate AQL v2.0
- Run ablation studies
- Prepare Paper 1 submission

### Success Metrics:
- **Training completes** without OOM errors
- **Final validation perplexity < 700** (target: 609)
- **Checkpoints saved** to Google Drive
- **GPU utilization > 90%** during training
- **Temperature < 80°C** (healthy range)

---

## 🔗 Links

- **Main Repository**: https://github.com/codenlighten/ai-algo-agents
- **Google Colab**: https://colab.research.google.com
- **Direct Notebook Link**: https://colab.research.google.com/github/codenlighten/ai-algo-agents/blob/main/colab_setup.ipynb

---

## 📞 Support

If you encounter issues:
1. Check **COLAB_SETUP.md** troubleshooting section
2. Run diagnostic Cell 6.5 in the notebook
3. Check GitHub Issues: https://github.com/codenlighten/ai-algo-agents/issues

---

## 🎉 Quick Win

**Fastest path to results**:
1. Upload `colab_setup.ipynb` to Colab
2. Select GPU runtime
3. Run all cells
4. Come back in 2 hours
5. Download checkpoints from Google Drive

That's it! Training will complete automatically and save results to your Drive.

---

**Last Updated**: December 11, 2025  
**Status**: ✅ Repository is public, ready to use  
**Tested On**: T4 GPU (Google Colab Free Tier)
