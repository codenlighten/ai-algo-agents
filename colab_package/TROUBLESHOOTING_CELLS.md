# 🧪 Colab Troubleshooting Cells

Add these cells to your Colab notebook for quick testing and debugging.

---

## Cell: Quick Test (Tiny Run)

Test that everything works with a minimal configuration:

```python
# Quick test: 20 steps, tiny model, should finish in ~2 minutes
!python3 -u experiments/sparsae_wikitext.py \
  --model_size tiny \
  --batch_size 2 \
  --max_steps 20 \
  --sparsity 0.8 \
  --checkpoint_dir /content/test_checkpoints \
  --checkpoint_interval 10 \
  --eval_interval 10
```

**Expected output**: Completes in ~2 minutes with perplexity dropping from ~50k to ~5k

---

## Cell: Check GPU Usage

Monitor GPU in real-time while training runs:

```python
import time
from IPython.display import clear_output

print("📊 GPU Monitoring (Ctrl+C to stop)\n")

try:
    while True:
        clear_output(wait=True)
        !nvidia-smi --query-gpu=timestamp,name,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total --format=csv
        print("\n✅ Healthy: Temp<80°C, Util>90%, Memory<90%")
        print("⏸️  Press Stop button to end monitoring")
        time.sleep(5)
except KeyboardInterrupt:
    print("\n⏹️  Monitoring stopped")
```

---

## Cell: Check Training Progress

View the most recent training output:

```python
# Check if training script is still running
import subprocess
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
python_procs = [line for line in result.stdout.split('\n') if 'sparsae_wikitext.py' in line]

if python_procs:
    print("✅ Training is running:")
    for proc in python_procs:
        print(f"   {proc}")
else:
    print("⚠️  No training process found")
    
# Check GPU memory (indicates if model is loaded)
print("\n📊 GPU Status:")
!nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
```

---

## Cell: View Partial Results

Check checkpoints saved so far:

```python
import os

checkpoint_dir = CONFIG.get('checkpoint_dir', '/content/checkpoints')

if os.path.exists(checkpoint_dir):
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
    
    if checkpoints:
        print(f"✅ Found {len(checkpoints)} checkpoint(s):\n")
        for cp in checkpoints:
            path = os.path.join(checkpoint_dir, cp)
            size_mb = os.path.getsize(path) / 1e6
            # Extract step number from filename
            step = cp.split('_')[-1].replace('.pt', '')
            print(f"   Step {step:>6}: {cp} ({size_mb:.1f} MB)")
    else:
        print("⚠️  No checkpoints yet (saved every 1000 steps)")
else:
    print(f"❌ Checkpoint directory doesn't exist: {checkpoint_dir}")
```

---

## Cell: Test Script Imports

Verify the training script can be imported:

```python
import sys
sys.path.insert(0, '/content/ai-algo-agents')

try:
    print("Testing imports...")
    from experiments import sparsae_wikitext
    print("✅ Script imports successfully")
    
    # Test argument parser
    print("\nTesting argument parser...")
    !python3 experiments/sparsae_wikitext.py --help | head -30
    
except Exception as e:
    print(f"❌ Import failed: {e}")
```

---

## Cell: Force Stop Training

If you need to stop a hung process:

```python
import subprocess
import signal

# Find training processes
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
lines = [line for line in result.stdout.split('\n') if 'sparsae_wikitext.py' in line]

if lines:
    print("🛑 Found training process(es):")
    for line in lines:
        # Extract PID (second column)
        pid = int(line.split()[1])
        print(f"   PID {pid}: Killing...")
        try:
            subprocess.run(['kill', '-9', str(pid)])
            print(f"   ✅ Killed PID {pid}")
        except Exception as e:
            print(f"   ❌ Failed to kill: {e}")
else:
    print("✅ No training processes found")
```

---

## Cell: Download Checkpoints Now

Download checkpoints even if training isn't finished:

```python
from google.colab import files
import os
import tarfile

checkpoint_dir = CONFIG.get('checkpoint_dir', '/content/checkpoints')

if os.path.exists(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if checkpoints:
        # Create tar.gz
        archive_name = 'partial_checkpoints.tar.gz'
        print(f"📦 Creating archive with {len(checkpoints)} checkpoint(s)...")
        
        with tarfile.open(archive_name, 'w:gz') as tar:
            for cp in checkpoints:
                tar.add(os.path.join(checkpoint_dir, cp), arcname=cp)
        
        size_mb = os.path.getsize(archive_name) / 1e6
        print(f"✅ Archive created: {size_mb:.1f} MB")
        print("📥 Downloading...")
        
        files.download(archive_name)
        print("✅ Download complete!")
    else:
        print("⚠️  No checkpoints to download yet")
else:
    print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
```

---

## Cell: Resume from Checkpoint

If training stopped, resume from last checkpoint:

```python
import os

checkpoint_dir = CONFIG.get('checkpoint_dir', '/content/checkpoints')

if os.path.exists(checkpoint_dir):
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
    
    if checkpoints:
        latest = checkpoints[-1]
        latest_path = os.path.join(checkpoint_dir, latest)
        step = latest.split('_')[-1].replace('.pt', '')
        
        print(f"📍 Found latest checkpoint: {latest} (step {step})")
        print(f"🚀 Resuming training...\n")
        
        !python3 -u experiments/sparsae_wikitext.py \
          --model_size {CONFIG['model_size']} \
          --batch_size {CONFIG['batch_size']} \
          --max_steps {CONFIG['max_steps']} \
          --sparsity {CONFIG['sparsity']} \
          --checkpoint_dir {CONFIG['checkpoint_dir']} \
          --checkpoint_interval {CONFIG['checkpoint_interval']} \
          --eval_interval {CONFIG['eval_interval']} \
          --resume_from {latest_path}
    else:
        print("⚠️  No checkpoints found to resume from")
else:
    print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
```

---

## Usage

1. Copy any of these cells into your Colab notebook
2. Run them as needed during or after training
3. They're independent and can run in any order

## Most Useful Cells

- **Quick Test** - Verify setup works (run first!)
- **Check GPU Usage** - Monitor while training (open in split view)
- **View Partial Results** - See progress without waiting for completion
- **Resume from Checkpoint** - Continue after disconnect

---

**Tip**: Add these to your notebook as hidden cells (View → Show/hide code) for quick access to diagnostics.
