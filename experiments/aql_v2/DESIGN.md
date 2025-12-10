# AQL v2.0 Design Document

**Project:** Adaptive Query-Based Learning - Version 2.0  
**Goal:** Optimize AQL for <3% computational overhead while maintaining 5x+ data efficiency  
**Date:** December 10, 2025  
**Status:** 🔄 Design Phase

---

## 📊 Current Status: AQL v1.0 Results

### What Works (MNIST Validation)
- ✅ **Accuracy improvement:** 94.00% vs 93.60% baseline (+0.4%)
- ✅ **More stable training:** Consistent learning progression
- ✅ **Concept validated:** Active learning works for deep learning
- ✅ **GPU compatible:** Runs on CUDA

### What Needs Improvement
- ⚠️ **Computational overhead:** 10% slower (Monte Carlo Dropout)
- ⚠️ **Uncertainty estimation:** 10 forward passes required
- ⚠️ **Memory:** Stores full dataset in memory
- ⚠️ **Scalability:** Hasn't been tested beyond small datasets

---

## 🎯 AQL v2.0 Objectives

### Primary Goals
1. **Reduce overhead to <3%** (currently 10%)
2. **Scale to 101M tokens** (WikiText-103)
3. **Achieve 5x data efficiency** (train on 20% of data)
4. **Maintain/improve accuracy** (match or beat baseline)

### Secondary Goals
5. Add curriculum learning integration
6. Implement streaming data selection (no full dataset in memory)
7. Meta-learn the selection policy
8. Support distributed training

---

## 🔬 Technical Design

### Component 1: Efficient Uncertainty Estimation

#### Problem with MC Dropout (v1.0)
```python
# Current approach (10% overhead)
def uncertainty_estimate_v1(model, data, n_samples=10):
    model.train()  # Keep dropout active
    predictions = []
    for _ in range(n_samples):  # 10 forward passes!
        outputs = model(data)
        predictions.append(outputs)
    return torch.var(torch.stack(predictions), dim=0)
```

#### Solution 1: Laplace Approximation (RECOMMENDED)
```python
# Proposed approach (~1% overhead)
class LaplacianUncertainty:
    """
    Single-pass uncertainty via Laplace approximation
    Based on local Hessian estimation
    """
    def __init__(self, model):
        self.model = model
        self.fisher_matrix = None  # Approximate Hessian
        
    def update_fisher(self, data, targets, ema_decay=0.95):
        """Update Fisher information matrix efficiently"""
        outputs = self.model(data)
        loss = F.cross_entropy(outputs, targets)
        
        # Compute gradients
        grads = torch.autograd.grad(loss, self.model.parameters(), 
                                     create_graph=False)
        
        # Update Fisher (moving average)
        if self.fisher_matrix is None:
            self.fisher_matrix = [g**2 for g in grads]
        else:
            self.fisher_matrix = [
                ema_decay * f + (1 - ema_decay) * g**2
                for f, g in zip(self.fisher_matrix, grads)
            ]
    
    def estimate_uncertainty(self, data):
        """Single forward pass with uncertainty"""
        outputs = self.model(data)
        
        # Uncertainty from Fisher diagonal (posterior variance)
        # Higher Fisher values = more certain
        # Return entropy as uncertainty measure
        probs = F.softmax(outputs, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        
        return entropy  # High entropy = high uncertainty

# Usage
uncertainty_estimator = LaplacianUncertainty(model)
uncertainty = uncertainty_estimator.estimate_uncertainty(data)  # Single pass!
```

**Advantages:**
- ✅ Single forward pass (vs. 10 in MC Dropout)
- ✅ ~1% computational overhead
- ✅ Theoretically grounded (Bayesian approximation)
- ✅ Efficient Fisher matrix update (moving average)

**Trade-offs:**
- Approximation quality depends on Fisher update frequency
- Requires storing Fisher matrix (one value per parameter)

#### Solution 2: Ensemble with Knowledge Distillation (ALTERNATIVE)
```python
class EnsembleUncertainty:
    """
    Fast ensemble using distilled small models
    """
    def __init__(self, main_model, n_ensemble=3):
        self.main_model = main_model
        # Create small student models (1/4 size)
        self.ensemble = [create_small_model() for _ in range(n_ensemble)]
        
    def train_ensemble(self, data, targets):
        """Distill main model into ensemble"""
        main_outputs = self.main_model(data).detach()
        
        for student in self.ensemble:
            student_outputs = student(data)
            # Knowledge distillation loss
            loss = F.kl_div(F.log_softmax(student_outputs / temp, dim=1),
                           F.softmax(main_outputs / temp, dim=1))
            loss.backward()
            # Update student
    
    def estimate_uncertainty(self, data):
        """Fast ensemble prediction"""
        predictions = []
        for student in self.ensemble:
            with torch.no_grad():
                predictions.append(student(data))
        
        # Variance across ensemble
        return torch.var(torch.stack(predictions), dim=0).sum(dim=1)

# Usage (3 small models ≈ 1.5 main model forward passes)
ensemble = EnsembleUncertainty(model, n_ensemble=3)
uncertainty = ensemble.estimate_uncertainty(data)  # ~3% overhead
```

**Advantages:**
- ✅ Better uncertainty estimates than single model
- ✅ Parallelizable (run ensemble in parallel)
- ✅ ~3% overhead with small models

**Trade-offs:**
- More memory (3 small models)
- Need to train/update ensemble periodically

---

### Component 2: Smart Data Selection

#### Streaming Selection (No Full Dataset in Memory)
```python
class StreamingAQL:
    """
    AQL that works with streaming data
    """
    def __init__(self, model, uncertainty_estimator, query_budget=0.2):
        self.model = model
        self.uncertainty = uncertainty_estimator
        self.query_budget = query_budget
        self.buffer_size = 10000  # Rolling buffer
        self.uncertainty_buffer = []
        self.data_buffer = []
        
    def process_batch(self, data, targets):
        """Process streaming batch and decide to keep or discard"""
        # Estimate uncertainty
        unc = self.uncertainty.estimate_uncertainty(data)
        
        # Add to rolling buffer
        self.uncertainty_buffer.extend(unc.tolist())
        self.data_buffer.append((data, targets))
        
        # Keep buffer size manageable
        if len(self.uncertainty_buffer) > self.buffer_size:
            # Select top uncertain samples
            threshold = np.percentile(self.uncertainty_buffer, 
                                     (1 - self.query_budget) * 100)
            
            # Filter data
            mask = torch.tensor(self.uncertainty_buffer) > threshold
            self.data_buffer = [d for d, m in zip(self.data_buffer, mask) if m]
            self.uncertainty_buffer = [u for u, m in zip(self.uncertainty_buffer, mask) if m]
        
        return self.data_buffer
```

---

### Component 3: Curriculum Learning Integration

```python
class CurriculumAQL:
    """
    Combine curriculum learning with active learning
    """
    def __init__(self, model, uncertainty_estimator):
        self.model = model
        self.uncertainty = uncertainty_estimator
        self.training_stage = 0  # 0=easy, 1=medium, 2=hard
        self.stage_thresholds = [0.3, 0.6, 1.0]  # Uncertainty thresholds
        
    def get_difficulty_range(self):
        """Return current acceptable uncertainty range"""
        if self.training_stage == 0:
            return (0.0, 0.3)  # Easy: low uncertainty
        elif self.training_stage == 1:
            return (0.3, 0.6)  # Medium: moderate uncertainty
        else:
            return (0.6, 1.0)  # Hard: high uncertainty
    
    def select_batch(self, data_stream):
        """Select batch based on current stage"""
        batch_uncertainties = []
        batch_data = []
        
        for data, targets in data_stream:
            unc = self.uncertainty.estimate_uncertainty(data)
            
            # Filter by difficulty range
            min_unc, max_unc = self.get_difficulty_range()
            mask = (unc >= min_unc) & (unc <= max_unc)
            
            if mask.any():
                batch_data.append((data[mask], targets[mask]))
                batch_uncertainties.extend(unc[mask].tolist())
            
            if len(batch_data) >= batch_size:
                break
        
        return batch_data
    
    def maybe_advance_stage(self, val_loss):
        """Advance to next difficulty stage if ready"""
        if val_loss < self.stage_thresholds[self.training_stage]:
            self.training_stage = min(2, self.training_stage + 1)
            print(f"📈 Advanced to stage {self.training_stage}")
```

---

### Component 4: Meta-Learned Selection Policy (Future)

```python
class MetaLearnedSelector:
    """
    Learn the selection policy itself
    Use RL to optimize data selection strategy
    """
    def __init__(self, model):
        self.model = model
        self.policy_net = PolicyNetwork()  # Small LSTM
        
    def train_policy(self, episodes):
        """
        Reward: Accuracy improvement per data point selected
        State: Model confidence, uncertainty distribution, training progress
        Action: Select or skip this data point
        """
        for data, targets in episodes:
            state = self.get_state(data)
            action = self.policy_net(state)  # Select or skip
            
            if action == "select":
                # Train on this data
                loss_before = self.evaluate()
                self.model.train_step(data, targets)
                loss_after = self.evaluate()
                
                reward = loss_before - loss_after  # Improvement
                self.policy_net.update(reward)
```

---

## 🏗️ Implementation Architecture

### File Structure
```
experiments/aql_v2/
├── __init__.py
├── uncertainty/
│   ├── __init__.py
│   ├── laplacian.py          # Laplace approximation
│   ├── ensemble.py            # Ensemble uncertainty
│   └── base.py                # Abstract base class
├── selection/
│   ├── __init__.py
│   ├── streaming.py           # Streaming data selection
│   ├── curriculum.py          # Curriculum integration
│   └── meta_policy.py         # Meta-learned policy (v3)
├── trainer.py                 # Main AQL trainer
├── config.py                  # Configuration
└── utils.py                   # Helper functions
```

### Main Training Loop
```python
# experiments/aql_v2/trainer.py
class AQLv2Trainer:
    """
    Main training loop for AQL v2.0
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Components
        self.uncertainty = LaplacianUncertainty(model)
        self.selector = StreamingAQL(model, self.uncertainty, 
                                     query_budget=config.query_budget)
        self.curriculum = CurriculumAQL(model, self.uncertainty)
        
        # Metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'data_selected': [],
            'overhead_time': []
        }
    
    def train_epoch(self, data_loader):
        """Train one epoch with AQL"""
        epoch_start = time.time()
        selection_time = 0
        
        for batch_idx, (data, targets) in enumerate(data_loader):
            # Standard training step
            train_start = time.time()
            outputs = self.model(data)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            train_time = time.time() - train_start
            
            # AQL selection (measure overhead)
            select_start = time.time()
            
            # Update uncertainty estimator
            self.uncertainty.update_fisher(data, targets)
            
            # Get next batch from curriculum-aware selector
            next_batch = self.curriculum.select_batch(data_loader)
            selection_time += time.time() - select_start
            
            # Log metrics
            if batch_idx % self.config.log_interval == 0:
                overhead = (selection_time / (time.time() - epoch_start)) * 100
                print(f"Batch {batch_idx}: Loss {loss:.4f}, Overhead {overhead:.1f}%")
        
        return self.metrics
```

---

## 📊 Experimental Plan

### Experiment 1: Uncertainty Estimation Comparison
**Goal:** Choose best uncertainty method

| Method | Forward Passes | Expected Overhead | Accuracy |
|--------|----------------|-------------------|----------|
| MC Dropout (baseline) | 10 | 10% | Baseline |
| Laplace Approximation | 1 | <2% | ? |
| Small Ensemble (3x) | 3 | ~3% | ? |

**Dataset:** WikiText-103 (subset, 10M tokens)  
**Model:** Small GPT-2 (100M params)  
**Duration:** 2-3 days

**Success Criteria:**
- ✅ Method with <3% overhead
- ✅ Maintains data efficiency (5x)
- ✅ Accuracy within 1% of MC Dropout

---

### Experiment 2: Curriculum Integration
**Goal:** Validate curriculum + AQL synergy

**Variants:**
1. AQL only (no curriculum)
2. Curriculum only (no active selection)
3. AQL + Curriculum (integrated)

**Hypothesis:** Combined approach > sum of parts

**Success Criteria:**
- ✅ Faster initial learning (first 10% of training)
- ✅ Better final accuracy
- ✅ More efficient convergence

---

### Experiment 3: Full WikiText-103 Training
**Goal:** Scale validation

**Setup:**
- Dataset: Full WikiText-103 (101M tokens)
- Model: GPT-2 small (124M params)
- Baseline: Standard training on all data
- AQL v2.0: Train on 20% selected data

**Metrics:**
- Perplexity (lower = better)
- Training time (hours)
- Total FLOPs
- Data efficiency (tokens needed)

**Success Criteria:**
- ✅ Match baseline perplexity with 5x less data
- ✅ <3% computational overhead
- ✅ 3-5x wall-clock speedup

---

## 🎯 Implementation Timeline

### Week 1: Foundation (Current)
- [x] Design document (this file)
- [ ] Implement Laplacian uncertainty
- [ ] Implement streaming selector
- [ ] Unit tests for components

### Week 2: Integration & Testing
- [ ] Integrate components into trainer
- [ ] Run Experiment 1 (uncertainty comparison)
- [ ] Optimize based on results
- [ ] Run Experiment 2 (curriculum)

### Week 3-4: Full Scale Validation
- [ ] Run Experiment 3 (full WikiText-103)
- [ ] Benchmark against baseline
- [ ] Document results
- [ ] Write up findings

---

## 🚀 Success Metrics

### Technical Targets
| Metric | v1.0 (MNIST) | v2.0 Target | Stretch |
|--------|--------------|-------------|---------|
| **Computational Overhead** | 10% | <3% | <1% |
| **Data Efficiency** | Untested | 5x | 10x |
| **Accuracy vs Baseline** | +0.4% | ±0% | +1% |
| **Memory Overhead** | High | Medium | Low |

### Validation Checkpoints
- ✅ **Checkpoint 1:** Uncertainty <3% overhead (Week 1)
- ⏳ **Checkpoint 2:** Curriculum improves learning (Week 2)
- ⏳ **Checkpoint 3:** Scale to 100M+ tokens (Week 3)
- ⏳ **Checkpoint 4:** 5x data efficiency achieved (Week 4)

---

## 🔧 Configuration Template

```python
# configs/aql_v2_config.yaml
model:
  type: "gpt2"
  n_layers: 12
  n_heads: 12
  d_model: 768
  vocab_size: 50257

training:
  batch_size: 32
  learning_rate: 3e-4
  warmup_steps: 500
  max_steps: 50000
  gradient_accumulation: 4

aql_v2:
  # Uncertainty estimation
  uncertainty_method: "laplace"  # or "ensemble"
  fisher_ema_decay: 0.95
  fisher_update_freq: 1  # Every batch
  
  # Data selection
  query_budget: 0.2  # Select top 20%
  buffer_size: 10000
  selection_strategy: "streaming"
  
  # Curriculum learning
  curriculum_enabled: true
  difficulty_stages: [0.3, 0.6, 1.0]
  stage_advance_threshold: 0.1  # Loss reduction needed
  
  # Performance
  target_overhead: 0.03  # <3%
  enable_caching: true
  parallelize_uncertainty: true

logging:
  log_every: 100
  eval_every: 1000
  save_every: 5000
  wandb_project: "efficient-llm-training"
  wandb_tags: ["aql-v2", "wikitext-103"]
```

---

## 📝 Next Actions

### Immediate (Today)
1. ✅ Complete this design document
2. ⏳ Implement `LaplacianUncertainty` class
3. ⏳ Write unit tests
4. ⏳ Create experiment configs

### This Week
1. Complete core components
2. Integration testing
3. Run first experiments
4. Iterate based on results

---

## 🤝 Dependencies

### External Libraries
- PyTorch 2.9.1+
- Hugging Face `datasets`
- Weights & Biases (`wandb`)
- `transformers` (for tokenizer)

### Internal Components
- `utils/device_manager.py` (GPU management)
- `experiments/experiment_framework.py` (experiment tracking)
- Existing optimizers and loss functions

---

**Document Status:** ✅ Complete - Ready for Implementation  
**Next Review:** After Week 1 experiments  
**Owner:** AI Research Lab Team
