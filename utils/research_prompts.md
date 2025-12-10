# AI Research Prompts

Collection of effective prompts for guiding the research agents.

## Optimization Research Prompts

### Novel Optimizers
```
Propose a novel optimization algorithm that:
- Goes beyond gradient descent and Adam
- Incorporates second-order information efficiently
- Is scalable to billion-parameter models
- Provides theoretical convergence guarantees

Consider:
- Memory constraints (GPU VRAM)
- Computational overhead
- Distributed training compatibility
- Robustness to hyperparameters
```

### Adaptive Learning Rates
```
Design an adaptive learning rate scheme that:
- Adjusts per-layer or per-parameter
- Responds to loss landscape geometry
- Handles varying batch sizes
- Works across different architectures

Explain the mathematical foundation and provide PyTorch implementation.
```

## Loss Function Prompts

### Robustness and Calibration
```
Propose a novel loss function that improves:
- Model calibration (predicted probabilities match actual frequencies)
- Robustness to label noise
- Out-of-distribution detection
- Generalization to test data

Compare against cross-entropy, focal loss, and label smoothing.
```

### Self-Supervised Learning
```
Design a contrastive loss for self-supervised representation learning that:
- Works with limited labeled data
- Learns transferable representations
- Scales to large batch sizes
- Is computationally efficient

Provide experimental validation on ImageNet.
```

## Architecture Prompts

### Efficiency and Expressiveness
```
Propose a novel architecture that:
- Achieves better accuracy-to-FLOPs ratio than Transformers
- Handles long sequences efficiently
- Incorporates useful inductive biases
- Scales to billions of parameters

Consider memory, computation, and parallelization constraints.
```

### Dynamic Networks
```
Design a network with adaptive computation that:
- Allocates more compute to difficult examples
- Grows or shrinks based on task complexity
- Maintains training stability
- Works with standard optimizers

Explain the halting mechanism and training procedure.
```

## Training Pipeline Prompts

### Curriculum Learning
```
Propose a data curriculum strategy that:
- Orders examples from easy to hard
- Adapts to model's current capability
- Improves final performance and convergence speed
- Is fully automated (no manual labeling of difficulty)

Provide implementation and validation experiments.
```

### Multi-Stage Training
```
Design a multi-stage training procedure that:
- Progressively increases model capacity or data difficulty
- Maintains stability during transitions
- Improves over standard end-to-end training
- Is compatible with large-scale distributed training
```

## Scalability Prompts

### Distributed Training
```
Analyze this proposal for distributed training at scale:
- How does communication overhead scale?
- Memory requirements per GPU/TPU?
- Batch size sensitivity?
- Synchronization requirements?
- Fault tolerance?

Provide concrete scaling analysis for 1B, 10B, 100B parameter models.
```

### Hardware Efficiency
```
Evaluate hardware utilization for this approach:
- GPU memory usage
- Compute intensity (FLOPs per byte)
- Tensor Core compatibility
- Mixed precision training
- Kernel fusion opportunities

Suggest optimizations for modern accelerators.
```

## Experimental Design Prompts

### Minimal Validation
```
Design a minimal experiment to validate this hypothesis:
- Smallest dataset that demonstrates effect
- Simplest baseline comparisons
- Key metrics to measure
- Compute budget (GPU-hours)
- Expected effect size

Make falsifiable predictions.
```

### Ablation Studies
```
Propose ablation studies to understand:
- Which components are critical?
- Sensitivity to hyperparameters
- Interaction between components
- Failure modes and edge cases

Design experiments to isolate each factor.
```

## Meta-Research Prompts

### Literature Review
```
Survey existing work on [topic]:
- Seminal papers and key contributions
- Current state-of-the-art
- Open problems and limitations
- Promising research directions

Identify gaps that our proposal could fill.
```

### Theoretical Analysis
```
Provide theoretical analysis of this method:
- Convergence guarantees
- Sample complexity
- Computational complexity
- Approximation error bounds

Connect to optimization theory, statistical learning theory, or information theory.
```
