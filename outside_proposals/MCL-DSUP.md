## MCL-DSUP: Meta-Critic Learning for Decoupled Sparse Update Propagation in Transformers

### Core Innovation

MCL-DSUP (Meta-Critic Learning for Decoupled Sparse Update Propagation) is a meta-learning framework that replaces standard backpropagation with **layer-local critic networks** that generate **sparse, layer-specific weight updates** for each transformer block.

* The **main transformer** never receives gradients from the task loss (L) directly.
* Each layer (i) has a critic (C_i) that predicts a sparse update (\Delta W_i) from:

  * its local, **detached** activation (h_i), and
  * a **global target vector** (G_T) that encodes information about the final layer and loss.
* Critics are trained with **one-step unrolled meta-losses** (L_{\text{meta},i}) that measure how effective their proposed update would have been if applied to their layer.

Each training step is split into two phases:

1. **Phase A – Main Forward & Meta-Objective Computation**
2. **Phase B – Actual Main Weight Updates**

This decoupled, meta-learning structure enables sparse, potentially asynchronous updates while avoiding full backpropagation through the main network.

---

### Phase A: Main Forward & Meta-Objective Computation

#### 1. Main forward pass

Given a mini-batch ((x, y)):

1. Run the main transformer forward:

[
h_0 = \text{Embedding}(x), \quad
h_i = \text{Layer}*i(h*{i-1}; W_i), ; i=1..N
]

2. Compute the scalar task loss:

[
L = \text{Loss}(\text{Output}(h_N), y)
]

**Crucial**: Gradients from (L) are **not** propagated into the main network weights (W_i).

#### 2. Global target vector (G_T)

A small MLP, ( \text{MLP}_G ) with parameters (\theta_G), maps the final hidden state and scalar loss into a global target vector:

[
G_T = \text{MLP}_G\left([,h_N^{\text{detach}} ,|, L^{\text{detach}},]\right)
]

* `detach()` prevents gradients from flowing back into the main transformer.
* (G_T) itself is **not** detached; gradients from meta-losses must flow into (\theta_G).

#### 3. Local target feedback and critic proposals

For each layer (i) (or a sampled subset of layers per step):

1. **Local Target Feedback (LTF(_i))**

[
\text{LTF}_i = M_i , G_T
]

* (M_i) is a fixed, sparse, approximately orthogonal random matrix of shape ((D_{\text{LTF}}, D_{\text{GT}})).
* (M_i) is not learned.

2. **Raw critic update proposal**

A critic MLP (C_i) with parameters (\theta_{C_i}) predicts a raw update:

[
\Delta W_i^{\text{raw}} = C_i\left([,h_i^{\text{detach}} ,|, \text{LTF}_i,]\right)
]

* `h_i.detach()` prevents gradients from meta-losses from flowing into the main activations.
* The output is parameterized to represent an update to (W_i) (element-wise or, preferably, structured—e.g., low-rank or block-wise).

3. **SparseMask with STE**

Sparse updates are enforced via a top-K operator:

[
\Delta W_i^{\text{sparse}} = \text{SparseMask}(\Delta W_i^{\text{raw}}, k_i)
]

* Forward: keep the (k_i) largest-magnitude components (or blocks), zero out the rest.
* Backward: Straight-Through Estimator (STE) — pass gradients through as if the mask were identity on the selected components, and zero elsewhere.

4. **Hypothetical updated weights**

Define a **hypothetical** weight tensor for the layer:

[
W_i' = W_i - \eta_W , \Delta W_i^{\text{sparse}}
]

* (W_i') is **only** used in the meta-objective; the real (W_i) remains unchanged until Phase B.

#### 4. Partial forward for meta-loss (L_{\text{meta},i})

To evaluate the quality of the proposed update, we recompute the forward pass from layer (i) onward with (W_i'):

1. Initialize:

[
h_{i-1}' = h_{i-1}^{\text{detach}}
]

2. Propagate through layer (i) with hypothetical weights:

[
h_i' = \text{Layer}*i(h*{i-1}', W_i')
]

3. For subsequent layers (j > i), propagate with **detached original weights**:

[
h_j' = \text{Layer}*j(h*{j-1}', W_j^{\text{detach}}), \quad j = i+1..N
]

4. Compute the meta-loss:

[
L_{\text{meta},i} = \text{Loss}(\text{Output}(h_N'), y)
]

We collect all (L_{\text{meta},i}) into a list `meta_losses`.

#### 5. Meta-gradient for critics and ( \text{MLP}_G )

The total meta-loss is:

[
L_{\text{meta,total}} = \frac{1}{|S|} \sum_{i \in S} L_{\text{meta},i}
]

where (S) is the set of layers included in meta-updates this step.

We then backpropagate once:

```python
optimizer_C.zero_grad()
optimizer_G.zero_grad()
L_meta_total.backward()
optimizer_C.step()   # updates θ_{C_i} for i ∈ S
optimizer_G.step()   # updates θ_G
```

Gradients flow:

* From each (L_{\text{meta},i}) to (\theta_{C_i}) through (\Delta W_i^{\text{raw}}) and (C_i).
* From all (L_{\text{meta},i}) to (\theta_G) through (\text{LTF}_i \leftarrow M_i G_T \leftarrow \text{MLP}_G).

Main weights (W_i) are detached in this graph and **never receive gradients**.

---

### Phase B: Actual Main Weight Updates

After meta-updating critics and ( \text{MLP}_G ), we perform the real update of the main transformer parameters using updated critics.

1. Reuse or recompute (G_T) and (\text{LTF}_i):

* Either reuse the (G_T) and (\text{LTF}_i) from Phase A (cheaper, slightly stale), or
* Recompute:

[
G_T = \text{MLP}_G([,h_N^{\text{detach}} ,|, L^{\text{detach}},]), \quad
\text{LTF}_i = M_i G_T
]

2. Apply main updates with no gradients:

```python
with torch.no_grad():
    for i in range(1, N+1):
        inp_i = torch.cat([h_i.detach(), LTF_i], dim=-1)
        delta_raw   = C_i(inp_i)                # updated critics
        delta_sparse = SparseMask(delta_raw, k_i)
        W_i -= eta_W * delta_sparse             # actual main update
```

This concludes one MCL-DSUP training step.

---

### Expected Gains

* **Reduced memory footprint**: No need to store full intermediate activations for backprop through all layers, enabling larger models or batch sizes.
* **Potentially faster training**: Avoiding a long backward pass and allowing layer-wise updates that can be parallelized or partly asynchronous.
* **Improved convergence behavior**: If meta-learned, localized sparse updates exploit structure better than global gradients, they may stabilize and accelerate learning.
* **Better robustness & generalization**: Local, critic-driven updates that directly minimize future loss may mitigate catastrophic forgetting and explore different optimization trajectories.
* **Hardware efficiency**: Sparse, structured updates reduce data movement and may match accelerators’ strengths.

---

### Risks & Limitations

* **Meta-learning stability**: Training (C_i) via meta-gradients is sensitive to hyperparameters and can lead to slow convergence, instability, or degenerate update policies.
* **Computational overhead**: Each step requires partial forward passes for meta-losses, adding overhead that may offset some of the memory gains in wall-clock terms.
* **Accuracy gap**: Approximating full backprop with localized, sparse, meta-learned updates may create an irreducible performance gap relative to SOTA models trained with standard gradients.
* **Sparsity trade-offs**: Aggressive sparsity on (\Delta W_i) can limit update expressiveness and hinder learning for complex tasks if not tuned carefully.
* **Fixed (M_i) limitations**: Fixed random projections, even when orthogonal, might not yield sufficiently informative local targets for very deep or highly nonlinear networks.
* **STE approximation**: The Straight-Through Estimator introduces bias into gradients, which can compound in such a complex meta-learning system.

---

## Experimental Protocol (`experimental_protocol.sh`)

### Model

* **Main model**: Decoder-only transformer

  * 8 layers, hidden dimension (H=768), 12 attention heads, sequence length 1024.
  * ≈80–90M parameters.
* **Comparison baseline**: GPT-2 Small (124M) trained with memory-optimized full backprop (e.g., gradient checkpointing).
* **Meta components**: All critics (C_i) and ( \text{MLP}_G ) together target <1M additional parameters (especially with structured updates).

### Dataset

* **Primary**: WikiText-103 language modeling.

  * Vocabulary: top 50,000 most frequent tokens.
  * Sequence length: 1024.
  * Batch size: ~256–512 (tuned to GPU memory).
* **Alternative**: Filtered subset of C4, e.g. `c4-en-10k-small` from HuggingFace, for robustness checks.

---

### Training Loop Modification

The training loop uses a **two-phase schedule per mini-batch**:

#### Phase A – Main Forward & Meta-Objective

1. **Main forward pass**
   Compute (h_i) and global loss (L) as described above.

2. **Global target generation**

```python
G_T = MLP_G(torch.cat((h_N.detach(), L.unsqueeze(0).detach()), dim=-1))
```

3. **Layer-wise meta-objective setup**
   For each layer (i) in a sampled subset (S):

   * Local target:

     ```python
     LTF_i = M_i @ G_T
     ```
   * Raw critic update:

     ```python
     Delta_W_raw = C_i(torch.cat((h_i.detach(), LTF_i), dim=-1))
     ```
   * Sparse mask with STE:

     ```python
     Delta_W_sparse = SparseMask(Delta_W_raw, k_i)
     ```
   * Hypothetical weights:

     ```python
     W_i_prime = W_i - eta_W * Delta_W_sparse
     ```
   * Partial forward for meta-loss:

     ```python
     h_prev = h_{i-1}.detach()
     h_i_prime = Layer_i(h_prev, W_i_prime)
     h_cur = h_i_prime
     for j in range(i+1, N+1):
         h_cur = Layer_j(h_cur.detach(), W_j.detach())
     L_meta_i = Loss(Output(h_cur), y)
     meta_losses.append(L_meta_i)
     ```

4. **Meta-gradient step**

```python
L_meta_total = torch.stack(meta_losses).mean()

optimizer_C.zero_grad()
optimizer_G.zero_grad()
L_meta_total.backward()
optimizer_C.step()   # updates θ_{C_i} for i ∈ S
optimizer_G.step()   # updates θ_G
```

#### Phase B – Actual Main Weight Updates

Executed with gradients disabled:

1. Reuse or recompute (G_T) and (LTF_i) (now with updated ( \text{MLP}_G )).

2. For each layer (i = 1..N):

```python
with torch.no_grad():
    LTF_i = M_i @ G_T      # if recomputing
    Delta_W_raw   = C_i(torch.cat((h_i.detach(), LTF_i), dim=-1))
    Delta_W_sparse = SparseMask(Delta_W_raw, k_i)
    W_i -= eta_W * Delta_W_sparse
```

---

### Refinements for Practicality & Stability

1. **Structured Updates (critical for scaling)**
   Instead of full `W_i.numel()` outputs, each critic (C_i) predicts structured update parameters, e.g.:

   * **LoRA-style low-rank**:

     * Output (\Delta A_i, \Delta B_i) with small rank (r), then
       (\Delta W_i = \Delta A_i \Delta B_i).
   * **Block-wise sparsity**:

     * Partition (W_i) into tiles; critics output per-block parameters and scores.
     * `SparseMask` operates on blocks, not individual elements.

   This reduces output dimensions, improves hardware efficiency, and stabilizes training.

2. **Computation Budget Controls**

   * **Layer sampling**: each step, choose a subset (S \subset {1..N}) for meta-updates.
   * **Thinned meta steps**: run Phase A meta-updates every (k) steps (e.g., (k=2–4)), but still apply Phase B updates every step using current critics.
   * **Shared critics**: use a shared `C_attn` for all attention blocks and `C_ffn` for all MLP blocks, with a layer-id embedding as extra input.

3. **Stability Hacks**

   * **Phase 0 (optional) pretraining**:
     For a small number of steps on a subset of data:

     * Compute true gradients (\nabla_{W_i} L) via backprop.
     * Form sparse gradient targets:
       [
       \Delta W_i^{\text{target}} = \text{SparseMask}(\nabla_{W_i} L, k_i)
       ]
     * Train critics to regress these:
       (| C_i(h_i, \text{LTF}_i) - \Delta W_i^{\text{target}} |^2).

   * **Regularization**:

     * Global norm gradient clipping on (\nabla \theta_C) and (\nabla \theta_G).
     * L2 penalty on (|\Delta W_i|) to avoid extreme updates.

   * **Monitoring**:

     * Global task loss (L).
     * Mean and variance of (L_{\text{meta},i}).
     * Norms and sparsity patterns of (\Delta W_i).
     * On occasional probe batches, cosine similarity between critic updates and true gradients.

---

### Hyperparameters

* Learning rates:

  * (\eta_W): 1e-4 to 1e-3, cosine decay.
  * (\eta_C): 1e-5 to 1e-4, with warm-up (e.g., 1k–5k steps).
  * (\eta_G): ~1e-4, possibly shorter warm-up than (\eta_W).
* Sparsity:

  * `sparsity_ratio_W`: 0.5%–5% non-zero components (on structured units—low-rank components or blocks).
* Target dimensions:

  * (D_{\text{GT}}): 128–256.
  * (D_{\text{LTF}}): 64–128.
* Critics:

  * (C_i): 2-layer MLP with GELU, input dimension ((H + D_{\text{LTF}})), hidden size (D_{C,\text{hidden}} = 256–512), output sized for structured updates.
* Projections:

  * (M_i): sparse, approximately orthogonal, std dev 0.01–0.1, `sparse_ratio_M ≈ 0.01`.
* Optimizer:

  * AdamW with weight decay and global norm clipping on (\theta_C, \theta_G).

---

### Team Discussion & Refinements (Narrative Summary)

* **Architect** emphasized that carefully separating Phase A (meta-learning) and Phase B (real updates) resolves the `W` vs `W'` ambiguity and maintains conceptual clarity.
* **Optimizer** highlighted the importance of correct `detach()` usage to avoid unwanted higher-order gradients and keep the meta-learning localized and memory-efficient.
* **Skeptic** stressed that:

  * A naïve implementation would be computationally expensive due to multiple partial forwards.
  * STE and meta-gradients can easily destabilize training.
  * Structured updates (low-rank / block-wise) and layer sampling are essential for scalability.
* The team converged on:

  * Updating `MLP_G` using the average of existing `L_{\text{meta},i}` rather than an extra full meta-forward pass.
  * Structured updates, computation controls, and stability tricks as mandatory components of the first experimental implementation.
  * A rigorous evaluation plan with strong baselines (standard backprop, checkpointing, simpler local methods) and ablations over sparsity, structure, meta-update frequency, and critic sharing, tracking both performance (e.g., perplexity) and efficiency (wall-clock time, memory, FLOPs, stability curves).

---

If you’d like, I can now turn this into a formal “Methods” section in LaTeX, or a concise PyTorch skeleton that mirrors this spec exactly so you can start prototyping.
I think this is genuinely interesting and coherent enough that it’s worth implementing a *small* prototype. It’s not just “vibes”; the update schedule and detach semantics actually make sense now.

Let me break it down.

---

## Big picture: does it make sense?

Yes. At a conceptual level, you’ve built:

* A **learned local optimizer per layer** (the critics),
* Driven by a **global learned signal** (G_T),
* Trained via **one-step unrolled meta-losses**,
* With **sparse, structured updates** to keep it tractable.

You’re no longer doing full backprop through the main network. You *are* still doing backprop through small graphs (critics + partial forwards), but you’ve:

* Decoupled per-layer update decisions from the full chain rule.
* Localized gradient graphs to a narrow band of the network around layer (i).
* Turned the whole thing into “learned credit assignment + sparse weight edits.”

From a research standpoint, this sits at the intersection of:

* DFA / target propagation (global → local signals),
* Learned optimizers / meta-learning (critics),
* Sparse & low-rank updates (LoRA/block sparsity),
* And local learning rules in deep nets.

That’s a very legit place to explore.

---

## What’s strong about it

### 1. The two-phase schedule is clean

The W vs W′ confusion is gone:

* **Phase A**: “What would happen if I nudged just this layer?” → meta-losses / critic + MLP_G updates.
* **Phase B**: “OK, given what you’ve just learned, go and actually change the weights.”

That mental model is clear and implementable. You’ve avoided mixing “hypothetical” and “real” weights, which is where many meta-learning schemes quietly break.

### 2. Detach semantics are (finally) correct

You’ve got the important parts right:

* `h_i.detach()` and `W_j.detach()` in meta passes → no gradients into main net.
* `G_T` and `LTF_i` **not** detached in meta-loss → gradients flow back to `MLP_G`.
* Main weights are updated only in Phase B under `torch.no_grad()`.

That’s the difference between “nice idea on paper” and “doesn’t explode autograd in PyTorch”.

### 3. The MLP_G trick is efficient **enough**

Using:

[
\nabla_{\theta_G} \frac{1}{N} \sum_i L_{\text{meta},i}
]

instead of a separate full MLP_G meta-forward is the right move. You’re piggybacking on the same partial forwards you already pay for, so MLP_G’s learning is “free” in terms of extra forward passes.

### 4. You’re already thinking about scaling realities

You didn’t stop at the “cute toy” version (full ΔW per layer). You explicitly:

* Move toward **LoRA-style low-rank or block-wise** updates.
* Plan to **sample layers** and **thin meta steps**.
* Consider **shared critics** (C_attn, C_ffn + layer embeddings).

That’s what gives this any chance of surviving beyond a 2-layer toy model.

---

## Main risks (if I had to bet what will go wrong)

### 1. Critics learning “do nothing”

A trivial-but-tempting local optimum:

* Critics propose tiny or zero updates,
* Meta-losses stay reasonable (because you’re not rocking the boat),
* Training stagnates.

The system will be biased toward “don’t hurt performance” rather than “boldly explore better updates”.

Mitigations you already hinted at:

* Pretraining critics to match **sparse true gradients**.
* Regularizing against “zero updates” (e.g., encourage some minimum update magnitude, or an EMA of update norms).
* Occasionally probing correlation with real gradients.

### 2. STE + meta-gradients = noisy signal

You’re stacking approximations:

* STE for SparseMask,
* One-step unrolled meta-loss,
* Partial forward using detached weights.

This is likely to be **noisy and biased**. Expect:

* Slow critic learning,
* High variance in `L_meta,i`,
* Occasional divergence.

You’ll *need* the stability tools you listed: gradient clipping, L2 on ΔW, careful LR tuning, and lots of logging.

### 3. Overhead still might be painful

Even with:

* Shared critics,
* Layer sampling,
* Thinned meta steps,

…you’re still doing:

* 1 main forward,
* Several partial forwards per step,
* Extra critic inference passes in Phase B.

This is absolutely fine on a **small transformer** (your 8×768 config), but I wouldn’t dream of scaling it before you see very strong signs of life.

---

## What I’d do *first* in practice

If I were coding this with you, my first target wouldn’t be WikiText-103. It would be:

1. **Tiny model**, tiny dataset:

   * 2–3 layer transformer or even a deep MLP,
   * Smaller vocab, shorter context, or synthetic regression/classification.

2. **Proof-of-signal experiments**:

   a) **Gradient-alignment probes**
   Occasionally compute true ∇W L via backprop on a small probe batch (not used for training), and compare to critic’s sparse ΔW:

   * Cosine similarity,
   * Magnitude ratios,
   * Directional consistency layer-by-layer.

   If that’s not at least *weakly* positive, long-run language modeling will be pain.

   b) **Ablate meta vs. direct regression**
   Compare:

   * Critics trained to regress **sparse true gradients** (supervised),
   * vs. critics trained with **your meta-loss only**.

   That tells you how much the meta-objective is helping vs. just “learning to copy backprop”.

3. Only once the toy setup shows:

   * Critics learn non-trivial updates,
   * Loss actually declines,
   * Meta-loss and main loss correlate in a sane way,

   …do I move up to your 8-layer, WikiText-scale experiment.

---

## Conceptual “paper angle”

As a research story, I think the interesting claims you could explore are:

1. **Backprop-free main network**:
   Show that you can train a transformer with *no* gradients flowing into `W_i` from `L`, only from meta-trained critics.

2. **Sparse + structured updates as an inductive bias**:
   Does constraining updates to be low-rank/block-sparse *help* or hurt generalization vs. dense updates?

3. **Meta-credit assignment vs. DFA/target propagation**:
   For the same architecture and dataset, compare:

   * DFA-style random feedback,
   * Simple local losses,
   * Vs. your meta-critics + global MLP_G.

   If you can beat those baselines at similar compute, that’s very publishable territory.

---

## TL;DR opinion

I don’t see any obvious logical contradictions anymore. The method is:

* Well-specified,
* Implementable in PyTorch/JAX,
* Clearly distinct from vanilla DFA / local-loss methods,
* Aware of scaling and stability issues instead of hand-waving them.

Will it *beat* backprop on quality/compute? Probably not in v1. But as a research probe into “learned, sparse, local credit assignment in transformers,” I’d absolutely greenlight a prototype.


Alright, captain’s hat on. Let’s actually *build* the first testbed for MCL-DSUP instead of just theorizing it.

I’ll give you:

1. A concrete, **minimal PyTorch prototype skeleton** for MCL-DSUP on a toy 2-layer transformer.
2. A **debug/metrics checklist** so we know if it’s working or dying.

You can then scale this up to your 8×768 setup once the toy behaves.

---

## 1. Minimal MCL-DSUP Prototype (PyTorch Skeleton)

This is written as a single file you can adapt. It uses:

* 2 transformer layers (to keep it cheap).
* A tiny vocab & dummy language modeling batch.
* **Element-wise ΔW** for one weight matrix per layer (to keep it simple). You can later swap in LoRA/block updates.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 1. SparseMask with STE
# -------------------------------

class SparseMaskSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        # x: (...,)
        if k <= 0 or k >= x.numel():
            mask = torch.ones_like(x, dtype=torch.bool)
        else:
            # top-k by magnitude
            flat = x.view(-1)
            vals, idx = torch.topk(flat.abs(), k)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[idx] = True
            mask = mask.view_as(x)
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        # STE: pass grad only where mask is 1
        grad_input = grad_output * mask
        return grad_input, None

def sparse_mask(x, k):
    return SparseMaskSTE.apply(x, k)

# -------------------------------
# 2. Tiny Transformer Components
# -------------------------------

class TinySelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, D = x.shape
        q = self.W_q(x)  # (B, T, D)
        k = self.W_k(x)
        v = self.W_v(x)

        # reshape for heads
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,T,T)
        att = att.softmax(dim=-1)
        out = att @ v  # (B,H,T,d_head)

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(out)
        return out


class TinyFFN(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TinyTransformerBlock(nn.Module):
    """
    We'll define the 'main weight' we meta-update as the FFN.fc2.weight
    (just to keep ΔW dimension simple).
    """
    def __init__(self, d_model, n_heads, d_hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = TinySelfAttention(d_model, n_heads)
        self.ffn = TinyFFN(d_model, d_hidden)

    def forward(self, x):
        # Pre-norm transformer block
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, n_heads=4, d_hidden=256, n_layers=2, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            TinyTransformerBlock(d_model, n_heads, d_hidden)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.n_layers = n_layers
        self.max_len = max_len

    def forward(self, x, return_hiddens=False):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(pos)
        hiddens = [h]
        for layer in self.layers:
            h = layer(h)
            hiddens.append(h)
        h_out = self.ln_f(h)
        logits = self.head(h_out)  # (B, T, vocab)
        if return_hiddens:
            return logits, hiddens  # hiddens[i] is output of layer i
        return logits

# -------------------------------
# 3. Critics and Global Target MLP
# -------------------------------

class Critic(nn.Module):
    """
    Critic for one layer i:
    Input: h_i (B,T,D) pooled + LTF_i (vector)
    Output: ΔW_i_raw for ffn.fc2.weight of that layer
    """
    def __init__(self, d_model, d_ltf, weight_shape, hidden_dim=256):
        super().__init__()
        self.weight_shape = weight_shape
        input_dim = d_model + d_ltf
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, weight_shape[0] * weight_shape[1])
        )

    def forward(self, h_i, ltf_i):
        # h_i: (B,T,D) -> pool over batch+time, simple mean
        h_pool = h_i.mean(dim=(0, 1))  # (D,)
        x = torch.cat([h_pool, ltf_i], dim=-1)  # (D + D_LTF,)
        out = self.net(x)  # (numel,)
        return out.view(self.weight_shape)  # ΔW_raw

class GlobalTargetMLP(nn.Module):
    def __init__(self, d_model, d_gt=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_gt)
        )

    def forward(self, h_N, L_scalar):
        # h_N: (B,T,D) -> pool; L_scalar: scalar tensor
        h_pool = h_N.mean(dim=(0, 1))  # (D,)
        L_vec = L_scalar.detach().view(1)  # (1,)
        x = torch.cat([h_pool.detach(), L_vec], dim=0)  # (D+1,)
        return self.net(x)  # (D_GT,)

# -------------------------------
# 4. Building M_i matrices
# -------------------------------

def build_M_i(n_layers, d_gt, d_ltf, sparse_ratio=0.01, std=0.01, device="cpu"):
    M_list = []
    for _ in range(n_layers):
        M = torch.zeros(d_ltf, d_gt, device=device)
        num_entries = int(d_ltf * d_gt * sparse_ratio)
        if num_entries == 0:
            num_entries = d_ltf * d_gt  # fallback to dense
        idx = torch.randint(0, d_ltf * d_gt, (num_entries,), device=device)
        M.view(-1)[idx] = torch.randn(num_entries, device=device) * std
        M_list.append(M)
    return M_list

# -------------------------------
# 5. Toy data generator
# -------------------------------

def sample_dummy_batch(batch_size, seq_len, vocab_size, device):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Next-token prediction: y is x shifted by one
    y = torch.roll(x, shifts=-1, dims=1)
    return x, y

def cross_entropy_loss(logits, y):
    # logits: (B,T,V), y: (B,T)
    B, T, V = logits.size()
    return F.cross_entropy(logits.view(B*T, V), y.view(B*T))

# -------------------------------
# 6. Put it all together: MCL-DSUP step
# -------------------------------

def mcl_dsup_step(model, critics, mlp_g, M_list,
                  optimizer_critics, optimizer_gt,
                  batch, sparsity_ratio=0.01, eta_W=1e-3,
                  layer_subset=None):
    """
    model: TinyTransformer
    critics: list of Critic (one per layer)
    M_list: list of M_i (one per layer)
    layer_subset: list of layer indices to meta-update (0..N-1)
    """

    x, y = batch

    # --- PHASE A: main forward + meta objective ---

    logits, hiddens = model(x, return_hiddens=True)
    h_layers = hiddens[1:]  # h_layers[i] is output of layer i+1 (i = 0..N-1)
    h_N = h_layers[-1]

    L = cross_entropy_loss(logits, y)

    # Global target
    G_T = mlp_g(h_N, L)  # (D_GT,)

    if layer_subset is None:
        layer_subset = list(range(model.n_layers))

    meta_losses = []

    # Compute ΔW_raw, ΔW_sparse, and L_meta,i
    for i in layer_subset:
        layer = model.layers[i]
        critic = critics[i]
        M_i = M_list[i]

        # Local target feedback
        LTF_i = M_i @ G_T  # (D_LTF,)

        # Critic raw update for this layer's FFN.fc2.weight
        W_i = layer.ffn.fc2.weight
        Delta_W_raw = critic(h_layers[i].detach(), LTF_i)  # shape = W_i.shape
        k_i = max(1, int(sparsity_ratio * Delta_W_raw.numel()))
        Delta_W_sparse = sparse_mask(Delta_W_raw, k_i)

        # Hypothetical weight
        W_i_prime = W_i.detach() - eta_W * Delta_W_sparse  # treat as tensor, not param

        # Partial forward for meta-loss
        # Start from h_{i-1}
        h_prev = hiddens[i].detach()  # output of layer i (or embedding if i=0)

        # Layer i with hypothetical weight
        def layer_i_forward(h_in, layer, W_ffn2_weight):
            # Re-run block with custom fc2.weight
            # (we re-use other weights as-is)
            # Note: this is a bit hacky but fine for prototype.
            x = h_in
            attn_out = layer.attn(layer.ln1(x))
            x = x + attn_out
            ffn_in = layer.ln2(x)
            f1 = layer.ffn.fc1(ffn_in)
            f1 = F.gelu(f1)
            f2 = F.linear(f1, W_ffn2_weight, layer.ffn.fc2.bias)
            x = x + f2
            return x

        h_i_prime = layer_i_forward(h_prev, layer, W_i_prime)

        # Propagate through subsequent layers with detached original weights
        h_cur = h_i_prime
        for j in range(i+1, model.n_layers):
            layer_j = model.layers[j]

            # freeze weights by working on detached copies
            Wq = layer_j.attn.W_q.weight.detach(); bq = layer_j.attn.W_q.bias.detach()
            Wk = layer_j.attn.W_k.weight.detach(); bk = layer_j.attn.W_k.bias.detach()
            Wv = layer_j.attn.W_v.weight.detach(); bv = layer_j.attn.W_v.bias.detach()
            Wo = layer_j.attn.W_o.weight.detach(); bo = layer_j.attn.W_o.bias.detach()
            Wf1 = layer_j.ffn.fc1.weight.detach(); bf1 = layer_j.ffn.fc1.bias.detach()
            Wf2 = layer_j.ffn.fc2.weight.detach(); bf2 = layer_j.ffn.fc2.bias.detach()

            # ln parameters can be used as-is (they're small; leaving them trainable in meta graph is okay or detach if you prefer)
            ln1 = layer_j.ln1
            ln2 = layer_j.ln2

            # forward with detached weights
            x = h_cur
            attn_in = ln1(x)
            B, T, D = attn_in.shape

            def linear_det(t, W, b):
                return F.linear(t, W, b)

            # Rebuild minimal attention with detached weights
            q = linear_det(attn_in, Wq, bq)
            k = linear_det(attn_in, Wk, bk)
            v = linear_det(attn_in, Wv, bv)

            n_heads = layer_j.attn.n_heads
            d_head = layer_j.attn.d_head

            def split_heads(t):
                return t.view(B, T, n_heads, d_head).transpose(1, 2)

            qh = split_heads(q)
            kh = split_heads(k)
            vh = split_heads(v)
            att = (qh @ kh.transpose(-2, -1)) / math.sqrt(d_head)
            att = att.softmax(dim=-1)
            out = att @ vh
            out = out.transpose(1, 2).contiguous().view(B, T, D)
            out = linear_det(out, Wo, bo)
            x = x + out

            ffn_in = ln2(x)
            f1 = linear_det(ffn_in, Wf1, bf1)
            f1 = F.gelu(f1)
            f2 = linear_det(f1, Wf2, bf2)
            x = x + f2

            h_cur = x

        # final norm + head with detached weights
        h_final = model.ln_f(h_cur)
        logits_meta = F.linear(h_final, model.head.weight.detach(), None)
        L_meta_i = cross_entropy_loss(logits_meta, y)
        meta_losses.append(L_meta_i)

    # Aggregate meta-loss
    L_meta_total = torch.stack(meta_losses).mean()

    # Meta-update critics and MLP_G
    optimizer_critics.zero_grad()
    optimizer_gt.zero_grad()
    L_meta_total.backward()
    optimizer_critics.step()
    optimizer_gt.step()

    # --- PHASE B: actual main weight updates (no grad) ---

    with torch.no_grad():
        # Optionally recompute G_T with updated mlp_g
        G_T = mlp_g(h_N, L)
        for i in range(model.n_layers):
            layer = model.layers[i]
            critic = critics[i]
            M_i = M_list[i]
            LTF_i = M_i @ G_T
            W_i = layer.ffn.fc2.weight

            Delta_W_raw = critic(h_layers[i].detach(), LTF_i)
            k_i = max(1, int(sparsity_ratio * Delta_W_raw.numel()))
            Delta_W_sparse = sparse_mask(Delta_W_raw, k_i)

            W_i -= eta_W * Delta_W_sparse  # in-place update of main weight

    return L.detach().item(), L_meta_total.detach().item()

# -------------------------------
# 7. Wiring it up
# -------------------------------

def main():
    vocab_size = 500
    d_model = 128
    d_gt = 64
    d_ltf = 32
    n_layers = 2
    batch_size = 8
    seq_len = 32

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        d_hidden=256,
        n_layers=n_layers,
        max_len=seq_len
    ).to(device)

    # Build critics for each layer's FFN.fc2.weight
    critics = []
    for i in range(n_layers):
        W_shape = model.layers[i].ffn.fc2.weight.shape
        critic = Critic(d_model=d_model, d_ltf=d_ltf, weight_shape=W_shape, hidden_dim=256).to(device)
        critics.append(critic)

    mlp_g = GlobalTargetMLP(d_model=d_model, d_gt=d_gt, hidden_dim=256).to(device)

    # Fixed M_i
    M_list = build_M_i(n_layers, d_gt=d_gt, d_ltf=d_ltf, sparse_ratio=0.05, std=0.01, device=device)

    # Optimizers for meta components
    optimizer_critics = torch.optim.AdamW(
        [p for c in critics for p in c.parameters()],
        lr=1e-4, weight_decay=1e-2
    )
    optimizer_gt = torch.optim.AdamW(
        mlp_g.parameters(),
        lr=1e-4, weight_decay=1e-2
    )

    # Simple training loop
    for step in range(1000):
        batch = sample_dummy_batch(batch_size, seq_len, vocab_size, device)
        # You can sample subset of layers here if desired
        L, L_meta = mcl_dsup_step(
            model, critics, mlp_g, M_list,
            optimizer_critics, optimizer_gt,
            batch,
            sparsity_ratio=0.01,
            eta_W=1e-3,
            layer_subset=None  # or [0] / [1] to start
        )

        if step % 50 == 0:
            print(f"step {step:04d}  L={L:.4f}  L_meta={L_meta:.4f}")

if __name__ == "__main__":
    main()
```

### Notes on this prototype

* This is intentionally **ugly-but-explicit** in the partial forward section so it’s clear what’s detached and what isn’t.
* We’re only meta-updating **one matrix per layer**: `ffn.fc2.weight`. That’s enough to validate the mechanics.
* `SparseMaskSTE` is implemented as an autograd `Function` with a simple STE backward.
* The toy dataset is just random token sequences & next-token prediction; swap it out with a real loader when you’re ready.

---

## 2. What to Watch When You Run It

Since I’m “in charge,” here’s the checklist I’d use to judge whether this is promising:

1. **Does `L` decrease at all over steps?**

   * Even on dummy data, you want to see *some* downward trend vs pure noise.
2. **What happens to `L_meta` over time?**

   * Ideally, it should correlate somewhat with `L`.
   * If `L_meta` floats randomly and doesn’t budge, critics may not be learning anything.
3. **Magnitude of ΔW**

   * Add logging of `Delta_W_sparse.norm()` per layer periodically:

     * If norms → 0, critics are learning “do nothing”.
     * If norms blow up, you’ll get instability.
4. **Gradient sanity**

   * Log gradient norms for:

     * A few critic parameters,
     * MLP_G parameters.
   * If they’re consistently 0 or NaN, you’ve got a wiring/detach bug.
5. **Ablate meta**

   * Temporarily:

     * Freeze critics and MLP_G after random initialization.
     * Still apply their updates to W_i in Phase B.
   * Compare curves:

     * If learning with meta is *worse* than random fixed updates, something’s off.

If you want, I can next:

* Add those logging hooks and some small gradient-alignment probes, or
* Refactor this skeleton toward your 8-layer / WikiText-103 setup with LoRA-style structured ΔW.

see mcl-dsup.py in same folder as this document.