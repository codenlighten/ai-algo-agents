# sparsae_wikitext.py - SparsAE with Real WikiText-103 Data
print(">>> [STARTUP] Entered sparsae_wikitext.py", flush=True)

import math
import copy
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path

print(">>> [STARTUP] Importing PyTorch...", flush=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
print(">>> [STARTUP] Importing datasets and transformers...", flush=True)
from datasets import load_dataset
from transformers import GPT2Tokenizer
print(">>> [STARTUP] All imports complete", flush=True)


####################################
# 1. WikiText-103 Dataset
####################################

class WikiTextDataset(Dataset):
    """WikiText-103 dataset with GPT-2 tokenization (memory-optimized)"""
    
    def __init__(self, split: str = "train", max_length: int = 512, cache_dir: str = "./data", max_examples: int = None):
        self.max_length = max_length
        print(f">>> [DATASET] Initializing GPT2 tokenizer...", flush=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f">>> [DATASET] Loading WikiText-103 {split} split (this may take 1-2 minutes on first run)...", flush=True)
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=split, cache_dir=cache_dir, streaming=False)
        print(f">>> [DATASET] Loaded {len(dataset)} raw documents", flush=True)
        
        # Tokenize and chunk (memory-optimized with limit)
        print(f">>> [DATASET] Tokenizing and chunking (this will take a few minutes)...", flush=True)
        self.examples = []
        
        for idx, item in enumerate(dataset):
            text = item["text"].strip()
            if len(text) < 50:  # Skip very short texts
                continue
            
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            
            # Chunk into max_length sequences
            for i in range(0, len(tokens) - max_length, max_length // 2):  # 50% overlap
                chunk = tokens[i:i + max_length]
                if len(chunk) == max_length:
                    self.examples.append(chunk)
                    
                    # Early exit if we hit max_examples limit (for memory constraints)
                    if max_examples and len(self.examples) >= max_examples:
                        print(f">>> [DATASET] Reached max_examples limit of {max_examples}", flush=True)
                        print(f">>> [DATASET] Created {len(self.examples)} examples from WikiText-103 {split}", flush=True)
                        return
        
        print(f">>> [DATASET] Created {len(self.examples)} examples from WikiText-103 {split}", flush=True)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


####################################
# 2. Tiny Transformer LM (debug)
####################################

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size=50257, d_model=256, n_heads=4, n_layers=4, d_ff=1024, max_seq_len=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


####################################
# 3. EMA Teacher
####################################

class EMATeacher:
    def __init__(self, model: nn.Module, beta: float = 0.999):
        self.beta = beta
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, student: nn.Module, masks: Dict[str, torch.Tensor]):
        t_params = dict(self.teacher.named_parameters())
        for name, p_s in student.named_parameters():
            if name not in t_params:
                continue
            p_t = t_params[name]
            if name in masks:
                m = masks[name].to(p_s.device)
                contrib = p_s * m
            else:
                contrib = p_s
            p_t.data.mul_(self.beta).add_(contrib, alpha=(1.0 - self.beta))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.teacher(input_ids)


####################################
# 4. KL between logits
####################################

def kl_divergence_logits(student_logits, teacher_logits):
    """KL(P || Q) where P = softmax(student), Q = softmax(teacher)"""
    p_log_probs = F.log_softmax(student_logits, dim=-1)
    q_log_probs = F.log_softmax(teacher_logits, dim=-1)
    p_probs = p_log_probs.exp()
    kl = (p_probs * (p_log_probs - q_log_probs)).sum(dim=-1)
    return kl.mean()


####################################
# 5. Sparsity Mask Manager + ES v1
####################################

class SparsaeMaskManager:
    """
    Implements:
    - Global k-sparsity
    - Per-layer sparsity bounds [min_s_l, max_s_l]
    - (1 + lambda)-ES over masks
    - Composite proxy P(M') with normalized CE, diversity, gradient activity
    - Global reset mutation
    - Tracking for stagnation detection
    """

    def __init__(
        self,
        model: nn.Module,
        global_sparsity: float = 0.8,
        lambda_: int = 3,
        p_mutate: float = 0.005,
        min_layer_sparsity: float = 0.5,
        max_layer_sparsity: float = 0.95,
        stats_beta: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.global_sparsity = global_sparsity
        self.lambda_ = lambda_
        self.p_mutate = p_mutate
        self.device = device or next(model.parameters()).device

        # layer grouping: we treat each "weight" param as one layer for now
        self.param_tensors: List[Tuple[str, nn.Parameter]] = []
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.dim() >= 2:
                self.param_tensors.append((name, p))

        self.num_layers = len(self.param_tensors)
        self.min_layer_sparsity = {name: min_layer_sparsity for name, _ in self.param_tensors}
        self.max_layer_sparsity = {name: max_layer_sparsity for name, _ in self.param_tensors}

        # masks and index helpers
        self.masks: Dict[str, torch.Tensor] = {}
        self._param_flat_indices: Dict[str, torch.Tensor] = {}
        self._init_masks_erdos_renyi()

        # proxy running stats
        self.stats_beta = stats_beta
        self.ce_mu = 0.0
        self.ce_sigma = 1.0
        self.div_mu = 0.0
        self.div_sigma = 1.0
        self.grad_mu = 0.0
        self.grad_sigma = 1.0

        # annealed weights (we update from trainer)
        self.w_ce = 1.0
        self.w_div = 0.01
        self.w_grad = 0.005

        # validation stagnation tracking
        self.val_history: List[float] = []
        self.stagnant_counter = 0
        self.p_global_reset_base = 0.01
        self.p_global_reset_boost = 0.1
        self.global_reset_boost_steps = 0

    def _init_masks_erdos_renyi(self):
        total_weights = sum(p.numel() for _, p in self.param_tensors)
        target_zeros = int(total_weights * self.global_sparsity)
        target_ones = total_weights - target_zeros

        flat_mask = torch.zeros(total_weights, dtype=torch.bool)
        one_indices = torch.randperm(total_weights)[:target_ones]
        flat_mask[one_indices] = True

        offset = 0
        for name, p in self.param_tensors:
            numel = p.numel()
            sub = flat_mask[offset:offset + numel]
            offset += numel
            self.masks[name] = sub.view_as(p).to(self.device)
            self._param_flat_indices[name] = torch.arange(numel, device=self.device)

        # biases / 1D params: keep dense (all ones)
        for name, p in self.model.named_parameters():
            if name not in self.masks and p.requires_grad:
                m = torch.ones_like(p, dtype=torch.bool, device=self.device)
                self.masks[name] = m
                self._param_flat_indices[name] = torch.arange(p.numel(), device=self.device)

    @torch.no_grad()
    def apply_to_model_(self):
        for name, p in self.model.named_parameters():
            if name in self.masks:
                p.mul_(self.masks[name].to(p.device))

    def _clone_masks(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.masks.items()}

    def _layer_sparsity(self, masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
        res = {}
        for name, _ in self.param_tensors:
            m = masks[name]
            total = m.numel()
            active = int(m.sum().item())
            sparsity = 1.0 - (active / max(total, 1))
            res[name] = sparsity
        return res

    def _mask_diversity(self, masks: Dict[str, torch.Tensor]) -> float:
        """L1_Mask_Diversity ~ sqrt(mean((s_l - global_target)^2))"""
        sparsities = self._layer_sparsity(masks)
        vals = torch.tensor(
            [sparsities[name] for name, _ in self.param_tensors],
            device=self.device,
        )
        target = self.global_sparsity
        diff = vals - target
        mse = (diff ** 2).mean()
        return float(torch.sqrt(mse + 1e-8).item())

    def _avg_layer_grad_activity(
        self,
        model: nn.Module,
        masks: Dict[str, torch.Tensor],
    ) -> float:
        """
        Compute (1/L) * sum_l ||grad(W_l ⊙ M'_l)||_2 / sqrt(active_params_l)
        Assumes grads already populated.
        """
        activities = []
        for name, p in model.named_parameters():
            if name not in masks:
                continue
            m = masks[name]
            if p.grad is None:
                continue
            g = p.grad * m  # only active weights
            active = int(m.sum().item())
            if active == 0:
                continue
            norm = g.norm().item() / math.sqrt(active)
            activities.append(norm)

        if not activities:
            return 0.0
        return sum(activities) / len(activities)

    def _update_stats(self, ce: float, div: float, grad: float):
        b = self.stats_beta

        def upd(mu, sig, x):
            mu_new = (1 - b) * mu + b * x
            sig_new = (1 - b) * sig + b * abs(x - mu_new)
            return mu_new, sig_new

        self.ce_mu, self.ce_sigma = upd(self.ce_mu, self.ce_sigma, ce)
        self.div_mu, self.div_sigma = upd(self.div_mu, self.div_sigma, div)
        self.grad_mu, self.grad_sigma = upd(self.grad_mu, self.grad_sigma, grad)

    def _normalize(self, x: float, mu: float, sigma: float) -> float:
        return (x - mu) / (sigma + 1e-8)

    def _proxy(
        self,
        ce: float,
        div: float,
        grad: float,
    ) -> float:
        ce_n = self._normalize(ce, self.ce_mu, self.ce_sigma)
        div_n = self._normalize(div, self.div_mu, self.div_sigma)
        grad_n = self._normalize(grad, self.grad_mu, self.grad_sigma)
        return (
            self.w_ce * ce_n
            + self.w_div * div_n
            - self.w_grad * grad_n
        )

    def _respect_layer_bounds(self, masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """If a layer's sparsity is outside [min, max], we adjust by flipping bits."""
        adjusted = {k: v.clone() for k, v in masks.items()}
        sparsities = self._layer_sparsity(adjusted)

        for name, _ in self.param_tensors:
            m = adjusted[name].view(-1)
            total = m.numel()
            if total == 0:
                continue
            active = int(m.sum().item())
            cur_s = 1.0 - active / total
            min_s = self.min_layer_sparsity[name]
            max_s = self.max_layer_sparsity[name]

            if cur_s < min_s:  # too dense: turn some 1s -> 0s
                target_active = int((1.0 - min_s) * total)
                to_prune = max(0, active - target_active)
                if to_prune > 0:
                    ones = torch.nonzero(m, as_tuple=False).view(-1)
                    if ones.numel() > 0:
                        idx = ones[torch.randperm(ones.numel())[:to_prune]]
                        m[idx] = False
            elif cur_s > max_s:  # too sparse: turn some 0s -> 1s
                target_active = int((1.0 - max_s) * total)
                to_regrow = max(0, target_active - active)
                if to_regrow > 0:
                    zeros = torch.nonzero(~m, as_tuple=False).view(-1)
                    if zeros.numel() > 0:
                        idx = zeros[torch.randperm(zeros.numel())[:to_regrow]]
                        m[idx] = True

        return adjusted

    def _swap_mutation(
        self,
        base_masks: Dict[str, torch.Tensor],
        global_reset: bool = False,
        reset_fraction: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """Swap mutation maintaining (approx) global sparsity."""
        mutated = {k: v.clone() for k, v in base_masks.items()}

        if global_reset:
            # Erdos-Renyi-like random reshuffle at same global sparsity
            total = 0
            active = 0
            for name, _ in self.param_tensors:
                m = mutated[name]
                total += m.numel()
                active += int(m.sum().item())
            cur_sparsity = 1.0 - active / max(total, 1)
            target_zeros = int(total * cur_sparsity)
            target_ones = total - target_zeros

            flat = torch.zeros(total, dtype=torch.bool, device=self.device)
            ones_idx = torch.randperm(total, device=self.device)[:target_ones]
            flat[ones_idx] = True
            offset = 0
            for name, _ in self.param_tensors:
                m = mutated[name]
                n = m.numel()
                mutated[name] = flat[offset:offset+n].view_as(m)
                offset += n
            return mutated

        # standard small swap mutation
        active_locs = []
        inactive_locs = []
        for name, _ in self.param_tensors:
            m = mutated[name].view(-1)
            idx = torch.arange(m.numel(), device=self.device)
            ones = idx[m.bool()]
            zeros = idx[~m.bool()]
            if ones.numel() > 0:
                active_locs.append((name, ones))
            if zeros.numel() > 0:
                inactive_locs.append((name, zeros))

        if not active_locs or not inactive_locs:
            return mutated

        total_active = sum(len(x[1]) for x in active_locs)
        n_swap = max(1, int(self.p_mutate * total_active))

        def flatten_pairs(lst):
            out = []
            for name, idxs in lst:
                for i in idxs:
                    out.append((name, int(i.item())))
            return out

        active_flat = flatten_pairs(active_locs)
        inactive_flat = flatten_pairs(inactive_locs)

        n_swap = min(n_swap, len(active_flat), len(inactive_flat))
        if n_swap == 0:
            return mutated

        perm_a = torch.randperm(len(active_flat))[:n_swap].tolist()
        perm_i = torch.randperm(len(inactive_flat))[:n_swap].tolist()

        for a_idx, i_idx in zip(perm_a, perm_i):
            name_a, pos_a = active_flat[a_idx]
            name_i, pos_i = inactive_flat[i_idx]
            ma = mutated[name_a].view(-1)
            mi = mutated[name_i].view(-1)
            ma[pos_a] = False
            mi[pos_i] = True

        mutated = self._respect_layer_bounds(mutated)
        return mutated

    def _maybe_global_reset_flag(self) -> bool:
        if self.global_reset_boost_steps > 0:
            p = self.p_global_reset_boost
        else:
            p = self.p_global_reset_base
        return torch.rand(1).item() < p

    def update_val_history(self, val_perplexity: float, stagnation_patience: int = 5):
        """Called from training loop after validation / mask update."""
        self.val_history.append(val_perplexity)
        if len(self.val_history) < stagnation_patience + 1:
            return

        recent = self.val_history[-(stagnation_patience + 1):]
        last = recent[-1]
        prev_best = min(recent[:-1])
        if last >= prev_best:
            self.stagnant_counter += 1
        else:
            self.stagnant_counter = 0

        if self.stagnant_counter >= stagnation_patience:
            self.global_reset_boost_steps = 2
            self.stagnant_counter = 0

    def es_step(
        self,
        model: nn.Module,
        micro_batch: Tuple[torch.Tensor, torch.Tensor],
    ):
        """(1 + lambda)-ES over masks with full proxy"""
        device = next(model.parameters()).device
        x_mb, y_mb = micro_batch
        x_mb = x_mb.to(device)
        y_mb = y_mb.to(device)

        # candidate 0: current mask
        candidate_masks: List[Dict[str, torch.Tensor]] = [self._clone_masks()]

        # lambda mutated candidates
        for _ in range(self.lambda_):
            global_reset = self._maybe_global_reset_flag()
            mutated = self._swap_mutation(
                base_masks=self.masks,
                global_reset=global_reset,
            )
            candidate_masks.append(mutated)

        ce_vals = []
        div_vals = []
        grad_vals = []

        # evaluate each candidate
        for masks in candidate_masks:
            original_masks = self.masks
            self.masks = masks
            
            # Apply mask without no_grad to allow backward pass
            for name, p in model.named_parameters():
                if name in self.masks:
                    p.data.mul_(self.masks[name].to(p.device))

            model.zero_grad(set_to_none=True)
            model.train()
            logits = model(x_mb)
            loss_ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_mb.view(-1),
                reduction="mean",
            )
            loss_ce.backward()

            ce_val = float(loss_ce.item())
            div_val = self._mask_diversity(masks)
            grad_val = self._avg_layer_grad_activity(model, masks)

            ce_vals.append(ce_val)
            div_vals.append(div_val)
            grad_vals.append(grad_val)

            # restore original masks
            self.masks = original_masks
            for name, p in model.named_parameters():
                if name in self.masks:
                    p.data.mul_(self.masks[name].to(p.device))

        # pick winner based on proxy
        self._update_stats(ce_vals[0], div_vals[0], grad_vals[0])

        proxies = [self._proxy(c, d, g) for c, d, g in zip(ce_vals, div_vals, grad_vals)]
        best_idx = int(torch.tensor(proxies).argmin().item())
        best_masks = candidate_masks[best_idx]

        # commit
        with torch.no_grad():
            self.masks = {k: v.to(self.device) for k, v in best_masks.items()}
            self.apply_to_model_()

        if self.global_reset_boost_steps > 0:
            self.global_reset_boost_steps -= 1


####################################
# 6. Training Loop
####################################

def parse_args():
    """Parse command-line arguments for Colab/flexible training"""
    parser = argparse.ArgumentParser(description="SparsAE Training on WikiText-103")
    
    # Model configuration
    parser.add_argument("--model_size", type=str, default="small", choices=["tiny", "small", "medium"],
                        help="Model size: tiny (49M), small (125M), medium (350M)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--seq_len", type=int, default=256,
                        help="Sequence length")
    
    # Training configuration
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay")
    
    # Memory optimization
    parser.add_argument("--max_train_examples", type=int, default=50000,
                        help="Max training examples to load (for OOM prevention)")
    parser.add_argument("--max_val_examples", type=int, default=5000,
                        help="Max validation examples to load (for OOM prevention)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (set 0 for Colab OOM issues)")
    
    # Sparsity configuration
    parser.add_argument("--sparsity", type=float, default=0.8,
                        help="Global sparsity ratio (0.0-1.0)")
    parser.add_argument("--lambda_", type=int, default=3,
                        help="Number of ES candidates")
    parser.add_argument("--mask_opt_every", type=int, default=1000,
                        help="Mask optimization frequency (steps)")
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Monitoring
    parser.add_argument("--eval_interval", type=int, default=200,
                        help="Evaluation frequency (steps)")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Logging frequency (steps)")
    
    return parser.parse_args()


def get_model_config(model_size: str):
    """Get model configuration based on size"""
    configs = {
        "tiny": {
            "d_model": 384,
            "n_layers": 6,
            "n_heads": 6,
            "d_ff": 1536,
            "description": "49M parameters"
        },
        "small": {
            "d_model": 768,
            "n_layers": 12,
            "n_heads": 12,
            "d_ff": 3072,
            "description": "125M parameters"
        },
        "medium": {
            "d_model": 1024,
            "n_layers": 24,
            "n_heads": 16,
            "d_ff": 4096,
            "description": "350M parameters"
        }
    }
    return configs[model_size]


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get model configuration
    model_config = get_model_config(args.model_size)
    print(f"\n🚀 Model: {args.model_size.upper()} ({model_config['description']})")
    print(f"📊 Config: {model_config}")

    # === Hyperparameters ===
    seq_len = args.seq_len
    batch_size = args.batch_size
    num_steps = args.max_steps

    base_lr = args.lr
    weight_decay = args.weight_decay

    global_sparsity = args.sparsity
    lambda_ = args.lambda_
    p_mutate = 0.005
    mask_opt_every = args.mask_opt_every
    stats_beta = 0.01

    # self-distillation
    alpha_initial = 0.5
    alpha_final = 0.1
    distill_anneal_frac = 0.7
    warmup_distill_steps = 500

    # burn-in
    N_burn_in = 5
    burn_in_lr_max_mult = 1.5

    # proxy weights annealing
    w_div_initial = 0.01
    w_grad_initial = 0.005
    proxy_anneal_frac = 0.7

    # === Data ===
    print("\n" + "="*60)
    print("Loading WikiText-103...")
    print("="*60)
    
    # Memory-optimized: limit number of examples loaded
    train_ds = WikiTextDataset(
        split="train", 
        max_length=seq_len+1,
        max_examples=args.max_train_examples
    )
    val_ds = WikiTextDataset(
        split="validation", 
        max_length=seq_len+1,
        max_examples=args.max_val_examples
    )

    # Use num_workers from args (default 0 for Colab to prevent OOM)
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=(args.num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=min(16, batch_size), 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=(args.num_workers > 0)
    )

    # micro-batch for ES proxy
    micro_batch = next(iter(val_loader))
    val_batch_stag = next(iter(val_loader))

    # === Model & opt ===
    print("\n" + "="*60)
    print(f"Initializing {args.model_size.upper()} Transformer LM...")
    print("="*60)
    
    vocab_size = 50257  # GPT-2 vocab size
    model = TinyTransformerLM(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        d_ff=model_config["d_ff"]
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # === Sparsae components ===
    print("\n" + "="*60)
    print("Initializing SparsAE...")
    print("="*60)
    
    mask_manager = SparsaeMaskManager(
        model=model,
        global_sparsity=global_sparsity,
        lambda_=lambda_,
        p_mutate=p_mutate,
        stats_beta=stats_beta,
        device=device,
    )
    mask_manager.apply_to_model_()
    
    active_params = sum(
        int(mask_manager.masks[name].sum().item()) 
        for name in mask_manager.masks 
        if name in [n for n, _ in mask_manager.param_tensors]
    )
    print(f"Active parameters: {active_params:,} ({100*(1-global_sparsity):.1f}% of total)")

    ema_teacher = EMATeacher(model, beta=0.999)

    burn_in_remaining: Dict[str, int] = {name: 0 for name, _ in model.named_parameters()}

    def schedule_alpha_kl(step: int) -> float:
        if step < warmup_distill_steps:
            return 0.0
        t = max(0, step - warmup_distill_steps)
        T_anneal = max(1, int(distill_anneal_frac * num_steps))
        ratio = min(1.0, t / T_anneal)
        return alpha_final + 0.5 * (alpha_initial - alpha_final) * (1 + math.cos(math.pi * ratio))

    def schedule_proxy_weights(mask_update_step: int, total_mask_updates: int):
        if total_mask_updates <= 0:
            return
        ratio = min(1.0, mask_update_step / max(1, int(proxy_anneal_frac * total_mask_updates)))
        decay = (1.0 - ratio) ** 2
        mask_manager.w_ce = 1.0
        mask_manager.w_div = w_div_initial * decay
        mask_manager.w_grad = w_grad_initial * decay

    def compute_val_perplexity(batch):
        model.eval()
        x_val, y_val = batch
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        with torch.no_grad():
            logits = model(x_val)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_val.view(-1),
                reduction="mean",
            )
        ppl = math.exp(min(loss.item(), 20))  # Cap for numerical stability
        return ppl

    step = 0
    mask_update_step = 0
    total_mask_updates_planned = max(1, num_steps // mask_opt_every)

    # Setup checkpointing
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Checkpoints will be saved to: {checkpoint_dir}")
    
    # Load from checkpoint if specified
    if args.resume_from:
        print(f"\n📂 Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        mask_manager.mask = checkpoint['mask']
        print(f"✅ Resumed from step {step}")

    train_iter = iter(train_loader)

    print("\n" + "="*60)
    print("Starting SparsAE Training on WikiText-103")
    print("="*60)
    print(f"Steps: {num_steps}")
    print(f"Sequence Length: {seq_len}")
    print(f"Batch Size: {batch_size}")
    print(f"Sparsity: {global_sparsity*100:.0f}%")
    print(f"ES lambda: {lambda_}")
    print(f"Mask updates every: {mask_opt_every} steps")
    print("="*60 + "\n")

    while step < num_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)

        mask_manager.apply_to_model_()

        model.train()
        logits_student = model(x)

        loss_ce = F.cross_entropy(
            logits_student.view(-1, logits_student.size(-1)),
            y.view(-1),
            reduction="mean",
        )

        alpha_kl = schedule_alpha_kl(step)
        if alpha_kl > 0.0:
            ema_teacher.teacher.eval()
            with torch.no_grad():
                logits_teacher = ema_teacher.teacher(x)
            loss_kl = kl_divergence_logits(logits_student, logits_teacher)
        else:
            loss_kl = torch.tensor(0.0, device=device)

        loss = loss_ce + alpha_kl * loss_kl

        optimizer.zero_grad()
        loss.backward()

        # burn-in LR scaling
        if N_burn_in > 0:
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                if burn_in_remaining.get(name, 0) > 0:
                    mult = 1.0 + (burn_in_lr_max_mult - 1.0) * (
                        1.0 - burn_in_remaining[name] / max(1, N_burn_in)
                    )
                    p.grad.mul_(mult)

        optimizer.step()
        mask_manager.apply_to_model_()
        ema_teacher.update(model, mask_manager.masks)

        # periodic ES update
        if (step + 1) % mask_opt_every == 0:
            mask_update_step += 1
            schedule_proxy_weights(mask_update_step, total_mask_updates_planned)

            print(f"\n[step {step+1}] Running ES mask optimization...")
            mask_manager.es_step(model, micro_batch)

            if N_burn_in > 0:
                for name, _ in model.named_parameters():
                    burn_in_remaining[name] = N_burn_in

            val_ppl = compute_val_perplexity(val_batch_stag)
            mask_manager.update_val_history(val_ppl, stagnation_patience=5)
            print(f"[step {step+1}] val_ppl={val_ppl:.3f}, w_div={mask_manager.w_div:.5f}, w_grad={mask_manager.w_grad:.5f}")

        if N_burn_in > 0:
            for name in burn_in_remaining:
                if burn_in_remaining[name] > 0:
                    burn_in_remaining[name] -= 1

        if step % 200 == 0:
            ppl_train = math.exp(min(loss_ce.item(), 20))
            print(
                f"step {step:5d} | ppl={ppl_train:7.2f} | loss_ce={loss_ce.item():.4f} | "
                f"loss_kl={loss_kl.item():.4f} | alpha_kl={alpha_kl:.4f}"
            )
        
        # Save checkpoint
        if (step + 1) % args.checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step+1}.pt"
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mask': mask_manager.mask,
                'args': vars(args),
                'val_ppl': val_ppl if 'val_ppl' in locals() else None,
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

        step += 1

    print("\n" + "="*60)
    print("Training finished!")
    print("="*60)
    
    # Final validation
    print("\nComputing final validation perplexity...")
    val_losses = []
    model.eval()
    with torch.no_grad():
        for i, (x_val, y_val) in enumerate(val_loader):
            if i >= 100:  # Limit to 100 batches
                break
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            logits = model(x_val)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_val.view(-1),
                reduction="mean",
            )
            val_losses.append(loss.item())
    
    final_val_loss = sum(val_losses) / len(val_losses)
    final_val_ppl = math.exp(min(final_val_loss, 20))
    print(f"Final validation perplexity: {final_val_ppl:.2f}")


if __name__ == "__main__":
    main()
