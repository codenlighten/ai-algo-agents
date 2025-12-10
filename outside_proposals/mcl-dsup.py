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
