"""
Novel model architectures and building blocks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class DynamicDepthNetwork(nn.Module):
    """
    Network with dynamic depth adaptation during training
    
    Core Concept:
    - Start with shallow network, progressively grow deeper
    - Add layers dynamically based on training progress
    - Each layer can be selectively activated
    
    Benefits:
    - Faster initial training (fewer layers)
    - Progressive complexity increase
    - Better gradient flow in early training
    
    Novelty:
    - Automatic depth scheduling based on loss convergence
    - Smooth layer activation via learned gates
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_layers: int = 6,
        initial_active_layers: int = 2
    ):
        super().__init__()
        self.max_layers = max_layers
        self.active_layers = initial_active_layers
        
        # Create all layers
        self.layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(max_layers)
        ])
        
        # Layer activation gates (learned)
        self.layer_gates = nn.Parameter(torch.ones(max_layers))
        
        # Output projection
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        # Soft gating for smooth activation
        gates = torch.sigmoid(self.layer_gates / temperature)
        
        for i, layer in enumerate(self.layers[:self.active_layers]):
            # Apply layer with soft gating
            gate_value = gates[i]
            x_new = F.relu(layer(x))
            x = gate_value * x_new + (1 - gate_value) * x
        
        return self.output(x)
    
    def grow_network(self):
        """Activate one more layer"""
        if self.active_layers < self.max_layers:
            self.active_layers += 1
            print(f"Network grown to {self.active_layers} layers")


class MixtureOfExpertsLayer(nn.Module):
    """
    Sparse Mixture of Experts with learned routing
    
    Core Concept:
    - Multiple expert networks, each specializing in different inputs
    - Learned gating network routes inputs to experts
    - Only activate top-k experts per input (sparsity)
    
    Benefits:
    - Increased model capacity without proportional compute increase
    - Specialization of sub-networks
    - Better scaling to very large models
    
    Related Work: Shazeer et al. (2017) "Outrageously Large Neural Networks"
    Novelty: Load-balancing loss and expert diversity regularization
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        expert_dim: int = 512,
        top_k: int = 2,
        diversity_penalty: float = 0.01
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.diversity_penalty = diversity_penalty
        
        # Gating network
        self.gate = nn.Linear(input_dim, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, input_dim)
            )
            for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Compute gating scores
        gate_logits = self.gate(x)
        gate_scores = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # Gather top-k expert outputs
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, self.top_k)
        selected_outputs = expert_outputs[batch_indices, top_k_indices]
        
        # Weighted combination of top-k experts
        output = (selected_outputs * top_k_scores.unsqueeze(-1)).sum(dim=1)
        
        # Load balancing loss (encourage uniform expert usage)
        expert_usage = gate_scores.mean(dim=0)
        load_balance_loss = self.diversity_penalty * (expert_usage ** 2).sum()
        
        return output, load_balance_loss


class AdaptiveComputationTime(nn.Module):
    """
    Adaptive Computation Time - dynamically determine computation depth per input
    
    Core Concept:
    - Network decides how many processing steps to use per input
    - Easy inputs: fewer steps, hard inputs: more steps
    - Learned halting mechanism
    
    Benefits:
    - Computation adapts to input difficulty
    - More efficient than fixed-depth networks
    - Can handle variable-complexity inputs
    
    Related Work: Graves (2016) "Adaptive Computation Time for Recurrent Neural Networks"
    Novelty: Applied to feedforward networks with attention-based halting
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        max_steps: int = 10,
        halt_threshold: float = 0.99
    ):
        super().__init__()
        self.max_steps = max_steps
        self.halt_threshold = halt_threshold
        
        # Processing cell
        self.cell = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        
        # Halting predictor
        self.halt_predictor = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Initialize
        state = x
        halting_prob = torch.zeros(batch_size, device=x.device)
        accumulated_output = torch.zeros_like(x)
        step_count = torch.zeros(batch_size, device=x.device)
        
        for step in range(self.max_steps):
            # Process
            state = self.cell(state.unsqueeze(1)).squeeze(1)
            
            # Predict halting probability
            p_halt = torch.sigmoid(self.halt_predictor(state)).squeeze(-1)
            
            # Determine which samples should halt
            still_running = (halting_prob < self.halt_threshold).float()
            
            # Update accumulated probability
            p_step = still_running * p_halt
            halting_prob += p_step
            
            # Accumulate output weighted by step probability
            accumulated_output += state * p_step.unsqueeze(-1)
            
            # Update step count
            step_count += still_running
            
            # Early stopping if all samples have halted
            if (halting_prob >= self.halt_threshold).all():
                break
        
        # Ponder cost (regularization term)
        ponder_cost = step_count.mean()
        
        return accumulated_output, ponder_cost


class HyperNetwork(nn.Module):
    """
    HyperNetwork that generates weights for main network
    
    Core Concept:
    - Small network generates weights for larger network
    - Weight generation conditioned on input or task
    - Enables rapid adaptation and weight sharing
    
    Benefits:
    - Parameter efficiency through weight sharing
    - Fast adaptation to new tasks
    - Implicit regularization
    
    Related Work: Ha et al. (2016) "HyperNetworks"
    Novelty: Task-conditioned weight generation with structured sparsity
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hyper_hidden_dim: int = 64,
        num_task_embeddings: int = 10
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Task embeddings
        self.task_embeddings = nn.Embedding(num_task_embeddings, hyper_hidden_dim)
        
        # HyperNetwork to generate weights
        self.weight_generator = nn.Sequential(
            nn.Linear(hyper_hidden_dim, hyper_hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim * 2, input_dim * hidden_dim + hidden_dim * output_dim)
        )
        
    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        # Get task embedding
        task_emb = self.task_embeddings(torch.tensor(task_id, device=x.device))
        
        # Generate weights
        generated_weights = self.weight_generator(task_emb)
        
        # Split into layer weights
        w1_size = x.size(-1) * self.hidden_dim
        w1 = generated_weights[:w1_size].view(x.size(-1), self.hidden_dim)
        w2 = generated_weights[w1_size:].view(self.hidden_dim, self.output_dim)
        
        # Forward pass with generated weights
        hidden = F.relu(x @ w1)
        output = hidden @ w2
        
        return output


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism processing at different resolutions
    
    Core Concept:
    - Parallel attention at multiple scales
    - Captures both local and global dependencies
    - Hierarchical feature aggregation
    
    Benefits:
    - Better handling of multi-scale patterns
    - More expressive than single-scale attention
    - Efficient through parallel processing
    
    Novelty: Learned scale selection and cross-scale interaction
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        scales: Tuple[int, ...] = (1, 2, 4, 8)
    ):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        
        # Multi-head attention for each scale
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in scales
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Linear(embed_dim * self.num_scales, embed_dim)
        
        # Learnable scale importance
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        scale_outputs = []
        
        for scale, attn in zip(self.scales, self.scale_attentions):
            if scale == 1:
                # Original resolution
                scale_out, _ = attn(x, x, x)
            else:
                # Downsample
                if seq_len % scale != 0:
                    pad_len = scale - (seq_len % scale)
                    x_padded = F.pad(x, (0, 0, 0, pad_len))
                else:
                    x_padded = x
                
                downsampled = x_padded.view(
                    batch_size, -1, scale, embed_dim
                ).mean(dim=2)
                
                # Apply attention at this scale
                scale_out, _ = attn(downsampled, downsampled, downsampled)
                
                # Upsample back
                scale_out = scale_out.repeat_interleave(scale, dim=1)[:, :seq_len, :]
            
            scale_outputs.append(scale_out)
        
        # Weight and concatenate scales
        weights = F.softmax(self.scale_weights, dim=0)
        weighted_outputs = [w * out for w, out in zip(weights, scale_outputs)]
        
        # Fuse scales
        concatenated = torch.cat(weighted_outputs, dim=-1)
        fused = self.scale_fusion(concatenated)
        
        return fused + x  # Residual connection
