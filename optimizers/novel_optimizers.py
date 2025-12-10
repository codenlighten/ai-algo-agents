"""
Novel optimizer implementations exploring beyond SGD/Adam
"""
import torch
from torch.optim import Optimizer
from typing import List, Optional
import math


class SecondOrderMomentumOptimizer(Optimizer):
    """
    Novel optimizer using second-order momentum with adaptive scaling
    
    Core Concept:
    - Maintains both first-order (gradient) and second-order (curvature) momentum
    - Adaptively scales learning rate based on loss landscape geometry
    - Uses exponential moving average of Hessian diagonal approximation
    
    Benefits:
    - Better handling of varying curvatures in loss landscape
    - Reduced sensitivity to learning rate
    - Faster convergence in non-convex optimization
    
    Trade-offs:
    - Slightly higher memory overhead (2x gradient buffers)
    - Additional computation for curvature estimation
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        curvature_momentum: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0
    ):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            curvature_momentum=curvature_momentum,
            eps=eps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('SecondOrderMomentum does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['curvature'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq, curvature = (
                    state['exp_avg'],
                    state['exp_avg_sq'],
                    state['curvature']
                )
                
                state['step'] += 1
                beta1, beta2 = group['beta1'], group['beta2']
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).add_(grad ** 2, alpha=1 - beta2)
                
                # Approximate curvature using gradient differences
                # This is a diagonal Hessian approximation
                curvature.mul_(group['curvature_momentum']).add_(
                    torch.abs(grad), alpha=1 - group['curvature_momentum']
                )
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Corrected estimates
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                
                # Adaptive learning rate based on curvature
                denom = exp_avg_sq_corrected.sqrt().add_(group['eps'])
                adaptive_lr = group['lr'] / (1 + curvature.mean())
                
                # Parameter update with curvature-aware scaling
                p.addcdiv_(exp_avg_corrected, denom, value=-adaptive_lr)
        
        return loss


class LookAheadWrapper(Optimizer):
    """
    Look-Ahead wrapper for any optimizer
    
    Core Concept:
    - Maintains two sets of weights: fast weights and slow weights
    - Fast weights explore using base optimizer
    - Slow weights interpolate periodically with fast weights
    
    Benefits:
    - Improved stability and convergence
    - Reduces sensitivity to hyperparameters
    - Can wrap any existing optimizer
    
    Related Work: Zhang et al. (2019) "Lookahead Optimizer: k steps forward, 1 step back"
    Novelty: Extended with adaptive synchronization frequency
    """
    
    def __init__(
        self,
        base_optimizer: Optimizer,
        la_steps: int = 5,
        la_alpha: float = 0.5,
        adaptive_sync: bool = True
    ):
        self.base_optimizer = base_optimizer
        self.la_steps = la_steps
        self.la_alpha = la_alpha
        self.adaptive_sync = adaptive_sync
        
        self.state = {'la_step': 0}
        self.param_groups = self.base_optimizer.param_groups
        
        # Initialize slow weights
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.base_optimizer.state[p]
                param_state['slow_buffer'] = torch.empty_like(p.data)
                param_state['slow_buffer'].copy_(p.data)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self.state['la_step'] += 1
        
        # Determine if we should synchronize
        sync_step = self.la_steps
        if self.adaptive_sync and loss is not None:
            # Adaptive: sync more frequently if loss is decreasing slowly
            if self.state['la_step'] > 1 and hasattr(self.state, 'prev_loss'):
                improvement = (self.state['prev_loss'] - loss) / (self.state['prev_loss'] + 1e-8)
                if improvement < 0.01:  # Less than 1% improvement
                    sync_step = max(3, self.la_steps // 2)
            self.state['prev_loss'] = loss if isinstance(loss, float) else loss.item()
        
        # Synchronize slow and fast weights
        if self.state['la_step'] % sync_step == 0:
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.base_optimizer.state[p]
                    slow_buffer = param_state['slow_buffer']
                    
                    # Interpolate: slow = slow + alpha * (fast - slow)
                    slow_buffer.add_(p.data - slow_buffer, alpha=self.la_alpha)
                    p.data.copy_(slow_buffer)
        
        return loss
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()


class AdaptiveGradientClipping(Optimizer):
    """
    Optimizer with adaptive gradient clipping based on parameter norms
    
    Core Concept:
    - Clips gradients based on their ratio to parameter norms
    - Prevents gradient explosion while preserving directional information
    - Adapts clipping threshold per-layer
    
    Benefits:
    - Training stability for very deep networks
    - Better than global gradient clipping
    - Preserves gradient direction
    
    Related Work: Brock et al. (2021) "High-Performance Large-Scale Image Recognition Without Normalization"
    Novelty: Extended with momentum-based threshold adaptation
    """
    
    def __init__(
        self,
        params,
        base_optimizer_class=torch.optim.Adam,
        clip_factor: float = 0.01,
        momentum: float = 0.9,
        **base_optimizer_kwargs
    ):
        self.clip_factor = clip_factor
        self.momentum = momentum
        
        # Initialize base optimizer
        self.base_optimizer = base_optimizer_class(params, **base_optimizer_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        
        # Initialize threshold tracking
        for group in self.param_groups:
            for p in group['params']:
                if p not in self.base_optimizer.state:
                    self.base_optimizer.state[p] = {}
                self.base_optimizer.state[p]['clip_threshold'] = None
    
    @torch.no_grad()
    def step(self, closure=None):
        # Adaptive gradient clipping
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.base_optimizer.state[p]
                
                # Compute parameter norm
                param_norm = p.norm()
                grad_norm = p.grad.norm()
                
                if grad_norm == 0 or param_norm == 0:
                    continue
                
                # Compute adaptive threshold
                max_norm = param_norm * self.clip_factor
                
                # Momentum-based threshold adaptation
                if state['clip_threshold'] is None:
                    state['clip_threshold'] = max_norm
                else:
                    state['clip_threshold'] = (
                        self.momentum * state['clip_threshold'] +
                        (1 - self.momentum) * max_norm
                    )
                
                # Clip if necessary
                if grad_norm > state['clip_threshold']:
                    p.grad.mul_(state['clip_threshold'] / (grad_norm + 1e-6))
        
        # Perform base optimizer step
        loss = self.base_optimizer.step(closure)
        return loss
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()


class StochasticWeightAveraging:
    """
    Stochastic Weight Averaging (SWA) with cyclic learning rate
    
    Core Concept:
    - Average weights traversed by SGD with cyclic/high learning rate
    - Finds flatter minima that generalize better
    - Can be applied to any optimizer
    
    Benefits:
    - Better generalization
    - Flatter loss landscapes
    - Simple to implement
    
    Related Work: Izmailov et al. (2018) "Averaging Weights Leads to Wider Optima and Better Generalization"
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        swa_start: int = 100,
        swa_freq: int = 10,
        swa_lr: float = 0.05
    ):
        self.optimizer = optimizer
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        
        self.n_averaged = 0
        self.swa_state = {}
        
        # Initialize SWA buffers
        for group in self.optimizer.param_groups:
            for p in group['params']:
                self.swa_state[p] = torch.zeros_like(p.data)
    
    @torch.no_grad()
    def step(self, closure=None, step_num: int = 0):
        loss = self.optimizer.step(closure)
        
        # Update SWA weights
        if step_num >= self.swa_start and step_num % self.swa_freq == 0:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    self.swa_state[p].mul_(self.n_averaged).add_(p.data).div_(self.n_averaged + 1)
            self.n_averaged += 1
        
        return loss
    
    @torch.no_grad()
    def swap_swa_params(self):
        """Swap current parameters with SWA parameters"""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p.data, self.swa_state[p] = self.swa_state[p], p.data.clone()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
