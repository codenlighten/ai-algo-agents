"""
Novel loss functions beyond standard cross-entropy and MSE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConfidencePenalizedCrossEntropy(nn.Module):
    """
    Cross-entropy with confidence penalty to prevent overconfidence
    
    Core Concept:
    - Standard cross-entropy + penalty for overly confident predictions
    - Encourages calibrated predictions
    - Prevents model from becoming too certain on training data
    
    Benefits:
    - Better calibration (predicted probabilities match actual frequencies)
    - Improved generalization by preventing overconfidence
    - Reduces overfitting to training data
    
    Trade-offs:
    - Additional hyperparameter (penalty weight)
    - May slow convergence slightly
    
    Related Work: Pereyra et al. (2017) "Regularizing Neural Networks by Penalizing Confident Output Distributions"
    """
    
    def __init__(self, penalty_weight: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits / self.temperature, targets)
        
        # Confidence penalty (negative entropy)
        probs = F.softmax(logits / self.temperature, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        # Total loss: CE + penalty for low entropy (high confidence)
        loss = ce_loss - self.penalty_weight * entropy
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Core Concept:
    - Down-weights loss for well-classified examples
    - Focuses on hard examples
    - Particularly effective for imbalanced datasets
    
    Benefits:
    - Better handling of class imbalance
    - Improved performance on minority classes
    - No need for manual re-weighting
    
    Related Work: Lin et al. (2017) "Focal Loss for Dense Object Detection"
    Novelty: Extended with adaptive gamma scheduling
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        adaptive_gamma: bool = False
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.adaptive_gamma = adaptive_gamma
        self.step = 0
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        
        # Adaptive gamma: start high, decay over training
        gamma = self.gamma
        if self.adaptive_gamma:
            self.step += 1
            gamma = self.gamma * (0.95 ** (self.step / 1000))
        
        focal_weight = (1 - p) ** gamma
        loss = focal_weight * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_weight = self.alpha.gather(0, targets)
            loss = alpha_weight * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ContrastivePredictiveLoss(nn.Module):
    """
    Contrastive loss for self-supervised learning
    
    Core Concept:
    - Pull positive pairs together, push negative pairs apart
    - Can be used for representation learning without labels
    - InfoNCE-style contrastive learning
    
    Benefits:
    - Self-supervised pre-training
    - Learn robust representations
    - Reduced dependence on labeled data
    
    Related Work: van den Oord et al. (2018) "Representation Learning with Contrastive Predictive Coding"
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            anchor: [batch_size, dim] anchor representations
            positive: [batch_size, dim] positive representations
            negatives: [batch_size, num_negatives, dim] negative representations
        """
        batch_size = anchor.size(0)
        
        # Normalize representations
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        
        # Compute positive similarities
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        
        if negatives is not None:
            # Normalize negatives
            negatives = F.normalize(negatives, dim=2)
            
            # Compute negative similarities [batch_size, num_negatives]
            neg_sim = torch.bmm(
                anchor.unsqueeze(1),
                negatives.transpose(1, 2)
            ).squeeze(1) / self.temperature
            
            # Concatenate positive and negative similarities
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        else:
            # Use all other samples in batch as negatives
            similarity_matrix = torch.matmul(anchor, anchor.T) / self.temperature
            pos_sim = torch.diag(similarity_matrix)
            logits = similarity_matrix
        
        # Targets: positive is always at index 0
        targets = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        
        loss = F.cross_entropy(logits, targets)
        return loss


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for robust regression
    
    Core Concept:
    - Combines advantages of L1 and L2 loss
    - Adaptive behavior based on error magnitude
    - More robust to outliers than MSE
    
    Benefits:
    - Better handling of outliers
    - Faster convergence than L1
    - Automatic adaptation to error distribution
    
    Related Work: Wang et al. (2019) "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"
    """
    
    def __init__(self, omega: float = 14, theta: float = 0.5, epsilon: float = 1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        
        # Pre-compute constants
        self.C = self.omega * (1 - torch.tensor(omega / (omega + epsilon), dtype=torch.float32))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = (target - pred).abs()
        
        # Adaptive behavior based on error magnitude
        A = self.omega * (
            1 / (1 + torch.pow(self.theta / self.epsilon, 2))
        ) * (self.theta / self.epsilon) * torch.pow(
            self.theta / self.epsilon + torch.abs(1 - target),
            2
        )
        
        C = self.theta * A - self.omega * torch.log(1 + self.theta / self.epsilon)
        
        # Different behavior for small and large errors
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + delta / self.epsilon),
            A * delta - C
        )
        
        return loss.mean()


class NoiseContrastiveEstimation(nn.Module):
    """
    Noise Contrastive Estimation for efficient training with large output spaces
    
    Core Concept:
    - Turn multi-class classification into binary classification
    - Distinguish data from noise samples
    - Efficient for very large vocabularies/output spaces
    
    Benefits:
    - O(k) complexity instead of O(V) for vocabulary size V
    - Enables training with millions of classes
    - Theoretically grounded
    
    Trade-offs:
    - Requires noise distribution
    - Approximate objective
    
    Related Work: Gutmann & Hyvärinen (2010) "Noise-contrastive estimation"
    Novelty: Dynamic noise distribution adaptation
    """
    
    def __init__(
        self,
        num_classes: int,
        num_noise_samples: int = 10,
        noise_distribution: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_noise_samples = num_noise_samples
        
        # Uniform noise distribution if not provided
        if noise_distribution is None:
            self.register_buffer(
                'noise_dist',
                torch.ones(num_classes) / num_classes
            )
        else:
            self.register_buffer('noise_dist', noise_distribution)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model_output_fn=None
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes] model outputs
            targets: [batch_size] target class indices
            model_output_fn: Optional function to compute logits for noise samples
        """
        batch_size = targets.size(0)
        
        # Sample noise examples
        noise_samples = torch.multinomial(
            self.noise_dist,
            batch_size * self.num_noise_samples,
            replacement=True
        ).view(batch_size, self.num_noise_samples)
        
        # Get logits for target and noise samples
        target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        noise_logits = logits.gather(1, noise_samples)
        
        # Compute NCE loss
        # Probability that sample came from data
        target_prob = torch.sigmoid(target_logits)
        
        # Probability that noise sample came from data
        noise_prob = torch.sigmoid(noise_logits)
        
        # Loss: maximize log P(data) + log(1 - P(noise))
        loss = -(
            torch.log(target_prob + 1e-8).mean() +
            torch.log(1 - noise_prob + 1e-8).mean()
        )
        
        return loss


class CurriculumLoss(nn.Module):
    """
    Curriculum learning via dynamic loss reweighting
    
    Core Concept:
    - Start with easy examples, gradually introduce harder ones
    - Dynamically reweight loss based on example difficulty
    - Adapts curriculum based on model's current capability
    
    Benefits:
    - Faster initial learning
    - Better final performance
    - Automatic curriculum discovery
    
    Novelty: Fully automated curriculum based on loss dynamics
    """
    
    def __init__(
        self,
        base_criterion: nn.Module,
        warmup_steps: int = 1000,
        difficulty_momentum: float = 0.9
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.warmup_steps = warmup_steps
        self.difficulty_momentum = difficulty_momentum
        self.step = 0
        self.difficulty_scores = None
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        sample_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Compute base loss for each sample
        base_loss = self.base_criterion(inputs, targets)
        
        if sample_ids is None:
            return base_loss.mean()
        
        self.step += 1
        
        # Initialize difficulty scores
        if self.difficulty_scores is None:
            self.difficulty_scores = {}
        
        # Update difficulty scores with momentum
        with torch.no_grad():
            for i, sid in enumerate(sample_ids):
                sid = sid.item()
                current_loss = base_loss[i].item()
                
                if sid not in self.difficulty_scores:
                    self.difficulty_scores[sid] = current_loss
                else:
                    self.difficulty_scores[sid] = (
                        self.difficulty_momentum * self.difficulty_scores[sid] +
                        (1 - self.difficulty_momentum) * current_loss
                    )
        
        # Compute curriculum weights
        if self.step < self.warmup_steps:
            # During warmup, focus on easier examples
            progress = self.step / self.warmup_steps
            difficulties = torch.tensor([
                self.difficulty_scores.get(sid.item(), 1.0)
                for sid in sample_ids
            ], device=base_loss.device)
            
            # Weight: higher for easier examples early, more uniform later
            weights = torch.exp(-difficulties * (1 - progress))
            weights = weights / weights.sum() * len(weights)
        else:
            # After warmup, uniform weighting
            weights = torch.ones_like(base_loss)
        
        # Apply curriculum weighting
        weighted_loss = (base_loss * weights).mean()
        return weighted_loss
