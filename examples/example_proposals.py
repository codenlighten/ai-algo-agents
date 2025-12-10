"""
Example research proposals demonstrating the system
"""
from research.proposal_system import (
    ResearchProposal,
    ExperimentalSetup,
    ResearchProposalBuilder,
    ProposalLibrary
)


def create_example_optimizer_proposal():
    """Example: Second-Order Momentum Optimizer"""
    
    experimental_setup = ExperimentalSetup(
        datasets=["CIFAR-10", "ImageNet-1k (subset)", "WikiText-103"],
        metrics=[
            "Training loss convergence",
            "Test accuracy",
            "Wall-clock time per epoch",
            "GPU memory usage",
            "Number of iterations to convergence"
        ],
        baselines=["SGD", "Adam", "AdamW", "LAMB"],
        expected_improvements={
            "convergence_speed": "15-25% faster to same loss threshold",
            "stability": "Lower variance in training loss trajectory",
            "final_performance": "0.5-1% better test accuracy"
        },
        minimal_compute_requirements="1x NVIDIA A100 (40GB), ~8 hours"
    )
    
    code_snippet = '''
import torch
from torch.optim import Optimizer

class SecondOrderMomentumOptimizer(Optimizer):
    """Optimizer using second-order momentum with adaptive scaling"""
    
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, 
                 curvature_momentum=0.9, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2,
                       curvature_momentum=curvature_momentum, eps=eps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['curvature'] = torch.zeros_like(p)
                
                grad = p.grad
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                curvature = state['curvature']
                
                # Update moments
                exp_avg.mul_(group['beta1']).add_(grad, alpha=1-group['beta1'])
                exp_avg_sq.mul_(group['beta2']).add_(grad**2, alpha=1-group['beta2'])
                
                # Update curvature estimate
                curvature.mul_(group['curvature_momentum']).add_(
                    torch.abs(grad), alpha=1-group['curvature_momentum']
                )
                
                # Compute update with adaptive scaling
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                adaptive_lr = group['lr'] / (1 + curvature.mean())
                
                p.addcdiv_(exp_avg, denom, value=-adaptive_lr)
        
        return None
'''
    
    proposal = ResearchProposalBuilder() \
        .set_title("Second-Order Momentum Optimizer with Adaptive Curvature Scaling") \
        .set_author("AI Algorithms Agent") \
        .set_core_concept(
            "An optimizer that maintains both gradient momentum (first-order) and "
            "curvature momentum (second-order approximation) to adaptively scale "
            "learning rates based on local loss landscape geometry. Uses diagonal "
            "Hessian approximation via gradient magnitude tracking."
        ) \
        .set_summary(
            "Extends Adam-style optimization with explicit curvature tracking to "
            "better handle varying loss surface geometries, particularly in regions "
            "with mixed curvatures."
        ) \
        .add_benefit("15-25% faster convergence in terms of wall-clock time") \
        .add_benefit("Lower sensitivity to learning rate hyperparameter") \
        .add_benefit("Better handling of saddle points and flat regions") \
        .add_benefit("Improved stability in large-batch training") \
        .add_risk("2x memory overhead for additional momentum buffers") \
        .add_risk("Slightly increased computation per step (~10-15%)") \
        .add_risk("May require tuning of curvature momentum hyperparameter") \
        .set_related_work(
            "Builds on Adam (Kingma & Ba, 2015) and AdaGrad (Duchi et al., 2011). "
            "Related to AdaHessian (Yao et al., 2020) which uses Hessian diagonal, "
            "but our approach uses cheaper gradient-magnitude based approximation. "
            "Also related to curvature-aware methods like K-FAC but computationally lighter."
        ) \
        .set_novelty(
            "Novel contribution: Lightweight second-order information via exponential "
            "moving average of gradient magnitudes, combined with adaptive per-parameter "
            "learning rate scaling. Does not require Hessian computation or storage."
        ) \
        .set_implementation("PyTorch", code_snippet) \
        .set_experimental_setup(experimental_setup) \
        .set_scalability(
            "Scales linearly with model parameters. Memory: O(3P) where P is parameter count "
            "(vs O(2P) for Adam). Compute: ~15% overhead vs Adam. Suitable for billion-parameter "
            "models. Fully compatible with distributed training (DDP, FSDP). No additional "
            "communication overhead beyond gradient synchronization."
        ) \
        .set_engineering_constraints(
            "Constraints: Requires 1.5x memory of Adam (acceptable trade-off). "
            "Not suitable for extremely memory-constrained scenarios. "
            "Works best with batch sizes >= 32. Compatible with mixed precision training (FP16/BF16). "
            "Can be combined with gradient accumulation and gradient checkpointing."
        ) \
        .set_reasoning_path(
            "1. Observation: Adam works well but can struggle in high-curvature regions\n"
            "2. Hypothesis: Explicit curvature tracking could help adaptive learning\n"
            "3. Challenge: Computing Hessian is prohibitively expensive\n"
            "4. Solution: Approximate via gradient magnitude EMA (captures local curvature)\n"
            "5. Implementation: Add curvature buffer, use for learning rate scaling\n"
            "6. Validation: Compare against Adam on diverse tasks"
        ) \
        .add_assumption("Gradient magnitude correlates with curvature (empirically true in practice)") \
        .add_assumption("EMA of gradients provides stable curvature estimate") \
        .add_assumption("Per-parameter adaptive LR is beneficial (supported by Adam/AdaGrad success)") \
        .build()
    
    return proposal


def create_example_loss_proposal():
    """Example: Confidence-Penalized Cross-Entropy"""
    
    experimental_setup = ExperimentalSetup(
        datasets=["CIFAR-100", "ImageNet", "Tiny-ImageNet"],
        metrics=[
            "Test accuracy",
            "Expected Calibration Error (ECE)",
            "Negative Log Likelihood (NLL)",
            "Brier score",
            "Overconfidence rate"
        ],
        baselines=["Cross-Entropy", "Label Smoothing", "Focal Loss"],
        expected_improvements={
            "calibration": "20-30% reduction in ECE",
            "overconfidence": "Significantly reduced max confidence on incorrect predictions",
            "generalization": "0.5-1.5% improvement in test accuracy"
        },
        minimal_compute_requirements="2x NVIDIA RTX 3090, ~12 hours"
    )
    
    code_snippet = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfidencePenalizedCrossEntropy(nn.Module):
    """Cross-entropy with explicit confidence penalty for calibration"""
    
    def __init__(self, penalty_weight=0.1, temperature=1.0):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.temperature = temperature
    
    def forward(self, logits, targets):
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits / self.temperature, targets)
        
        # Confidence penalty (negative entropy)
        probs = F.softmax(logits / self.temperature, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        # Encourage higher entropy (less confidence)
        loss = ce_loss - self.penalty_weight * entropy
        return loss

# Usage
model = ...  # your model
criterion = ConfidencePenalizedCrossEntropy(penalty_weight=0.1)
output = model(input)
loss = criterion(output, target)
'''
    
    proposal = ResearchProposalBuilder() \
        .set_title("Confidence-Penalized Cross-Entropy for Improved Calibration") \
        .set_author("AI Algorithms Agent") \
        .set_core_concept(
            "A loss function that combines standard cross-entropy with an explicit "
            "penalty on overconfident predictions. Encourages the model to produce "
            "well-calibrated probability estimates by maximizing entropy on the "
            "predicted distribution."
        ) \
        .set_summary(
            "Addresses the overconfidence problem in modern neural networks by "
            "adding a regularization term that penalizes low-entropy (high-confidence) "
            "predictions, leading to better-calibrated models."
        ) \
        .add_benefit("Significantly improved calibration (20-30% ECE reduction)") \
        .add_benefit("Better uncertainty estimates for downstream tasks") \
        .add_benefit("Reduced overconfidence on out-of-distribution examples") \
        .add_benefit("Simple to implement, single hyperparameter") \
        .add_risk("May slightly slow convergence (trade-off for calibration)") \
        .add_risk("Requires tuning penalty_weight hyperparameter") \
        .add_risk("May reduce top-1 accuracy by 0.1-0.3% in some cases") \
        .set_related_work(
            "Related to Label Smoothing (Szegedy et al., 2016) which also prevents "
            "overconfidence. Similar to entropy regularization in RL. Connects to "
            "temperature scaling (Guo et al., 2017) for calibration. Our approach "
            "integrates calibration into training rather than post-hoc."
        ) \
        .set_novelty(
            "Novel: Direct optimization for calibration during training via entropy "
            "maximization. Unlike label smoothing (which modifies targets), we modify "
            "the loss function. Unlike temperature scaling (post-hoc), we train for "
            "calibration from the start."
        ) \
        .set_implementation("PyTorch", code_snippet) \
        .set_experimental_setup(experimental_setup) \
        .set_scalability(
            "Zero additional memory overhead. Negligible computational overhead "
            "(entropy calculation is O(C) where C is number of classes). "
            "Scales to any model size. Works with distributed training. "
            "Compatible with all standard optimizers."
        ) \
        .set_engineering_constraints(
            "No constraints. Drop-in replacement for cross-entropy. "
            "Works with mixed precision training. Compatible with label smoothing "
            "(can be combined). Suitable for production deployment."
        ) \
        .set_reasoning_path(
            "1. Problem: Modern NNs are overconfident, poor calibration\n"
            "2. Observation: Overconfidence = low entropy in predictions\n"
            "3. Idea: Penalize low entropy directly in loss function\n"
            "4. Implementation: Add negative entropy term to CE loss\n"
            "5. Expected outcome: More uncertain predictions, better calibration\n"
            "6. Validation: Measure ECE, NLL, Brier score"
        ) \
        .add_assumption("Entropy regularization improves calibration (supported by theory)") \
        .add_assumption("Slight accuracy trade-off is acceptable for better calibration") \
        .build()
    
    return proposal


def save_example_proposals():
    """Save example proposals to library"""
    library = ProposalLibrary()
    
    # Create and save proposals
    optimizer_proposal = create_example_optimizer_proposal()
    loss_proposal = create_example_loss_proposal()
    
    library.add_proposal(optimizer_proposal)
    library.add_proposal(loss_proposal)
    
    print(f"✅ Saved optimizer proposal: {optimizer_proposal.proposal_id}")
    print(f"✅ Saved loss proposal: {loss_proposal.proposal_id}")
    
    return library


if __name__ == "__main__":
    library = save_example_proposals()
    
    print("\n" + "="*80)
    print("EXAMPLE PROPOSALS CREATED")
    print("="*80)
    
    for proposal in library.list_proposals():
        print(f"\n📄 {proposal.title}")
        print(f"   ID: {proposal.proposal_id}")
        print(f"   Author: {proposal.author_agent}")
