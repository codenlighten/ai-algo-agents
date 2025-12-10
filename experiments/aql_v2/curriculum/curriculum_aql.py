"""
Curriculum Learning Integration for AQL
Orders training data from easy to hard, integrated with active learning
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class DifficultyMetrics:
    """Metrics for assessing sample difficulty"""
    loss: float
    uncertainty: float
    gradient_norm: float
    perplexity: float
    
    def get_combined_score(self, weights: Dict[str, float] = None) -> float:
        """
        Combine metrics into single difficulty score
        
        Args:
            weights: Weight for each metric. Defaults to equal weights.
        """
        if weights is None:
            weights = {
                'loss': 0.4,
                'uncertainty': 0.3,
                'gradient_norm': 0.2,
                'perplexity': 0.1
            }
        
        score = (
            weights['loss'] * self.loss +
            weights['uncertainty'] * self.uncertainty +
            weights['gradient_norm'] * self.gradient_norm +
            weights['perplexity'] * self.perplexity
        )
        return score


class CurriculumScheduler:
    """
    Manages curriculum progression from easy to hard samples
    
    Key features:
    - Automatic difficulty assessment
    - Smooth progression schedule
    - Integration with uncertainty-based selection
    
    Usage:
        scheduler = CurriculumScheduler(
            total_steps=10000,
            warmup_ratio=0.2,
            pacing_function='root'
        )
        
        for step in range(total_steps):
            difficulty_threshold = scheduler.get_threshold(step)
            # Select samples below threshold
            selected = samples[difficulties < difficulty_threshold]
    """
    
    def __init__(
        self,
        total_steps: int,
        warmup_ratio: float = 0.2,
        pacing_function: str = 'root',
        min_difficulty: float = 0.0,
        max_difficulty: float = 1.0
    ):
        """
        Args:
            total_steps: Total training steps
            warmup_ratio: Fraction of training to spend on easy samples
            pacing_function: How to progress ('linear', 'root', 'exponential')
            min_difficulty: Minimum difficulty threshold (easiest samples)
            max_difficulty: Maximum difficulty threshold (hardest samples)
        """
        self.total_steps = total_steps
        self.warmup_ratio = warmup_ratio
        self.pacing_function = pacing_function
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        
        self.warmup_steps = int(total_steps * warmup_ratio)
    
    def get_threshold(self, current_step: int) -> float:
        """
        Get difficulty threshold for current step
        
        Args:
            current_step: Current training step
            
        Returns:
            difficulty_threshold: Samples with difficulty <= threshold should be used
        """
        if current_step < self.warmup_steps:
            # Warmup phase: only easy samples
            progress = current_step / self.warmup_steps
            threshold = self.min_difficulty + (self.max_difficulty - self.min_difficulty) * 0.3 * progress
        else:
            # Main training: gradually increase difficulty
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            
            if self.pacing_function == 'linear':
                factor = progress
            elif self.pacing_function == 'root':
                # Square root - faster initial increase, slower later
                factor = np.sqrt(progress)
            elif self.pacing_function == 'exponential':
                # Exponential - slower initial increase, faster later
                factor = progress ** 2
            else:
                factor = progress
            
            threshold = self.min_difficulty + (self.max_difficulty - self.min_difficulty) * (0.3 + 0.7 * factor)
        
        return threshold
    
    def get_stage(self, current_step: int) -> str:
        """Get current curriculum stage"""
        threshold = self.get_threshold(current_step)
        
        if threshold < 0.4:
            return 'easy'
        elif threshold < 0.7:
            return 'medium'
        else:
            return 'hard'


class CurriculumAQL:
    """
    Combines curriculum learning with active query-based learning
    
    Strategy:
    1. Assess difficulty of all samples
    2. Progress from easy to hard based on curriculum schedule
    3. Within each difficulty level, use uncertainty for selection
    
    This gives us:
    - Better early training (start with learnable examples)
    - Efficient sample selection (focus on informative samples)
    - Robust late training (handle hard examples when ready)
    """
    
    def __init__(
        self,
        model: nn.Module,
        uncertainty_estimator,
        curriculum_scheduler: CurriculumScheduler,
        difficulty_weights: Dict[str, float] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Neural network
            uncertainty_estimator: Uncertainty estimation module
            curriculum_scheduler: Curriculum progression scheduler
            difficulty_weights: Weights for combining difficulty metrics
            device: Computation device
        """
        self.model = model
        self.uncertainty_estimator = uncertainty_estimator
        self.curriculum_scheduler = curriculum_scheduler
        self.difficulty_weights = difficulty_weights
        self.device = device
        
        # Cached difficulty scores
        self.difficulty_cache: Dict[int, float] = {}
    
    def assess_difficulty(
        self, 
        data: torch.Tensor, 
        targets: torch.Tensor,
        sample_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Assess difficulty of samples
        
        Args:
            data: Input batch
            targets: Target labels
            sample_indices: Optional indices for caching
            
        Returns:
            difficulty_scores: Difficulty for each sample [batch_size]
        """
        self.model.eval()
        
        difficulties = []
        
        with torch.enable_grad():
            for i in range(len(data)):
                sample = data[i:i+1].to(self.device)
                target = targets[i:i+1].to(self.device)
                
                # Check cache
                if sample_indices is not None and sample_indices[i] in self.difficulty_cache:
                    difficulties.append(self.difficulty_cache[sample_indices[i]])
                    continue
                
                # Forward pass
                output = self.model(sample)
                loss = nn.functional.cross_entropy(output, target)
                
                # Uncertainty
                uncertainty = self.uncertainty_estimator.estimate(sample).item()
                
                # Gradient norm (complexity indicator)
                self.model.zero_grad()
                loss.backward()
                grad_norm = sum(
                    p.grad.norm().item() 
                    for p in self.model.parameters() 
                    if p.grad is not None
                )
                
                # Perplexity
                perplexity = torch.exp(loss).item()
                
                # Combine metrics
                metrics = DifficultyMetrics(
                    loss=loss.item(),
                    uncertainty=uncertainty,
                    gradient_norm=grad_norm,
                    perplexity=min(perplexity, 100.0)  # Cap perplexity
                )
                
                difficulty = metrics.get_combined_score(self.difficulty_weights)
                
                # Normalize to [0, 1]
                difficulty = np.tanh(difficulty / 10.0)  # Smooth normalization
                
                # Cache
                if sample_indices is not None:
                    self.difficulty_cache[sample_indices[i]] = difficulty
                
                difficulties.append(difficulty)
        
        return torch.tensor(difficulties)
    
    def select_samples(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        current_step: int,
        n_select: int,
        sample_indices: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Select samples using curriculum + uncertainty
        
        Args:
            data: Input data
            targets: Target labels
            current_step: Current training step
            n_select: Number of samples to select
            sample_indices: Optional indices for tracking
            
        Returns:
            selected_data, selected_targets, selected_indices
        """
        # Get difficulty threshold from curriculum
        difficulty_threshold = self.curriculum_scheduler.get_threshold(current_step)
        
        # Assess difficulties
        difficulties = self.assess_difficulty(data, targets, sample_indices)
        
        # Filter by curriculum (only samples within difficulty range)
        curriculum_mask = difficulties <= difficulty_threshold
        curriculum_indices = torch.where(curriculum_mask)[0]
        
        if len(curriculum_indices) == 0:
            # No samples meet curriculum criteria, return empty
            return torch.tensor([]), torch.tensor([]), []
        
        # Among curriculum-appropriate samples, select by uncertainty
        curriculum_data = data[curriculum_indices]
        curriculum_targets = targets[curriculum_indices]
        
        uncertainties = self.uncertainty_estimator.estimate(
            curriculum_data.to(self.device)
        ).cpu()
        
        # Select top-k by uncertainty
        n_select = min(n_select, len(curriculum_data))
        selected_local_indices = torch.topk(uncertainties, k=n_select).indices
        
        # Map back to original indices
        selected_global_indices = curriculum_indices[selected_local_indices]
        
        selected_data = data[selected_global_indices]
        selected_targets = targets[selected_global_indices]
        
        if sample_indices is not None:
            selected_sample_indices = [sample_indices[i.item()] for i in selected_global_indices]
        else:
            selected_sample_indices = selected_global_indices.tolist()
        
        return selected_data, selected_targets, selected_sample_indices
    
    def get_statistics(self, current_step: int) -> Dict:
        """Get curriculum statistics"""
        threshold = self.curriculum_scheduler.get_threshold(current_step)
        stage = self.curriculum_scheduler.get_stage(current_step)
        
        return {
            'current_step': current_step,
            'difficulty_threshold': threshold,
            'curriculum_stage': stage,
            'cached_difficulties': len(self.difficulty_cache)
        }


def test_curriculum_aql():
    """Test curriculum learning + AQL integration"""
    print("Testing Curriculum AQL")
    print("=" * 50)
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Mock uncertainty estimator
    class MockUncertainty:
        def estimate(self, data):
            return torch.rand(len(data))
    
    uncertainty_estimator = MockUncertainty()
    
    # Create curriculum scheduler
    scheduler = CurriculumScheduler(
        total_steps=1000,
        warmup_ratio=0.2,
        pacing_function='root'
    )
    
    print("\nTesting curriculum progression...")
    test_steps = [0, 100, 200, 500, 800, 999]
    for step in test_steps:
        threshold = scheduler.get_threshold(step)
        stage = scheduler.get_stage(step)
        print(f"  Step {step:4d}: threshold={threshold:.3f}, stage={stage}")
    
    # Create curriculum AQL
    curriculum_aql = CurriculumAQL(
        model=model,
        uncertainty_estimator=uncertainty_estimator,
        curriculum_scheduler=scheduler,
        device='cpu'
    )
    
    print("\nTesting sample selection...")
    # Generate synthetic data
    data = torch.randn(100, 10)
    targets = torch.randint(0, 5, (100,))
    
    # Test selection at different stages
    for step in [0, 200, 500, 900]:
        selected_data, selected_targets, selected_indices = curriculum_aql.select_samples(
            data=data,
            targets=targets,
            current_step=step,
            n_select=20
        )
        
        stats = curriculum_aql.get_statistics(step)
        print(f"\n  Step {step}:")
        print(f"    Stage: {stats['curriculum_stage']}")
        print(f"    Threshold: {stats['difficulty_threshold']:.3f}")
        print(f"    Selected: {len(selected_indices)} samples")
        print(f"    Cached difficulties: {stats['cached_difficulties']}")
    
    print("\n✅ Curriculum AQL test passed!")


if __name__ == "__main__":
    test_curriculum_aql()
