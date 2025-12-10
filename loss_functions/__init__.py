"""Package initialization for loss_functions module"""
from loss_functions.novel_losses import (
    ConfidencePenalizedCrossEntropy,
    FocalLoss,
    ContrastivePredictiveLoss,
    AdaptiveWingLoss,
    NoiseContrastiveEstimation,
    CurriculumLoss
)

__all__ = [
    'ConfidencePenalizedCrossEntropy',
    'FocalLoss',
    'ContrastivePredictiveLoss',
    'AdaptiveWingLoss',
    'NoiseContrastiveEstimation',
    'CurriculumLoss'
]
