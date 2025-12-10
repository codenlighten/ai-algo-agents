"""Package initialization for models module"""
from models.novel_architectures import (
    DynamicDepthNetwork,
    MixtureOfExpertsLayer,
    AdaptiveComputationTime,
    HyperNetwork,
    MultiScaleAttention
)

__all__ = [
    'DynamicDepthNetwork',
    'MixtureOfExpertsLayer',
    'AdaptiveComputationTime',
    'HyperNetwork',
    'MultiScaleAttention'
]
