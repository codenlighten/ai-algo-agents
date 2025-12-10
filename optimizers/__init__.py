"""Package initialization for optimizers module"""
from optimizers.novel_optimizers import (
    SecondOrderMomentumOptimizer,
    LookAheadWrapper,
    AdaptiveGradientClipping,
    StochasticWeightAveraging
)

__all__ = [
    'SecondOrderMomentumOptimizer',
    'LookAheadWrapper',
    'AdaptiveGradientClipping',
    'StochasticWeightAveraging'
]
