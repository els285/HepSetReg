from hepsetreg.losses.regression import RegressionLoss
from hepsetreg.losses.distribution import (
    HistogramKLDivergenceLoss,
    KNNKLDivergenceLoss,
    MMDLoss,
    SlicedWassersteinLoss,
)
from hepsetreg.losses.flow_matching import ConditionalFlowMatchingLoss
from hepsetreg.losses.consistency import PhysicsConsistencyLoss
from hepsetreg.losses.composite import CompositeLoss, LossTermConfig

__all__ = [
    "RegressionLoss",
    "HistogramKLDivergenceLoss",
    "KNNKLDivergenceLoss",
    "MMDLoss",
    "SlicedWassersteinLoss",
    "ConditionalFlowMatchingLoss",
    "PhysicsConsistencyLoss",
    "CompositeLoss",
    "LossTermConfig",
]
