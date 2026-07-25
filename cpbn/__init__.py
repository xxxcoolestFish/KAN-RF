"""Control-equivalent ProtoKAN cognition for continual dynamics adaptation."""

from cpbn.generic_affine_kan import (
    AffineKANContext,
    AffineKANPosterior,
    AffinePosteriorPullback,
    CompactInteractionKANDictionary,
    LearnedMLPDictionary,
    RecursiveAffineKANEstimator,
    fit_affine_kan_context,
)
from cpbn.hopper_source_twin import (
    HopperSourceAffineTwin,
    JointStateSupportCalibrator,
    SparseComposableKANTwin,
)

__all__ = [
    "AffineKANContext",
    "AffineKANPosterior",
    "AffinePosteriorPullback",
    "CompactInteractionKANDictionary",
    "HopperSourceAffineTwin",
    "JointStateSupportCalibrator",
    "LearnedMLPDictionary",
    "RecursiveAffineKANEstimator",
    "SparseComposableKANTwin",
    "fit_affine_kan_context",
]
