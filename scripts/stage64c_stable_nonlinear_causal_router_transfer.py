"""Stable nonlinear residual-router entry point for Stage 64."""

from kanrf.protokan_causal_router_stable import (
    StableProtoKANNonlinearEdgeRouter,
)
from scripts import stage64b_nonlinear_causal_router_transfer as corrected


corrected.experiment.ProtoKANNonlinearEdgeRouter = (
    StableProtoKANNonlinearEdgeRouter
)


if __name__ == "__main__":
    corrected.experiment.main()
