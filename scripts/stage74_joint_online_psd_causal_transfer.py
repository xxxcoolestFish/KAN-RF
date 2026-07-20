"""Stage 72 joint online protocol using the Stage 73 PSD causal actor."""

from kanrf.protokan_causal_router_stable import (
    StableProtoKANNonlinearEdgeRouter,
)
from scripts import stage72_joint_online_equivariant_transfer as experiment
from scripts.stage23_multistep_terminal_value import SimpleCognitiveKAN
from scripts.stage73_psd_causal_preconditioner_actor import (
    PSDCausalPreconditionerActor,
)


def build_actor(config):
    cognitive = SimpleCognitiveKAN(hidden_dim=32, n_prototypes=8)
    router = StableProtoKANNonlinearEdgeRouter(delta=config["edge_delta"])
    return PSDCausalPreconditionerActor(
        cognitive, router, config["route_horizon"], config["hidden_dim"],
        config["temperature"], config["route_scale"], config["step_size"],
        config["metric_rank"], config["min_diagonal"],
        config["max_diagonal"],
    )


experiment.build_actor = build_actor


if __name__ == "__main__":
    experiment.main()
