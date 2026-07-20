"""Corrected entry point for Stage 62 temporal causal route validation."""

from kanrf.temporal_causal_routing_fixed import temporal_reachability_route
from scripts import stage62_temporal_causal_route_validation as validation


validation.temporal_reachability_route = temporal_reachability_route


if __name__ == "__main__":
    validation.main()
