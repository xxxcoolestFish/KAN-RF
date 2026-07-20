"""Entry point for the stable parameter-transport validation."""

from scripts import stage27_parameter_transport_v3 as experiment
from physics_transfer.parameter_transport_v3 import ParameterTransport, TransportedMLPPolicy


experiment.ParameterTransport = ParameterTransport
experiment.TransportedMLPPolicy = TransportedMLPPolicy


if __name__ == "__main__":
    experiment.main()
