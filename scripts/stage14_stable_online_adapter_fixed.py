"""Corrected launcher for Stage 14 trust-region projection.

The trust region is measured relative to the runtime residual's initial
parameters, not relative to the absolute norm of its randomly initialized
hidden layer.
"""

from __future__ import annotations

import torch

import scripts.stage14_stable_online_adapter as stage14


_REFERENCES = {}


def project_runtime_relative(decision, radius: float):
    parameters = list(decision.runtime_residual.parameters())
    key = id(decision)
    if key not in _REFERENCES:
        _REFERENCES[key] = [parameter.detach().clone() for parameter in parameters]
    references = _REFERENCES[key]
    with torch.no_grad():
        difference = torch.cat([
            (parameter - reference).flatten()
            for parameter, reference in zip(parameters, references)
        ])
        norm = torch.linalg.vector_norm(difference)
        if norm > radius:
            scale = radius / (norm + 1e-8)
            for parameter, reference in zip(parameters, references):
                parameter.copy_(reference + scale * (parameter - reference))
        return min(norm.item(), radius)


stage14.project_runtime = project_runtime_relative


if __name__ == "__main__":
    stage14.main()
