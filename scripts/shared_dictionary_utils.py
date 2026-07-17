"""Shared-dictionary experiment helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from physics_transfer.losses import cognitive_prediction_loss, token_consistency_loss
from physics_transfer.multifactor_data import sample_multifactor_batch
from scripts.stage2_lowrank_gap_calibration import FACTORS, _pair


def dynamics_terms(model, batch, index):
    output = model(batch["history"], batch["state"], batch["action"])
    standard = cognitive_prediction_loss(output["next_state"], batch["next_state"])
    factor_a, factor_b = FACTORS[index % len(FACTORS)], FACTORS[(index + 1) % len(FACTORS)]
    pair = _pair(factor_a, factor_b, batch_size=16)
    out_a = model(pair["history_a"], pair["state"], pair["action"])
    out_b = model(pair["history_b"], pair["state"], pair["action"])
    absolute = cognitive_prediction_loss(out_a["next_state"], pair["target_a"]) + cognitive_prediction_loss(out_b["next_state"], pair["target_b"])
    gap = F.mse_loss(
        torch.linalg.vector_norm(out_a["next_state"] - out_b["next_state"], dim=-1),
        torch.linalg.vector_norm(pair["target_a"] - pair["target_b"], dim=-1),
    )
    same_a = sample_multifactor_batch(16, 8, (factor_a,))
    same_b = sample_multifactor_batch(16, 8, (factor_a,))
    same_out_a = model(same_a["history"], same_a["state"], same_a["action"])
    same_out_b = model(same_b["history"], same_b["state"], same_b["action"])
    token = token_consistency_loss(same_out_a["physics_pooled"], same_out_b["physics_pooled"], same_out_a["physics_gates"], same_out_b["physics_gates"])
    coefficient = F.mse_loss(same_out_a["physics_coefficients"], same_out_b["physics_coefficients"])
    return pair, output, out_a, out_b, standard, absolute, gap, token, coefficient
