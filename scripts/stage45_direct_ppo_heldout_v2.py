"""Corrected runner for the direct PPO held-out comparison."""

from __future__ import annotations

from scripts import stage45_direct_ppo_heldout as base
from scripts.stage44_ppo_embedded_cognitive_v2 import evaluate_states


base.evaluate = evaluate_states
base.main()
