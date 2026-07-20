"""Stage 56 entry point with source-factor default for the oracle model."""

from __future__ import annotations

from scripts import stage56_oracle_ppo as oracle
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR


class OracleCognitiveDefault(oracle.OracleCognitive):
    def __init__(self, factor=None):
        super().__init__(PRETRAIN_FACTOR[0] if factor is None else factor)


oracle.base.ContextCognitiveKAN = OracleCognitiveDefault
oracle.main()
