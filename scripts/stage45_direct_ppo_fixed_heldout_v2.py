"""Direct PPO held-out evaluator with a selectable factor preset."""

from scripts import stage45_direct_ppo_fixed_heldout as base

# The second held-out factor is deliberately farther from the source factor.
base.HELDOUT_FACTOR = (13.475, 0.06, 0.90, 1.10)
base.main()
