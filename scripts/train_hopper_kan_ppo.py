"""Train a KAN-based PPO policy on source Hopper.

Uses SB3's CustomActorCriticPolicy with a B-spline feature extractor.
Smoothness regularisation is applied via a post-update callback.
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cpbn.kan_sb3_policy import BsplineFeaturesExtractor
from cpbn.protokan_sb3_policy import ProtoKANFeaturesExtractor
from scripts.prescreen_hopper_physics_shifts import ENVS, SHIFTS, make_shifted_env


class SmoothnessRegularisationCallback(BaseCallback):
    def __init__(self, smooth_lambda=10.0, verbose=0):
        super().__init__(verbose)
        self.smooth_lambda = float(smooth_lambda)

    def _on_step(self) -> bool:
        if self.smooth_lambda <= 0.0:
            return True
        extractor = self.model.policy.features_extractor
        loss = self.smooth_lambda * extractor.smoothness_loss()
        if loss.item() > 0.0:
            self.model.policy.optimizer.zero_grad()
            loss.backward()
            self.model.policy.optimizer.step()
        return True


@torch.no_grad()
def evaluate(shift_name, shift, model, seed, episodes, env_name, norm_path):
    base = DummyVecEnv([make_shifted_env(shift, seed + 10000, env_name)])
    env = VecNormalize.load(norm_path, base)
    env.training = False
    env.norm_reward = False
    returns = []
    for ep in range(episodes):
        obs = env.reset()
        total = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total += float(reward[0])
            if done[0]:
                break
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    # ── source environment ────────────────────────────────────────────
    def _make():
        return make_shifted_env(SHIFTS["source"], args.seed + i, args.env)()
    vec_env = DummyVecEnv([
        lambda i=i: make_shifted_env(SHIFTS["source"], args.seed + i, args.env)()
        for i in range(8)
    ])
    vec_env = VecNormalize(vec_env, training=True, norm_obs=True, norm_reward=True)

    # ── PPO with KAN / ProtoKAN feature extractor ────────────────────
    if args.feat_type == "protokan":
        feat_cls = ProtoKANFeaturesExtractor
        feat_kwargs = dict(n_prototypes=16, out_dim=256, grid_range=1.0)
    else:
        feat_cls = BsplineFeaturesExtractor
        feat_kwargs = dict(
            grid_size=args.kan_grid_size,
            spline_order=args.kan_spline_order,
            pair_modes=args.kan_pair_modes,
            state_scale=5.0,
        )

    # Parse net_arch: "" → [], "256,256" → dict(pi=[256,256], vf=[256,256])
    if args.net_arch.strip():
        arch = [int(x) for x in args.net_arch.split(",") if x.strip()]
        net_arch_cfg = dict(pi=arch, vf=arch)
    else:
        net_arch_cfg = []

    policy_kwargs = dict(
        features_extractor_class=feat_cls,
        features_extractor_kwargs=feat_kwargs,
        net_arch=net_arch_cfg,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.rollout_steps,
        batch_size=args.minibatch_size,
        n_epochs=args.update_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_ratio,
        ent_coef=args.entropy_coefficient,
        seed=args.seed,
        device=device,
        verbose=0,
        policy_kwargs=policy_kwargs,
    )

    feature_dim = model.policy.features_extractor.features_dim
    print(f"KAN feature_dim={feature_dim}", flush=True)

    # Apply smoothness manually after training (simple first-pass approach)
    # ── Real-time progress callback ────────────────────────────────────
    class ProgressCallback(BaseCallback):
        def __init__(self, total, verbose=0):
            super().__init__(verbose)
            self.total = total
            self.last_print = 0
        def _on_step(self):
            steps = self.num_timesteps
            if steps - self.last_print >= 16384:
                self.last_print = steps
                print(f"  {steps}/{self.total} steps ({100*steps/self.total:.0f}%)",
                      flush=True)
            return True

    print(f"Training {args.total_transitions} steps on source Hopper ...",
          flush=True)
    model.learn(
        total_timesteps=args.total_transitions,
        callback=ProgressCallback(args.total_transitions),
    )

    # ── Explicit smoothness regularisation step ────────────────────────
    if args.smooth_lambda > 0.0 and hasattr(
        model.policy.features_extractor, "smoothness_loss",
    ):
        extractor = model.policy.features_extractor
        for _ in range(100):
            loss = args.smooth_lambda * extractor.smoothness_loss()
            if loss.item() <= 0.0:
                break
            model.policy.optimizer.zero_grad()
            loss.backward()
            model.policy.optimizer.step()

    # ── evaluate ──────────────────────────────────────────────────────
    # Save norm first so evaluate() can load it
    vec_env.save(args.norm_out)
    model.save(args.model_out)

    source_r, source_std = evaluate(
        "source", SHIFTS["source"], model, args.seed,
        args.evaluation_episodes, args.env, args.norm_out,
    )
    print(f"Source: {source_r:.1f} ± {source_std:.1f}", flush=True)

    output = {
        "experiment": "HopperKANPolicyPPO",
        "seed": args.seed,
        "source_return": source_r,
        "source_std": source_std,
        "kan_feature_dim": feature_dim,
        "config": vars(args),
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved → {args.json_out}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--env", choices=tuple(ENVS), default="hopper")
    parser.add_argument("--total-transitions", type=int, default=500_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--feat-type", choices=("bspline", "protokan"), default="bspline")
    parser.add_argument("--net-arch", type=str, default="",
                        help="'256,256' for MLP, empty for direct linear")
    parser.add_argument("--kan-grid-size", type=int, default=6)
    parser.add_argument("--kan-spline-order", type=int, default=3)
    parser.add_argument("--kan-pair-modes", type=int, default=3)
    parser.add_argument("--smooth-lambda", type=float, default=10.0)
    parser.add_argument("--model-out", default="results/hopper_kan_ppo_source")
    parser.add_argument("--norm-out",
                        default="results/hopper_kan_ppo_source_vecnorm.pkl")
    parser.add_argument("--json-out",
                        default="results/hopper_kan_ppo_source.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
