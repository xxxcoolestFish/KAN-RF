"""Train a physics-conditioned PPO policy with FiLM modulation.

The policy  π(s, z) receives the oracle physics vector z at every step.
z modulates hidden features via FiLM (Feature-wise Linear Modulation):
    h' = h * γ(z) + β(z)

This forces the policy to attend to z — ignoring z degrades performance
across all physics configurations simultaneously, so the policy MUST
learn to use z.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpbn.physics_randomized_env import PhysicsRandomizedHopper
from scripts.prescreen_hopper_physics_shifts import ENVS, SHIFTS, make_shifted_env


# ── FiLM-conditioned policy network ───────────────────────────────────
class FiLMPolicy(nn.Module):
    """MLP policy with FiLM conditioning from physics vector z."""

    def __init__(
        self,
        state_dim: int = 11,
        z_dim: int = 3,
        action_dim: int = 3,
        hidden_dim: int = 256,
        depth: int = 3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        # ── state encoder ─────────────────────────────────────────────
        state_layers = []
        in_dim = state_dim
        for _ in range(depth):
            state_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        self.state_net = nn.Sequential(*state_layers)

        # ── FiLM generators: z → (γ, β) for each hidden layer ─────────
        self.film_generators = nn.ModuleList()
        for _ in range(depth):
            self.film_generators.append(nn.Sequential(
                nn.Linear(z_dim, hidden_dim * 2),
                nn.Tanh(),  # bounded modulation
            ))

        # ── actor head ────────────────────────────────────────────────
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # ── critic head ───────────────────────────────────────────────
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        """Extract features from [s, z] concatenated observation."""
        s = obs[:, :self.state_dim]
        z = obs[:, self.state_dim:self.state_dim + self.z_dim]
        h = s
        for layer_idx in range(self.depth):
            h = self.state_net[layer_idx * 2](h)           # Linear
            # ── FiLM modulation ────────────────────────────────────────
            film = self.film_generators[layer_idx](z)
            gamma, beta = film.chunk(2, dim=1)
            h = h * (1.0 + gamma) + beta
            h = self.state_net[layer_idx * 2 + 1](h)       # ReLU
        return h

    def forward_actor(self, obs, deterministic=False):
        features = self.forward(obs)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if deterministic:
            return mean
        return dist.sample()

    def forward_critic(self, obs):
        return self.critic(self.forward(obs))

    def evaluate_actions(self, obs, actions):
        features = self.forward(obs)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(features)
        return values, log_prob, entropy


# ── Environment: returns [s, z] as observation ────────────────────────
class PhysicsConditionedHopper(gym.Wrapper):
    """Wraps PhysicsRandomizedHopper to return [s, z] as observation."""

    def __init__(self, *, mass_range, friction_range, actuator_range, seed=0):
        env = PhysicsRandomizedHopper(
            mass_range=mass_range,
            friction_range=friction_range,
            actuator_range=actuator_range,
            seed=seed,
        )
        super().__init__(env)
        obs_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf,
            shape=(obs_dim + 3,),  # state + z
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return np.concatenate([obs, self.env.current_z]), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return np.concatenate([obs, self.env.current_z]), reward, term, trunc, info


# ── Evaluation on held-out shifts ─────────────────────────────────────
@torch.no_grad()
def evaluate_on_shift(policy, shift_name, shift_dict, z_vector, episodes, seed):
    """Evaluate policy conditioned on z_vector on a named physics shift."""
    env_factory = make_shifted_env(shift_dict, seed + 10000, "hopper")
    env = env_factory()
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 10000 + ep)
        total = 0.0
        while True:
            obs_with_z = np.concatenate([obs, z_vector])
            action, _ = policy.predict(obs_with_z, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            total += float(reward)
            if term or trunc:
                break
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


# ── Main training ─────────────────────────────────────────────────────
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # ── Environments ──────────────────────────────────────────────────
    def make_env(rank):
        def _factory():
            return PhysicsConditionedHopper(
                mass_range=(0.85, 1.5),
                friction_range=(0.65, 1.1),
                actuator_range=(0.6, 1.0),
                seed=args.seed + rank,
            )
        return _factory

    vec_env = DummyVecEnv([make_env(i) for i in range(args.parallel_envs)])
    # NOTE: z stays raw — only reward is normalised
    vec_env = VecNormalize(vec_env, training=True, norm_obs=False, norm_reward=True)

    # ── PPO training ──────────────────────────────────────────────────
    model = PPO(
        "MlpPolicy",  # SB3's MLP policy (receives concatenated [s, z])
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
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
    )

    # ── Periodic evaluation ───────────────────────────────────────────
    eval_shifts = [
        ("source", SHIFTS["source"], np.array([1.0, 1.0, 1.0], dtype=np.float32)),
        ("payload_125", SHIFTS["payload_125"], np.array([1.25, 1.0, 1.0], dtype=np.float32)),
        ("payload_150", SHIFTS["payload_150"], np.array([1.50, 1.0, 1.0], dtype=np.float32)),
        ("friction_070", SHIFTS["friction_070"], np.array([1.0, 0.70, 1.0], dtype=np.float32)),
        ("actuator_080", SHIFTS["actuator_080"], np.array([1.0, 1.0, 0.80], dtype=np.float32)),
        ("actuator_065", SHIFTS["actuator_065"], np.array([1.0, 1.0, 0.65], dtype=np.float32)),
        ("combo_medium", SHIFTS["combo_medium"], np.array([1.35, 0.70, 0.75], dtype=np.float32)),
    ]

    eval_history = []
    for step in range(0, args.total_transitions + 1, args.evaluate_every):
        if step > 0:
            model.learn(
                total_timesteps=args.evaluate_every,
                reset_num_timesteps=(step == args.evaluate_every),
            )
        record = {"step": step}
        for name, shift_dict, z in eval_shifts:
            mean_r, std_r = evaluate_on_shift(
                model, name, shift_dict, z,
                args.evaluation_episodes, args.seed,
            )
            record[name] = mean_r
            record[f"{name}_std"] = std_r
        eval_history.append(record)
        summary = {k: round(v, 1) for k, v in record.items() if isinstance(v, float)}
        print(summary, flush=True)

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        "experiment": "PhysicsConditionedPPO",
        "seed": args.seed,
        "config": vars(args),
        "eval_history": eval_history,
        "final_eval": eval_history[-1] if eval_history else {},
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8",
    )
    model.save(args.model_out)
    vec_env.save(args.norm_out)
    print(f"Saved → {args.model_out}, {args.json_out}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--total-transitions", type=int, default=2_000_000)
    parser.add_argument("--parallel-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--evaluate-every", type=int, default=200_000)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--model-out", default="results/physics_conditioned_ppo")
    parser.add_argument("--norm-out", default="results/physics_conditioned_ppo_vecnorm.pkl")
    parser.add_argument("--json-out", default="results/physics_conditioned_ppo.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
