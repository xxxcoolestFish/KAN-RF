"""Fast closed-loop gate for cognition-to-policy low-rank modulation.

The environments are anonymous during training.  A shared modulated ProtoKAN
learns their transition laws with one trainable latent per source environment.
For a held-out environment, the latent is fitted using reward-free transitions
only.  Three policies are then compared without target reward updates:

* pooled: state-only MLP;
* concat: MLP receiving the inferred cognitive latent;
* lowrank: the same latent directly changes the state-to-action matrix.

This is a deliberately small continuous-control mechanism test, not a final
benchmark.  The coupled second-order system is non-trivial but cheap enough for
multi-seed diagnostics in minutes.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kanrf import ActionModulatedProtoKAN, CognitionModulatedActor


@dataclass(frozen=True)
class Physics:
    mass_1: float
    mass_2: float
    actuator_gain: float
    actuator_angle: float = 0.0
    coupling: float = 0.18
    damping: float = 0.16


SOURCE_PHYSICS = (
    Physics(0.75, 1.20, 1.00, -0.45),
    Physics(0.75, 1.20, 1.00, +0.45),
    Physics(1.25, 0.80, 1.00, -0.45),
    Physics(1.25, 0.80, 1.00, +0.45),
)
TARGET_PHYSICS = {
    "interpolation": Physics(1.00, 1.00, 1.00, 0.00),
    "mass_extrapolation": Physics(1.55, 0.65, 1.00, 0.00),
    "angle_extrapolation": Physics(1.00, 1.00, 1.00, +0.75),
    "unseen_combination": Physics(1.50, 0.68, 1.00, -0.75),
}


def system_matrices(physics: Physics, dt: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    """Return a coupled two-axis discrete second-order system."""
    mass = np.array(
        [[physics.mass_1, physics.coupling],
         [physics.coupling, physics.mass_2]],
        dtype=np.float64,
    )
    stiffness = np.array([[0.32, -0.12], [-0.12, 0.28]])
    damping = np.array(
        [[physics.damping, 0.025], [0.025, 1.15 * physics.damping]]
    )
    inv_mass = np.linalg.inv(mass)
    a = np.block(
        [
            [np.eye(2) - dt * dt * inv_mass @ stiffness,
             dt * np.eye(2) - dt * dt * inv_mass @ damping],
            [-dt * inv_mass @ stiffness,
             np.eye(2) - dt * inv_mass @ damping],
        ]
    )
    cosine = np.cos(physics.actuator_angle)
    sine = np.sin(physics.actuator_angle)
    actuator_direction = np.array(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    b_velocity = (
        dt * inv_mass @ actuator_direction * physics.actuator_gain
    )
    b = np.vstack((dt * b_velocity, b_velocity))
    return a.astype(np.float32), b.astype(np.float32)


def lqr_gain(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    q = np.diag([8.0, 8.0, 0.7, 0.7]).astype(np.float64)
    r = np.diag([0.12, 0.12]).astype(np.float64)
    p = q.copy()
    for _ in range(500):
        gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        updated = q + a.T @ p @ (a - b @ gain)
        if np.max(np.abs(updated - p)) < 1e-11:
            p = updated
            break
        p = updated
    return np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a).astype(np.float32)


def transition(
    states: np.ndarray,
    actions: np.ndarray,
    physics: Physics,
) -> np.ndarray:
    a, b = system_matrices(physics)
    return states @ a.T + actions @ b.T


def paired_transition_bank(
    count: int,
    rng: np.random.Generator,
    physics_set: tuple[Physics, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = rng.uniform(-1.5, 1.5, size=(count, 4)).astype(np.float32)
    actions = rng.uniform(-2.0, 2.0, size=(count, 2)).astype(np.float32)
    inputs = np.concatenate((states, actions), axis=1)
    targets = np.stack(
        [transition(states, actions, physics) - states for physics in physics_set]
    )
    return inputs, targets, states


class PooledPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, states: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        del latent
        return 2.0 * torch.tanh(self.net(states))


class ConcatPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, states: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[0] == 1 and states.shape[0] != 1:
            latent = latent.expand(states.shape[0], -1)
        return 2.0 * torch.tanh(self.net(torch.cat((states, latent), dim=-1)))


def reward_free_support_rollout(
    count: int,
    rng: np.random.Generator,
    physics: Physics,
    episode_length: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect a real transition prefix without reward or state cloning."""
    inputs = []
    deltas = []
    state = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
    previous_action = np.zeros(2, dtype=np.float32)
    for step in range(count):
        if step > 0 and step % episode_length == 0:
            state = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
            previous_action.fill(0.0)
        proposal = rng.uniform(-2.0, 2.0, size=2).astype(np.float32)
        action = np.clip(
            0.65 * previous_action + 0.35 * proposal, -2.0, 2.0
        ).astype(np.float32)
        next_state = transition(
            state[None, :], action[None, :], physics
        )[0]
        inputs.append(np.concatenate((state, action)))
        deltas.append(next_state - state)
        state = next_state
        previous_action = action
    return np.stack(inputs), np.stack(deltas)


def cognitive_loss(
    model: ActionModulatedProtoKAN,
    embedding: nn.Embedding,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    root_indices: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    env_count = targets.shape[0]
    batch_inputs = inputs[root_indices]
    repeated_inputs = batch_inputs.repeat(env_count, 1)
    env_ids = torch.arange(env_count, device=inputs.device).repeat_interleave(
        len(root_indices)
    )
    prediction = model(repeated_inputs, embedding(env_ids)).reshape(
        env_count, len(root_indices), -1
    )
    truth = targets[:, root_indices]
    pointwise = F.mse_loss(prediction, truth)
    counterfactual = F.mse_loss(
        prediction - prediction.mean(dim=0, keepdim=True),
        truth - truth.mean(dim=0, keepdim=True),
    )
    z = embedding.weight
    covariance = z.T @ z / env_count
    identity = torch.eye(z.shape[1], device=z.device)
    coordinate = z.mean(dim=0).square().mean() + (
        covariance - identity
    ).square().mean()
    loss = pointwise + 5.0 * counterfactual + 0.05 * coordinate
    return loss, {
        "pointwise": float(pointwise.detach()),
        "counterfactual": float(counterfactual.detach()),
        "coordinate": float(coordinate.detach()),
    }


def train_cognition(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    latent_dim: int,
    steps: int,
    batch_size: int,
    seed: int,
    log_every: int,
) -> tuple[ActionModulatedProtoKAN, nn.Embedding]:
    torch.manual_seed(seed + 11)
    model = ActionModulatedProtoKAN(
        input_dim=6,
        hidden_dim=24,
        output_dim=4,
        action_start=4,
        action_dim=2,
        latent_dim=latent_dim,
        n_prototypes=7,
        grid_range=4.0,
    ).to(inputs.device)
    embedding = nn.Embedding(len(SOURCE_PHYSICS), latent_dim).to(inputs.device)
    nn.init.normal_(embedding.weight, std=0.7)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(embedding.parameters()), lr=2e-3
    )
    generator = torch.Generator(device=inputs.device).manual_seed(seed + 12)
    for step in range(1, steps + 1):
        roots = torch.randint(
            len(inputs),
            (min(batch_size, len(inputs)),),
            generator=generator,
            device=inputs.device,
        )
        loss, diagnostics = cognitive_loss(
            model, embedding, inputs, targets, roots
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            print(
                f"COGNITION step={step:04d}/{steps} loss={float(loss):.6f} "
                f"pred={diagnostics['pointwise']:.6f} "
                f"effect={diagnostics['counterfactual']:.6f} "
                f"coord={diagnostics['coordinate']:.6f}",
                flush=True,
            )
    return model, embedding


def train_policy(
    policy: nn.Module,
    states: torch.Tensor,
    teacher_actions: torch.Tensor,
    source_latents: torch.Tensor,
    steps: int,
    batch_size: int,
    seed: int,
    log_every: int,
) -> list[float]:
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    generator = torch.Generator(device=states.device).manual_seed(seed + 31)
    env_count, root_count, _ = states.shape
    trace = []
    for step in range(1, steps + 1):
        roots = torch.randint(
            root_count,
            (min(batch_size, root_count),),
            generator=generator,
            device=states.device,
        )
        env_ids = torch.randint(
            env_count,
            (len(roots),),
            generator=generator,
            device=states.device,
        )
        batch_states = states[env_ids, roots]
        batch_targets = teacher_actions[env_ids, roots]
        batch_latents = source_latents[env_ids]
        prediction = policy(batch_states, batch_latents)
        loss = F.mse_loss(prediction, batch_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            value = float(loss.detach())
            trace.append(value)
            print(
                f"POLICY name={policy.__class__.__name__:24s} "
                f"step={step:04d}/{steps} bc={value:.6f}",
                flush=True,
            )
    return trace


def infer_latent(
    model: ActionModulatedProtoKAN,
    inputs: torch.Tensor,
    target_delta: torch.Tensor,
    latent_dim: int,
    steps: int,
    seed: int,
    log_every: int,
) -> tuple[torch.Tensor, list[float]]:
    torch.manual_seed(seed + 41)
    latent = nn.Parameter(torch.zeros(1, latent_dim, device=inputs.device))
    optimizer = torch.optim.Adam([latent], lr=5e-2)
    trace = []
    for step in range(1, steps + 1):
        loss = F.mse_loss(model(inputs, latent), target_delta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            value = float(loss.detach())
            trace.append(value)
            print(
                f"IDENTIFY step={step:03d}/{steps} loss={value:.6f} "
                f"latent={latent.detach().cpu().numpy().round(3).tolist()}",
                flush=True,
            )
    return latent.detach(), trace


@torch.no_grad()
def evaluate_policy(
    policy: nn.Module,
    latent: torch.Tensor,
    physics: Physics,
    initial_states: np.ndarray,
    horizon: int,
    device: torch.device,
) -> dict[str, float]:
    a, b = system_matrices(physics)
    states = torch.as_tensor(initial_states, device=device)
    a_tensor = torch.as_tensor(a, device=device)
    b_tensor = torch.as_tensor(b, device=device)
    total_cost = torch.zeros(len(states), device=device)
    max_norm = states.norm(dim=-1)
    for _ in range(horizon):
        action = policy(states, latent)
        total_cost += (
            8.0 * states[:, :2].square().sum(-1)
            + 0.7 * states[:, 2:].square().sum(-1)
            + 0.12 * action.square().sum(-1)
        )
        states = states @ a_tensor.T + action @ b_tensor.T
        max_norm = torch.maximum(max_norm, states.norm(dim=-1))
    final_norm = states.norm(dim=-1)
    stable = (final_norm < 0.20) & (max_norm < 8.0)
    return {
        "mean_return": -float(total_cost.mean()),
        "median_cost": float(total_cost.median()),
        "final_norm": float(final_norm.mean()),
        "stable_rate": float(stable.float().mean()),
    }


def run_seed(args: argparse.Namespace, seed: int, device: torch.device) -> dict:
    print(f"SEED_BEGIN seed={seed} device={device}", flush=True)
    rng = np.random.default_rng(seed)
    train_x_np, train_y_np, _ = paired_transition_bank(
        args.transition_bank, rng, SOURCE_PHYSICS
    )
    train_x = torch.as_tensor(train_x_np, device=device)
    train_y = torch.as_tensor(train_y_np, device=device)
    cognition, embedding = train_cognition(
        train_x,
        train_y,
        args.latent_dim,
        args.cognition_steps,
        args.batch_size,
        seed,
        args.log_every,
    )
    source_latents = embedding.weight.detach()
    print(
        f"SOURCE_LATENTS {source_latents.cpu().numpy().round(3).tolist()}",
        flush=True,
    )

    policy_states_np = rng.uniform(
        -2.0, 2.0, size=(len(SOURCE_PHYSICS), args.policy_bank, 4)
    ).astype(np.float32)
    teacher_actions_np = []
    for env_index, physics in enumerate(SOURCE_PHYSICS):
        a, b = system_matrices(physics)
        gain = lqr_gain(a, b)
        teacher_actions_np.append(
            np.clip(-policy_states_np[env_index] @ gain.T, -2.0, 2.0)
        )
    policy_states = torch.as_tensor(policy_states_np, device=device)
    teacher_actions = torch.as_tensor(np.stack(teacher_actions_np), device=device)
    torch.manual_seed(seed + 50)
    policies = {
        "pooled": PooledPolicy(4, 2, args.policy_hidden).to(device),
        "concat": ConcatPolicy(
            4, 2, args.latent_dim, args.policy_hidden
        ).to(device),
        "cognitive_lowrank": CognitionModulatedActor(
            state_dim=4,
            action_dim=2,
            cognition_dim=args.latent_dim,
            hidden_dims=(args.policy_hidden,),
            rank=args.rank,
            action_limit=2.0,
        ).to(device),
    }
    with torch.no_grad():
        policies["cognitive_lowrank"].layers[0].base.load_state_dict(
            policies["pooled"].net[0].state_dict()
        )
        policies["cognitive_lowrank"].layers[1].base.load_state_dict(
            policies["pooled"].net[2].state_dict()
        )
    print(
        "POLICY_PARAMETERS "
        + " ".join(
            f"{name}={sum(p.numel() for p in policy.parameters())}"
            for name, policy in policies.items()
        ),
        flush=True,
    )
    policy_traces = {}
    for offset, (name, policy) in enumerate(policies.items()):
        policy_traces[name] = train_policy(
            policy,
            policy_states,
            teacher_actions,
            source_latents,
            args.policy_steps,
            args.batch_size,
            seed + 100 * offset,
            args.log_every,
        )

    initial_states = rng.uniform(
        -1.25, 1.25, size=(args.eval_episodes, 4)
    ).astype(np.float32)
    cognition.requires_grad_(False)
    targets = {}
    for target_index, (name, physics) in enumerate(TARGET_PHYSICS.items()):
        support_inputs_np, support_delta_np = reward_free_support_rollout(
            args.support_transitions, rng, physics
        )
        support_inputs = torch.as_tensor(
            support_inputs_np, device=device
        )
        support_delta = torch.as_tensor(
            support_delta_np, device=device
        )
        print(f"TARGET_BEGIN name={name}", flush=True)
        latent, identify_trace = infer_latent(
            cognition,
            support_inputs,
            support_delta,
            args.latent_dim,
            args.identify_steps,
            seed + target_index,
            args.log_every,
        )
        methods = {}
        for method, policy in policies.items():
            methods[method] = evaluate_policy(
                policy,
                latent,
                physics,
                initial_states,
                args.horizon,
                device,
            )
            values = methods[method]
            print(
                f"CONTROL target={name:20s} method={method:17s} "
                f"return={values['mean_return']:.3f} "
                f"final={values['final_norm']:.4f} "
                f"stable={values['stable_rate']:.3f}",
                flush=True,
            )
        zero_latent = torch.zeros_like(latent)
        methods["cognitive_lowrank_zero"] = evaluate_policy(
            policies["cognitive_lowrank"],
            zero_latent,
            physics,
            initial_states,
            args.horizon,
            device,
        )
        zero_values = methods["cognitive_lowrank_zero"]
        print(
            f"CONTROL target={name:20s} "
            "method=cognitive_lowrank_zero "
            f"return={zero_values['mean_return']:.3f} "
            f"final={zero_values['final_norm']:.4f} "
            f"stable={zero_values['stable_rate']:.3f}",
            flush=True,
        )
        target_a, target_b = system_matrices(physics)
        target_gain = torch.as_tensor(
            lqr_gain(target_a, target_b), device=device
        )

        class OraclePolicy(nn.Module):
            def forward(self, states, latent):  # noqa: ANN001
                del latent
                return torch.clamp(-states @ target_gain.T, -2.0, 2.0)

        methods["target_lqr_oracle"] = evaluate_policy(
            OraclePolicy(),
            latent,
            physics,
            initial_states,
            args.horizon,
            device,
        )
        targets[name] = {
            "physics": asdict(physics),
            "inferred_latent": latent.cpu().numpy().tolist(),
            "identify_trace": identify_trace,
            "methods": methods,
        }
    return {
        "seed": seed,
        "source_latents": source_latents.cpu().numpy().tolist(),
        "policy_traces": policy_traces,
        "targets": targets,
    }


def aggregate(seed_results: list[dict]) -> dict:
    output = {}
    for target in TARGET_PHYSICS:
        output[target] = {}
        for method in (
            "pooled",
            "concat",
            "cognitive_lowrank",
            "cognitive_lowrank_zero",
            "target_lqr_oracle",
        ):
            output[target][method] = {}
            for metric in ("mean_return", "final_norm", "stable_rate"):
                values = np.asarray(
                    [
                        result["targets"][target]["methods"][method][metric]
                        for result in seed_results
                    ]
                )
                output[target][method][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[321, 322, 323])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--transition-bank", type=int, default=1600)
    parser.add_argument("--policy-bank", type=int, default=2400)
    parser.add_argument("--support-transitions", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--policy-hidden", type=int, default=32)
    parser.add_argument("--cognition-steps", type=int, default=700)
    parser.add_argument("--policy-steps", type=int, default=700)
    parser.add_argument("--identify-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--output", default="results/_cognitive_lowrank_lqr_gate.json"
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    all_results = []
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        all_results.append(run_seed(args, seed, device))
    summary = aggregate(all_results)
    for target, methods in summary.items():
        for method, metrics in methods.items():
            print(
                f"SUMMARY target={target:20s} method={method:17s} "
                f"return={metrics['mean_return']['mean']:.3f}"
                f"+-{metrics['mean_return']['std']:.3f} "
                f"final={metrics['final_norm']['mean']:.4f} "
                f"stable={metrics['stable_rate']['mean']:.3f}",
                flush=True,
            )
    payload = {
        "experiment": "anonymous_cognitive_lowrank_lqr_gate",
        "claim_boundary": (
            "No target reward updates and no physical parameter input. "
            "The target latent uses reward-free transitions only."
        ),
        "configuration": vars(args),
        "source_physics_for_audit_only": [asdict(p) for p in SOURCE_PHYSICS],
        "target_physics_for_audit_only": {
            name: asdict(p) for name, p in TARGET_PHYSICS.items()
        },
        "seed_results": all_results,
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"WROTE {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
