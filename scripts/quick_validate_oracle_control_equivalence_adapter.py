"""Quick Oracle gate for a critic-weighted control-equivalence adapter.

The source SAC actor and critic are frozen.  Exact source/target MuJoCo
dynamics are used only to isolate the proposed interface from learned-model
error.  At every target-environment state, the adapter asks for the smallest
bounded action correction whose local target effect matches the effect that
the source actor would have produced under source physics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import mujoco
import numpy as np
import torch
from stable_baselines3 import SAC

from kanrf.control_equivalence_adapter import (
    normalized_task_metric,
    solve_local_control_equivalence,
    weighted_norm,
)
from kanrf.pusher_oracle import object_goal_distance


@dataclass(frozen=True)
class StepDiagnostics:
    raw_error_before: float
    raw_error_after: float
    task_error_before: float
    task_error_after: float
    action_correction: float
    predicted_residual: float
    condition_number: float
    saturated_fraction: float
    adapter_ms: float


def make_env(max_steps: int = 100) -> gym.Env:
    return gym.make("Pusher-v5", max_episode_steps=max_steps)


def apply_physics(env: gym.Env, variation: str) -> None:
    """Apply a reproducible target-physics intervention in place."""
    base = env.unwrapped
    model = base.model
    if variation == "source":
        return
    qpos = np.asarray(base.data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(base.data.qvel, dtype=np.float64).copy()
    if variation == "weak_actuator":
        model.actuator_gear[:, 0] *= 0.65
    elif variation == "heavy_arm":
        body_ids = np.arange(4, 11)
        model.body_mass[body_ids] *= 1.75
        model.body_inertia[body_ids] *= 1.75
    elif variation == "heavy_object_friction":
        object_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        model.body_mass[object_body] *= 4.0
        model.body_inertia[object_body] *= 4.0
        object_geoms = np.flatnonzero(model.geom_bodyid == object_body)
        model.geom_friction[object_geoms, 0] *= 1.75
    elif variation == "combined":
        model.actuator_gear[:, 0] *= 0.72
        body_ids = np.arange(4, 11)
        model.body_mass[body_ids] *= 1.5
        model.body_inertia[body_ids] *= 1.5
        object_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        model.body_mass[object_body] *= 3.0
        model.body_inertia[object_body] *= 3.0
        object_geoms = np.flatnonzero(model.geom_bodyid == object_body)
        model.geom_friction[object_geoms, 0] *= 1.5
    else:
        raise ValueError(f"unknown physics variation: {variation}")
    mujoco.mj_setConst(model, base.data)
    restore(base, qpos, qvel)


def restore(base, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    base.set_state(qpos.copy(), qvel.copy())
    if base.data.ctrl.size:
        base.data.ctrl[:] = 0.0
    mujoco.mj_forward(base.model, base.data)
    return np.asarray(base._get_obs(), dtype=np.float64).copy()


def rollout_effect(
    env: gym.Env,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    horizon: int,
) -> np.ndarray:
    base = env.unwrapped
    initial = restore(base, qpos, qvel)
    observation = initial
    for _ in range(horizon):
        observation, _, terminated, truncated, _ = base.step(
            np.asarray(action, dtype=np.float32)
        )
        if terminated or truncated:
            break
    effect = np.asarray(observation, dtype=np.float64) - initial
    restore(base, qpos, qvel)
    return effect


def action_jacobian(
    env: gym.Env,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    horizon: int,
    epsilon: float,
) -> np.ndarray:
    columns = []
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    for index in range(action.size):
        plus = np.asarray(action, dtype=np.float64).copy()
        minus = plus.copy()
        plus[index] = min(plus[index] + epsilon, high[index])
        minus[index] = max(minus[index] - epsilon, low[index])
        denominator = plus[index] - minus[index]
        if denominator < 1e-10:
            columns.append(
                np.zeros(env.observation_space.shape[0], dtype=np.float64)
            )
            continue
        plus_effect = rollout_effect(
            env, qpos, qvel, plus, horizon
        )
        minus_effect = rollout_effect(
            env, qpos, qvel, minus, horizon
        )
        columns.append((plus_effect - minus_effect) / denominator)
    return np.stack(columns, axis=1)


def critic_gradients(
    sac: SAC,
    observation: np.ndarray,
    samples: int,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    device = sac.device
    state = torch.as_tensor(
        observation, dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        center = sac.actor(state, deterministic=True)
    actions = center.repeat(samples, 1)
    if samples > 1:
        perturbation = torch.as_tensor(
            rng.normal(0.0, noise, size=tuple(actions[1:].shape)),
            dtype=torch.float32,
            device=device,
        )
        actions[1:] = (actions[1:] + perturbation).clamp(-1.0, 1.0)
    states = state.repeat(samples, 1).detach().requires_grad_(True)
    q_values = sac.critic(states, actions)
    conservative_q = torch.minimum(q_values[0], q_values[1])
    gradients = torch.autograd.grad(conservative_q.sum(), states)[0]
    return gradients.detach().cpu().numpy().astype(np.float64)


def adapted_action(
    sac: SAC,
    observation: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    source_oracle: gym.Env,
    target_oracle: gym.Env,
    metric_kind: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, StepDiagnostics]:
    started = perf_counter()
    reference, _ = sac.predict(observation, deterministic=True)
    reference = np.asarray(reference, dtype=np.float64)
    desired = rollout_effect(
        source_oracle, qpos, qvel, reference, args.effect_horizon
    )
    target_before = rollout_effect(
        target_oracle, qpos, qvel, reference, args.effect_horizon
    )
    jacobian = action_jacobian(
        target_oracle,
        qpos,
        qvel,
        reference,
        args.effect_horizon,
        args.fd_epsilon,
    )
    if metric_kind == "identity":
        metric = np.eye(observation.size, dtype=np.float64)
    elif metric_kind == "critic":
        gradients = critic_gradients(
            sac,
            observation,
            args.critic_samples,
            args.critic_action_noise,
            rng,
        )
        metric = normalized_task_metric(
            gradients,
            observation.size,
            isotropic_floor=args.metric_floor,
        )
    else:
        raise ValueError(metric_kind)
    solution = solve_local_control_equivalence(
        reference,
        desired,
        target_before,
        jacobian,
        metric,
        source_oracle.action_space.low,
        source_oracle.action_space.high,
        regularization=args.regularization,
        trust_radius=args.trust_radius,
    )
    target_after = rollout_effect(
        target_oracle,
        qpos,
        qvel,
        solution.action,
        args.effect_horizon,
    )
    before_error = target_before - desired
    after_error = target_after - desired
    diagnostics = StepDiagnostics(
        raw_error_before=float(np.linalg.norm(before_error)),
        raw_error_after=float(np.linalg.norm(after_error)),
        task_error_before=weighted_norm(before_error, metric),
        task_error_after=weighted_norm(after_error, metric),
        action_correction=float(np.linalg.norm(solution.correction)),
        predicted_residual=solution.predicted_residual_norm,
        condition_number=solution.condition_number,
        saturated_fraction=solution.saturated_fraction,
        adapter_ms=1000.0 * (perf_counter() - started),
    )
    return solution.action, diagnostics


def evaluate_episode(
    sac: SAC,
    seed: int,
    variation: str,
    controller: str,
    args: argparse.Namespace,
) -> dict:
    live = make_env(args.max_steps)
    source_oracle = make_env(args.max_steps)
    target_oracle = make_env(args.max_steps)
    observation, _ = live.reset(seed=seed)
    source_oracle.reset(seed=seed)
    target_oracle.reset(seed=seed)
    apply_physics(live, variation)
    apply_physics(target_oracle, variation)
    rng = np.random.default_rng(
        args.seed + seed * 1009 + sum(ord(char) for char in controller)
    )
    initial_distance = object_goal_distance(live)
    distances = [initial_distance]
    total_reward = 0.0
    diagnostics: list[StepDiagnostics] = []
    for step in range(args.max_steps):
        if controller == "source_actor":
            action, _ = sac.predict(observation, deterministic=True)
        else:
            base = live.unwrapped
            action, diagnostic = adapted_action(
                sac,
                np.asarray(observation, dtype=np.float32),
                np.asarray(base.data.qpos, dtype=np.float64).copy(),
                np.asarray(base.data.qvel, dtype=np.float64).copy(),
                source_oracle,
                target_oracle,
                controller,
                args,
                rng,
            )
            diagnostics.append(diagnostic)
            if args.debug_every > 0 and step % args.debug_every == 0:
                print(
                    f"STEP variation={variation} controller={controller} "
                    f"seed={seed} t={step:03d} "
                    f"task_err={diagnostic.task_error_before:.4g}"
                    f"->{diagnostic.task_error_after:.4g} "
                    f"|da|={diagnostic.action_correction:.3f} "
                    f"cond={diagnostic.condition_number:.2e} "
                    f"ms={diagnostic.adapter_ms:.1f}",
                    flush=True,
                )
        observation, reward, terminated, truncated, _ = live.step(action)
        total_reward += float(reward)
        distances.append(object_goal_distance(live))
        if terminated or truncated:
            break
    live.close()
    source_oracle.close()
    target_oracle.close()
    record = {
        "seed": seed,
        "variation": variation,
        "controller": controller,
        "return": total_reward,
        "initial_distance": initial_distance,
        "minimum_distance": float(np.min(distances)),
        "final_distance": distances[-1],
        "progress": initial_distance - distances[-1],
        "success_any_005": float(np.min(distances) < args.success_distance),
    }
    if diagnostics:
        for field in StepDiagnostics.__dataclass_fields__:
            values = [getattr(item, field) for item in diagnostics]
            record[f"mean_{field}"] = float(np.mean(values))
        record["task_error_reduction"] = float(
            1.0
            - record["mean_task_error_after"]
            / max(record["mean_task_error_before"], 1e-12)
        )
        record["raw_error_reduction"] = float(
            1.0
            - record["mean_raw_error_after"]
            / max(record["mean_raw_error_before"], 1e-12)
        )
    return record


def source_successful_seeds(
    sac: SAC,
    args: argparse.Namespace,
) -> tuple[list[int], list[dict]]:
    successful = []
    records = []
    for seed in range(args.seed_scan):
        record = evaluate_episode(
            sac, seed, "source", "source_actor", args
        )
        records.append(record)
        if record["success_any_005"] > 0.5:
            successful.append(seed)
        print(
            f"SOURCE_SCAN seed={seed:02d} "
            f"min_d={record['minimum_distance']:.4f} "
            f"return={record['return']:.2f} "
            f"success={int(record['success_any_005'])}",
            flush=True,
        )
        if len(successful) >= args.eval_seeds:
            break
    if len(successful) < args.eval_seeds:
        raise RuntimeError(
            f"only found {len(successful)} successful source seeds"
        )
    return successful[: args.eval_seeds], records


def summarize(records: list[dict]) -> dict:
    summaries = {}
    keys = (
        "success_any_005",
        "return",
        "progress",
        "minimum_distance",
        "final_distance",
        "task_error_reduction",
        "raw_error_reduction",
        "mean_action_correction",
        "mean_adapter_ms",
    )
    groups = sorted(
        {(record["variation"], record["controller"]) for record in records}
    )
    for variation, controller in groups:
        group = [
            record
            for record in records
            if record["variation"] == variation
            and record["controller"] == controller
        ]
        summary = {"episodes": len(group)}
        for key in keys:
            values = [record[key] for record in group if key in record]
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
        summaries[f"{variation}/{controller}"] = summary
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actor", default="results/pusher-v5-SAC-expert.zip"
    )
    parser.add_argument(
        "--output",
        default=(
            "results/"
            "pusher_oracle_control_equivalence_adapter_quick.json"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7727)
    parser.add_argument("--seed-scan", type=int, default=30)
    parser.add_argument("--eval-seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--success-distance", type=float, default=0.05)
    parser.add_argument("--effect-horizon", type=int, default=2)
    parser.add_argument("--fd-epsilon", type=float, default=0.08)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--trust-radius", type=float, default=0.75)
    parser.add_argument("--critic-samples", type=int, default=6)
    parser.add_argument("--critic-action-noise", type=float, default=0.18)
    parser.add_argument("--metric-floor", type=float, default=0.01)
    parser.add_argument("--debug-every", type=int, default=25)
    parser.add_argument(
        "--variations",
        nargs="+",
        default=["weak_actuator", "heavy_arm", "combined"],
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["source_actor", "identity", "critic"],
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU", flush=True)
        args.device = "cpu"
    sac = SAC.load(args.actor, device=args.device)
    successful_seeds, scan_records = source_successful_seeds(sac, args)
    print(
        f"GATE device={sac.device} selected_seeds={successful_seeds} "
        f"horizon={args.effect_horizon}",
        flush=True,
    )
    records = []
    for variation in args.variations:
        for controller in args.controllers:
            for seed in successful_seeds:
                record = evaluate_episode(
                    sac, seed, variation, controller, args
                )
                records.append(record)
                print(
                    f"EP variation={variation} controller={controller} "
                    f"seed={seed} success={int(record['success_any_005'])} "
                    f"return={record['return']:.2f} "
                    f"min_d={record['minimum_distance']:.4f} "
                    f"final_d={record['final_distance']:.4f} "
                    f"progress={record['progress']:.4f}",
                    flush=True,
                )
    summaries = summarize(records)
    for name, summary in summaries.items():
        success = summary["success_any_005"]["mean"]
        progress = summary["progress"]["mean"]
        minimum = summary["minimum_distance"]["mean"]
        suffix = ""
        if "task_error_reduction" in summary:
            suffix = (
                f" task_reduce="
                f"{summary['task_error_reduction']['mean']:.3f}"
                f" action_delta="
                f"{summary['mean_action_correction']['mean']:.3f}"
            )
        print(
            f"SUMMARY {name} success={success:.3f} "
            f"progress={progress:.4f} min_d={minimum:.4f}{suffix}",
            flush=True,
        )
    payload = {
        "experiment": "oracle_control_equivalence_adapter_quick_gate",
        "hypothesis": (
            "Matching the source actor's task-relevant local effect under "
            "target dynamics improves immediate closed-loop transfer."
        ),
        "oracle_scope": (
            "Both source and target dynamics are exact MuJoCo models; "
            "ProtoKAN approximation and online adaptation are intentionally "
            "excluded from this upper-bound gate."
        ),
        "configuration": vars(args),
        "selected_source_successful_seeds": successful_seeds,
        "source_scan": scan_records,
        "episodes": records,
        "summary": summaries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
