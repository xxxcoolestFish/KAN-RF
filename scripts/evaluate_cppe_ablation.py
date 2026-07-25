"""Evaluate CPPE ablation models on source + target physics shifts."""

from __future__ import annotations

import argparse, json, sys, os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpbn.cognitive_pca import CognitivePCA, PCARanges
from cpbn.cppe_env import PhysicsConditionedEnv
from scripts.prescreen_hopper_physics_shifts import ENVS, SHIFTS, make_shifted_env


def evaluate_on_shift(model_path, norm_path, shift, z_value, device, n_episodes=10, seed=1911):
    """Evaluate a conditioned model on a physics shift with given z."""
    def make():
        base = make_shifted_env(shift, seed, "hopper")()
        cond = PhysicsConditionedEnv(base, z_dim=len(z_value))
        cond.set_z(z_value)
        return cond

    vec_env = DummyVecEnv([make])
    vec_env = VecNormalize.load(norm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env, device=device)

    returns = []
    for _ in range(n_episodes):
        obs = vec_env.reset()
        total = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            total += float(reward[0])
            if dones[0]:
                break
        returns.append(total)
    vec_env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="results/cppe_training")
    parser.add_argument("--pca-model", default="results/cppe_pca_model.npz")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--json-out", default="results/cppe_ablation_eval.json")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load PCA
    pca_data = np.load(args.pca_model)
    pc_ranges = PCARanges(mins=pca_data["pc_mins"], maxs=pca_data["pc_maxs"])
    pca = CognitivePCA(args.pca_model, k=5, pc_ranges=pc_ranges)
    z_source = pca.z_source.astype(np.float32)

    # Compute z for each target shift using observed values
    shift_z = {}
    for i, name in enumerate(pca_data["shift_names"]):
        shift_z[str(name)] = pca_data["z_values"][i, :5].astype(np.float32)

    print(f"z_source: {z_source}")
    for name in ["payload_125", "friction_070", "combo_medium"]:
        print(f"z_{name}: {shift_z[name]}")

    # Shifts to evaluate
    target_shifts = {
        "source": ("source", z_source),
        "payload_125": (SHIFTS["payload_125"], shift_z["payload_125"]),
        "friction_070": (SHIFTS["friction_070"], shift_z["friction_070"]),
        "combo_medium": (SHIFTS["combo_medium"], shift_z["combo_medium"]),
    }

    # Ablations to test
    ablations = ["ppo_only", "random_z", "pca_only", "full"]

    results = {}
    for ablation in ablations:
        model_path = os.path.join(args.model_dir, f"cppe_{ablation}_seed1811.zip")
        norm_path = os.path.join(args.model_dir, f"cppe_{ablation}_norm_seed1811.pkl")

        if not os.path.exists(model_path):
            print(f"SKIP {ablation}: model not found")
            continue

        print(f"\n=== {ablation} ===")
        results[ablation] = {}
        for shift_name, (shift, z_val) in target_shifts.items():
            r, s = evaluate_on_shift(model_path, norm_path, shift, z_val,
                                      device, n_episodes=args.n_episodes)
            print(f"  {shift_name:16s}: {r:8.1f} +/- {s:.1f}")
            results[ablation][shift_name] = {"mean": r, "std": s}

    # Summary: improvement over ppo_only on target shifts
    print("\n=== Target shift improvement vs ppo_only ===")
    baseline = results.get("ppo_only", {})
    for ablation in ["random_z", "pca_only", "full"]:
        if ablation not in results:
            continue
        for shift_name in ["payload_125", "friction_070", "combo_medium"]:
            if shift_name in baseline and shift_name in results[ablation]:
                diff = results[ablation][shift_name]["mean"] - baseline[shift_name]["mean"]
                print(f"  {ablation:12s} {shift_name:16s}: {diff:+.1f}")

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    json.dump(results, open(args.json_out, "w"), indent=2)
    print(f"\nSaved: {args.json_out}")


if __name__ == "__main__":
    main()
