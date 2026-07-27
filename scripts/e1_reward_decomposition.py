"""E1: Reward decomposition — forward/healthy/control, survival curves, per-step analysis.

Reads results/e0_per_step_data.json (produced by E0).
Produces:
  1. Reward decomposition table (forward, healthy, control cost per step)
  2. First-100-step return, last-50-step return
  3. Survival curves S(t) plot data
  4. Per-step mean reward curves for visual comparison
  5. Verdict: Case A (early termination) vs B (low per-step quality) vs C (high control cost)
"""
import json, os, numpy as np, sys
from pathlib import Path
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

OUTPUT_JSON = "results/e1_reward_decomposition.json"


def compute_stats(episodes):
    """Compute per-step reward decomposition stats from episode data."""
    n = len(episodes)
    total_R = np.array([ep["total_return"] for ep in episodes])
    lengths = np.array([ep["length"] for ep in episodes])
    fell = np.array([ep["fell"] for ep in episodes])

    # Aggregate per-step components
    all_fwd = []; all_healthy = []; all_ctrl = []
    all_rewards = []
    for ep in episodes:
        all_fwd.extend(ep.get("forward_r", []))
        all_healthy.extend(ep.get("healthy_r", []))
        all_ctrl.extend(ep.get("ctrl_cost", []))
        all_rewards.extend(ep.get("rewards", []))

    all_fwd = np.array(all_fwd); all_healthy = np.array(all_healthy)
    all_ctrl = np.array(all_ctrl); all_rewards = np.array(all_rewards)

    # Per-step means
    per_step_r = np.array([ep["total_return"] / max(ep["length"], 1) for ep in episodes])

    # First-100 and last-50 returns
    first100 = []; last50 = []
    for ep in episodes:
        rs = ep.get("rewards", [])
        if len(rs) >= 100: first100.append(sum(rs[:100]))
        else: first100.append(sum(rs))
        if len(rs) >= 50: last50.append(sum(rs[-50:]))
        elif len(rs) > 0: last50.append(sum(rs))

    # Survival curve
    max_T = int(max(lengths))
    survival = np.ones(max_T + 1)
    for t in range(1, max_T + 1):
        survival[t] = np.mean(lengths >= t)

    # Per-step reward curve (align to episode start)
    min_len = min(lengths)
    psr_curve = np.zeros(min_len)
    for t in range(min_len):
        vals = [ep["rewards"][t] for ep in episodes if t < len(ep.get("rewards", []))]
        psr_curve[t] = np.mean(vals) if vals else 0

    # Healthy rate over time
    healthy_curve = np.zeros(min_len)
    for t in range(min_len):
        vals = [ep["healthy_r"][t] for ep in episodes if t < len(ep.get("healthy_r", []))]
        healthy_curve[t] = np.mean(vals) if vals else 0

    return {
        "n_episodes": n,
        "mean_return": float(np.mean(total_R)), "std_return": float(np.std(total_R)),
        "mean_length": float(np.mean(lengths)), "std_length": float(np.std(lengths)),
        "mean_per_step_r": float(np.mean(per_step_r)),
        "mean_forward_per_step": float(np.mean(all_fwd)),
        "mean_healthy_per_step": float(np.mean(all_healthy)),
        "mean_ctrl_cost_per_step": float(np.mean(all_ctrl)),
        "total_forward": float(np.sum(all_fwd)),
        "total_healthy": float(np.sum(all_healthy)),
        "total_ctrl_cost": float(np.sum(all_ctrl)),
        "mean_first100": float(np.mean(first100)) if first100 else None,
        "mean_last50": float(np.mean(last50)) if last50 else None,
        "fell_rate": float(np.mean(fell)),
        "survival_curve": survival.tolist(),
        "per_step_reward_curve": psr_curve.tolist(),
        "healthy_rate_curve": healthy_curve.tolist(),
        "lengths": lengths.tolist(),
        "returns": total_R.tolist(),
    }


def main():
    data_path = Path("results/e0_per_step_data.json")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run E0 first.", flush=True)
        print("Will attempt to read from e0_unified_eval.json instead...", flush=True)
        # Fallback: E0 summary has some stats
        summary_path = Path("results/e0_unified_eval.json")
        if summary_path.exists():
            summary = json.load(open(summary_path))
            print("Available keys:", list(summary.get("summary", {}).keys()), flush=True)
            print("\nFrom E0 summary:", flush=True)
            for k, v in summary.get("summary", {}).items():
                print(f"  {k}: R={v['mean_return']:.1f} T={v['mean_length']:.1f} "
                      f"R̄={v['mean_per_step_r']:.4f} Fell={v['fell_rate']:.0%}",
                      flush=True)
        return

    raw = json.load(open(data_path))
    per_step = raw["per_step"]
    methods = list(per_step.keys())

    print("=" * 72)
    print("E1: Reward Decomposition Analysis")
    print(f"  Methods: {methods}")
    print("=" * 72)

    all_stats = {}
    for method in methods:
        print(f"\n  Computing stats for {method}...", flush=True)
        all_stats[method] = compute_stats(per_step[method])

    # ── Main comparison table ──────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  E1: Reward Decomposition")
    print(f"  {'Method':35s} {'R/step':>8s} {'Fwd/step':>10s} "
          f"{'Healthy%':>9s} {'Ctrl/step':>10s} {'F100':>8s} {'L50':>8s}")
    print(f"  {'-'*75}")

    for method in methods:
        s = all_stats[method]
        healthy_pct = s["mean_healthy_per_step"]  # 0-1
        print(f"  {method:35s} {s['mean_per_step_r']:>8.4f} "
              f"{s['mean_forward_per_step']:>10.4f} "
              f"{healthy_pct:>8.1%} "
              f"{s['mean_ctrl_cost_per_step']:>10.6f} "
              f"{s.get('mean_first100', 0) or 0:>8.1f} "
              f"{s.get('mean_last50', 0) or 0:>8.1f}")

    # ── Survival analysis ──────────────────────────────────────────────
    print(f"\n  Survival at key milestones (% episodes still alive):")
    print(f"  {'Method':35s} {'T=100':>8s} {'T=200':>8s} {'T=400':>8s} "
          f"{'T=800':>8s} {'T=1000':>8s}")
    print(f"  {'-'*65}")
    for method in methods:
        s = all_stats[method]
        surv = s["survival_curve"]
        def S(t): return surv[min(t, len(surv)-1)]
        print(f"  {method:35s} {S(100):>7.1%} {S(200):>7.1%} {S(400):>7.1%} "
              f"{S(800):>7.1%} {S(1000):>7.1%}")

    # ── Forward reward efficiency ──────────────────────────────────────
    print(f"\n  Forward reward (accumulated):")
    print(f"  {'Method':35s} {'Total Fwd':>12s} {'Fwd/step':>10s} {'Fwd ratio':>10s}")
    src_fwd_per_step = all_stats.get("source_on_source", {}).get("mean_forward_per_step", 1.0)
    for method in methods:
        s = all_stats[method]
        ratio = s["mean_forward_per_step"] / max(src_fwd_per_step, 1e-10)
        print(f"  {method:35s} {s['total_forward']:>12.1f} "
              f"{s['mean_forward_per_step']:>10.4f} {ratio:>9.1%}")

    # ── Verdict ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Verdict:")

    kan = all_stats.get("kan_ilc_c010", {})
    src = all_stats.get("source_on_source", {})
    trans = all_stats.get("transport", {})

    kan_rbar = kan.get("mean_per_step_r", 0)
    src_rbar = src.get("mean_per_step_r", 0)
    kan_T = kan.get("mean_length", 0)
    src_T = src.get("mean_length", 0)
    kan_f100 = kan.get("mean_first100", 0) or 0
    src_f100 = src.get("mean_first100", 0) or 0

    # Compare per-step reward ratios
    rbar_ratio = kan_rbar / max(src_rbar, 1e-10)
    T_ratio = kan_T / max(src_T, 1e-10)
    f100_ratio = kan_f100 / max(src_f100, 1e-10)

    print(f"  KAN-ILC per-step reward: {kan_rbar:.4f} vs Source: {src_rbar:.4f} ({rbar_ratio:.1%})")
    print(f"  KAN-ILC episode length:   {kan_T:.0f} vs Source: {src_T:.0f} ({T_ratio:.1%})")
    print(f"  KAN-ILC first-100 return:  {kan_f100:.1f} vs Source: {src_f100:.1f} ({f100_ratio:.1%})")

    if rbar_ratio > 0.8 and T_ratio < 0.5:
        print(f"\n  >>> Case A: Per-step quality OK ({rbar_ratio:.0%}), but early termination ({T_ratio:.0%}).")
        print(f"      Main bottleneck: long-term stability and recovery.")
    elif rbar_ratio < 0.6:
        print(f"\n  >>> Case B: Per-step reward significantly lower ({rbar_ratio:.0%}).")
        print(f"      Main bottleneck: nominal control behavior quality.")
    elif kan.get("mean_ctrl_cost_per_step", 0) > src.get("mean_ctrl_cost_per_step", 0) * 1.5:
        print(f"\n  >>> Case C: Control cost much higher. ILC finds effective but inefficient behavior.")
    else:
        print(f"\n  >>> Mixed: rbar_ratio={rbar_ratio:.0%}, T_ratio={T_ratio:.0%}")

    # Save
    json.dump({"methods": methods, "stats": all_stats}, open(OUTPUT_JSON, "w"), indent=2)
    print(f"\n  Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
