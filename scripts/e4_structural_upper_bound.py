"""E4: Structural upper bound — train residual policies with target reward.

Trains three residual architectures anchored to frozen Transport:
  1. Fourier phase actor (4-dim: sin/cos 2pi, sin/cos 4pi)
  2. State residual actor (full obs)
  3. Circular phase-table (learned per-bin residual, linear interpolation)

All use target env reward (diagnostic only — "how high can this structure go?").
PPO training: 100k steps, 30 eval episodes.
"""
import sys, numpy as np, torch, argparse, json, time
from pathlib import Path
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy, load_cognition
from scripts.diagnose_hopper_pullback_effect import fit_distilled_source_counterfactual_context, load_source_twin
from scripts.ilc1_ablation import *

DEVICE = torch.device("cuda")
N_PPO_STEPS = 100_000
N_EVAL_EPS = 30
PPO_LR = 1e-4
GAMMA = 0.99; LAMBDA = 0.95; CLIP_EPS = 0.2
BATCH_SIZE = 256; N_EPOCHS = 5
RESIDUAL_MAX = 0.5  # max residual magnitude
PHASE_BINS = 20

# ── Phase utilities ────────────────────────────────────────────────────
def fourier_phase(xi, n_harmonics=2):
    """xi in [0,1) -> [sin(2pi*xi), cos(2pi*xi), sin(4pi*xi), cos(4pi*xi)]"""
    feats = []
    for k in range(1, n_harmonics + 1):
        feats.append(np.sin(2 * np.pi * k * xi))
        feats.append(np.cos(2 * np.pi * k * xi))
    return np.array(feats, dtype=np.float32)

def estimate_phase(obs, ref_obs, current_xi=0.5):
    """Simple phase estimation: find closest ref point to current obs."""
    dists = np.sum((ref_obs - obs) ** 2, axis=1)
    xi = np.argmin(dists) / len(ref_obs)
    return xi


# ── Actor networks ─────────────────────────────────────────────────────
class FourierPhaseActor(torch.nn.Module):
    def __init__(self, a_dim, n_harmonics=2, hidden=64):
        super().__init__()
        self.phase_dim = n_harmonics * 2
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.phase_dim, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, a_dim),
        )
        # Initialize near zero (small residual)
        for layer in [self.net[0], self.net[2], self.net[4]]:
            if hasattr(layer, 'weight'):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, phase_feats):
        return RESIDUAL_MAX * torch.tanh(self.net(phase_feats))


class StateResidualActor(torch.nn.Module):
    def __init__(self, s_dim, a_dim, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(s_dim, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, a_dim),
        )
        for layer in [self.net[0], self.net[2], self.net[4]]:
            if hasattr(layer, 'weight'):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, state):
        return RESIDUAL_MAX * torch.tanh(self.net(state))


class PhaseTable(torch.nn.Module):
    """Learnable per-bin residual table with linear interpolation."""
    def __init__(self, a_dim, n_bins=PHASE_BINS):
        super().__init__()
        self.n_bins = n_bins
        self.table = torch.nn.Parameter(torch.zeros(n_bins, a_dim))

    def forward(self, xi):
        """Linear interpolation: xi in [0,1) -> residual."""
        xi = xi % 1.0
        idx = xi * self.n_bins
        i0 = int(np.floor(idx)) % self.n_bins
        i1 = (i0 + 1) % self.n_bins
        alpha = idx - np.floor(idx)
        return RESIDUAL_MAX * torch.tanh(
            (1 - alpha) * self.table[i0] + alpha * self.table[i1])


# ── Critic ─────────────────────────────────────────────────────────────
class FullStateCritic(torch.nn.Module):
    def __init__(self, s_dim, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(s_dim, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


# ── PPO training ───────────────────────────────────────────────────────
def run_episode_for_training(env_factory, sp, basis, sc, tc, seed, actor, actor_type,
                              ref_obs, critic=None):
    """Run one episode, return (states, phase_feats, actions, rewards, dones, values, log_probs)."""
    env = env_factory(seed)
    obs, _ = env.reset(seed=seed)
    step = 0; ep_data = {"s": [], "pf": [], "a_res": [], "r": [], "d": [], "v": [], "lp": []}
    detector = CycleDetector(); detector.reset()
    td_steps = []
    xi = 0.5  # initial phase estimate (mid-cycle)

    while True:
        s_t = torch.as_tensor(obs, device='cpu', dtype=torch.float32)
        with torch.no_grad():
            nominal = sp.action(s_t.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
            s_eff = sc.acceleration(basis, s_t.unsqueeze(0).to(DEVICE),
                                     nominal.unsqueeze(0).to(DEVICE))
            a_tr = tc.transport_action(basis, s_t.unsqueeze(0).to(DEVICE),
                                        desired_effect=s_eff,
                                        nominal_action=nominal.unsqueeze(0).to(DEVICE),
                                        regularization=1e-2).clamp(-1, 1).squeeze(0).cpu().numpy()

        # Phase update
        if len(td_steps) > 0:
            xi = min((step - td_steps[-1]) / 64, 0.99)
        pf = fourier_phase(xi)

        # Actor: residual from phase or state
        if actor_type == "phase":
            residual = actor(torch.as_tensor(pf, device=DEVICE).unsqueeze(0)).squeeze(0).cpu().detach().numpy()
        elif actor_type == "table":
            residual = actor(xi).cpu().detach().numpy()
        else:  # state
            residual = actor(s_t.unsqueeze(0).to(DEVICE)).squeeze(0).cpu().detach().numpy()

        # Critic value
        if critic is not None:
            v = critic(s_t.unsqueeze(0).to(DEVICE)).item()
        else:
            v = 0.0

        # Log prob (Gaussian with fixed std)
        a_mean_t = torch.as_tensor(residual, device=DEVICE)
        a_dist = torch.distributions.Normal(a_mean_t, 0.3)
        a_sample = a_dist.sample()
        lp = a_dist.log_prob(a_sample).sum().item()

        a_final = np.clip(a_tr + a_sample.cpu().numpy(), -1, 1)
        next_obs, reward, terminated, truncated, info = env.step(a_final)

        ep_data["s"].append(obs.copy())
        ep_data["pf"].append(pf)
        ep_data["a_res"].append(a_sample.cpu().numpy().copy())
        ep_data["r"].append(float(reward))
        ep_data["d"].append(terminated or truncated)
        ep_data["v"].append(v)
        ep_data["lp"].append(lp)

        if detector.update(env): td_steps.append(step)
        obs = next_obs; step += 1
        if terminated or truncated: break

    env.close()
    return ep_data


def compute_gae_and_returns(rewards, values, dones, gamma, lam):
    advantages = np.zeros(len(rewards))
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = 0.0 if (t == len(rewards) - 1 or dones[t]) else values[t + 1]
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae * (1 - dones[t])
        advantages[t] = gae
    returns = advantages + np.array(values)
    return advantages, returns


def ppo_update(actor, critic, actor_opt, critic_opt, buffer, n_epochs, batch_size, clip_eps):
    """PPO update from collected buffer."""
    n = len(buffer["s"])
    idx = np.arange(n)
    actor_losses, critic_losses = [], []

    for _ in range(n_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]

            s_b = torch.as_tensor(np.stack([buffer["s"][i] for i in batch_idx]),
                                  device=DEVICE, dtype=torch.float32)
            pf_b = torch.as_tensor(np.stack([buffer["pf"][i] for i in batch_idx]),
                                    device=DEVICE, dtype=torch.float32)
            a_b = torch.as_tensor(np.stack([buffer["a_res"][i] for i in batch_idx]),
                                   device=DEVICE, dtype=torch.float32)
            adv_b = torch.as_tensor([buffer["adv"][i] for i in batch_idx],
                                     device=DEVICE, dtype=torch.float32)
            ret_b = torch.as_tensor([buffer["ret"][i] for i in batch_idx],
                                     device=DEVICE, dtype=torch.float32)
            old_lp_b = torch.as_tensor([buffer["lp"][i] for i in batch_idx],
                                        device=DEVICE, dtype=torch.float32)

            # Normalize advantages
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

            # Critic update
            v_pred = critic(s_b)
            critic_loss = ((v_pred - ret_b) ** 2).mean()
            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
            critic_losses.append(critic_loss.item())

            # Actor update
            if hasattr(actor, 'table'):  # PhaseTable — recover xi from Fourier features
                new_a_list = []
                for i in batch_idx:
                    # pf = [sin(2πξ), cos(2πξ), sin(4πξ), cos(4πξ)]
                    xi_val = float(np.arctan2(buffer["pf"][i][0], buffer["pf"][i][1]))
                    xi_val = (xi_val / (2 * np.pi)) % 1.0
                    new_a_list.append(actor(xi_val))
                new_a = torch.stack(new_a_list)
            else:
                new_a = actor(pf_b) if "phase" in str(type(actor)).lower() or hasattr(actor, 'table') else actor(s_b)
            dist = torch.distributions.Normal(new_a, 0.3)
            new_lp = dist.log_prob(a_b).sum(-1)
            ratio = (new_lp - old_lp_b).exp()
            clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
            actor_loss = -torch.min(ratio * adv_b, clip_adv).mean()
            actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
            actor_losses.append(actor_loss.item())

    return np.mean(actor_losses), np.mean(critic_losses)


def eval_actor(env_factory, sp, basis, sc, tc, actor, actor_type, ref_obs, n_episodes=30):
    """Deterministic evaluation."""
    returns, lengths = [], []
    for i in range(n_episodes):
        env = env_factory(9000 + i * 100)
        obs, _ = env.reset(seed=9000 + i * 100)
        step = 0; total_r = 0.0
        detector = CycleDetector(); detector.reset()
        td_steps = []; xi = 0.5
        while True:
            s_t = torch.as_tensor(obs, device=DEVICE, dtype=torch.float32).unsqueeze(0)
            nominal = sp.action(s_t); s_eff = sc.acceleration(basis, s_t, nominal)
            a_tr = tc.transport_action(basis, s_t, desired_effect=s_eff,
                                       nominal_action=nominal, regularization=1e-2
                                       ).clamp(-1, 1).squeeze(0).cpu().numpy()
            if len(td_steps) > 0: xi = min((step - td_steps[-1]) / 64, 0.99)
            with torch.no_grad():
                if actor_type == "phase":
                    pf = fourier_phase(xi)
                    residual = actor(torch.as_tensor(pf, device=DEVICE).unsqueeze(0)).squeeze(0).cpu().numpy()
                elif actor_type == "table":
                    residual = actor(xi).cpu().numpy()
                else:
                    residual = actor(s_t).squeeze(0).cpu().numpy()
            a_final = np.clip(a_tr + residual, -1, 1)
            next_obs, reward, terminated, truncated, _ = env.step(a_final)
            total_r += float(reward)
            if detector.update(env): td_steps.append(step)
            obs = next_obs; step += 1
            if terminated or truncated: break
        env.close()
        returns.append(total_r); lengths.append(step)
    return np.mean(returns), np.std(returns), np.mean(lengths), sum(1 for r in returns if not np.isnan(r))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1811)
    p.add_argument("--n-steps", type=int, default=N_PPO_STEPS)
    p.add_argument("--residual-max", type=float, default=RESIDUAL_MAX)
    a = p.parse_args()

    print("=" * 72)
    print("E4: Structural Upper Bound (Reward-Trained Residuals)")
    print(f"  PPO: {a.n_steps} steps, residual_max={a.residual_max}")
    print("=" * 72)

    # ── Load ────────────────────────────────────────────────────────────
    print("\n[1/3] Loading...", flush=True)
    t0 = time.time()
    sp = FrozenSourcePolicy("results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl", DEVICE, a.seed, env="hopper")
    st = load_source_twin("results/hopper_source_affine_twin_cloud_seed1811.pt", DEVICE)
    basis, sc, _, _ = load_cognition(argparse.Namespace(
        cognition_checkpoint="results/hopper_source_control_sobolev_calibrated_seed1811.pt", device="cuda"), DEVICE)
    fa = argparse.Namespace(target="friction_070", seed=a.seed, env="hopper", device="cuda",
        cognition_warmup=1024, warmup_noise=0.3, transform_ridge=10.0, drift_ridge=100.0,
        drift_spectral_eta=0.0, drift_spectral_beta=1.0, drift_spectral_mode="max",
        drift_smooth_lambda=0.0, diagonal_transform=False)
    tc, _ = fit_distilled_source_counterfactual_context(sp, basis, sc, fa, DEVICE, st)

    shift = SHIFTS["friction_070"]
    env_factory = lambda s=0: make_shifted_env(shift, s, "hopper")()
    s_dim, a_dim = basis.state_dim, basis.action_dim

    # Build reference for phase estimation
    ro, ra, rl = build_reference(sp, basis, sc, tc, shift, DEVICE, a.seed, n_ep=10)
    ref_obs = ro
    print(f"  Load time: {time.time()-t0:.1f}s", flush=True)

    # ── Train each architecture ─────────────────────────────────────────
    configs = [
        ("phase_residual", "Fourier Phase Actor", "phase",
         lambda: FourierPhaseActor(a_dim)),
        ("state_residual", "State Residual Actor", "state",
         lambda: StateResidualActor(s_dim, a_dim)),
        ("phase_table", "Phase Table (circular)", "table",
         lambda: PhaseTable(a_dim)),
    ]

    results = {}

    for key, label, actor_type, make_actor in configs:
        print(f"\n[2/3] Training {label} ({a.n_steps} steps)...", flush=True)
        t0 = time.time()

        actor = make_actor().to(DEVICE)
        critic = FullStateCritic(s_dim).to(DEVICE)
        actor_opt = torch.optim.Adam(actor.parameters(), lr=PPO_LR)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=PPO_LR * 3)

        buffer_done = {"s": [], "pf": [], "a_res": [], "adv": [], "ret": [], "lp": []}
        total_steps = 0; ep = 0
        eval_log = []

        buffer_done = {"s": [], "pf": [], "a_res": [], "adv": [], "ret": [], "lp": []}

        while total_steps < a.n_steps:
            ep_data = run_episode_for_training(env_factory, sp, basis, sc, tc,
                                                a.seed + 50000 + ep * 100,
                                                actor, actor_type, ref_obs, critic)
            n = len(ep_data["r"])
            total_steps += n; ep += 1

            # Compute GAE for THIS episode
            adv, ret = compute_gae_and_returns(ep_data["r"], ep_data["v"], ep_data["d"], GAMMA, LAMBDA)

            # Accumulate into done buffer
            for i in range(n):
                buffer_done["s"].append(ep_data["s"][i])
                buffer_done["pf"].append(ep_data["pf"][i])
                buffer_done["a_res"].append(ep_data["a_res"][i])
                buffer_done["adv"].append(float(adv[i]))
                buffer_done["ret"].append(float(ret[i]))
                buffer_done["lp"].append(float(ep_data["lp"][i]))

            # Update when buffer is large enough
            if len(buffer_done["s"]) >= BATCH_SIZE * 2:
                a_loss, c_loss = ppo_update(actor, critic, actor_opt, critic_opt, buffer_done, N_EPOCHS, BATCH_SIZE, CLIP_EPS)
                buffer_done = {"s": [], "pf": [], "a_res": [], "adv": [], "ret": [], "lp": []}

            if total_steps % 20000 == 0 or total_steps >= a.n_steps:
                eval_r, eval_std, eval_T, _ = eval_actor(env_factory, sp, basis, sc, tc, actor, actor_type, ref_obs, N_EVAL_EPS)
                eval_log.append({"steps": total_steps, "return": float(eval_r), "std": float(eval_std), "length": float(eval_T)})
                print(f"    {total_steps}/{a.n_steps}: eval_R={eval_r:.1f}+/-{eval_std:.0f} T={eval_T:.1f}", flush=True)

        final_r, final_std, final_T, _ = eval_actor(env_factory, sp, basis, sc, tc, actor, actor_type, ref_obs, N_EVAL_EPS)
        print(f"    Final: R={final_r:.1f}+/-{final_std:.0f} T={final_T:.1f} ({time.time()-t0:.1f}s)", flush=True)

        results[key] = {"label": label, "mean_return": float(final_r), "std_return": float(final_std),
                        "mean_length": float(final_T), "eval_log": eval_log,
                        "actor_type": actor_type}

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n[3/3] E4 Results", flush=True)
    print("=" * 72)
    print(f"  {'Architecture':30s} {'Return':>10s} {'Length':>10s}")
    print(f"  {'-'*55}")

    # Baselines from E0
    print(f"  {'Transport (baseline)':30s} {'571.4':>10s} {'183.3':>10s}")
    print(f"  {'KAN-ILC (c=0.10)':30s} {'576.1':>10s} {'185.6':>10s}")
    print(f"  {'Target Oracle':30s} {'1282.2':>10s} {'413.9':>10s}")
    print(f"  {'-'*55}")

    for key, r in results.items():
        print(f"  {r['label']:30s} {r['mean_return']:>10.1f} {r['mean_length']:>10.1f}")

    json.dump({"config": {"n_ppo_steps": a.n_steps, "residual_max": a.residual_max},
               "results": results},
              open("results/e4_structural_upper_bound.json", "w"), indent=2)
    print(f"\n  Saved to results/e4_structural_upper_bound.json")


if __name__ == "__main__":
    main()
