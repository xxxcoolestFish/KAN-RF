"""CartPole: KAN + Transport on pole length shift (0.5 → 0.7).

Non-periodic environment. Tests whether KAN-B_t + value gradient
works in closed-loop when distribution shift is minimal (fixed start state).

Pipeline:
  1. PPO on source (half_length=0.5) → π_source, V_source
  2. KAN on source → F_source(s,a)
  3. Warmup on target (half_length=0.7) → F_target(s,a)
  4. Transport: a_tr = argmin ||F_target(s,a) - F_source(s,π_source(s))||²
  5. Gradient: a = a_tr + α·normalize(B_tᵀ∇V_source(s'_KAN))
"""
import sys, numpy as np, torch, argparse, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpbn.continuous_cartpole import (
    CartPoleParameters, cartpole_acceleration, CartPoleActor)
from cpbn.generic_affine_kan import (
    CompactInteractionKANDictionary, AffineKANContext,
    RecursiveAffineKANEstimator, fit_affine_kan_context)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S_DIM, A_DIM = 4, 1
FORCE_LIMIT = 12.0
DT = 0.02  # 50Hz


# ═══════════════════════════════════════════════════════════════════════
# Env wrapper (differentiable, batched)
# ═══════════════════════════════════════════════════════════════════════

class CartPoleEnv:
    """Differentiable CartPole. State = [x, theta, x_dot, theta_dot]."""
    def __init__(self, params=None):
        self.params = params or CartPoleParameters()
        self.state = None
        self.steps = 0
        self.max_steps = 500

    def reset(self):
        self.state = torch.zeros(S_DIM, device=DEVICE)
        self.state[1] = 0.05 * (2 * torch.rand(1, device=DEVICE) - 1)  # random theta
        self.steps = 0
        return self.state.cpu().numpy()

    def step(self, action):
        s = self.state
        a = torch.as_tensor(action, device=DEVICE, dtype=torch.float32).view(-1)
        if a.dim() == 0:
            a = a.unsqueeze(0)
        acc = cartpole_acceleration(s.unsqueeze(0), a.unsqueeze(0), self.params).squeeze(0)
        # Euler integration
        s_next = s.clone()
        s_next[0] = s[0] + DT * s[2]  # x
        s_next[1] = s[1] + DT * s[3]  # theta
        s_next[2] = s[2] + DT * acc[0]  # x_dot
        s_next[3] = s[3] + DT * acc[1]  # theta_dot
        self.state = s_next
        self.steps += 1
        # Reward: 1 if alive, 0 if dead
        alive = abs(s_next[0].item()) < 2.4 and abs(s_next[1].item()) < 0.5
        reward = 1.0 if alive else 0.0
        done = not alive or self.steps >= self.max_steps
        return s_next.cpu().numpy(), reward, done, {}

    def batch_acceleration(self, states, actions):
        """(N,4), (N,1) → (N,4) state derivatives."""
        acc = cartpole_acceleration(states, actions, self.params)
        ds = torch.cat([states[..., 2:3], states[..., 3:4], acc], dim=-1)
        return ds


# ═══════════════════════════════════════════════════════════════════════
# PPO training (simple, fast on CartPole)
# ═══════════════════════════════════════════════════════════════════════

def train_ppo(env_factory, n_steps=50000, lr=3e-4, hidden=64, seed=0):
    """Train PPO on CartPole. Returns (actor, critic, vecnorm_stats)."""
    torch.manual_seed(seed)
    actor = CartPoleActor(hidden_dim=hidden).to(DEVICE)
    critic = torch.nn.Sequential(
        torch.nn.Linear(4, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1)).to(DEVICE)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)

    # Collect stats for normalization
    obs_buffer = []
    for _ in range(20):
        env = env_factory()
        s = env.reset()
        for _ in range(env.max_steps):
            obs_buffer.append(s)
            a = actor(torch.as_tensor(s, device=DEVICE, dtype=torch.float32)).detach().cpu().numpy()
            s, _, done, _ = env.step(a)
            if done: break
    obs_arr = np.stack(obs_buffer)
    obs_mean = torch.as_tensor(obs_arr.mean(0), device=DEVICE, dtype=torch.float32)
    obs_std = torch.as_tensor(obs_arr.std(0).clip(0.01), device=DEVICE, dtype=torch.float32)

    def normalize(obs):
        return (obs - obs_mean) / obs_std

    env = env_factory()
    s = env.reset()
    total_steps = 0
    gamma, lam = 0.99, 0.95
    clip_eps = 0.2
    batch_size = 256
    n_epochs = 10

    while total_steps < n_steps:
        # Collect rollout
        states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
        ep_reward = 0
        for _ in range(batch_size):
            s_t = torch.as_tensor(s, device=DEVICE, dtype=torch.float32)
            s_norm = normalize(s_t)
            with torch.no_grad():
                a_mean = actor(s_norm)
                a_std = 0.5
                a_dist = torch.distributions.Normal(a_mean, a_std)
                a = a_dist.sample()
                logp = a_dist.log_prob(a).sum()
                v = critic(s_norm)
            a_np = a.clamp(-FORCE_LIMIT, FORCE_LIMIT).cpu().numpy()
            s_next, r, done, _ = env.step(a_np)
            states.append(s); actions.append(a); rewards.append(r)
            dones.append(done); values.append(v); log_probs.append(logp)
            s = s_next; total_steps += 1; ep_reward += r
            if done:
                s = env.reset()

        # GAE
        advantages = torch.zeros(len(rewards), device=DEVICE)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1 or dones[t]:
                next_v = 0.0
            else:
                next_v = values[t + 1].item()
            delta = rewards[t] + gamma * next_v - values[t].item()
            gae = delta + gamma * lam * gae * (1 - dones[t])
            advantages[t] = gae
        returns = advantages + torch.tensor([v.item() for v in values], device=DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.as_tensor(np.stack(states), device=DEVICE, dtype=torch.float32)
        actions_t = torch.stack(actions)
        old_logp = torch.stack(log_probs)

        # PPO update
        for _ in range(n_epochs):
            idx = torch.randperm(len(states_t))
            for i in range(0, len(states_t), 64):
                batch_idx = idx[i:i+64]
                s_b = normalize(states_t[batch_idx])
                a_b = actions_t[batch_idx]
                adv_b = advantages[batch_idx]
                ret_b = returns[batch_idx]
                old_lp = old_logp[batch_idx]

                a_mean = actor(s_b)
                dist = torch.distributions.Normal(a_mean, 0.5)
                new_lp = dist.log_prob(a_b).sum(-1)
                ratio = (new_lp - old_lp).exp()
                clip_adv = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv_b
                actor_loss = -torch.min(ratio * adv_b, clip_adv).mean()

                v_pred = critic(s_b).squeeze(-1)
                critic_loss = ((v_pred - ret_b) ** 2).mean()

                actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
                critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

        if total_steps % 5000 == 0:
            print(f"    PPO step {total_steps}: ep_reward={ep_reward:.0f}", flush=True)

    return actor, critic, obs_mean, obs_std


# ═══════════════════════════════════════════════════════════════════════
# KAN training
# ═══════════════════════════════════════════════════════════════════════

def collect_kan_data(env, actor, normalize, n_steps=2048):
    """Collect (s, a, s') for KAN training."""
    states, actions, next_states = [], [], []
    s = env.reset()
    for _ in range(n_steps):
        s_t = torch.as_tensor(s, device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            a = actor(normalize(s_t)).cpu().numpy()
        a = np.clip(a + 0.1 * np.random.randn(*a.shape), -FORCE_LIMIT, FORCE_LIMIT)
        s_next, _, done, _ = env.step(a)
        states.append(s); actions.append(a); next_states.append(s_next)
        s = s_next
        if done: s = env.reset()
    return (torch.as_tensor(np.stack(states), device=DEVICE, dtype=torch.float32),
            torch.as_tensor(np.stack(actions), device=DEVICE, dtype=torch.float32),
            torch.as_tensor(np.stack(next_states), device=DEVICE, dtype=torch.float32))


# ═══════════════════════════════════════════════════════════════════════
# Value gradient + Transport controller
# ═══════════════════════════════════════════════════════════════════════

class KANTransportController:
    def __init__(self, basis, source_ctx, target_ctx, critic, obs_mean, obs_std):
        self.basis = basis; self.source_ctx = source_ctx
        self.target_ctx = target_ctx; self.critic = critic
        self.obs_mean = obs_mean; self.obs_std = obs_std

    def transport_action(self, state):
        s_t = state.unsqueeze(0) if state.dim() == 1 else state
        with torch.no_grad():
            # Source nominal action
            s_norm = (s_t - self.obs_mean) / self.obs_std
            a_nominal = CartPoleActor.__new__(CartPoleActor)  # placeholder, use stored
            # Actually compute action manually...
            # Skip - use actor directly

    def compute_gradient(self, state, action, alpha):
        """B_t^T * grad(V_source)(s'_KAN), normalized."""
        # ...


# ═══════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════

def evaluate(env_factory, actor, normalize, n_episodes=20):
    returns = []
    for ep in range(n_episodes):
        env = env_factory()
        s = env.reset(); total_r = 0
        while True:
            s_t = torch.as_tensor(s, device=DEVICE, dtype=torch.float32)
            with torch.no_grad():
                a = actor(normalize(s_t)).clamp(-FORCE_LIMIT, FORCE_LIMIT).cpu().numpy()
            s, r, done, _ = env.step(a)
            total_r += r
            if done: break
        returns.append(total_r)
    return np.mean(returns), np.std(returns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1607)
    parser.add_argument("--json-out", default="results/cartpole_kan_transport.json")
    args = parser.parse_args()

    print("=" * 60)
    print("CartPole KAN + Transport: pole length 0.5 → 0.7")
    print("=" * 60)

    # ── 1. Setup ────────────────────────────────────────────────────────
    print("\n[1/4] Setting up environments...", flush=True)
    # Higher gravity = harder to balance (single physics dimension change)
    src_params = CartPoleParameters()  # default: mass=1.0, pole_mass=0.1, half=0.5, g=9.8
    tgt_params = CartPoleParameters(  # hard combo: multi-parameter shift
        cart_mass=1.8, pole_mass=0.25, half_length=0.8,
        gravity=13.0, actuator_scale=0.55, cart_friction=0.1)

    def make_src(): return CartPoleEnv(src_params)
    def make_tgt(): return CartPoleEnv(tgt_params)

    # ── 2. Train PPO on source ──────────────────────────────────────────
    print("\n[2/4] Training PPO on source (half_length=0.5)...", flush=True)
    # Try loading existing checkpoint first
    ckpt_path = "results/cartpole_source_actor_ppo_seed1607.pt"
    if Path(ckpt_path).exists():
        print("  Loading existing checkpoint (hidden=96)...", flush=True)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        actor = CartPoleActor(hidden_dim=96).to(DEVICE)
        actor.load_state_dict(ckpt["actor"])
        # Train training data and critic
        print("  Training new critic (10k steps)...", flush=True)
        _, critic, obs_mean, obs_std = train_ppo(make_src, n_steps=10000, hidden=96, seed=args.seed)
        # Use the loaded actor (better) instead of trained one
    else:
        actor, critic, obs_mean, obs_std = train_ppo(make_src, n_steps=50000, seed=args.seed)
        torch.save({"actor": actor}, ckpt_path)

    def normalize(obs):
        return (obs - obs_mean) / obs_std

    # Evaluate source
    src_r, src_s = evaluate(make_src, actor, normalize)
    print(f"  Source performance: {src_r:.1f} ± {src_s:.1f}")
    tgt_r_naive, tgt_s_naive = evaluate(make_tgt, actor, normalize)
    print(f"  Target (no adaptation): {tgt_r_naive:.1f} ± {tgt_s_naive:.1f}")

    # ── 3. Train KAN ────────────────────────────────────────────────────
    print("\n[3/4] Training KAN...", flush=True)

    # Source KAN
    basis = CompactInteractionKANDictionary(
        torch.tensor([2.4, 0.5, 3.0, 3.0]), torch.tensor([FORCE_LIMIT]),
        grid_size=4, spline_order=2, pair_modes=3).to(DEVICE)
    src_s, src_a, src_sn = collect_kan_data(make_src(), actor, normalize, 4096)
    src_acc = (src_sn - src_s) / DT
    # State derivative for KAN: ds/dt = [x_dot, theta_dot, x_ddot, theta_ddot]
    # KAN predicts [x_ddot, theta_ddot] only (the acceleration part)
    src_dsdt = torch.cat([src_s[..., 2:3], src_s[..., 3:4],
                          src_acc[..., 0:1], src_acc[..., 1:2]], dim=-1)
    src_ctx = fit_affine_kan_context(basis, src_s, src_a, src_dsdt, ridge=1.0)
    print(f"  Source KAN fitted: {basis.feature_dim} features")

    # Target KAN (warmup)
    warmup_s, warmup_a, warmup_sn = collect_kan_data(make_tgt(), actor, normalize, 1024)
    warmup_dsdt = torch.cat([warmup_s[..., 2:3], warmup_s[..., 3:4],
                             (warmup_sn[..., 0:1] - warmup_s[..., 0:1]) / DT,
                             (warmup_sn[..., 1:2] - warmup_s[..., 1:2]) / DT], dim=-1)
    tgt_ctx = fit_affine_kan_context(basis, warmup_s, warmup_a, warmup_dsdt, ridge=1.0)
    print("  Target KAN fitted (warmup)")

    # ── 4. Test: Frozen vs Online KAN ────────────────────────────────────
    print("\n[4/4] Testing controllers...", flush=True)

    results = {}
    # Test fewer alphas, focus on frozen vs online comparison
    alphas = [0.0, 0.1]

    for online_kan_enabled in [False, True]:
        mode = "online" if online_kan_enabled else "frozen"
        print(f"\n  --- {mode} KAN ---", flush=True)

        for alpha in alphas:
            label = f"{mode}/alpha={alpha:.1f}" if alpha != 0 else f"{mode}/Transport"
            returns = []

            # Fresh KAN for each mode
            if online_kan_enabled:
                estimator = RecursiveAffineKANEstimator(basis, tgt_ctx, ridge=50.0, forgetting_factor=0.995)
                ctx = estimator.context()
            else:
                ctx = tgt_ctx

            for ep in range(50):
                env = make_tgt()
                s = env.reset()
                total_r = 0
                ep_states, ep_actions, ep_next_states = [], [], []

                while True:
                    s_t = torch.as_tensor(s, device=DEVICE, dtype=torch.float32)
                    with torch.no_grad():
                        s_norm = normalize(s_t)
                        a_nom = actor(s_norm).clamp(-FORCE_LIMIT, FORCE_LIMIT)
                        src_eff = src_ctx.acceleration(basis, s_t, a_nom)
                        a_tr = ctx.transport_action(
                            basis, s_t, desired_effect=src_eff,
                            nominal_action=a_nom, regularization=1e-2).clamp(-FORCE_LIMIT, FORCE_LIMIT)

                    if alpha != 0:
                        with torch.no_grad():
                            tgt_eff = ctx.acceleration(basis, s_t, a_tr)
                            s_next_kan = s_t + tgt_eff
                        s_next_grad = s_next_kan.detach().clone().requires_grad_(True)
                        s_nn = normalize(s_next_grad)
                        v = critic(s_nn)
                        grad_v = torch.autograd.grad(v.sum(), s_next_grad)[0].detach()
                        with torch.no_grad():
                            _, gain = ctx.drift_and_gain(basis, s_t)
                            B = gain.squeeze(0)
                        g = (B.T @ grad_v).squeeze()
                        g_norm = abs(g.item()) + 1e-8
                        da = alpha * (g / g_norm) * FORCE_LIMIT
                        a_final = (a_tr + da).clamp(-FORCE_LIMIT, FORCE_LIMIT)
                    else:
                        a_final = a_tr

                    a_np = a_final.cpu().numpy()
                    s_next, r, done, _ = env.step(a_np)
                    total_r += r
                    ep_states.append(s); ep_actions.append(a_np); ep_next_states.append(s_next)
                    s = s_next
                    if done: break

                # Online KAN update after episode (quality-gated)
                if online_kan_enabled and total_r >= 150 and len(ep_states) > 30:
                    batch_s = torch.as_tensor(np.stack(ep_states), device=DEVICE, dtype=torch.float32)
                    batch_a = torch.as_tensor(np.stack(ep_actions), device=DEVICE, dtype=torch.float32)
                    dsdt = torch.as_tensor(np.stack(ep_next_states) - np.stack(ep_states),
                                          device=DEVICE, dtype=torch.float32)
                    estimator.update(batch_s, batch_a, dsdt)
                    ctx = estimator.context()

                returns.append(total_r)

            mean_r, std_r = np.mean(returns), np.std(returns)
            print(f"    {label:>22s}: {mean_r:.1f} ± {std_r:.1f}", flush=True)
            results[label] = {"mean": float(mean_r), "std": float(std_r),
                             "returns": [float(r) for r in returns]}

    # Save
    summary = {"source_return": float(src_r), "target_no_adapt": float(tgt_r_naive),
               "results": results}
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.json_out, "w"), indent=2)
    print(f"\n  Saved to {args.json_out}")


if __name__ == "__main__":
    main()
