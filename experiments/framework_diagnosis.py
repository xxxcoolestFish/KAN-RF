"""
Framework Diagnosis: Systematic analysis of design flaws in CDPN framework.

Theoretical hypotheses tested:
  A. Abstract Dynamics Fidelity Gap: Control term has systematic bias (11x in unsaturated regime)
  B. Execute Transfer Function Nonlinearity: Clamping creates nonlinearity that abstract model cannot capture
  C. Bridge Information Bottleneck: 12-dim h -> 3 scalars loses critical information
  D. Jacobian Spatial Variation: Global J_mean != J(s) for many states
  E. Gradient Misalignment: Abstract dynamics gradient direction differs from true gradient
  F. Failure Mode Classification: Systematic patterns in the 20% failures
"""
import torch, torch.nn as nn, numpy as np, time, sys, os, gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN, ProtoKANLayer
from control.cdpn import (discover_tier0, CausalDecomposedPolicy, Execute,
    CausalBridge, AbstractPendulumDynamics, AbstractPlannerTrainer,
    CognitiveTrainer, AdaptivePolicy, make_env_params, train_cognitive_head)

PI_2 = np.pi / 2
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])
G = 10.0; DEVICE = 'cpu'
torch.manual_seed(42); np.random.seed(42)

# ---- Real Pendulum Dynamics (Gymnasium compatible) ----
def real_pendulum_step(s, a, g=G):
    """Real pendulum physics. s: (...,3) [cos(theta), sin(theta), thetadot/8]"""
    cos_th, sin_th = s[..., 0], s[..., 1]
    thd = s[..., 2] * 8.0
    u = a[..., 0].clamp(-1, 1) * 2.0
    thd_new = thd + (3.0 * g / 2.0 * sin_th + 3.0 * u) * 0.05
    thd_new = thd_new.clamp(-8.0, 8.0)
    th = torch.atan2(sin_th, cos_th)
    th_new = th + thd_new * 0.05
    return torch.stack([torch.cos(th_new), torch.sin(th_new), thd_new / 8.0], dim=-1)

def pendulum_with_execute(s, v_des, execute, g=G):
    """v_des -> Execute -> a -> real dynamics -> s_next"""
    a = execute(v_des, s)
    return real_pendulum_step(s, a, g)

# ---- Phase 0: Setup ----
def setup_master():
    from experiments.baseline_sweep import generate_pendulum_data, train_wm as twm
    from experiments.baseline_sweep import generate_policy_states as gps
    print("="*70); print("PHASE 0: SETUP"); print("="*70)
    X, Y = generate_pendulum_data(5000, seed=42)
    wm, wm_val = twm(X.to(DEVICE), Y.to(DEVICE))
    print(f"  WM val_mse = {wm_val:.6f}")
    s_pol = gps(10000, seed=42).to(DEVICE)
    tier0, mask, jac_norms, thresh = discover_tier0(wm, 3, device=DEVICE)
    print(f"  Tier 0: {[['cos','sin','thd'][i] for i in tier0]}")
    bridge = CausalBridge(wm, 3, tier0, s_pol, device=DEVICE, g_true=G)
    execute = Execute(wm, 3, tier0, s_pol, damping=0.1, device=DEVICE, bridge=bridge)
    print(f"  max_delta = {bridge.max_delta.cpu().tolist()}")
    print(f"  a_fit = {bridge.a_fit:.2f}")
    print(f"  J = {execute.J.squeeze().cpu().tolist()}")
    return wm, bridge, execute, s_pol, tier0

# ---- EXPERIMENT A: Execute Transfer Nonlinearity ----
def experiment_A(execute, bridge):
    print("\n"+"="*70); print("EXPERIMENT A: Execute Transfer Nonlinearity"); print("="*70)
    v_grid = torch.linspace(-1.0, 1.0, 201).unsqueeze(1)
    a_eval = execute(v_grid, None)
    ctrl_abs = v_grid * bridge.max_delta[0].item() * 8.0
    s_base = torch.tensor([[0.0, 1.0, 0.0]]).expand(v_grid.shape[0], -1)
    s_pass = real_pendulum_step(s_base, torch.zeros_like(v_grid))
    s_act = real_pendulum_step(s_base, a_eval)
    ctrl_real = (s_act[:, 2]*8.0 - s_base[:, 2]*8.0) - (s_pass[:, 2]*8.0 - s_base[:, 2]*8.0)
    print(f"\n  {'v_des':>8} {'a':>8} {'abs_abstract':>14} {'abs_real':>12} {'Ratio':>8}")
    print(f"  {'-'*48}")
    for vi in [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]:
        vi = min(vi, len(v_grid)-1)
        ra = abs(ctrl_real[vi].item()) / (abs(ctrl_abs[vi,0].item()) + 1e-10)
        print(f"  {v_grid[vi,0].item():+8.4f} {a_eval[vi,0].item():+8.4f} {abs(ctrl_abs[vi,0].item()):14.6f} {abs(ctrl_real[vi].item()):12.6f} {ra:8.2f}x")
    # Small-signal analysis
    sat_th = None
    for i in range(len(v_grid)):
        if abs(a_eval[i,0].item()) >= 1.99:
            sat_th = sat_th or abs(v_grid[i,0].item())
    unsat = v_grid.abs() < (sat_th or 0.1)
    if unsat.any():
        sr = (ctrl_real[unsat.squeeze()].abs() / (ctrl_abs[unsat.squeeze()].abs()+1e-10))
        print(f"\n  Small-signal (|v|<{sat_th:.4f}): mean ratio = {sr.mean().item():.2f}x  +/- {sr.std().item():.2f}x")
        print(f"  Theoretical prediction: ~11.2x")
        print(f"  {'CONFIRMED: Abstract dynamics underestimates control by ~10x near goal' if sr.mean().item() > 5 else 'Ratio differs from prediction'}")
    else:
        print("  (All points saturated)")
    return {'ratio': sr.mean().item() if unsat.any() else None}

# ---- EXPERIMENT B: Abstract Dynamics Fidelity Map ----
def experiment_B(wm, bridge, execute):
    print("\n"+"="*70); print("EXPERIMENT B: Abstract Dynamics Fidelity Map"); print("="*70)
    abstract = AbstractPendulumDynamics(bridge, S_TARGET)
    n_th, n_td, n_v = 20, 12, 10
    thetas = torch.linspace(-np.pi, np.pi, n_th)
    thds = torch.linspace(-8.0, 8.0, n_td)
    v_grid = torch.linspace(-1.0, 1.0, n_v)
    errs, agrees, info = [], [], []
    for th in thetas:
        for td in thds:
            s = torch.tensor([[np.cos(th.item()), np.sin(th.item()), td.item()/8.0]])
            for vv in v_grid:
                vt = torch.tensor([[vv.item()]])
                sp_abs = abstract.predict_next(s, vt)
                sp_real = pendulum_with_execute(s, vt, execute)
                err = (sp_real - sp_abs).pow(2).sum().item()
                errs.append(err)
                da = (sp_abs[0,2] - s[0,2]).item()
                dr = (sp_real[0,2] - s[0,2]).item()
                agree = 1 if da*dr >= 0 else 0
                agrees.append(agree); info.append({'th':th.item(),'td':td.item(),'v':vv.item(),'err':err})
    errs = np.array(errs); agrees = np.array(agrees)
    p50, p90, p95 = np.percentile(errs, [50, 90, 95])
    print(f"\n  N={n_th*n_td*n_v}, Direction agreement: {agrees.mean()*100:.1f}%")
    print(f"  Prediction MSE: median={p50:.6f}  p90={p90:.6f}  p95={p95:.6f}")
    sv = [s for s in info if abs(s['v'])<0.1]; lv = [s for s in info if abs(s['v'])>=0.1]
    if sv: print(f"  |v|<0.1: MSE={np.mean([s['err'] for s in sv]):.6f}")
    if lv: print(f"  |v|>=0.1: MSE={np.mean([s['err'] for s in lv]):.6f}")
    return {'errs': errs, 'agrees': agrees}

# ---- EXPERIMENT C: Bridge Information Bottleneck ----
def experiment_C(wm, bridge, execute, s_pol):
    print("\n"+"="*70); print("EXPERIMENT C: Bridge Information Bottleneck"); print("="*70)
    N = 2000
    h_l, p_l, d_l = [], [], []
    for _ in range(N):
        idx = torch.randint(0, len(s_pol), (1,))
        s = s_pol[idx].to(DEVICE)
        v = torch.rand(1, 1, device=DEVICE) * 2 - 1
        with torch.no_grad():
            a = execute(v, s)
            h = wm.layers[0](torch.cat([s, a], dim=-1))
            s_next = wm(torch.cat([s, a], dim=-1))
        ep = make_env_params(bridge, 1, DEVICE)
        h_l.append(h.squeeze(0)); p_l.append(ep.squeeze(0)); d_l.append((s_next-s).squeeze(0))
    H = torch.stack(h_l); P = torch.stack(p_l); D = torch.stack(d_l)
    mh = nn.Sequential(nn.Linear(12,32),nn.SiLU(),nn.Linear(32,3)).to(DEVICE)
    mp = nn.Sequential(nn.Linear(3,32),nn.SiLU(),nn.Linear(32,3)).to(DEVICE)
    mse = nn.MSELoss()
    oh = torch.optim.Adam(mh.parameters(), lr=1e-3)
    op = torch.optim.Adam(mp.parameters(), lr=1e-3)
    for _ in range(500):
        perm = torch.randperm(N)
        for i in range(0, N, 128):
            idx = perm[i:i+128]
            oh.zero_grad(); mse(mh(H[idx]),D[idx]).backward(); oh.step()
            op.zero_grad(); mse(mp(P[idx]),D[idx]).backward(); op.step()
    with torch.no_grad():
        mse_h = mse(mh(H), D).item()
        mse_p = mse(mp(P), D).item()
    print(f"\n  h (12-dim) -> delta_s decoder MSE: {mse_h:.8f}")
    print(f"  env_params (3-dim) -> delta_s decoder MSE: {mse_p:.8f}")
    print(f"  Ratio: {mse_p/(mse_h+1e-12):.2f}x")
    return {'mse_h':mse_h, 'mse_p':mse_p}

# ---- EXPERIMENT D: Jacobian Spatial Variation ----
def experiment_D(wm, s_pol, tier0):
    print("\n"+"="*70); print("EXPERIMENT D: Jacobian Spatial Variation"); print("="*70)
    wm.eval()
    was_frozen = not next(wm.parameters()).requires_grad
    if was_frozen:
        for p in wm.parameters(): p.requires_grad = True
    N = min(500, len(s_pol))
    idx = torch.randperm(len(s_pol))[:N]
    ss = s_pol[idx].clone().to(DEVICE)
    jv = []
    for i in range(N):
        s = ss[i:i+1]; a = torch.zeros(1,1,device=DEVICE,requires_grad=True)
        sp = wm(torch.cat([s,a], dim=-1))
        for j, dim in enumerate(tier0):
            g = torch.autograd.grad(sp[0,dim], a, retain_graph=True)[0]
            jv.append(g[0,0].abs().item())
    if was_frozen:
        for p in wm.parameters(): p.requires_grad = False
    jv = np.array(jv)
    mean_j, std_j = np.mean(jv), np.std(jv)
    cv = std_j / (mean_j + 1e-12)
    print(f"\n  N={N}, Mean J={mean_j:.6f}, Std={std_j:.6f}")
    print(f"  CV = {cv:.2%}")
    p5, p95 = np.percentile(jv, [5, 95])
    print(f"  P5={p5:.6f}, P95={p95:.6f}, P95/P5={p95/(p5+1e-12):.1f}x")
    print(f"  {'CRITICAL: J varies >30% across states' if cv > 0.3 else 'J variation moderate'}")
    return {'jv': jv, 'cv': cv}

# ---- EXPERIMENT E: Gradient Alignment ----
def experiment_E(wm, bridge, execute, s_pol):
    print("\n"+"="*70); print("EXPERIMENT E: Gradient Alignment"); print("="*70)
    abstract = AbstractPendulumDynamics(bridge, S_TARGET.to(DEVICE))
    wm.eval()
    for p in wm.parameters(): p.requires_grad = False
    N = 200
    idx = torch.randperm(len(s_pol))[:N]
    sb = s_pol[idx].clone().to(DEVICE)
    coss = []
    for i in range(N):
        s = sb[i:i+1].clone().detach()
        v = torch.randn(1,1,device=DEVICE,requires_grad=True) * 0.5
        st = S_TARGET.to(DEVICE)
        la = (abstract.predict_next(s,v) - st).pow(2).sum()
        ga = torch.autograd.grad(la, v, retain_graph=True)[0].detach().clone()
        v.grad = None
        a = execute(v, s)
        lr = (real_pendulum_step(s,a) - st).pow(2).sum()
        gr = torch.autograd.grad(lr, v, retain_graph=True)[0].detach().clone()
        coss.append(((ga/(ga.norm()+1e-10))*(gr/(gr.norm()+1e-10))).sum().item())
    coss = np.array(coss)
    print(f"\n  N={N}")
    print(f"  Cosine similarity: mean={np.mean(coss):.3f} +/- {np.std(coss):.3f}")
    print(f"  Median={np.median(coss):.3f}, %>0.8={(coss>0.8).mean()*100:.0f}%")
    print(f"  % WRONG DIRECTION (<0): {(coss<0).mean()*100:.0f}%")
    return {'coss': coss}

# ---- EXPERIMENT F: Failure Mode Classification ----
def experiment_F(wm, bridge, execute, s_pol, tier0):
    print("\n"+"="*70); print("EXPERIMENT F: Failure Mode Analysis"); print("="*70)
    print("\n  Training policy...")
    with torch.no_grad():
        h_goal = wm.layers[0](torch.cat([S_TARGET.to(DEVICE), torch.zeros(1,1,device=DEVICE)], dim=-1))
    head = train_cognitive_head(wm, bridge, execute, s_pol, n_epochs=300, device=DEVICE)
    policy = AdaptivePolicy(h_dim=12, e_dim=3, hidden=24, n_layers=2).to(DEVICE)
    trainer = CognitiveTrainer(wm, bridge, execute, policy, head, h_goal, dr_range=(5.,25.), device=DEVICE)
    for ep in range(1, 201):
        ld = trainer.train_epoch(s_pol, H=5, batch_size=128)
        if ep % 50 == 0: print(f"    Epoch {ep:4d}  loss={ld['total']:.4f}")
    print("\n  Evaluating 50 trials...")
    n_trials = 50; trials = []
    for trial in range(n_trials):
        env = gym.make('Pendulum-v1')
        obs, _ = env.reset(seed=42 + trial*100)
        success = False; ngt = 0; overshoot = 0; traj = []
        for step in range(300):
            sn = np.array([obs[0], obs[1], obs[2]/8.0], dtype=np.float32)
            a = trainer.get_action(sn)
            obs, _, _, _, _ = env.step([a*2.0])
            err = min(abs(np.arctan2(obs[1],obs[0])-PI_2), 2*np.pi-abs(np.arctan2(obs[1],obs[0])-PI_2))
            traj.append(err)
            if err < 0.3: ngt += 1
            if err < 0.2: success = True; break
            if len(traj)>2 and traj[-1]<traj[-2] and traj[-2]<0.3:
                overshoot += 1
        env.close()
        if success: ft = 'success'
        elif np.mean(traj[-20:]) if len(traj)>=20 else 10 > 2.0: ft = 'stuck_bottom'
        elif ngt > 20: ft = 'unstable_balance'
        elif overshoot > 3: ft = 'overshoot'
        else: ft = 'unknown'
        trials.append({'success':success,'type':ft,'traj':traj,'ngt':ngt})
    succ = sum(1 for t in trials if t['success'])
    from collections import Counter
    ft = Counter(t['type'] for t in trials if not t['success'])
    print(f"\n  {succ}/{n_trials} ({succ*100/n_trials:.0f}%) successful")
    for ftn, cnt in ft.most_common():
        print(f"    {ftn}: {cnt}/{(n_trials-succ)}")
    return {'succ': succ, 'ft': dict(ft)}

# ---- EXPERIMENT ORACLE: True Dynamics Upper Bound ----
def experiment_oracle(s_pol):
    print("\n"+"="*70); print("EXPERIMENT ORACLE: True dynamics training (upper bound)"); print("="*70)
    policy = nn.Sequential(nn.Linear(6,32),nn.Tanh(),nn.Linear(32,24),nn.Tanh(),nn.Linear(24,1),nn.Tanh()).to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    for ep in range(1, 301):
        N = len(s_pol); nb = max(1, N//128); tl = 0.0
        for _ in range(nb):
            idx = torch.randint(0, N, (128,))
            sc = s_pol[idx].to(DEVICE)
            policy.train(); opt.zero_grad(); loss = 0.0
            for t in range(8):
                v = policy(torch.cat([sc, S_TARGET.expand(sc.shape[0],-1)], dim=-1))
                sc = real_pendulum_step(sc, v)
                loss += (0.9**t)*(sc-S_TARGET.to(DEVICE)).pow(2).sum(dim=-1).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(),10.); opt.step(); tl+=loss.item()
        if ep % 100 == 0: print(f"    Epoch {ep:4d}  loss={tl/nb:.4f}")
    env = gym.make('Pendulum-v1')
    succ = 0; steps = []
    for trial in range(10):
        obs, _ = env.reset(seed=42+trial*100)
        ok = False
        for step in range(300):
            sn = torch.tensor([[obs[0], obs[1], obs[2]/8.0]], dtype=torch.float32)
            with torch.no_grad():
                v = policy(torch.cat([sn, S_TARGET], dim=-1))
            obs, _, _, _, _ = env.step([v.item()*2.0])
            err = min(abs(np.arctan2(obs[1],obs[0])-PI_2), 2*np.pi-abs(np.arctan2(obs[1],obs[0])-PI_2))
            if err < 0.2: succ+=1; steps.append(step+1); ok=True; break
        if not ok: steps.append(300)
    env.close()
    print(f"\n  ORACLE RESULT: {succ}/10")
    print(f"  If oracle < 10/10: problem is deeper than abstract dynamics (policy capacity, H, eval)")
    print(f"  If oracle = 10/10 but our framework < 10/10: framework IS the bottleneck")
    return {'succ':succ,'steps':steps}

# ---- MAIN ----
def main():
    print("="*70); print("FRAMEWORK DIAGNOSIS"); print("="*70)
    wm, bridge, execute, s_pol, tier0 = setup_master()
    ra = experiment_A(execute, bridge)
    rb = experiment_B(wm, bridge, execute)
    rc = experiment_C(wm, bridge, execute, s_pol)
    rd = experiment_D(wm, s_pol, tier0)
    re = experiment_E(wm, bridge, execute, s_pol)
    rf = experiment_F(wm, bridge, execute, s_pol, tier0)
    orc = experiment_oracle(s_pol)
    print("\n"+"="*70); print("SYNTHESIS"); print("="*70)
    print(f"\n  A: abstract/real ratio = {ra.get('ratio','?'):.2f}x")
    print(f"  B: direction agreement = {np.mean(rb['agrees'])*100:.1f}%")
    print(f"  C: Bridge ratio = {rc['mse_p']/(rc['mse_h']+1e-12):.2f}x")
    print(f"  D: Jacobian CV = {rd['cv']:.2%}")
    print(f"  E: Gradient cos = {np.mean(re['coss']):.3f}")
    print(f"  F: Failures = {rf['ft']}")
    print(f"  ORACLE = {orc['succ']}/10")
    print(f"\n  KEY: IF ORACLE > CURRENT BEST -> FRAMEWORK IS THE BOTTLENECK")
    print(f"       IF ORACLE == CURRENT BEST -> PENDULUM TASK IS THE LIMIT")

if __name__ == '__main__':
    main()
