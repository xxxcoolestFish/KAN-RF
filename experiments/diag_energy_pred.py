"""Diagnose: compare energy predictions of ProtoKAN vs CWS KAN WM."""
import torch, numpy as np, gymnasium as gym, sys
sys.path.insert(0, '.')
from kanrf import ProtoKAN, KAN

proto = ProtoKAN([4, 12, 3], n_prototypes=16)
proto.load_state_dict(torch.load('/tmp/protokAN_wm_pendulum.pt', weights_only=True))
proto.eval()

kan = KAN([4, 12, 3], grid_size=5, spline_order=3)
kan.load_state_dict(torch.load('/tmp/kanrf_cl_exp/kan_cws_cl.pt', weights_only=True))
kan.eval()

env = gym.make('Pendulum-v1')
env.reset()

print("Energy prediction from hanging (E_current=-10):")
print(f"{'Action':>8s}  {'True':>8s}  {'Proto':>8s}  {'KAN':>8s}  {'True_dE':>8s}  {'Proto_dE':>8s}  {'KAN_dE':>8s}")

for a_norm in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    theta, thd = -np.pi/2, 0.0
    s_norm = np.array([np.cos(theta), np.sin(theta), thd/8.0], dtype=np.float32)
    E_cur = 0.5*thd**2 + 10.0*np.sin(theta)

    env.unwrapped.state = (theta, thd)
    obs, _, _, _, _ = env.step([a_norm * 2.0])
    true_E = 0.5*obs[2]**2 + 10.0*obs[1]

    s_t = torch.tensor([s_norm], dtype=torch.float32)
    a_t = torch.tensor([[a_norm]], dtype=torch.float32)
    with torch.no_grad():
        sp_p = proto(torch.cat([s_t, a_t], dim=-1)).squeeze(0)
        sp_k = kan(torch.cat([s_t, a_t], dim=-1)).squeeze(0)
    proto_E = 0.5*(sp_p[2].item()*8.0)**2 + 10.0*sp_p[1].item()
    kan_E = 0.5*(sp_k[2].item()*8.0)**2 + 10.0*sp_k[1].item()

    print(f"{a_norm:8.1f}  {true_E:8.4f}  {proto_E:8.4f}  {kan_E:8.4f}  "
          f"{true_E-E_cur:8.4f}  {proto_E-E_cur:8.4f}  {kan_E-E_cur:8.4f}")

# Bulk test
print("\nBulk energy error across 500 random states:")
proto_errs = []; kan_errs = []
for _ in range(500):
    theta = np.random.uniform(-np.pi, np.pi)
    thd = np.random.uniform(-8.0, 8.0)
    a_norm = np.random.uniform(-1, 1)
    s_norm = np.array([np.cos(theta), np.sin(theta), thd/8.0], dtype=np.float32)
    E_cur = 0.5*thd**2 + 10.0*np.sin(theta)

    env.unwrapped.state = (theta, thd)
    obs, _, _, _, _ = env.step([a_norm * 2.0])
    true_E = 0.5*obs[2]**2 + 10.0*obs[1]

    s_t = torch.tensor([s_norm], dtype=torch.float32)
    a_t = torch.tensor([[a_norm]], dtype=torch.float32)
    with torch.no_grad():
        sp_p = proto(torch.cat([s_t, a_t], dim=-1)).squeeze(0)
        sp_k = kan(torch.cat([s_t, a_t], dim=-1)).squeeze(0)
    proto_E = 0.5*(sp_p[2].item()*8.0)**2 + 10.0*sp_p[1].item()
    kan_E = 0.5*(sp_k[2].item()*8.0)**2 + 10.0*sp_k[1].item()

    proto_errs.append(abs(proto_E - true_E))
    kan_errs.append(abs(kan_E - true_E))

env.close()
print(f"  ProtoKAN WM mean |E_error|: {np.mean(proto_errs):.4f}")
print(f"  CWS KAN WM  mean |E_error|: {np.mean(kan_errs):.4f}")
print(f"  ProtoKAN is {'better' if np.mean(proto_errs) < np.mean(kan_errs) else 'worse'}")

# Also check: gradient of energy w.r.t. action
print("\nEnergy gradient dE/da test (Jacobian in energy space):")
cos_sims_p = []; cos_sims_k = []
for _ in range(100):
    theta = np.random.uniform(-np.pi, np.pi)
    thd = np.random.uniform(-8.0, 8.0)
    s_norm = np.array([np.cos(theta), np.sin(theta), thd/8.0], dtype=np.float32)

    # True dE/da via finite difference
    eps = 0.01
    env.unwrapped.state = (theta, thd)
    obs_p, _, _, _, _ = env.step([eps*2.0])
    E_p = 0.5*obs_p[2]**2 + 10.0*obs_p[1]
    env.unwrapped.state = (theta, thd)
    obs_m, _, _, _, _ = env.step([-eps*2.0])
    E_m = 0.5*obs_m[2]**2 + 10.0*obs_m[1]
    true_dE = (E_p - E_m) / (2*eps)

    # WM dE/da via autograd
    s_t = torch.tensor([s_norm], dtype=torch.float32)
    a_t = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)

    sp = proto(torch.cat([s_t, a_t], dim=-1)).squeeze(0)
    E_p = 0.5*(sp[2]*8.0)**2 + 10.0*sp[1]
    proto_dE = torch.autograd.grad(E_p, a_t)[0].item()

    sp = kan(torch.cat([s_t, a_t], dim=-1)).squeeze(0)
    E_k = 0.5*(sp[2]*8.0)**2 + 10.0*sp[1]
    kan_dE = torch.autograd.grad(E_k, a_t)[0].item()

    # Sign agreement
    if true_dE * proto_dE > 0:
        cos_sims_p.append(1.0)
    else:
        cos_sims_p.append(-1.0)
    if true_dE * kan_dE > 0:
        cos_sims_k.append(1.0)
    else:
        cos_sims_k.append(-1.0)

env.close()
print(f"  ProtoKAN WM dE/da sign correct: {np.mean(cos_sims_p)*100:.0f}%")
print(f"  CWS KAN WM  dE/da sign correct: {np.mean(cos_sims_k)*100:.0f}%")
