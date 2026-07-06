"""Multi-step rollout accuracy: ProtoKAN vs KAN WM on CartPole."""
import torch, numpy as np, sys
sys.path.insert(0, '.')
from kanrf import ProtoKAN, KAN
from experiments.cartpole_protokAN_compare import step_cartpole, X_S, XD_S, TH_S, THD_S

device = 'cpu'
proto_wm = ProtoKAN([5, 16, 4], n_prototypes=16)
proto_wm.load_state_dict(torch.load('/tmp/cartpole_proto_wm.pt', weights_only=True))
proto_wm.eval()
kan_wm = KAN([5, 16, 4], grid_size=5, spline_order=3)
kan_wm.load_state_dict(torch.load('/tmp/cartpole_kan_wm.pt', weights_only=True))
kan_wm.eval()

np.random.seed(42)
H = 20
n_traj = 200
proto_errors = [[] for _ in range(H)]
kan_errors = [[] for _ in range(H)]

for _ in range(n_traj):
    x = np.random.uniform(-1.0, 1.0)
    xd = np.random.uniform(-1.0, 1.0)
    th = np.random.uniform(-0.2, 0.2)
    thd = np.random.uniform(-1.0, 1.0)

    s_true = torch.tensor([[x, xd, th, thd]], dtype=torch.float32)
    s_norm_p = s_true.clone()
    s_norm_p[:, 0] /= X_S; s_norm_p[:, 1] /= XD_S
    s_norm_p[:, 2] /= TH_S; s_norm_p[:, 3] /= THD_S
    s_norm_k = s_norm_p.clone()

    for h in range(H):
        a = np.random.uniform(-1, 1)
        a_t = torch.tensor([a], dtype=torch.float32)

        # True
        s_true = step_cartpole(s_true, a_t)

        # ProtoKAN
        with torch.no_grad():
            sp = proto_wm(torch.cat([s_norm_p, a_t.unsqueeze(0)], dim=-1))
        s_norm_p = sp
        sp_raw = sp.clone()
        sp_raw[:, 0] *= X_S; sp_raw[:, 1] *= XD_S
        sp_raw[:, 2] *= TH_S; sp_raw[:, 3] *= THD_S
        proto_errors[h].append((sp_raw - s_true).pow(2).mean().item())

        # KAN
        with torch.no_grad():
            sk = kan_wm(torch.cat([s_norm_k, a_t.unsqueeze(0)], dim=-1))
        s_norm_k = sk
        sk_raw = sk.clone()
        sk_raw[:, 0] *= X_S; sk_raw[:, 1] *= XD_S
        sk_raw[:, 2] *= TH_S; sk_raw[:, 3] *= THD_S
        kan_errors[h].append((sk_raw - s_true).pow(2).mean().item())

print("Multi-step rollout MSE (raw state space):")
print(f"{'Step':>5s}  {'ProtoKAN':>10s}  {'KAN':>10s}  {'Ratio':>10s}")
for h in [0, 4, 9, 14, 19]:
    pe = np.mean(proto_errors[h])
    ke = np.mean(kan_errors[h])
    print(f"{h+1:5d}  {pe:10.6f}  {ke:10.6f}  {ke/pe:10.2f}x")

pe_all = np.mean([np.mean(e) for e in proto_errors])
ke_all = np.mean([np.mean(e) for e in kan_errors])
print(f"\nOverall avg over {H} steps:")
print(f"  ProtoKAN: {pe_all:.6f}")
print(f"  KAN:      {ke_all:.6f}")
print(f"  ProtoKAN is {ke_all/pe_all:.1f}x more accurate in multi-step rollouts")
