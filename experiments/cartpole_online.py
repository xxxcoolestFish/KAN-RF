"""CartPole Online Continual Learning: gravity switch 9.8→15.0."""
import torch, numpy as np, time, sys
from collections import deque
sys.path.insert(0, '.')
from kanrf import KAN
from control.kan_policy_net import KANPolicy

# CartPole with configurable gravity
G_DEFAULT = 9.8; MC=1.0; MP=0.1; DT=0.02; TOTAL=MC+MP; FM=10.0
X_S=2.5; XD_S=3.0; TH_S=0.3; THD_S=3.0

def step_cp(state, a_norm, g=G_DEFAULT):
    """CartPole step. state: [x,xd,theta,thd] raw. a_norm ∈ [-1,1]."""
    x,xd,th,thd = state[:,0],state[:,1],state[:,2],state[:,3]
    force = a_norm * FM
    PML = MP * 0.5  # pole_mass * length
    costh, sinth = torch.cos(th), torch.sin(th)
    temp = (force + PML*thd**2*sinth) / TOTAL
    denom = 0.5 * (4/3 - MP*costh**2/TOTAL)
    th_acc = (g*sinth - costh*temp) / (denom + 1e-8)
    x_acc = temp - PML*th_acc*costh/TOTAL
    xd_n = xd + x_acc*DT; thd_n = thd + th_acc*DT
    x_n = x + xd_n*DT; th_n = th + thd_n*DT
    return torch.stack([x_n, xd_n, th_n, thd_n], dim=-1)

def normalize(s_raw):
    s = s_raw.clone()
    s[:,0]/=X_S; s[:,1]/=XD_S; s[:,2]/=TH_S; s[:,3]/=THD_S
    return s

device = torch.device('cpu')

# Load models
wm = KAN([5,12,4],grid_size=5,spline_order=3)
wm.load_state_dict(torch.load('/tmp/kanrf_cl_cp/cartpole_kan_cws.pt',weights_only=True))
wm.to(device); wm.eval()
pol = KANPolicy(4,1,12,2)
pol.load_state_dict(torch.load('/tmp/cartpole_kan_policy.pt',weights_only=True))
pol.to(device); pol.eval()

# Online learning buffers
buf_x = deque(maxlen=200); buf_y = deque(maxlen=200)
pol_losses = deque(maxlen=200)
pol_opt = torch.optim.Adam(pol.parameters(), lr=2e-4)
switch_step = 800; total_steps = 2000; g = G_DEFAULT

print(f"CartPole Online: g=9.8→15.0 at step {switch_step}")
survivals = []; ep_step = 0
s_raw = torch.tensor([[0.,0.,0.05,0.]], dtype=torch.float32)

for step in range(total_steps):
    if step == switch_step:
        g = 15.0
        print(f"\n  *** g=9.8 → 15.0 ***\n")

    s_n = normalize(s_raw)
    with torch.no_grad(): a = pol(s_n).item()
    s_raw = step_cp(s_raw, torch.tensor([a]), g); ep_step += 1

    # WM online update
    sn, st = s_n.squeeze(0), normalize(s_raw).squeeze(0)
    x_t = torch.cat([sn.unsqueeze(0), torch.tensor([[a]])], -1)
    y_t = st.unsqueeze(0)
    wm.train(); pred = wm(x_t); wm_loss = torch.nn.functional.mse_loss(pred, y_t)
    wm_loss.backward()
    with torch.no_grad():
        for p in wm.parameters():
            if p.grad is not None: p -= 5e-4 * p.grad; p.grad.zero_()
    wm.eval()

    # Policy online update (via buffer)
    buf_x.append(x_t.detach()); buf_y.append(y_t.detach())
    if len(buf_x) >= 16:
        idx = np.random.choice(len(buf_x), min(32,len(buf_x)))
        xb = torch.cat([buf_x[i][:,:4] for i in idx])
        ab = torch.cat([buf_x[i][:,4:] for i in idx])
        pol.train(); pol_opt.zero_grad()
        ap = pol(xb); sp = wm(torch.cat([xb, ap], -1))
        loss = sp[:,2].pow(2).mean() + 0.1*sp[:,0].pow(2).mean() + 0.01*ap.pow(2).mean()
        loss.backward(); pol_opt.step(); pol.eval()
        pol_losses.append(loss.item())

    # Episode check
    th, x = s_raw[0,2].item(), s_raw[0,0].item()
    if abs(th) > 0.21 or abs(x) > 2.4 or ep_step >= 500:
        survivals.append(ep_step); ep_step = 0
        s_raw = torch.tensor([[0.,0.,np.random.uniform(-0.05,0.05),0.]])

    if step % 150 == 0 or step == switch_step + 10:
        pl = np.mean(list(pol_losses)[-50:]) if pol_losses else 0
        sr = np.mean([s >= 500 for s in survivals[-5:]]) * 100 if survivals else 0
        print(f"  Step {step:4d}  g={g:.1f}  pol_loss={pl:.4f}  recent_sr={sr:.0f}%")

# Report
n = len(survivals)
pre = survivals[:n//3]; mid = survivals[n//3:2*n//3]; late = survivals[2*n//3:]
print(f"\n  Episodes: {n}")
print(f"  g=9.8 (early): mean={np.mean(pre):.0f} sr={np.mean([s>=500 for s in pre])*100:.0f}%")
print(f"  g=15  (mid):   mean={np.mean(mid):.0f} sr={np.mean([s>=500 for s in mid])*100:.0f}%")
print(f"  g=15  (late):  mean={np.mean(late):.0f} sr={np.mean([s>=500 for s in late])*100:.0f}%")
