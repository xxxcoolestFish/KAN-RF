"""CartPole CDPN v4: WM-gradient training with auto-damped Execute.

Key innovation: Policy trained through differentiable ProtoKAN WM gradient
(no abstract dynamics). Execute uses auto-damping (max_gain=2.0) to prevent
Bang-Bang saturation. Policy in h-space with Bridge-provided env_params.
"""
import torch, torch.nn as nn, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
from control.cdpn import (discover_tier0, Execute, CausalBridge,
    make_env_params)
from experiments.cartpole_continual import (
    step_cartpole, generate_wm_data, generate_policy_states,
    X_S, XD_S, TH_S, THD_S, G_DEF)

DEVICE = "cpu"
S_TARGET = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
torch.manual_seed(42); np.random.seed(42)

class CartpolePolicy(nn.Module):
    """Policy: (h:16, env_params:3, h_goal:16) -> v_des:(tier0_dim,)."""
    def __init__(self, h_dim=16, e_dim=3, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim+e_dim+h_dim, 32), nn.Tanh(),
            nn.Linear(32, 24), nn.Tanh(),
            nn.Linear(24, out_dim), nn.Tanh())
    def forward(self, h, env_p, h_g):
        return self.net(torch.cat([h,env_p,h_g], dim=-1))

def setup():
    print("="*60); print("SETUP"); print("="*60)
    X,Y=generate_wm_data(g=G_DEF,n=5000,device=DEVICE)
    n_tr=int(len(X)*0.85)
    wm=ProtoKAN([5,16,4],n_prototypes=16).to(DEVICE)
    for l in wm.layers: l.log_sigma.data.fill_(-1.5)
    mse=nn.MSELoss()
    opt=torch.optim.LBFGS(wm.parameters(),lr=1.0,max_iter=20,
        history_size=50,line_search_fn="strong_wolfe")
    bv=float("inf");bs=None
    def c():
        opt.zero_grad();l=mse(wm(X[:n_tr]),Y[:n_tr]);l.backward();return l
    for _ in range(100):
        opt.step(c)
        v=mse(wm(X[n_tr:]),Y[n_tr:]).item()
        if v<bv: bv=v; bs={k:v.clone() for k,v in wm.state_dict().items()}
    wm.load_state_dict(bs);wm.eval()
    print(f"  WM val_mse = {bv:.6f}")
    s_pol=generate_policy_states(15000,device=DEVICE)
    tier0,_,jn,_=discover_tier0(wm,4,device=DEVICE)
    names=['x','xd','th','thd']
    print(f"  Tier 0: {[names[i] for i in tier0]}")
    bridge=CausalBridge(wm,4,tier0,s_pol,device=DEVICE)
    execute=Execute(wm,4,tier0,s_pol,damping=0.1,
        device=DEVICE,bridge=bridge,max_gain=2.0)
    print(f"  Execute J = {execute.J.squeeze().cpu().tolist()}")
    return wm,bridge,execute,s_pol,tier0

def train(wm,bridge,execute,s_pol,tier0,n_epochs=200):
    print("\n"+"="*60); print("WM-GRADIENT TRAINING (BPTT H=8)"); print("="*60)
    with torch.no_grad():
        h_goal=wm.layers[0](torch.cat(
            [S_TARGET.to(DEVICE),torch.zeros(1,1,device=DEVICE)],dim=-1))
    policy=CartpolePolicy(h_dim=16,e_dim=len(tier0)+1,out_dim=len(tier0)).to(DEVICE)
    opt=torch.optim.Adam(policy.parameters(),lr=3e-3)
    wm.eval()
    for p in wm.parameters(): p.requires_grad=False
    for ep in range(1,n_epochs+1):
        N=len(s_pol);nb=max(1,N//128);tl=0.0
        for _ in range(nb):
            idx=torch.randint(0,N,(128,));s_cur=s_pol[idx].to(DEVICE)
            env_p=make_env_params(bridge,128,DEVICE)
            policy.train();opt.zero_grad()
            total_loss=torch.tensor(0.0,device=DEVICE)
            for t in range(8):
                with torch.no_grad():
                    h_cur=wm.layers[0](torch.cat(
                        [s_cur,torch.zeros(128,1,device=DEVICE)],dim=-1))
                v=policy(h_cur,env_p,h_goal.expand(128,-1))
                a=execute(v,s_cur)
                s_pred=wm(torch.cat([s_cur,a],dim=-1))
                h_pred=wm.layers[0](torch.cat(
                    [s_pred,torch.zeros(128,1,device=DEVICE)],dim=-1))
                sl=(h_pred-h_goal.expand(128,-1)).pow(2).sum(dim=-1).mean()
                sl=sl+0.01*a.pow(2).mean()
                total_loss=total_loss+(0.9**t)*sl
                s_cur=s_pred
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),10.0)
            opt.step()
            tl+=total_loss.item()
        if ep%50==0: print(f"  Epoch {ep:4d}  loss={tl/nb:.4f}")
    return policy

def evaluate(wm,bridge,execute,policy,n_trials=20,label=""):
    succ=0;steps=[]
    for t in range(n_trials):
        np.random.seed(42+t*100)
        th=np.random.uniform(-0.05,0.05)
        s=torch.tensor([[0.,0.,th/TH_S,0.]],dtype=torch.float32)
        with torch.no_grad():
            hg=wm.layers[0](torch.cat(
                [S_TARGET.to(DEVICE),torch.zeros(1,1,device=DEVICE)],dim=-1))
            ep=make_env_params(bridge,1,DEVICE)
        for step in range(500):
            with torch.no_grad():
                h=wm.layers[0](torch.cat([s,torch.zeros(1,1)],dim=-1))
                v=policy(h,ep,hg);a=execute(v,s)
            sr=s.clone()
            sr[:,0]*=X_S;sr[:,1]*=XD_S;sr[:,2]*=TH_S;sr[:,3]*=THD_S
            sn=step_cartpole(sr,a,g=G_DEF)
            sn[:,0]/=X_S;sn[:,1]/=XD_S;sn[:,2]/=TH_S;sn[:,3]/=THD_S
            s=sn
            if abs(s[0,2].item()*TH_S)>0.21 or abs(s[0,0].item()*X_S)>2.4: break
        steps.append(step+1)
        if step+1>=500: succ+=1
    print(f"  [{label}] {succ}/{n_trials}  mean_steps={np.mean(steps):.0f}")
    return succ,steps

if __name__=="__main__":
    wm,bridge,execute,s_pol,t0=setup()
    policy=train(wm,bridge,execute,s_pol,t0,n_epochs=200)
    evaluate(wm,bridge,execute,policy,n_trials=20,label="CDPNv4 CartPole")
    print("\nDone.")