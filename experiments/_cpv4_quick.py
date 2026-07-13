import torch,torch.nn as nn,numpy as np,sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
from control.cdpn import (discover_tier0,Execute,CausalBridge,make_env_params)
from experiments.cartpole_continual import (step_cartpole,generate_wm_data,
    generate_policy_states,X_S,XD_S,TH_S,THD_S,G_DEF)
DEVICE="cpu";S_TARGET=torch.tensor([[0.,0.,0.,0.]])
torch.manual_seed(42);np.random.seed(42)

class CPolicy(nn.Module):
    def __init__(self,out_d=1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(35,32),nn.Tanh(),nn.Linear(32,24),nn.Tanh(),nn.Linear(24,out_d),nn.Tanh())
    def forward(self,h,e,hg):return self.net(torch.cat([h,e,hg],dim=-1))

print("SETUP")
X,Y=generate_wm_data(n=5000,device=DEVICE)
nt=int(len(X)*.85)
wm=ProtoKAN([5,16,4],n_prototypes=16).to(DEVICE)
for l in wm.layers:l.log_sigma.data.fill_(-1.5)
mse=nn.MSELoss()
opt=torch.optim.LBFGS(wm.parameters(),lr=1.,max_iter=20,history_size=50,line_search_fn="strong_wolfe")
bv=float("inf");bs=None
def c():opt.zero_grad();l=mse(wm(X[:nt]),Y[:nt]);l.backward();return l
for _ in range(80):
    opt.step(c);v=mse(wm(X[nt:]),Y[nt:]).item()
    if v<bv:bv=v;bs={k:v.clone() for k,v in wm.state_dict().items()}
wm.load_state_dict(bs);wm.eval()
print(f"WM val_mse={bv:.6f}")

sp=generate_policy_states(15000,device=DEVICE)
t0,_,_,_=discover_tier0(wm,4,device=DEVICE)
bridge=CausalBridge(wm,4,t0,sp,device=DEVICE)
executor=Execute(wm,4,t0,sp,damping=.1,device=DEVICE,bridge=bridge,max_gain=2.)
print(f"J={executor.J.squeeze().cpu().tolist()}")

with torch.no_grad():
    hg=wm.layers[0](torch.cat([S_TARGET.to(DEVICE),torch.zeros(1,1,device=DEVICE)],dim=-1))
pol=CPolicy(out_d=len(t0)).to(DEVICE)
op=torch.optim.Adam(pol.parameters(),lr=3e-3)
wm.eval();[p.requires_grad_(False) for p in wm.parameters()]

print("Single-step training...")
for ep in range(1,51):
    N=len(sp);nb=max(1,N//128);tl=0.
    for _ in range(nb):
        idx=torch.randint(0,N,(128,));sc=sp[idx]
        ep_=make_env_params(bridge,128,DEVICE)
        pol.train();op.zero_grad()
        hc=wm.layers[0](torch.cat([sc,torch.zeros(128,1,device=DEVICE)],dim=-1)).detach_()
        v=pol(hc.detach(),ep_,hg.expand(128,-1))
        a=executor(v,sc)
        sp_=wm(torch.cat([sc,a],dim=-1))
        hp=wm.layers[0](torch.cat([sp_,torch.zeros(128,1,device=DEVICE)],dim=-1))
        ls=(hp-hg.expand(128,-1)).pow(2).sum(dim=-1).mean()+.01*a.pow(2).mean()
        ls.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(),10.)
        op.step();tl+=ls.item()
    if ep%20==0:print(f"  ep{ep:4d} loss={tl/nb:.4f}")

print("Eval (20 trials)...")
succ=0;steps=[]
for t in range(20):
    np.random.seed(42+t*100);th=np.random.uniform(-.05,.05)
    s=torch.tensor([[0.,0.,th/TH_S,0.]],dtype=torch.float32)
    with torch.no_grad():
        hg=wm.layers[0](torch.cat([S_TARGET.to(DEVICE),torch.zeros(1,1,device=DEVICE)],dim=-1))
        ep_=make_env_params(bridge,1,DEVICE)
    for step in range(500):
        with torch.no_grad():
            h=wm.layers[0](torch.cat([s,torch.zeros(1,1)],dim=-1))
            v=pol(h,ep_,hg);a=executor(v,s)
        sr=s.clone();sr[:,0]*=X_S;sr[:,1]*=XD_S;sr[:,2]*=TH_S;sr[:,3]*=THD_S
        sn=step_cartpole(sr,a.squeeze(-1),g=G_DEF)
        sn[:,0]/=X_S;sn[:,1]/=XD_S;sn[:,2]/=TH_S;sn[:,3]/=THD_S
        s=sn
        if abs(s[0,2].item()*TH_S)>.21 or abs(s[0,0].item()*X_S)>2.4:break
    steps.append(step+1)
    if step+1>=500:succ+=1
print(f"\nCDPNv4 CartPole: {succ}/20 ({succ*5}%) mean_steps={np.mean(steps):.0f}")