"""CDPNv4+ES: Comprehensive validation.
   Pendulum adaptation (g=10->15) + CartPole validation.
"""
import torch,torch.nn as nn,numpy as np,sys,os,gymnasium as gym
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
from experiments.baseline_sweep import generate_pendulum_data,train_wm

DEVICE="cpu";PI_2=np.pi/2;ST=torch.tensor([[0.,1.,0.]])
torch.manual_seed(42);np.random.seed(42)

# === UTILS ===
class Policy(nn.Module):
    def __init__(self,in_d,hid=32,out_d=1):
        super().__init__()
        self.n=nn.Sequential(nn.Linear(in_d,hid),nn.Tanh(),nn.Linear(hid,24),nn.Tanh(),nn.Linear(24,out_d),nn.Tanh())
    def forward(self,s,sg):return self.n(torch.cat([s,sg],dim=-1))

class ES:
    def __init__(self,policy,pop=20,sigma=.1,lr=.01):
        self.p=policy;self.P=pop;self.s=sigma;self.lr=lr
        self.d=sum(p.numel() for p in policy.parameters())
        self.m=torch.cat([p.data.view(-1) for p in policy.parameters()]).clone()
    def setp(self,params):
        i=0
        for p in self.p.parameters():
            n=p.numel();p.data.copy_(params[i:i+n].reshape(p.shape));i+=n
    def fitness(self,policy,wm,s0,H,wr):
        if wr>1:f=sum(self._roll(policy,wm,s0.clone(),H) for _ in range(wr))/wr
        else:f=self._roll(policy,wm,s0.clone(),H)
        return f
    def _roll(self,policy,wm,s0,H):
        s=s0;c=0.
        for _ in range(H):
            a=policy(s,ST);s=wm(torch.cat([s,a],dim=-1));c+=((s-ST)**2).sum().item()
        return -c
    def train(self,wm,H=20,gens=100,wr=3,report=20):
        for g in range(1,gens+1):
            noise=torch.randn(self.P,self.d)
            fits=[]
            for i in range(self.P):
                self.setp(self.m+self.s*noise[i])
                fits.append(self.fitness(self.p,wm,self._rands(),H,wr))
            ft=torch.tensor(fits)
            ft=(ft-ft.mean())/(ft.std()+1e-8)
            grad=(noise.T@ft)/(self.P*self.s)
            self.m+=self.lr*grad;self.setp(self.m)
            if g%report==0:print(f"    Gen {g:3d}  fit={ft.mean():+.3f}")
    def _rands(self):
        th=np.random.uniform(-np.pi,np.pi);td=np.random.uniform(-1.,1.)
        return torch.tensor([[np.cos(th),np.sin(th),td/8.]],dtype=torch.float32)

def real_step(s,a,g=10.):
    c_t,s_t=s[...,0],s[...,1];td=s[...,2]*8.
    u=a[...,0].clamp(-1.,1.)*2.
    td_n=td+(1.5*g*s_t+3.*u)*.05;td_n=td_n.clamp(-8.,8.)
    t=torch.atan2(s_t,c_t);t_n=t+td_n*.05
    return torch.stack([torch.cos(t_n),torch.sin(t_n),td_n/8.],dim=-1)

def eval_pendulum(policy,g=10.,n=20,label=""):
    env=gym.make("Pendulum-v1");succ=0;steps=[]
    for t in range(n):
        obs,_=env.reset(seed=42+t*100);ok=False
        for st in range(300):
            sn=torch.tensor([[obs[0],obs[1],obs[2]/8.]],dtype=torch.float32)
            with torch.no_grad():a=policy(sn,ST).item()
            obs,_,_,_,_=env.step([a*2.])
            err=min(abs(np.arctan2(obs[1],obs[0])-PI_2),2*np.pi-abs(np.arctan2(obs[1],obs[0])-PI_2))
            if err<.2:succ+=1;steps.append(st+1);ok=True;break
        if not ok:steps.append(300)
    env.close()
    print(f"  [{label}] {succ}/{n} ({succ*100//n}%) steps={np.mean(steps):.0f}")
    return succ

# === PHASE 1: PENDULUM ADAPTATION ===
print("="*70);print("PHASE 1: PENDULUM");print("="*70)
print("\n[1.1] Train WM on g=10")
X10,Y10=generate_pendulum_data(5000,seed=42)
X15,Y15=generate_pendulum_data(5000,seed=43)
wm10,_=train_wm(X10.to(DEVICE),Y10.to(DEVICE))

print("\n[1.2] ES train policy with WM(g=10)")
pol=Policy(6,32,1)
es=ES(pol,pop=20,sigma=.1,lr=.01)
es.train(wm10,H=20,gens=100,wr=3)

print("\n[1.3] Evaluate")
eval_pendulum(pol,g=10.,label="g=10 baseline")
eval_pendulum(pol,g=15.,label="g=15 zero-shot")

print("\n[1.4] WM continual learning: g=10 -> g=15")
opt=torch.optim.Adam(wm10.parameters(),lr=1e-3)
wm10.train()
for _ in range(50):
    idx=torch.randint(0,len(X15),(256,))
    l=nn.MSELoss()(wm10(X15[idx].to(DEVICE)),Y15[idx].to(DEVICE))
    opt.zero_grad();l.backward();opt.step()
wm10.eval()
print("  WM adapted to g=15")

eval_pendulum(pol,g=15.,label="g=15 after WM adapt (no policy retrain)")

print("\n[1.5] Quick ES adaptation (30 gens with adapted WM)")
es.train(wm10,H=20,gens=30,wr=3)
eval_pendulum(pol,g=15.,label="g=15 after ES quick adapt")
eval_pendulum(pol,g=10.,label="g=10 forgetting test")

# === PHASE 2: CARTIPOLE ===
print("\n"+"="*70);print("PHASE 2: CARTIPOLE");print("="*70)

# Cartpole dynamics
G=9.8;MC=1.;MP=.1;L=.5;DT=.02
TM=MC+MP;PML=MP*L;FM=10.
XS=2.5;XDS=3.;THS=.3;THDS=3.
S_CP=torch.tensor([[0.,0.,0.,0.]])

def cp_step(sr,a,g=G,mp=MP,l=L):
    a=a.squeeze(-1) if a.dim()>1 else a
    x,xd,th,thd=sr[:,0],sr[:,1],sr[:,2],sr[:,3]
    f=a*FM;ct,st=torch.cos(th),torch.sin(th)
    tm_=MC+mp;pml_=mp*l
    tmp=(f+pml_*thd**2*st)/tm_
    th_a=(g*st-ct*tmp)/(.5*(4./3.-mp*ct**2/tm_)+1e-8)
    x_a=tmp-pml_*th_a*ct/tm_
    return torch.stack([x+xd*DT+x_a*DT**2/2,xd+x_a*DT,
        th+thd*DT+th_a*DT**2/2,thd+th_a*DT],dim=-1)

def gen_cp(n=5000,mp=MP,l=L):
    xs,ys=[],[]
    for _ in range(n):
        x=np.random.uniform(-2.4,2.4);xd=np.random.uniform(-3.,3.)
        th=np.random.uniform(-.3,.3);thd=np.random.uniform(-3.,3.)
        a=np.random.uniform(-1.,1.)
        sr=torch.tensor([[x,xd,th,thd]],dtype=torch.float32)
        sn=cp_step(sr,torch.tensor([[a]]),mp=mp,l=l)
        sn_=sn.clone();sn_[:,0]/=XS;sn_[:,1]/=XDS;sn_[:,2]/=THS;sn_[:,3]/=THDS
        sr_=sr.clone();sr_[:,0]/=XS;sr_[:,1]/=XDS;sr_[:,2]/=THS;sr_[:,3]/=THDS
        xs.append(torch.cat([sr_,torch.tensor([[a]])],dim=-1).float())
        ys.append(sn_.float())
    return torch.cat(xs),torch.cat(ys)

def train_wm_cp(X,Y,np_=16,nl=100):
    nt=int(len(X)*.85)
    wm=ProtoKAN([5,np_,4],n_prototypes=np_).to(DEVICE)
    for l in wm.layers:l.log_sigma.data.fill_(-1.5)
    mse=nn.MSELoss()
    opt=torch.optim.LBFGS(wm.parameters(),lr=1.,max_iter=20,history_size=50,line_search_fn="strong_wolfe")
    bv=float("inf");bs=None
    def c():opt.zero_grad();l=mse(wm(X[:nt]),Y[:nt]);l.backward();return l
    for _ in range(nl):
        opt.step(c);v=mse(wm(X[nt:]),Y[nt:]).item()
        if v<bv:bv=v;bs={k:v.clone() for k,v in wm.state_dict().items()}
    wm.load_state_dict(bs);wm.eval()
    return wm

class ES_CP:
    def __init__(self,policy,pop=30,sigma=.1,lr=.01):
        self.p=policy;self.P=pop;self.s=sigma;self.lr=lr
        self.d=sum(p.numel() for p in policy.parameters())
        self.m=torch.cat([p.data.view(-1) for p in policy.parameters()]).clone()
    def setp(self,params):
        i=0
        for p in self.p.parameters():
            n=p.numel();p.data.copy_(params[i:i+n].reshape(p.shape));i+=n
    def _rs(self):
        return torch.tensor([[np.random.uniform(-.25,.25)/XS,np.random.uniform(-1.5,1.5)/XDS,
            np.random.uniform(-.15,.15)/THS,np.random.uniform(-1.5,1.5)/THDS]],dtype=torch.float32)
    def _roll(self,policy,wm,s0,H):
        s=s0;c=0.
        for _ in range(H):
            a=policy(s,S_CP);s=wm(torch.cat([s,a],dim=-1))
            c+=((s-S_CP)**2).sum().item()
        return -c
    def train(self,wm,H=20,gens=100,wr=3,report=20):
        for g in range(1,gens+1):
            noise=torch.randn(self.P,self.d)
            fits=[]
            for i in range(self.P):
                self.setp(self.m+self.s*noise[i])
                f=0.
                for _ in range(wr):f+=self._roll(self.p,wm,self._rs(),H)
                fits.append(f/wr)
            ft=torch.tensor(fits)
            ft=(ft-ft.mean())/(ft.std()+1e-8)
            grad=(noise.T@ft)/(self.P*self.s)
            self.m+=self.lr*grad;self.setp(self.m)
            if g%report==0:print(f"    Gen {g:3d}  fit={ft.mean():+.3f}")

def eval_cp(policy,mp=MP,l=L,n=20,label=""):
    succ=0;steps=[]
    for t in range(n):
        np.random.seed(42+t*100);th=np.random.uniform(-.05,.05)
        s=torch.tensor([[0.,0.,th/THS,0.]],dtype=torch.float32)
        for st in range(500):
            sr=s.clone();sr[:,0]*=XS;sr[:,1]*=XDS;sr[:,2]*=THS;sr[:,3]*=THDS
            with torch.no_grad():a=policy(s,S_CP).item()
            sn=cp_step(sr,torch.tensor([[a]]),mp=mp,l=l)
            sn[:,0]/=XS;sn[:,1]/=XDS;sn[:,2]/=THS;sn[:,3]/=THDS
            s=sn
            if abs(s[0,2].item()*THS)>.21 or abs(s[0,0].item()*XS)>2.4:break
        steps.append(st+1)
        if st+1>=500:succ+=1
    print(f"  [{label}] {succ}/{n} ({succ*100//n}%) steps={np.mean(steps):.0f}")
    return succ

# Train WM
print("\n[2.1] Train CartPole WM")
Xcp,Ycp=gen_cp(5000)
wmc=train_wm_cp(Xcp.to(DEVICE),Ycp.to(DEVICE))
print(f"  WM trained")

# ES train policy
print("\n[2.2] ES train CartPole policy")
pol_cp=Policy(8,32,1)
es_cp=ES_CP(pol_cp,pop=30,sigma=.1,lr=.01)
es_cp.train(wmc,H=20,gens=100,wr=3)

print("\n[2.3] Evaluate CartPole")
eval_cp(pol_cp,label="CartPole default (m=0.1, L=0.5)")

print("\n[2.4] WM adapt: heavy pole (m=0.1 -> 0.2)")
Xh,Yh=gen_cp(2000,mp=0.2)
opt=torch.optim.Adam(wmc.parameters(),lr=1e-3)
wmc.train()
for _ in range(50):
    idx=torch.randint(0,len(Xh),(256,))
    l=nn.MSELoss()(wmc(Xh[idx].to(DEVICE)),Yh[idx].to(DEVICE))
    opt.zero_grad();l.backward();opt.step()
wmc.eval()
eval_cp(pol_cp,mp=0.2,label="m=0.2 after WM adapt (no policy retrain)")

print("\n[2.5] Quick ES adaptation (30 gens with adapted WM)")
es_cp.train(wmc,H=20,gens=30,wr=3)
eval_cp(pol_cp,mp=0.2,label="m=0.2 after ES quick adapt")
eval_cp(pol_cp,label="m=0.1 forgetting test")

print("\n"+"="*70);print("DONE");print("="*70)