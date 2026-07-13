"""CDPNv4 + ES: WM as forward simulator, Evolution Strategies for policy."""
import torch,torch.nn as nn,numpy as np,sys,os,gymnasium as gym
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
from experiments.baseline_sweep import generate_pendulum_data,train_wm
DEVICE="cpu";PI_2=np.pi/2
S_TARGET=torch.tensor([[0.,1.,0.]])
torch.manual_seed(42);np.random.seed(42)

class SwingPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.n=nn.Sequential(nn.Linear(6,32),nn.Tanh(),nn.Linear(32,24),nn.Tanh(),nn.Linear(24,1),nn.Tanh())
    def forward(self,s,sg):return self.n(torch.cat([s,sg],dim=-1))

class ESTrainer:
    def __init__(self,policy,pop=20,sigma=.1,lr=.01):
        self.p=policy;self.P=pop;self.s=sigma;self.lr=lr
        self.d=sum(p.numel() for p in policy.parameters())
        self.m=torch.cat([p.data.view(-1) for p in policy.parameters()]).clone()
    def setp(self,params):
        i=0
        for p in self.p.parameters():
            n=p.numel();p.data.copy_(params[i:i+n].reshape(p.shape));i+=n
    def eval(self,policy,wm,H=20):
        th=np.random.uniform(-np.pi,np.pi);td=np.random.uniform(-1.,1.)
        s=torch.tensor([[np.cos(th),np.sin(th),td/8.]],dtype=torch.float32)
        cost=0.
        for _ in range(H):
            a=policy(s,S_TARGET)
            s=wm(torch.cat([s,a],dim=-1))
            cost+=((s-S_TARGET)**2).sum().item()
        return -cost
    def step(self,wm,nr=3,H=20):
        noise=torch.randn(self.P,self.d)
        fits=[]
        for i in range(self.P):
            self.setp(self.m+self.s*noise[i])
            f=sum(self.eval(self.p,wm,H) for _ in range(nr))/nr
            fits.append(f)
        fits=torch.tensor(fits)
        fits=(fits-fits.mean())/(fits.std()+1e-8)
        grad=(noise.T@fits)/(self.P*self.s)
        self.m+=self.lr*grad;self.setp(self.m)
        return float(fits.mean())

print("="*60);print("CDPNv4+ES: Pendulum");print("="*60)
print("\n[1] Train WM")
X,Y=generate_pendulum_data(5000)
wm,_=train_wm(X.to(DEVICE),Y.to(DEVICE))

print("\n[2] ES training (100 gens, pop=20, H=20)")
pol=SwingPolicy()
es=ESTrainer(pol,pop=20,sigma=.1,lr=.01)
for g in range(1,101):
    f=es.step(wm,nr=3,H=20)
    if g%20==0:print(f"  Gen {g:3d}  fit={f:.3f}")

print("\n[3] Evaluate")
succ=0;steps=[];env=gym.make("Pendulum-v1")
for t in range(20):
    obs,_=env.reset(seed=42+t*100);ok=False
    for st in range(300):
        sn=torch.tensor([[obs[0],obs[1],obs[2]/8.]],dtype=torch.float32)
        with torch.no_grad():a=pol(sn,S_TARGET).item()
        obs,_,_,_,_=env.step([a*2.])
        err=min(abs(np.arctan2(obs[1],obs[0])-PI_2),2*np.pi-abs(np.arctan2(obs[1],obs[0])-PI_2))
        if err<.2:succ+=1;steps.append(st+1);ok=True;break
    if not ok:steps.append(300)
env.close()
print(f"\nRESULT: {succ}/20 ({succ*5}%)  mean_steps={np.mean(steps):.0f}")