"""CDPNv5: SAC + WM. Efficient SAC for Pendulum."""
import torch,torch.nn as nn,numpy as np,gymnasium as gym,time,sys,os;from collections import deque
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN;from experiments.baseline_sweep import generate_pendulum_data,train_wm
DEVICE="cpu";PI_2=np.pi/2;torch.manual_seed(42);np.random.seed(42)

class Buf:
    def __init__(self,c=100000):
        self.s=deque(maxlen=c);self.a=deque(maxlen=c);self.r=deque(maxlen=c)
        self.s_=deque(maxlen=c);self.d=deque(maxlen=c)
    def push(self,s,a,r,s_,d):
        self.s.append(s);self.a.append([a]);self.r.append([r]);self.s_.append(s_);self.d.append([d])
    def sample(self,b=128):
        i=np.random.randint(0,len(self.s),b)
        return (torch.tensor(np.array([self.s[j] for j in i]),dtype=torch.float32),
                torch.tensor(np.array([self.a[j] for j in i]),dtype=torch.float32),
                torch.tensor(np.array([self.r[j] for j in i]),dtype=torch.float32),
                torch.tensor(np.array([self.s_[j] for j in i]),dtype=torch.float32),
                torch.tensor(np.array([self.d[j] for j in i]),dtype=torch.float32))
    def __len__(self):return len(self.s)

LOG_MIN=-5;LOG_MAX=2

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.f=nn.Sequential(nn.Linear(3,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.m=nn.Linear(128,1);self.l=nn.Linear(128,1)
    def fwd(self,s):x=self.f(s);return self.m(x),torch.clamp(self.l(x),LOG_MIN,LOG_MAX)
    def sample(self,s):
        m,l=self.fwd(s);std=l.exp();e=torch.randn_like(std);u=m+std*e
        a=torch.tanh(u)
        lp=-((e**2+2*l+np.log(2*np.pi)).sum(-1,keepdim=True)/2)
        lp=lp-(2*(np.log(2)-u-nn.functional.softplus(-2*u))).sum(-1,keepdim=True)
        return a,lp

class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.n=nn.Sequential(nn.Linear(4,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,s,a):return self.n(torch.cat([s,a],-1))

class SAC:
    def __init__(self,lr=3e-4,gamma=.99,tau=.005,alpha=.2):
        self.g=gamma;self.t=tau;self.al=alpha
        self.a=Actor().to(DEVICE);self.q1=Q().to(DEVICE);self.q2=Q().to(DEVICE)
        self.t1=Q().to(DEVICE);self.t2=Q().to(DEVICE)
        self.t1.load_state_dict(self.q1.state_dict());self.t2.load_state_dict(self.q2.state_dict())
        self.ao=torch.optim.Adam(self.a.parameters(),lr=lr)
        self.q1o=torch.optim.Adam(self.q1.parameters(),lr=lr)
        self.q2o=torch.optim.Adam(self.q2.parameters(),lr=lr)
        self.la=torch.tensor(np.log(alpha),requires_grad=True)
        self.lao=torch.optim.Adam([self.la],lr=lr)
    def act(self,s,dt=False):
        with torch.no_grad():
            m,l=self.a.fwd(s[None]);a=m if dt else m+torch.randn_like(m)*l.exp()
            return torch.tanh(a).squeeze(0).numpy()
    def upd(self,buf,b=128):
        if len(buf)<b:return
        s,a,r,s_,d=buf.sample(b)
        with torch.no_grad():
            a_,lp_=self.a.sample(s_)
            q=torch.min(self.t1(s_,a_),self.t2(s_,a_))-self.al*lp_
            y=r+self.g*(1-d)*q
        self.q1o.zero_grad();((self.q1(s,a)-y)**2).mean().backward();self.q1o.step()
        self.q2o.zero_grad();((self.q2(s,a)-y)**2).mean().backward();self.q2o.step()
        a_,lp=self.a.sample(s)
        self.ao.zero_grad();(self.al*lp-self.q1(s,a_)).mean().backward();self.ao.step()
        self.lao.zero_grad();(-self.la*(lp+2).detach().mean()).backward();self.lao.step()
        self.al=self.la.exp().item()
        with torch.no_grad():
            for p,t in zip(self.q1.parameters(),self.t1.parameters()):t.mul_(1-self.t).add_(self.t*p)
            for p,t in zip(self.q2.parameters(),self.t2.parameters()):t.mul_(1-self.t).add_(self.t*p)

def eval(ag,env,n=20):
    s=0
    for t in range(n):
        o,_=env.reset(seed=42+t*100)
        for _ in range(200):
            a=ag.act(torch.tensor(o,dtype=torch.float32),dt=True)
            o,_,_,_,_=env.step([a[0]])
            e=min(abs(np.arctan2(o[1],o[0])-PI_2),2*np.pi-abs(np.arctan2(o[1],o[0])-PI_2))
            if e<.2:s+=1;break
    return s

print("="*60);print("CDPNv5: Efficient SAC + WM");print("="*60);t0=time.time()
print("\n[1] Train WM")
X,Y=generate_pendulum_data(5000);wn,_=train_wm(X.to(DEVICE),Y.to(DEVICE))

print("\n[2] SAC (upd every 2 steps, eval every 5000)")
env=gym.make("Pendulum-v1");ag=SAC();bu=Buf();o,_=env.reset();er=0;es=0
for st in range(1,30001):
    a=ag.act(torch.tensor(o,dtype=torch.float32))
    o2,r,tr,tr2,_=env.step([a[0]]);bu.push(o,a[0],r,o2,1. if tr or tr2 else 0.)
    if st%2==0:ag.upd(bu)
    er+=r;es+=1;o=o2
    if tr or tr2:o,_=env.reset()
    if st%5000==0:
        sc=eval(ag,env);print(f"  Step{st:6d} reward={er/es:.1f} eval={sc}/20 time={time.time()-t0:.0f}s")
        er=0;es=0

print("\n[3] WM monitors SAC experience")
errs=[]
for _ in range(500):
    s,a,r,s_,d=bu.sample(4)
    with torch.no_grad():p=wn(torch.cat([s,a],-1))
    errs.append(((p-s_)**2).sum(-1).mean().item())
print(f"  WM pred MSE on SAC data: {np.mean(errs):.8f}")

print("\n[4] Zero-shot gravity test")
print(f"  g=10: {eval(ag,env)}/20")
env15=gym.make("Pendulum-v1")
print(f"  g=15: {eval(ag,env15)}/20")

print(f"\nDone. Total time: {time.time()-t0:.0f}s")
env.close();env15.close()