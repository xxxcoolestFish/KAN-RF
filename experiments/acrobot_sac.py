"""Acrobot SAC: continuous double pendulum + SAC + WM."""
import torch,torch.nn as nn,numpy as np,sys,os,time;from collections import deque
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
DEVICE="cpu";torch.manual_seed(42);np.random.seed(42)
LOG_MIN=-5;LOG_MAX=2

# Continuous Acrobot dynamics
G=9.8;M1=M2=1.;L1=L2=1.;LC1=LC2=.5;I1=I2=1.;DT=.05;MAX_V1=6.;MAX_V2=8.;TARGET_H=1.0

def acrobot_step(sr,a):
    """Continuous Acrobot step: s=(B,6) [cos1,sin1,cos2,sin2,d1/6,d2/8], a=torque[-1,1]"""
    cos1,sin1,cos2,sin2=sr[:,0],sr[:,1],sr[:,2],sr[:,3]
    d1=sr[:,4]*MAX_V1;d2=sr[:,5]*MAX_V2
    # Compute accelerations (standard double pendulum dynamics)
    th1=torch.atan2(sin1,cos1);th2=torch.atan2(sin2,cos2)
    st1,st2=torch.sin(th1),torch.sin(th2)
    ct1,ct2=torch.cos(th1),torch.cos(th2)
    st12=torch.sin(th1+th2);ct12=torch.cos(th1+th2)
    # Matrix elements
    d11=M1*LC1**2+M2*(L1**2+LC2**2+2*L1*LC2*ct2)+I1+I2
    d22=M2*LC2**2+I2
    d12=M2*(LC2**2+L1*LC2*ct2)+I2
    # Gravity torques
    phi2=M2*G*LC2*st12
    phi1=-(M1*LC1+M2*L1)*G*st1-phi2
    # Coriolis
    c1=2*M2*L1*LC2*st2
    h=c1*d1
    # Accelerations
    th2a=(a[...,0]+d12/d11*phi1-h*d2-c1*d1*d2-phi2)/(d22-d12**2/d11+1e-6)
    th1a=-(d12*th2a+phi1)/d11
    d1=d1+th1a*DT;d2=d2+th2a*DT
    d1=d1.clamp(-MAX_V1,MAX_V1);d2=d2.clamp(-MAX_V2,MAX_V2)
    th1=th1+d1*DT;th2=th2+d2*DT
    return torch.stack([torch.cos(th1),torch.sin(th1),torch.cos(th2),torch.sin(th2),d1/MAX_V1,d2/MAX_V2],-1)

def tip_height(s):
    """Height of the end of second link: -l1*cos(th1) - l2*cos(th1+th2)"""
    cos1,sin1,cos2,sin2=s[:,0],s[:,1],s[:,2],s[:,3]
    th1=torch.atan2(sin1,cos1);th1_2=torch.atan2(sin2,cos2)+th1
    return -L1*torch.cos(th1)-L2*torch.cos(th1_2)

# SAC networks (same architecture as Pendulum, just d_in=6)
class ActorNet(nn.Module):
    def __init__(self,d):super().__init__();self.f=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU());self.m=nn.Linear(128,1);self.l=nn.Linear(128,1)
    def fwd(self,s):x=self.f(s);return self.m(x),torch.clamp(self.l(x),LOG_MIN,LOG_MAX)
    def sample(self,s):m,l=self.fwd(s);std=l.exp();e=torch.randn_like(std);u=m+std*e;a=torch.tanh(u);lp=-((e**2+2*l+np.log(2*np.pi)).sum(-1,keepdim=True)/2);lp=lp-(2*(np.log(2)-u-nn.functional.softplus(-2*u))).sum(-1,keepdim=True);return a,lp

class QNet(nn.Module):
    def __init__(self,d):super().__init__();self.n=nn.Sequential(nn.Linear(d+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,s,a):return self.n(torch.cat([s,a],-1))

class Buf:
    def __init__(self,c=100000,sd=6):
        self.s=np.zeros((c,sd),dtype=np.float32);self.a=np.zeros((c,1),dtype=np.float32)
        self.r=np.zeros((c,1),dtype=np.float32);self.s_=np.zeros((c,sd),dtype=np.float32)
        self.d=np.zeros((c,1),dtype=np.float32);self.p=0;self.n=0;self.c=c
    def push(self,s,a,r,s_,d):
        self.s[self.p]=s;self.a[self.p]=a;self.r[self.p]=r;self.s_[self.p]=s_;self.d[self.p]=d
        self.p=(self.p+1)%self.c;self.n=min(self.n+1,self.c)
    def sample(self,b=128):
        i=np.random.randint(0,self.n,b)
        return(torch.tensor(self.s[i]),torch.tensor(self.a[i]),torch.tensor(self.r[i]),torch.tensor(self.s_[i]),torch.tensor(self.d[i]))
    def __len__(self):return self.n

class SAC:
    def __init__(self,d_in=6,lr=3e-4,gamma=.99,tau=.005,alpha=.2):
        self.g=gamma;self.t=tau;self.al=alpha
        self.a=ActorNet(d_in).to(DEVICE);self.q1=QNet(d_in).to(DEVICE);self.q2=QNet(d_in).to(DEVICE)
        self.t1=QNet(d_in).to(DEVICE);self.t2=QNet(d_in).to(DEVICE)
        self.t1.load_state_dict(self.q1.state_dict());self.t2.load_state_dict(self.q2.state_dict())
        self.ao=torch.optim.Adam(self.a.parameters(),lr=lr);self.q1o=torch.optim.Adam(self.q1.parameters(),lr=lr);self.q2o=torch.optim.Adam(self.q2.parameters(),lr=lr)
        self.la=torch.tensor(np.log(alpha),requires_grad=True);self.lao=torch.optim.Adam([self.la],lr=lr)
    def act(self,s,dt=False):
        with torch.no_grad():m,l=self.a.fwd(s[None]);a=m if dt else m+torch.randn_like(m)*l.exp();return torch.tanh(a).squeeze(0).numpy()
    def upd(self,buf,b=128):
        if len(buf)<b:
            return
        s,a,r,s_next,d=buf.sample(b)
        with torch.no_grad():
            a_next,lp_next=self.a.sample(s_next)
            q=torch.min(self.t1(s_next,a_next),self.t2(s_next,a_next))-self.al*lp_next
            y=r+self.g*(1-d)*q
        self.q1o.zero_grad();((self.q1(s,a)-y)**2).mean().backward();self.q1o.step()
        self.q2o.zero_grad();((self.q2(s,a)-y)**2).mean().backward();self.q2o.step()
        a_,lp=self.a.sample(s);self.ao.zero_grad();(self.al*lp-self.q1(s,a_)).mean().backward();self.ao.step()
        self.lao.zero_grad();(-self.la*(lp+2).detach().mean()).backward();self.lao.step();self.al=self.la.exp().item()
        with torch.no_grad():
            for p,t in zip(self.q1.parameters(),self.t1.parameters()):t.mul_(1-self.t).add_(self.t*p)
            for p,t in zip(self.q2.parameters(),self.t2.parameters()):t.mul_(1-self.t).add_(self.t*p)

def eval_acrobot(ag,n=10):
    sc=0
    for t in range(n):
        th1=np.random.uniform(-.1,.1);th2=np.random.uniform(-.1,.1)
        s=torch.tensor([[np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),0.,0.]],dtype=torch.float32)
        for _ in range(500):
            a=ag.act(s.squeeze(0),dt=True)
            s=acrobot_step(s,torch.tensor([a]))
            if tip_height(s).item()>TARGET_H:sc+=1;break
    return sc

print("Acrobot SAC+WM");t0=time.time()

# Train WM
print("\n[1] Train WM on random Acrobot data")
cpx,cpy=[],[]
for _ in range(5000):
    th1=np.random.uniform(-np.pi,np.pi);th2=np.random.uniform(-np.pi,np.pi)
    d1=np.random.uniform(-MAX_V1,MAX_V1)/MAX_V1;d2=np.random.uniform(-MAX_V2,MAX_V2)/MAX_V2
    a=np.random.uniform(-1.,1.)
    s=torch.tensor([[np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),d1,d2]],dtype=torch.float32)
    sn=acrobot_step(s,torch.tensor([[a]]))
    cpx.append(torch.cat([s,torch.tensor([[a]])],-1).float());cpy.append(sn.float())
cpx=torch.cat(cpx);cpy=torch.cat(cpy);nt=int(len(cpx)*.85);mse=nn.MSELoss()
wm=ProtoKAN([7,16,6],n_prototypes=16).to(DEVICE)
for l in wm.layers:l.log_sigma.data.fill_(-1.5)
opt=torch.optim.LBFGS(wm.parameters(),lr=1.,max_iter=20,history_size=50,line_search_fn="strong_wolfe")
bv=float("inf");bs=None
def c():opt.zero_grad();l=mse(wm(cpx[:nt]),cpy[:nt]);l.backward();return l
for _ in range(80):
    opt.step(c);v=mse(wm(cpx[nt:]),cpy[nt:]).item()
    if v<bv:bv=v;bs={k:v.clone() for k,v in wm.state_dict().items()}
wm.load_state_dict(bs);wm.eval();print(f"  WM val_mse={bv:.6f}")

# SAC training
print("\n[2] SAC training")
ag=SAC(6);bu=Buf();rng=np.random.RandomState(42)
th1,th2=rng.uniform(-.1,.1),rng.uniform(-.1,.1)
o=torch.tensor([[np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),0.,0.]],dtype=torch.float32)
for st in range(1,20001):
    a=ag.act(o.squeeze(0))
    sn=acrobot_step(o,torch.tensor([a]))
    h=tip_height(sn).item();d=1. if h>TARGET_H else 0.
    bu.push(o.squeeze(0).numpy(),float(a),h,sn.squeeze(0).numpy(),d)
    if st%2==0:ag.upd(bu)
    o=sn
    if d:th1,th2=rng.uniform(-.1,.1),rng.uniform(-.1,.1);o=torch.tensor([[np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),0.,0.]],dtype=torch.float32)
    if st%5000==0:print(f"  Step{st:6d} eval={eval_acrobot(ag)}/10 reward={h:.3f} time={time.time()-t0:.0f}s")

print(f"\nFinal: {eval_acrobot(ag,20)}/20")
print(f"Time: {time.time()-t0:.0f}s")