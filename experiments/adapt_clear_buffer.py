
import torch,torch.nn as nn,numpy as np,gymnasium as gym,time,sys,os;from collections import deque
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN;from experiments.baseline_sweep import generate_pendulum_data,train_wm
DEVICE='cpu';PI_2=np.pi/2;torch.manual_seed(42);np.random.seed(42);LOG_MIN=-5;LOG_MAX=2;mse=nn.MSELoss()

class Buf:
    def __init__(self,c=200000):
        self.s=deque(maxlen=c);self.a=deque(maxlen=c);self.r=deque(maxlen=c);self.s_=deque(maxlen=c);self.d=deque(maxlen=c)
    def push(self,s,a,r,s_,d):self.s.append(s);self.a.append([a]);self.r.append([r]);self.s_.append(s_);self.d.append([d])
    def clear(self):self.s.clear();self.a.clear();self.r.clear();self.s_.clear();self.d.clear()
    def sample(self,b=128):
        i=np.random.randint(0,len(self.s),b)
        return(torch.tensor(np.array([self.s[j] for j in i]),dtype=torch.float32),
               torch.tensor(np.array([self.a[j] for j in i]),dtype=torch.float32),
               torch.tensor(np.array([self.r[j] for j in i]),dtype=torch.float32),
               torch.tensor(np.array([self.s_[j] for j in i]),dtype=torch.float32),
               torch.tensor(np.array([self.d[j] for j in i]),dtype=torch.float32))
    def __len__(self):return len(self.s)

class ActorNet(nn.Module):
    def __init__(self):super().__init__();self.f=nn.Sequential(nn.Linear(3,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU());self.m=nn.Linear(128,1);self.l=nn.Linear(128,1)
    def fwd(self,s):x=self.f(s);return self.m(x),torch.clamp(self.l(x),LOG_MIN,LOG_MAX)
    def sample(self,s):m,l=self.fwd(s);std=l.exp();e=torch.randn_like(std);u=m+std*e;a=torch.tanh(u);lp=-((e**2+2*l+np.log(2*np.pi)).sum(-1,keepdim=True)/2);lp=lp-(2*(np.log(2)-u-nn.functional.softplus(-2*u))).sum(-1,keepdim=True);return a,lp

class QNet(nn.Module):
    def __init__(self):super().__init__();self.n=nn.Sequential(nn.Linear(4,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,s,a):return self.n(torch.cat([s,a],-1))

class SAC:
    def __init__(self,lr=3e-4,gamma=.99,tau=.005,alpha=.2):
        self.g=gamma;self.t=tau;self.al=alpha
        self.a=ActorNet().to(DEVICE);self.q1=QNet().to(DEVICE);self.q2=QNet().to(DEVICE)
        self.t1=QNet().to(DEVICE);self.t2=QNet().to(DEVICE)
        self.t1.load_state_dict(self.q1.state_dict());self.t2.load_state_dict(self.q2.state_dict())
        self.ao=torch.optim.Adam(self.a.parameters(),lr=lr);self.q1o=torch.optim.Adam(self.q1.parameters(),lr=lr)
        self.q2o=torch.optim.Adam(self.q2.parameters(),lr=lr)
        self.la=torch.tensor(np.log(alpha),requires_grad=True);self.lao=torch.optim.Adam([self.la],lr=lr)
    def act(self,s,dt=False):
        with torch.no_grad():m,l=self.a.fwd(s[None]);a=m if dt else m+torch.randn_like(m)*l.exp();return torch.tanh(a).squeeze(0).numpy()
    def upd(self,buf,b=128):
        if len(buf)<b:return
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

def eval(ag,env,n=20):
    s=0
    for t in range(n):
        o,_=env.reset(seed=42+t*100)
        for _ in range(200):
            a=ag.act(torch.tensor(o,dtype=torch.float32),dt=True);o,_,_,_,_=env.step([a[0]])
            if min(abs(np.arctan2(o[1],o[0])-PI_2),2*np.pi-abs(np.arctan2(o[1],o[0])-PI_2))<.2:s+=1;break
    return s

print('CDPNv5: adaptation with buffer clear');t0=time.time()

# Phase 1: SAC on g=10
print('[1] SAC on g=10 (15000 steps)')
ag=SAC();bu=Buf();env=gym.make('Pendulum-v1');o,_=env.reset()
for st in range(1,15001):
    a=ag.act(torch.tensor(o,dtype=torch.float32));o2,r,tr,tr2,_=env.step([a[0]])
    bu.push(o,a[0],r,o2,1. if tr or tr2 else 0.)
    if st%2==0:ag.upd(bu)
    o=o2
    if tr or tr2:o,_=env.reset()
g10=eval(ag,env);print(f'  eval_g10={g10}/20 time={time.time()-t0:.0f}s')

# Phase 2: Clear buffer + SAC on g=15
print('[2] BUFFER CLEARED + SAC on g=15 (15000 steps)')
bu.clear();env15=gym.make('Pendulum-v1');o,_=env15.reset()
for st in range(1,15001):
    a=ag.act(torch.tensor(o,dtype=torch.float32));o2,r,tr,tr2,_=env15.step([a[0]])
    bu.push(o,a[0],r,o2,1. if tr or tr2 else 0.)
    if st%2==0:ag.upd(bu)
    o=o2
    if tr or tr2:o,_=env15.reset()
g15=eval(ag,env15);print(f'  eval_g15={g15}/20 time={time.time()-t0:.0f}s')

# Forgetting test
g10f=eval(ag,env);print(f'  eval_g10_forgetting={g10f}/20')
print(f'Time: {time.time()-t0:.0f}s')
