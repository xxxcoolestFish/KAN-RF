import torch,torch.nn as nn,numpy as np,gymnasium as gym,time,sys,os;from collections import deque
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN,KANLayer
DEVICE="cpu";PI_2=np.pi/2;torch.manual_seed(42);np.random.seed(42);LOG_MIN=-5;LOG_MAX=2
class Buf:
    def __init__(self,c=200000):
        self.s=deque(maxlen=c);self.a=deque(maxlen=c);self.r=deque(maxlen=c);self.s_=deque(maxlen=c);self.d=deque(maxlen=c)
    def push(self,s,a,r,s_,d):self.s.append(s);self.a.append([a]);self.r.append([r]);self.s_.append(s_);self.d.append([d])
    def sample(self,b=128):
        i=np.random.randint(0,len(self.s),b)
        return(torch.tensor(np.array([self.s[j] for j in i]),dtype=torch.float32),
               torch.tensor(np.array([self.a[j] for j in i]),dtype=torch.float32),
               torch.tensor(np.array([self.r[j] for j in i]),dtype=torch.float32),
               torch.tensor(np.array([self.s_[j] for j in i]),dtype=torch.float32),
               torch.tensor(np.array([self.d[j] for j in i]),dtype=torch.float32))
    def __len__(self):return len(self.s)

def actor_samp(actor,s):
    m,l=actor(s);std=l.exp();e=torch.randn_like(std);u=m+std*e;a=torch.tanh(u)
    lp=-((e**2+2*l+np.log(2*np.pi)).sum(-1,keepdim=True)/2)
    lp=lp-(2*(np.log(2)-u-nn.functional.softplus(-2*u))).sum(-1,keepdim=True)
    return a,lp

class SAC:
    def __init__(self,make_actor,make_q,lr=3e-4,gamma=.99,tau=.005,alpha=.2):
        self.g=gamma;self.t=tau;self.al=alpha
        self.a=make_actor().to(DEVICE);self.q1=make_q().to(DEVICE);self.q2=make_q().to(DEVICE)
        self.t1=make_q().to(DEVICE);self.t2=make_q().to(DEVICE)
        self.t1.load_state_dict(self.q1.state_dict());self.t2.load_state_dict(self.q2.state_dict())
        self.ao=torch.optim.Adam(self.a.parameters(),lr=lr)
        self.q1o=torch.optim.Adam(self.q1.parameters(),lr=lr);self.q2o=torch.optim.Adam(self.q2.parameters(),lr=lr)
        self.la=torch.tensor(np.log(alpha),requires_grad=True);self.lao=torch.optim.Adam([self.la],lr=lr)
    def act(self,s,dt=False):
        with torch.no_grad():m,l=self.a(s[None]);x=m if dt else m+torch.randn_like(m)*l.exp();return torch.tanh(x).squeeze(0).numpy()
    def upd(self,buf,b=128):
        if len(buf)<b:return
        s,a,r,s_next,d=buf.sample(b)
        with torch.no_grad():
            a_next,lp_next=actor_samp(self.a,s_next)
            q=torch.min(self.t1(s_next,a_next),self.t2(s_next,a_next))-self.al*lp_next
            y=r+self.g*(1-d)*q
        self.q1o.zero_grad();((self.q1(s,a)-y)**2).mean().backward();self.q1o.step()
        self.q2o.zero_grad();((self.q2(s,a)-y)**2).mean().backward();self.q2o.step()
        a2,lp2=actor_samp(self.a,s)
        self.ao.zero_grad();(self.al*lp2-self.q1(s,a2)).mean().backward();self.ao.step()
        self.lao.zero_grad();(-self.la*(lp2+2).detach().mean()).backward();self.lao.step();self.al=self.la.exp().item()
        with torch.no_grad():
            for p,t in zip(self.q1.parameters(),self.t1.parameters()):t.mul_(1-self.t).add_(self.t*p)
            for p,t in zip(self.q2.parameters(),self.t2.parameters()):t.mul_(1-self.t).add_(self.t*p)

class MLPActor(nn.Module):
    def __init__(self):super().__init__();self.f=nn.Sequential(nn.Linear(3,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU());self.m=nn.Linear(128,1);self.l=nn.Linear(128,1)
    def forward(self,s):x=self.f(s);return self.m(x),torch.clamp(self.l(x),LOG_MIN,LOG_MAX)
class MLPCritic(nn.Module):
    def __init__(self):super().__init__();self.n=nn.Sequential(nn.Linear(4,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,s,a):return self.n(torch.cat([s,a],-1))

class KANActor(nn.Module):
    def __init__(self):super().__init__();self.kan=KAN([3,16,2],grid_size=5,spline_order=3)
    def forward(self,s):x=self.kan(s);return x[:,0:1],torch.clamp(x[:,1:2],LOG_MIN,LOG_MAX)
class KANCritic(nn.Module):
    def __init__(self):super().__init__();self.kan=KAN([4,16,1],grid_size=5,spline_order=3)
    def forward(self,s,a):return self.kan(torch.cat([s,a],-1))

def eval_p(ag,env,n=20):
    sc=0
    for t in range(n):
        o,_=env.reset(seed=42+t*100)
        for _ in range(200):
            a=ag.act(torch.tensor(o,dtype=torch.float32),dt=True);o,_,_,_,_=env.step([a[0]])
            if min(abs(np.arctan2(o[1],o[0])-PI_2),2*np.pi-abs(np.arctan2(o[1],o[0])-PI_2))<.2:sc+=1;break
    return sc

def train_ph(ag,env,bu,steps,label):
    o,_=env.reset();t0=time.time()
    for s in range(1,steps+1):
        a=ag.act(torch.tensor(o,dtype=torch.float32));o2,r,tr,tr2,_=env.step([a[0]])
        bu.push(o,a[0],r,o2,1. if tr or tr2 else 0.)
        if s%2==0:ag.upd(bu)
        o=o2
        if tr or tr2:o,_=env.reset()
    sc=eval_p(ag,env);print(f"  {label}: {sc}/20 time={time.time()-t0:.0f}s");return sc

print("KAN-SAC vs MLP-SAC Adaptation");t0=time.time()
print("\n[MLP-SAC] g=10 (10000 steps)")
m_ag=SAC(lambda:MLPActor(),lambda:MLPCritic());m_bu=Buf();m_env=gym.make("Pendulum-v1")
m_g10=train_ph(m_ag,m_env,m_bu,10000,"MLP g=10")
print("\n[KAN-SAC] g=10 (10000 steps)")
k_ag=SAC(lambda:KANActor(),lambda:KANCritic());k_bu=Buf();k_env=gym.make("Pendulum-v1")
k_g10=train_ph(k_ag,k_env,k_bu,10000,"KAN g=10")

print("\n[ADAPT] g=15 (15000 more steps)")
env15=gym.make("Pendulum-v1")
m_g15=train_ph(m_ag,env15,m_bu,15000,"MLP g=15");k_g15=train_ph(k_ag,env15,k_bu,15000,"KAN g=15")
print(f"\nSUMMARY:  MLP {m_g10}/20 -> {m_g15}/20  KAN {k_g10}/20 -> {k_g15}/20")
