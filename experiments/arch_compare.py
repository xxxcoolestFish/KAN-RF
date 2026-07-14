"""Compare MLP, KAN, ProtoKAN as policy network in SAC + WM error."""
import torch,torch.nn as nn,numpy as np,gymnasium as gym,time,sys,os;from collections import deque
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN,KAN
from experiments.baseline_sweep import generate_pendulum_data,train_wm
DEVICE="cpu";PI_2=np.pi/2;torch.manual_seed(42);np.random.seed(42);LOG_MIN=-5;LOG_MAX=2;mse=nn.MSELoss()

class Buf:
    def __init__(self,c=100000):
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
    lp=lp-(2*(np.log(2)-u-nn.functional.softplus(-2*u))).sum(-1,keepdim=True);return a,lp

def make_sac(actor_cls,critic_cls,lr=3e-4,gamma=.99,tau=.005,alpha=.2):
    class SAC:
        def __init__(self):
            self.g=gamma;self.t=tau;self.al=alpha;self.a=actor_cls().to(DEVICE)
            self.q1=critic_cls().to(DEVICE);self.q2=critic_cls().to(DEVICE)
            self.t1=critic_cls().to(DEVICE);self.t2=critic_cls().to(DEVICE)
            self.t1.load_state_dict(self.q1.state_dict());self.t2.load_state_dict(self.q2.state_dict())
            self.ao=torch.optim.Adam(self.a.parameters(),lr=lr)
            self.q1o=torch.optim.Adam(self.q1.parameters(),lr=lr);self.q2o=torch.optim.Adam(self.q2.parameters(),lr=lr)
            self.la=torch.tensor(np.log(alpha),requires_grad=True);self.lao=torch.optim.Adam([self.la],lr=lr)
        def act(self,s,dt=False):
            with torch.no_grad():m,l=self.a(s[None]);x=m if dt else m+torch.randn_like(m)*l.exp();return torch.tanh(x).squeeze(0).numpy()
        def upd(self,buf,b=128):
            if len(buf)<b:return;s,a,r,s_,d=buf.sample(b)
            with torch.no_grad():a_,lp_=actor_samp(self.a,s_);q=torch.min(self.t1(s_,a_),self.t2(s_,a_))-self.al*lp_;y=r+self.g*(1-d)*q
            self.q1o.zero_grad();((self.q1(s,a)-y)**2).mean().backward();self.q1o.step()
            self.q2o.zero_grad();((self.q2(s,a)-y)**2).mean().backward();self.q2o.step()
            a2,lp2=actor_samp(self.a,s);self.ao.zero_grad();(self.al*lp2-self.q1(s,a2)).mean().backward();self.ao.step()
            self.lao.zero_grad();(-self.la*(lp2+2).detach().mean()).backward();self.lao.step();self.al=self.la.exp().item()
            with torch.no_grad():
                for p,t in zip(self.q1.parameters(),self.t1.parameters()):t.mul_(1-self.t).add_(self.t*p)
                for p,t in zip(self.q2.parameters(),self.t2.parameters()):t.mul_(1-self.t).add_(self.t*p)
    return SAC()

class MLPA(nn.Module):
    def __init__(self):super().__init__();self.f=nn.Sequential(nn.Linear(3,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU());self.m=nn.Linear(128,1);self.l=nn.Linear(128,1)
    def forward(self,s):x=self.f(s);return self.m(x),torch.clamp(self.l(x),LOG_MIN,LOG_MAX)
class KANA(nn.Module):
    def __init__(self):super().__init__();self.kan=KAN([3,16,2],grid_size=5,spline_order=3)
    def forward(self,s):x=self.kan(s);return x[:,0:1],torch.clamp(x[:,1:2],LOG_MIN,LOG_MAX)
class PKA(nn.Module):
    def __init__(self):super().__init__();self.kan=ProtoKAN([3,16,2],n_prototypes=16,grid_range=1.0)
    def forward(self,s):x=self.kan(s);return x[:,0:1],torch.clamp(x[:,1:2],LOG_MIN,LOG_MAX)
class Critic(nn.Module):
    def __init__(self):super().__init__();self.n=nn.Sequential(nn.Linear(4,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,s,a):return self.n(torch.cat([s,a],-1))

def eval_p(ag,env,n=20):
    sc=0
    for t in range(n):
        o,_=env.reset(seed=42+t*100)
        for _ in range(200):
            a=ag.act(torch.tensor(o,dtype=torch.float32),dt=True);o,_,_,_,_=env.step([a[0]])
            if min(abs(np.arctan2(o[1],o[0])-PI_2),2*np.pi-abs(np.arctan2(o[1],o[0])-PI_2))<.2:sc+=1;break
    return sc

def train_and_measure(name,actor_cls,steps=10000):
    print(f"\n[{name}] training {steps} steps")
    ag=make_sac(actor_cls,lambda:Critic());bu=Buf();env=gym.make("Pendulum-v1");o,_=env.reset();t0=time.time()
    for s in range(1,steps+1):
        a=ag.act(torch.tensor(o,dtype=torch.float32));o2,r,tr,tr2,_=env.step([a[0]])
        bu.push(o,a[0],r,o2,1. if tr or tr2 else 0.)
        if s%2==0:ag.upd(bu);o=o2
        if tr or tr2:o,_=env.reset()
    sc=eval_p(ag,env);tm=time.time()-t0
    with torch.no_grad():
        s,a,_,s_,_=bu.sample(min(2000,len(bu)));wp=wm(torch.cat([s,a],-1));we=mse(wp,s_).item()
    print(f"  eval={sc}/20  wm_err={we:.6f}  time={tm:.0f}s")
    return ag,bu,sc,we,tm

print("Architecture Comparison: MLP vs KAN vs ProtoKAN");t0=time.time()
print("\n[WM] Train on g=10");X,Y=generate_pendulum_data(5000);wm,_=train_wm(X.to(DEVICE),Y.to(DEVICE))

mlp_ag,mlp_bu,mlp_sc,mlp_we,mlp_tm=train_and_measure("MLP",MLPA,10000)
kan_ag,kan_bu,kan_sc,kan_we,kan_tm=train_and_measure("KAN",KANA,10000)
pka_ag,pka_bu,pka_sc,pka_we,pka_tm=train_and_measure("ProtoKAN",PKA,10000)

print("\n"+"="*60);print("SUMMARY");print("="*60)
print(f"  {'Arch':>10} {'Eval':>6} {'WM_err':>10} {'Time':>6}")
print(f"  {'MLP':>10} {mlp_sc:>5}/20 {mlp_we:>10.6f} {mlp_tm:>5.0f}s")
print(f"  {'KAN':>10} {kan_sc:>5}/20 {kan_we:>10.6f} {kan_tm:>5.0f}s")
print(f"  {'ProtoKAN':>10} {pka_sc:>5}/20 {pka_we:>10.6f} {pka_tm:>5.0f}s")
print(f"\nWM error on KAN data:  {kan_we:.6f}  {'CLEAN' if kan_we<0.1 else 'NOISY'}")
print(f"WM error on ProtoKAN data: {pka_we:.6f}  {'CLEAN' if pka_we<0.1 else 'NOISY'}")
