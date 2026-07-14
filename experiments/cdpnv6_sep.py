"""CDPNv6: physics_head/decision_head separation + WM consistency."""
import torch,torch.nn as nn,numpy as np,gymnasium as gym,time,sys,os;from collections import deque
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN;from experiments.baseline_sweep import generate_pendulum_data,train_wm
DEVICE="cpu";PI_2=np.pi/2;torch.manual_seed(42);np.random.seed(42);LOG_MIN=-5;LOG_MAX=2;mse=nn.MSELoss()

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

class PhysicsHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(3,32),nn.ReLU(),nn.Linear(32,16))
    def forward(self,s):return self.net(s)

class DecisionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(3+16,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.m=nn.Linear(128,1);self.l=nn.Linear(128,1)
    def forward(self,s,f):x=torch.cat([s,f],-1);x=self.net(x);return self.m(x),torch.clamp(self.l(x),LOG_MIN,LOG_MAX)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(16+1,32),nn.ReLU(),nn.Linear(32,3))
    def forward(self,f,a):return self.net(torch.cat([f,a],-1))

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.ph=PhysicsHead();self.dh=DecisionHead()
    def forward(self,s):f=self.ph(s);m,l=self.dh(s,f);return m,l
    def get_f(self,s):return self.ph(s)

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.n=nn.Sequential(nn.Linear(4,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,s,a):return self.n(torch.cat([s,a],-1))

class SAC:
    def __init__(self,lr=3e-4,gamma=.99,tau=.005,alpha=.2):
        self.g=gamma;self.t=tau;self.al=alpha
        self.a=Actor().to(DEVICE);self.decoder=Decoder().to(DEVICE)
        self.q1=QNet().to(DEVICE);self.q2=QNet().to(DEVICE)
        self.t1=QNet().to(DEVICE);self.t2=QNet().to(DEVICE)
        self.t1.load_state_dict(self.q1.state_dict());self.t2.load_state_dict(self.q2.state_dict())
        self.ao=torch.optim.Adam(list(self.a.parameters())+list(self.decoder.parameters()),lr=lr)
        self.q1o=torch.optim.Adam(self.q1.parameters(),lr=lr)
        self.q2o=torch.optim.Adam(self.q2.parameters(),lr=lr)
        self.la=torch.tensor(np.log(alpha),requires_grad=True);self.lao=torch.optim.Adam([self.la],lr=lr)
    def act(self,s,dt=False):
        with torch.no_grad():m,l=self.a(s[None]);a=m if dt else m+torch.randn_like(m)*l.exp();return torch.tanh(a).squeeze(0).numpy()
    def upd(self,buf,wm,lam_wm=0.5,b=128):
        if len(buf)<b:
            return
        s,a,r,s_,d=buf.sample(b)
        with torch.no_grad():a_,lp_=self.a(s_);q=torch.min(self.t1(s_,a_),self.t2(s_,a_))-self.al*lp_;y=r+self.g*(1-d)*q
        self.q1o.zero_grad();((self.q1(s,a)-y)**2).mean().backward();self.q1o.step()
        self.q2o.zero_grad();((self.q2(s,a)-y)**2).mean().backward();self.q2o.step()
        a_,lp=self.a(s);sac_loss=(self.al*lp-self.q1(s,a_)).mean()
        f=self.a.get_f(s);s_pred_wm=wm(torch.cat([s,a],-1));s_pred_d=self.decoder(f,a)
        wm_loss=mse(s_pred_d,s_pred_wm.detach())
        total_loss=sac_loss+lam_wm*wm_loss
        self.ao.zero_grad();total_loss.backward();torch.nn.utils.clip_grad_norm_(self.a.parameters(),10.);self.ao.step()
        self.lao.zero_grad();(-self.la*(lp+2).detach().mean()).backward();self.lao.step();self.al=self.la.exp().item()
        with torch.no_grad():
            for p,t in zip(self.q1.parameters(),self.t1.parameters()):t.mul_(1-self.t).add_(self.t*p)
            for p,t in zip(self.q2.parameters(),self.t2.parameters()):t.mul_(1-self.t).add_(self.t*p)

def eval_sac(ag,env,n=20):
    sc=0
    for t in range(n):
        o,_=env.reset(seed=42+t*100)
        for _ in range(200):
            a=ag.act(torch.tensor(o,dtype=torch.float32),dt=True);o,_,_,_,_=env.step([a[0]])
            if min(abs(np.arctan2(o[1],o[0])-PI_2),2*np.pi-abs(np.arctan2(o[1],o[0])-PI_2))<.2:sc+=1;break
    return sc

print("CDPNv6: Physics/Decision separation + WM consistency");t0=time.time()

print("\n[1] Train WM on g=10");X,Y=generate_pendulum_data(5000);wm,_=train_wm(X.to(DEVICE),Y.to(DEVICE))

print("\n[2] SAC on g=10 (15000 steps)")
ag=SAC();bu=Buf();env=gym.make("Pendulum-v1");o,_=env.reset()
for st in range(1,15001):
    a=ag.act(torch.tensor(o,dtype=torch.float32));o2,r,tr,tr2,_=env.step([a[0]])
    bu.push(o,a[0],r,o2,1. if tr or tr2 else 0.)
    if st%2==0:ag.upd(bu,wm,lam_wm=0.5)
    o=o2
    if tr or tr2:o,_=env.reset()
g10=eval_sac(ag,env);print(f"  Step15000 eval_g10={g10}/20 time={time.time()-t0:.0f}s")

print("\n[3] WM adapt: g=10 -> g=15");X15,Y15=generate_pendulum_data(5000,seed=43)
optwm=torch.optim.Adam(wm.parameters(),lr=1e-4);wm.train()
for _ in range(200):
    idx=torch.randint(0,len(X15),(256,));l=mse(wm(X15[idx].to(DEVICE)),Y15[idx].to(DEVICE))
    optwm.zero_grad();l.backward();optwm.step()
wm.eval();print("  WM fine-tuned to g=15")

print("\n[4] Freeze decision_head, adapt physics_head+decoder")
for p in ag.a.dh.parameters():p.requires_grad=False
ag.decoder=Decoder().to(DEVICE)
opt_ad=torch.optim.Adam(list(ag.decoder.parameters())+list(ag.a.ph.parameters()),lr=1e-3)
env15=gym.make("Pendulum-v1");o,_=env15.reset()
for step in range(2000):
    a=ag.act(torch.tensor(o,dtype=torch.float32));o2,r,tr,tr2,_=env15.step([a[0]])
    s_t=torch.tensor(o2,dtype=torch.float32).unsqueeze(0);a_t=torch.tensor([[a[0]]])
    f=ag.a.get_f(s_t);s_wm=wm(torch.cat([s_t,a_t],-1));s_d=ag.decoder(f,a_t)
    l=mse(s_d,s_wm.detach());opt_ad.zero_grad();l.backward();opt_ad.step()
    o=o2
    if tr or tr2:o,_=env15.reset()
    if step%500==0:print(f"  adapt_step{step:4d} wm_loss={l.item():.6f}")
for p in ag.a.dh.parameters():p.requires_grad=True

g15=eval_sac(ag,env15);g10f=eval_sac(ag,env)
print(f"  eval_g15={g15}/20  eval_g10_forgetting={g10f}/20")
print(f"Time: {time.time()-t0:.0f}s")
