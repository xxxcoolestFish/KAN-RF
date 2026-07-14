"""KAN-SAC vs MLP-SAC on Acrobot with GRAVITY change (harder test)."""
import torch,torch.nn as nn,numpy as np,sys,os,time;from collections import deque
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN
DEVICE="cpu";torch.manual_seed(42);np.random.seed(42);LOG_MIN=-5;LOG_MAX=2
L1=L2=1.;LC1=LC2=.5;I1=I2=1.;DT=.05;MAX_V1=6.;MAX_V2=8.;TARGET_H=1.0

def acrobot_step(sr,a,m1=1.,m2=1.,g=9.8):
    c1,s1,c2,s2=sr[:,0],sr[:,1],sr[:,2],sr[:,3];t1=torch.atan2(s1,c1);t2=torch.atan2(s2,c2)
    st1,st2=torch.sin(t1),torch.sin(t2);ct1,ct2=torch.cos(t1),torch.cos(t2);st12=torch.sin(t1+t2)
    d11=m1*LC1**2+m2*(L1**2+LC2**2+2*L1*LC2*ct2)+I1+I2
    d22=m2*LC2**2+I2;d12=m2*(LC2**2+L1*LC2*ct2)+I2
    p2=m2*g*LC2*st12;p1=-(m1*LC1+m2*L1)*g*st1-p2
    c1h=2*m2*L1*LC2*st2;h=c1h*(sr[:,4]*MAX_V1);d1=sr[:,4]*MAX_V1;d2=sr[:,5]*MAX_V2
    t2a=(a[...,0]+d12/d11*p1-h*d2-c1h*d1*d2-p2)/(d22-d12**2/d11+1e-6);t1a=-(d12*t2a+p1)/d11
    d1=d1+t1a*DT;d2=d2+t2a*DT;d1=d1.clamp(-MAX_V1,MAX_V1);d2=d2.clamp(-MAX_V2,MAX_V2)
    t1=t1+d1*DT;t2=t2+d2*DT
    return torch.stack([torch.cos(t1),torch.sin(t1),torch.cos(t2),torch.sin(t2),d1/MAX_V1,d2/MAX_V2],-1)

def tip_h(s):
    t1=torch.atan2(s[:,1],s[:,0]);t12=torch.atan2(s[:,3],s[:,2])+t1
    return -L1*torch.cos(t1)-L2*torch.cos(t12)

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
    lp=lp-(2*(np.log(2)-u-nn.functional.softplus(-2*u))).sum(-1,keepdim=True);return a,lp
class SAC:
    def __init__(self,make_a,make_q,lr=3e-4,gamma=.99,tau=.005,alpha=.2):
        self.g=gamma;self.t=tau;self.al=alpha
        self.a=make_a().to(DEVICE);self.q1=make_q().to(DEVICE);self.q2=make_q().to(DEVICE)
        self.t1=make_q().to(DEVICE);self.t2=make_q().to(DEVICE)
        self.t1.load_state_dict(self.q1.state_dict());self.t2.load_state_dict(self.q2.state_dict())
        self.ao=torch.optim.Adam(self.a.parameters(),lr=lr)
        self.q1o=torch.optim.Adam(self.q1.parameters(),lr=lr);self.q2o=torch.optim.Adam(self.q2.parameters(),lr=lr)
        self.la=torch.tensor(np.log(alpha),requires_grad=True);self.lao=torch.optim.Adam([self.la],lr=lr)
    def act(self,s,dt=False):
        with torch.no_grad():m,l=self.a(s[None]);x=m if dt else m+torch.randn_like(m)*l.exp();return torch.tanh(x).squeeze(0).numpy()
    def upd(self,buf,b=128):
        if len(buf)<b:
            return
        s,a,r,s_next,d=buf.sample(b)
        s_=s_next
        with torch.no_grad():a_,lp_=actor_samp(self.a,s_);q=torch.min(self.t1(s_,a_),self.t2(s_,a_))-self.al*lp_;y=r+self.g*(1-d)*q
        self.q1o.zero_grad();((self.q1(s,a)-y)**2).mean().backward();self.q1o.step()
        self.q2o.zero_grad();((self.q2(s,a)-y)**2).mean().backward();self.q2o.step()
        a2,lp2=actor_samp(self.a,s);self.ao.zero_grad();(self.al*lp2-self.q1(s,a2)).mean().backward();self.ao.step()
        self.lao.zero_grad();(-self.la*(lp2+2).detach().mean()).backward();self.lao.step();self.al=self.la.exp().item()
        with torch.no_grad():
            for p,t in zip(self.q1.parameters(),self.t1.parameters()):t.mul_(1-self.t).add_(self.t*p)
            for p,t in zip(self.q2.parameters(),self.t2.parameters()):t.mul_(1-self.t).add_(self.t*p)

class MLPA(nn.Module):
    def __init__(self):super().__init__();self.f=nn.Sequential(nn.Linear(6,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU());self.m=nn.Linear(128,1);self.l=nn.Linear(128,1)
    def forward(self,s):x=self.f(s);return self.m(x),torch.clamp(self.l(x),LOG_MIN,LOG_MAX)
class MLPQ(nn.Module):
    def __init__(self):super().__init__();self.n=nn.Sequential(nn.Linear(7,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,s,a):return self.n(torch.cat([s,a],-1))
class KANA(nn.Module):
    def __init__(self):super().__init__();self.kan=KAN([6,16,2],grid_size=5,spline_order=3)
    def forward(self,s):x=self.kan(s);return x[:,0:1],torch.clamp(x[:,1:2],LOG_MIN,LOG_MAX)
class KANQ(nn.Module):
    def __init__(self):super().__init__();self.kan=KAN([7,16,1],grid_size=5,spline_order=3)
    def forward(self,s,a):return self.kan(torch.cat([s,a],-1))

def eval_ac(ag,n=10,g=9.8):
    sc=0
    for t in range(n):
        th1=np.random.uniform(-.1,.1);th2=np.random.uniform(-.1,.1)
        s=torch.tensor([[np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),0.,0.]],dtype=torch.float32)
        for _ in range(500):
            a=ag.act(s.squeeze(0),dt=True);s=acrobot_step(s,torch.tensor([a]),g=g)
            if tip_h(s).item()>TARGET_H:sc+=1;break
    return sc

def train_ph(ag,bu,steps,g,label):
    th1=np.random.uniform(-.1,.1);th2=np.random.uniform(-.1,.1)
    o=torch.tensor([[np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),0.,0.]],dtype=torch.float32)
    t0=time.time()
    for s in range(1,steps+1):
        a=ag.act(o.squeeze(0));o2=acrobot_step(o,torch.tensor([a]),g=g)
        h=tip_h(o2).item();d=1. if h>TARGET_H else 0.
        bu.push(o.squeeze(0).numpy(),float(a),h,o2.squeeze(0).numpy(),d);o=o2
        if s%2==0:ag.upd(bu)
        if d:th1=np.random.uniform(-.1,.1);th2=np.random.uniform(-.1,.1);o=torch.tensor([[np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),0.,0.]],dtype=torch.float32)
    sc=eval_ac(ag,10,g);print(f"  {label}: {sc}/10 time={time.time()-t0:.0f}s");return sc

print("KAN-SAC vs MLP-SAC: Acrobot GRAVITY change (g=9.8->19.6)");t0=time.time()
print("\n[MLP] g=9.8 (10000 steps)")
ma=SAC(lambda:MLPA(),lambda:MLPQ());mb=Buf()
mg1=train_ph(ma,mb,10000,9.8,"MLP g=9.8")
print("\n[KAN] g=9.8 (10000 steps)")
ka=SAC(lambda:KANA(),lambda:KANQ());kb=Buf()
kg1=train_ph(ka,kb,10000,9.8,"KAN g=9.8")
print("\n[ADAPT] Both to g=19.6 (15000 more steps)")
mg2=train_ph(ma,mb,15000,19.6,"MLP g=19.6")
kg2=train_ph(ka,kb,15000,19.6,"KAN g=19.6")
print(f"\nSUMMARY: MLP {mg1}/10->{mg2}/10  KAN {kg1}/10->{kg2}/10")
