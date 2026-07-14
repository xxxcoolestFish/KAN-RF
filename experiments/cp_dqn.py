"""CartPole: use DISCRETE actions (DQN-style) instead of SAC's continuous."""
import torch,torch.nn as nn,numpy as np,sys,os,time
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
DEVICE="cpu";torch.manual_seed(42);np.random.seed(42)
G=9.8;MC=1.;MP=.1;L=.5;DT=.02;FM=10.;XS=2.5;XDS=3.;THS=.3;THDS=3.

def cp_step(sr,a):
    x,xd,th,thd=sr[:,0],sr[:,1],sr[:,2],sr[:,3]
    f=a*FM;ct,st=torch.cos(th),torch.sin(th);tm_=MC+MP;pml_=MP*L
    tmp=(f+pml_*thd**2*st)/tm_;denom=.5*(4./3.-MP*ct**2/tm_)+1e-8
    th_a=(G*st-ct*tmp)/denom;x_a=tmp-pml_*th_a*ct/tm_
    return torch.stack([x+xd*DT+x_a*DT**2/2,xd+x_a*DT,th+thd*DT+th_a*DT**2/2,thd+th_a*DT],-1)

class DQN(torch.nn.Module):
    def __init__(self):super().__init__();self.n=nn.Sequential(nn.Linear(4,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,2))
    def forward(self,s):return self.n(s)

class Buf:
    def __init__(self,c=100000):
        self.s=deque(maxlen=c);self.a=deque(maxlen=c);self.r=deque(maxlen=c)
        self.s_=deque(maxlen=c);self.d=deque(maxlen=c)
    def push(self,s,a,r,s_,d):self.s.append(s);self.a.append([a]);self.r.append([r]);self.s_.append(s_);self.d.append([d])
    def sample(self,b=128):
        i=np.random.randint(0,len(self.s),b)
        return(torch.tensor(np.array([self.s[j] for j in i]),dtype=torch.float32),
               torch.tensor(np.array([self.a[j] for j in i]),dtype=torch.long),
               torch.tensor(np.array([self.r[j] for j in i]),dtype=torch.float32),
               torch.tensor(np.array([self.s_[j] for j in i]),dtype=torch.float32),
               torch.tensor(np.array([self.d[j] for j in i]),dtype=torch.float32))
    def __len__(self):return len(self.s)

from collections import deque
print("DQN on CartPole (discrete: push left/right)")
ag=DQN();tg=DQN();tg.load_state_dict(ag.state_dict())
opt=torch.optim.Adam(ag.parameters(),lr=3e-4);bu=Buf();mse=nn.MSELoss()
eps=1.;rng=np.random.RandomState(42);t0=time.time()

o=torch.tensor([[0.,0.,rng.uniform(-.05,.05)/THS,0.]],dtype=torch.float32)
for st in range(1,100001):
    if rng.rand()<eps:
        a=rng.randint(0,2)
        fwd=1 if a==1 else -1
    else:
        with torch.no_grad():fv=ag(o).squeeze(0);a=fv.argmax().item();fwd=1 if a==1 else -1
    sr=o.clone();sr[:,0]*=XS;sr[:,1]*=XDS;sr[:,2]*=THS;sr[:,3]*=THDS
    sn=cp_step(sr,torch.tensor([float(fwd)]))
    sn[:,0]/=XS;sn[:,1]/=XDS;sn[:,2]/=THS;sn[:,3]/=THDS
    d=1. if abs(sn[0,2].item()*THS)>.21 or abs(sn[0,0].item()*XS)>2.4 else 0.
    bu.push(o.squeeze(0).numpy(),a,1-d,sn.squeeze(0).numpy(),d)
    o=sn
    if d:o=torch.tensor([[0.,0.,rng.uniform(-.05,.05)/THS,0.]],dtype=torch.float32)
    eps=max(.01,eps*.995)
    if st%4==0 and len(bu)>128:
        s,a,r,s_,d=bu.sample(128)
        with torch.no_grad():tq=r+.99*(1-d)*tg(s_).max(1,keepdim=True)[0]
        l=((ag(s).gather(1,a)-tq)**2).mean();opt.zero_grad();l.backward();opt.step()
    if st%4==0:tg.load_state_dict(ag.state_dict())
    if st%20000==0:
        sc=0
        for _ in range(10):
            th_=rng.uniform(-.05,.05);s=torch.tensor([[0.,0.,th_/THS,0.]],dtype=torch.float32)
            for _s in range(500):
                with torch.no_grad():a=ag(s).argmax().item()
                sr2=s.clone();sr2[:,0]*=XS;sr2[:,1]*=XDS;sr2[:,2]*=THS;sr2[:,3]*=THDS
                sn2=cp_step(sr2,torch.tensor([1. if a==1 else -1.]));sn2[:,0]/=XS;sn2[:,1]/=XDS;sn2[:,2]/=THS;sn2[:,3]/=THDS;s=sn2
                if abs(s[0,2].item()*THS)>.21 or abs(s[0,0].item()*XS)>2.4:break
            if _s>=499:sc+=1
        print(f"  Step{st:6d} eps={eps:.3f} eval={sc}/10 time={time.time()-t0:.0f}s")
print(f"Done. {time.time()-t0:.0f}s")