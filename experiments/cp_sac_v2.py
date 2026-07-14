import torch,torch.nn as nn,numpy as np,sys,os,time
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
DEVICE='cpu';torch.manual_seed(42);np.random.seed(42)
LOG_MIN=-5;LOG_MAX=2;G=9.8;MC=1.;MP=.1;L=.5;DT=.02;FM=10.;XS=2.5;XDS=3.;THS=.3;THDS=3.;ACT_SCALE=0.5

def cp_step(sr,a):
    if a.dim()>1:a=a.squeeze(-1)
    x,xd,th,thd=sr[:,0],sr[:,1],sr[:,2],sr[:,3]
    f=a*FM;ct,st=torch.cos(th),torch.sin(th);tm_=MC+MP;pml_=MP*L
    tmp=(f+pml_*thd**2*st)/tm_;denom=.5*(4./3.-MP*ct**2/tm_)+1e-8
    th_a=(G*st-ct*tmp)/denom;x_a=tmp-pml_*th_a*ct/tm_
    return torch.stack([x+xd*DT+x_a*DT**2/2,xd+x_a*DT,th+thd*DT+th_a*DT**2/2,thd+th_a*DT],-1)

class Buf:
    def __init__(self,c=200000,sd=4):
        self.s=np.zeros((c,sd),dtype=np.float32);self.a=np.zeros((c,1),dtype=np.float32)
        self.r=np.zeros((c,1),dtype=np.float32);self.s_=np.zeros((c,sd),dtype=np.float32)
        self.d=np.zeros((c,1),dtype=np.float32);self.p=0;self.n=0;self.c=c
    def push(self,s,a,r,s_,d):
        self.s[self.p]=s;self.a[self.p]=a;self.r[self.p]=r;self.s_[self.p]=s_;self.d[self.p]=d
        self.p=(self.p+1)%self.c;self.n=min(self.n+1,self.c)
    def sample(self,b=256):
        i=np.random.randint(0,self.n,b)
        return(torch.tensor(self.s[i]),torch.tensor(self.a[i]),torch.tensor(self.r[i]),torch.tensor(self.s_[i]),torch.tensor(self.d[i]))
    def __len__(self):return self.n

class ActorNet(nn.Module):
    def __init__(self,d):super().__init__();self.f=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU());self.m=nn.Linear(128,1);self.l=nn.Linear(128,1)
    def fwd(self,s):x=self.f(s);return self.m(x),torch.clamp(self.l(x),LOG_MIN,LOG_MAX)
    def sample(self,s):m,l=self.fwd(s);std=l.exp();e=torch.randn_like(std);u=m+std*e;a=torch.tanh(u);lp=-((e**2+2*l+np.log(2*np.pi)).sum(-1,keepdim=True)/2);lp=lp-(2*(np.log(2)-u-nn.functional.softplus(-2*u))).sum(-1,keepdim=True);return a,lp

class QNet(nn.Module):
    def __init__(self,d):super().__init__();self.n=nn.Sequential(nn.Linear(d+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,s,a):return self.n(torch.cat([s,a],-1))

class SAC:
    def __init__(self,d_in=4,lr=3e-4,gamma=.99,tau=.005,alpha=.2):
        self.g=gamma;self.t=tau;self.al=alpha
        self.a=ActorNet(d_in).to(DEVICE);self.q1=QNet(d_in).to(DEVICE);self.q2=QNet(d_in).to(DEVICE)
        self.t1=QNet(d_in).to(DEVICE);self.t2=QNet(d_in).to(DEVICE)
        self.t1.load_state_dict(self.q1.state_dict());self.t2.load_state_dict(self.q2.state_dict())
        self.ao=torch.optim.Adam(self.a.parameters(),lr=lr);self.q1o=torch.optim.Adam(self.q1.parameters(),lr=lr);self.q2o=torch.optim.Adam(self.q2.parameters(),lr=lr)
        self.la=torch.tensor(np.log(alpha),requires_grad=True);self.lao=torch.optim.Adam([self.la],lr=lr)
    def act(self,s,dt=False):
        with torch.no_grad():m,l=self.a.fwd(s[None]);a=m if dt else m+torch.randn_like(m)*l.exp();return torch.tanh(a).squeeze(0).numpy()
    def upd(self,buf,b=256):
        if len(buf)<b:
            return
        s,a,r,s_next,d=buf.sample(b)
        with torch.no_grad():
            a_next,lp_next=self.a.sample(s_next)
            q=torch.min(self.t1(s_next,a_next),self.t2(s_next,a_next))-self.al*lp_next
            y=r+self.g*(1-d)*q
        self.q1o.zero_grad();((self.q1(s,a)-y)**2).mean().backward();self.q1o.step()
        self.q2o.zero_grad();((self.q2(s,a)-y)**2).mean().backward();self.q2o.step()
        a_,lp=self.a.sample(s)
        self.ao.zero_grad();(self.al*lp-self.q1(s,a_)).mean().backward();self.ao.step()
        self.lao.zero_grad();(-self.la*(lp+2).detach().mean()).backward();self.lao.step();self.al=self.la.exp().item()
        with torch.no_grad():
            for p,t in zip(self.q1.parameters(),self.t1.parameters()):t.mul_(1-self.t).add_(self.t*p)
            for p,t in zip(self.q2.parameters(),self.t2.parameters()):t.mul_(1-self.t).add_(self.t*p)

print('Train WM...')
cpx,cpy=[],[]
for _ in range(5000):
    import numpy.random as rng
    x=rng.uniform(-2.4,2.4);xd=rng.uniform(-3.,3.);th=rng.uniform(-.3,.3);thd=rng.uniform(-3.,3.)
    ar=rng.uniform(-1.,1.);sr=torch.tensor([[x,xd,th,thd]],dtype=torch.float32)
    sn=cp_step(sr,torch.tensor([ar*ACT_SCALE]))
    sn[:,0]/=XS;sn[:,1]/=XDS;sn[:,2]/=THS;sn[:,3]/=THDS
    sr[:,0]/=XS;sr[:,1]/=XDS;sr[:,2]/=THS;sr[:,3]/=THDS
    cpx.append(torch.cat([sr,torch.tensor([[ar*ACT_SCALE]])],-1).float());cpy.append(sn.float())
cpx=torch.cat(cpx);cpy=torch.cat(cpy);nt=int(len(cpx)*.85);mse=nn.MSELoss()
wm=ProtoKAN([5,16,4],n_prototypes=16).to(DEVICE)
for l in wm.layers:l.log_sigma.data.fill_(-1.5)
opt=torch.optim.LBFGS(wm.parameters(),lr=1.,max_iter=20,history_size=50,line_search_fn='strong_wolfe')
bv=float('inf');bs=None
def c():opt.zero_grad();l=mse(wm(cpx[:nt]),cpy[:nt]);l.backward();return l
for _ in range(80):
    opt.step(c);v=mse(wm(cpx[nt:]),cpy[nt:]).item()
    if v<bv:bv=v;bs={k:v.clone() for k,v in wm.state_dict().items()}
wm.load_state_dict(bs);wm.eval();print(f'  WM val_mse={bv:.6f}')

print('SAC on CartPole (act_scale=0.5, H=100k)...')
ag=SAC(4);bu=Buf();rng=np.random.RandomState(42)
o=torch.tensor([[0.,0.,rng.uniform(-.05,.05)/THS,0.]],dtype=torch.float32)
t0=time.time()
for st in range(1,100001):
    a=ag.act(o.squeeze(0))
    sr=o.clone();sr[:,0]*=XS;sr[:,1]*=XDS;sr[:,2]*=THS;sr[:,3]*=THDS
    sn=cp_step(sr,torch.tensor([a*ACT_SCALE]))
    sn[:,0]/=XS;sn[:,1]/=XDS;sn[:,2]/=THS;sn[:,3]/=THDS
    d=1. if abs(sn[0,2].item()*THS)>.21 or abs(sn[0,0].item()*XS)>2.4 else 0.
    bu.push(o.squeeze(0).numpy(),a*ACT_SCALE,-(sn[0,2].item()**2),sn.squeeze(0).numpy(),d)
    if st%3==0:ag.upd(bu)
    o=sn
    if d:o=torch.tensor([[0.,0.,rng.uniform(-.05,.05)/THS,0.]],dtype=torch.float32)
    if st%20000==0:
        with torch.no_grad():s,a,_,s_,_=bu.sample(256);wp=wm(torch.cat([s,a],-1));we=mse(wp,s_).item()
        sc=0
        for _ in range(10):
            th_=rng.uniform(-.05,.05);s=torch.tensor([[0.,0.,th_/THS,0.]],dtype=torch.float32)
            for _s in range(500):
                a2=ag.act(s.squeeze(0),True)
                sr2=s.clone();sr2[:,0]*=XS;sr2[:,1]*=XDS;sr2[:,2]*=THS;sr2[:,3]*=THDS
                sn2=cp_step(sr2,torch.tensor([a2*ACT_SCALE]));sn2[:,0]/=XS;sn2[:,1]/=XDS;sn2[:,2]/=THS;sn2[:,3]/=THDS;s=sn2
                if abs(s[0,2].item()*THS)>.21 or abs(s[0,0].item()*XS)>2.4:break
            if _s>=499:sc+=1
        print(f'  Step{st:6d} wm_err={we:.6f} eval={sc}/10 time={time.time()-t0:.0f}s')
print(f'Done.')
