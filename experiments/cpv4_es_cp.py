"""CartPole CDPNv4+ES only (adapted from cpv4_es_adapt.py)."""
import torch,torch.nn as nn,numpy as np,sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
DEVICE="cpu";torch.manual_seed(42);np.random.seed(42)

G=9.8;MC=1.;MP=.1;L=.5;DT=.02;TM=MC+MP;PML=MP*L;FM=10.
XS=2.5;XDS=3.;THS=.3;THDS=3.
S_CP=torch.tensor([[0.,0.,0.,0.]])

def cp_step(sr,a,g=G,mp=MP,l=L):
    if a.dim()>1:a=a.squeeze(-1)
    x,xd,th,thd=sr[:,0],sr[:,1],sr[:,2],sr[:,3]
    f=a*FM;ct,st=torch.cos(th),torch.sin(th)
    tm_=MC+mp;pml_=mp*l
    tmp=(f+pml_*thd**2*st)/tm_
    th_a=(g*st-ct*tmp)/(.5*(4./3.-mp*ct**2/tm_)+1e-8)
    x_a=tmp-pml_*th_a*ct/tm_
    return torch.stack([x+xd*DT+x_a*DT**2/2,xd+x_a*DT,th+thd*DT+th_a*DT**2/2,thd+th_a*DT],dim=-1)

def gen_cp(n=5000,mp=MP,l=L):
    xs,ys=[],[]
    for _ in range(n):
        x=np.random.uniform(-2.4,2.4);xd=np.random.uniform(-3.,3.)
        th=np.random.uniform(-.3,.3);thd=np.random.uniform(-3.,3.)
        a=np.random.uniform(-1.,1.)
        sr=torch.tensor([[x,xd,th,thd]],dtype=torch.float32)
        sn=cp_step(sr,torch.tensor([a]),mp=mp,l=l)
        sn[:,0]/=XS;sn[:,1]/=XDS;sn[:,2]/=THS;sn[:,3]/=THDS
        sr[:,0]/=XS;sr[:,1]/=XDS;sr[:,2]/=THS;sr[:,3]/=THDS
        xs.append(torch.cat([sr,torch.tensor([[a]])],dim=-1).float())
        ys.append(sn.float())
    return torch.cat(xs),torch.cat(ys)

def train_wm(X,Y,np_=16,nl=100):
    nt=int(len(X)*.85)
    wm=ProtoKAN([5,np_,4],n_prototypes=np_).to(DEVICE)
    for l in wm.layers:l.log_sigma.data.fill_(-1.5)
    mse=nn.MSELoss();opt=torch.optim.LBFGS(wm.parameters(),lr=1.,max_iter=20,history_size=50,line_search_fn="strong_wolfe")
    bv=float("inf");bs=None
    def c():opt.zero_grad();l=mse(wm(X[:nt]),Y[:nt]);l.backward();return l
    for _ in range(nl):
        opt.step(c);v=mse(wm(X[nt:]),Y[nt:]).item()
        if v<bv:bv=v;bs={k:v.clone() for k,v in wm.state_dict().items()}
    wm.load_state_dict(bs);wm.eval();return wm

class Policy(nn.Module):
    def __init__(self,in_d,hid=32,out_d=1):
        super().__init__()
        self.n=nn.Sequential(nn.Linear(in_d,hid),nn.Tanh(),nn.Linear(hid,24),nn.Tanh(),nn.Linear(24,out_d),nn.Tanh())
    def forward(self,s,sg):return self.n(torch.cat([s,sg],dim=-1))

class ES:
    def __init__(self,policy,pop=30,sigma=.1,lr=.01):
        self.p=policy;self.P=pop;self.s=sigma;self.lr=lr
        self.d=sum(p.numel() for p in policy.parameters())
        self.m=torch.cat([p.data.view(-1) for p in policy.parameters()]).clone()
    def setp(self,params):
        i=0
        for p in self.p.parameters():
            n=p.numel();p.data.copy_(params[i:i+n].reshape(p.shape));i+=n
    def _rs(self):
        return torch.tensor([[np.random.uniform(-.25,.25)/XS,np.random.uniform(-1.5,1.5)/XDS,
            np.random.uniform(-.15,.15)/THS,np.random.uniform(-1.5,1.5)/THDS]],dtype=torch.float32)
    def _roll(self,policy,wm,s0,H):
        s=s0;c=0.
        for _ in range(H):
            a=policy(s,S_CP);s=wm(torch.cat([s,a],dim=-1));c+=((s-S_CP)**2).sum().item()
        return -c
    def train(self,wm,H=20,gens=100,wr=3,report=20):
        for g in range(1,gens+1):
            noise=torch.randn(self.P,self.d);fits=[]
            for i in range(self.P):
                self.setp(self.m+self.s*noise[i])
                f=0.
                for _ in range(wr):f+=self._roll(self.p,wm,self._rs(),H)
                fits.append(f/wr)
            ft=torch.tensor(fits);ft=(ft-ft.mean())/(ft.std()+1e-8)
            grad=(noise.T@ft)/(self.P*self.s);self.m+=self.lr*grad;self.setp(self.m)
            if g%report==0:print(f"    Gen {g:3d}  fit={ft.mean():+.3f}")

def eval_cp(policy,mp=MP,l=L,n=20,label=""):
    succ=0;steps=[]
    for t in range(n):
        np.random.seed(42+t*100);th=np.random.uniform(-.05,.05)
        s=torch.tensor([[0.,0.,th/THS,0.]],dtype=torch.float32)
        for st in range(500):
            sr=s.clone();sr[:,0]*=XS;sr[:,1]*=XDS;sr[:,2]*=THS;sr[:,3]*=THDS
            with torch.no_grad():a=policy(s,S_CP).item()
            sn=cp_step(sr,torch.tensor([a]),mp=mp,l=l)
            sn[:,0]/=XS;sn[:,1]/=XDS;sn[:,2]/=THS;sn[:,3]/=THDS
            s=sn
            if abs(s[0,2].item()*THS)>.21 or abs(s[0,0].item()*XS)>2.4:break
        steps.append(st+1)
        if st+1>=500:succ+=1
    print(f"  [{label}] {succ}/{n} ({succ*100//n}%) steps={np.mean(steps):.0f}")
    return succ

print("="*60);print("CARTIPOLE CDPNv4+ES");print("="*60)
print("\n[1] Train WM")
Xcp,Ycp=gen_cp(5000);wmc=train_wm(Xcp.to(DEVICE),Ycp.to(DEVICE))

print("\n[2] ES train policy (100 gens)")
pol=Policy(8,32,1);es=ES(pol,pop=30,sigma=.1,lr=.01)
es.train(wmc,H=20,gens=100,wr=3)

print("\n[3] Evaluate")
eval_cp(pol,label="CartPole default (m=0.1, L=0.5)")

print("\n[4] WM adapt: pole mass 0.1 -> 0.2")
Xh,Yh=gen_cp(2000,mp=0.2)
opt=torch.optim.Adam(wmc.parameters(),lr=1e-3);wmc.train()
for _ in range(50):
    idx=torch.randint(0,len(Xh),(256,));l=nn.MSELoss()(wmc(Xh[idx].to(DEVICE)),Yh[idx].to(DEVICE))
    opt.zero_grad();l.backward();opt.step()
wmc.eval()
eval_cp(pol,mp=0.2,label="m=0.2 after WM adapt (no policy retrain)")

print("\n[5] Quick ES adaptation (30 gens)")
es.train(wmc,H=20,gens=30,wr=3)
eval_cp(pol,mp=0.2,label="m=0.2 after ES quick adapt")
eval_cp(pol,label="m=0.1 forgetting test")
print("\nDone.")