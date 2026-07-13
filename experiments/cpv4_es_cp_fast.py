import torch,numpy as np,sys,os;sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G=9.8;MC=1.;MP=.1;L=.5;DT=.02;FM=10.;XS=2.5;XDS=3.;THS=.3;THDS=3.;S_CP=torch.tensor([[0.,0.,0.,0.]])
def cp_step(sr,a):
    x,xd,th,thd=sr[:,0],sr[:,1],sr[:,2],sr[:,3];f=a*FM;ct,st=torch.cos(th),torch.sin(th)
    tm_=MC+MP;pml_=MP*L;tmp=(f+pml_*thd**2*st)/tm_
    denom=.5*(4./3.-MP*ct**2/tm_)+1e-8;th_a=(G*st-ct*tmp)/denom;x_a=tmp-pml_*th_a*ct/tm_
    return torch.stack([x+xd*DT+x_a*DT**2/2,xd+x_a*DT,th+thd*DT+th_a*DT**2/2,thd+th_a*DT],-1)

class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n=torch.nn.Sequential(torch.nn.Linear(8,64),torch.nn.Tanh(),torch.nn.Linear(64,48),torch.nn.Tanh(),torch.nn.Linear(48,1),torch.nn.Tanh())
    def forward(self,s,sg):return self.n(torch.cat([s,sg],dim=-1))

def eval_policy(pol,max_steps=200,nr=5):
    total=0
    for _ in range(nr):
        th=np.random.uniform(-.05,.05)
        s=torch.tensor([[0.,0.,th/THS,0.]],dtype=torch.float32)
        for st in range(max_steps):
            a=pol(s,S_CP).item()
            sr=s*S_CP.new_tensor([XS,XDS,THS,THDS])
            s=cp_step(sr,torch.tensor([a]))/S_CP.new_tensor([XS,XDS,THS,THDS])
            if abs(s[0,2].item()*THS)>.21 or abs(s[0,0].item()*XS)>2.4:break
        total+=st
    return total/nr

pol=Policy();d=sum(p.numel() for p in pol.parameters())
m=torch.cat([p.data.view(-1) for p in pol.parameters()]).clone()
print("ES pop=32, nr=5 (5 rollouts avg), H=200, gens=100")
import time;t0=time.time()
for gen in range(1,101):
    noise=torch.randn(32,d)*.4;fits=[]
    for i in range(32):
        idx=0
        for p in pol.parameters():
            n=p.numel();p.data.copy_((m+noise[i])[idx:idx+n].reshape(p.shape));idx+=n
        fits.append(eval_policy(pol,200,5))  # nr=5 average
    for p in pol.parameters():p.data.copy_(m[:p.numel()].reshape(p.shape))
    ft=torch.tensor(fits,dtype=torch.float32);ft=(ft-ft.mean())/(ft.std()+1e-8)
    m+=.1*(noise.T@ft)/32
    idx=0
    for p in pol.parameters():
        n=p.numel();p.data.copy_(m[idx:idx+n].reshape(p.shape));idx+=n
    if gen%20==0:print(f"  Gen{gen:3d} best={max(fits):.0f} median={np.median(fits):.0f}")

print("\nFinal eval:")
succ=0;steps=[]
for t in range(20):
    np.random.seed(42+t*100);th=np.random.uniform(-.05,.05)
    s=torch.tensor([[0.,0.,th/THS,0.]],dtype=torch.float32)
    for st in range(500):
        a=pol(s,S_CP).item()
        sr=s*S_CP.new_tensor([XS,XDS,THS,THDS])
        sn=cp_step(sr,torch.tensor([a]))/S_CP.new_tensor([XS,XDS,THS,THDS]);s=sn
        if abs(s[0,2].item()*THS)>.21 or abs(s[0,0].item()*XS)>2.4:break
    steps.append(st+1)
    if st+1>=500:succ+=1
print(f"CartPole: {succ}/20 ({succ*5}%) steps={np.mean(steps):.0f} max={max(steps)}")
print(f"Time: {time.time()-t0:.0f}s")