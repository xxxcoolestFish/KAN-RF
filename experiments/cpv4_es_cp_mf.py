import torch,torch.nn as nn,numpy as np,sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))));DEVICE="cpu"
torch.manual_seed(42);np.random.seed(42)

G=9.8;MC=1.;MP=.1;L=.5;DT=.02;TM=MC+MP;PML=MP*L;FM=10.
XS=2.5;XDS=3.;THS=.3;THDS=3.;S_CP=torch.tensor([[0.,0.,0.,0.]])
def cp_step(sr,a,g=G,mp=MP,l=L):
    if a.dim()>1:a=a.squeeze(-1);x,xd,th,thd=sr[:,0],sr[:,1],sr[:,2],sr[:,3]
    f=a*FM;ct,st=np.cos(th),np.sin(th);tm_=MC+mp;pml_=mp*l
    tmp=(f+pml_*thd**2*st)/tm_;denom=.5*(4./3.-mp*ct**2/tm_)+1e-8
    th_a=(g*st-ct*tmp)/denom;x_a=tmp-pml_*th_a*ct/tm_
    return np.stack([x+xd*DT+x_a*DT**2/2,xd+x_a*DT,th+thd*DT+th_a*DT**2/2,thd+th_a*DT],-1)

class Policy:
    def __init__(self,dim=8):
        self.w1=np.random.randn(dim,64)*.01;self.b1=np.zeros(64)
        self.w2=np.random.randn(64,48)*.01;self.b2=np.zeros(48)
        self.w3=np.random.randn(48,1)*.01;self.b3=np.zeros(1)
    def forward(self,s,sg):
        x=np.concatenate([s,sg]);x=np.tanh(x@self.w1+self.b1)
        x=np.tanh(x@self.w2+self.b2);return np.tanh(x@self.w3+self.b3).item()
    def get_params(self):return np.concatenate([p.ravel() for p in [self.w1,self.b1,self.w2,self.b2,self.w3,self.b3]])
    def set_params(self,p):
        i=0
        for arr in [self.w1,self.b1,self.w2,self.b2,self.w3,self.b3]:
            n=arr.size;arr[:]=p[i:i+n].reshape(arr.shape);i+=n

def eval_policy(policy,horizon=200,nr=2):
    total=0.
    for _ in range(nr):
        th=np.random.uniform(-.05,.05)
        s=np.array([[0.,0.,th/THS,0.]],dtype=np.float32)
        for st in range(horizon):
            a=policy.forward(s[0],S_CP[0].numpy())
            sr=s*np.array([[XS,XDS,THS,THDS]])
            sn=cp_step(sr,np.array([a]))
            s=sn/np.array([XS,XDS,THS,THDS])
            if abs(s[0,2]*THS)>.21 or abs(s[0,0]*XS)>2.4:break
        total+=st
    return total/nr

print("Model-Free ES for CartPole (numpy, fast)")
pol=Policy()
n_dim=pol.get_params().shape[0]
mean=pol.get_params().copy();sigma=.2;lr=.05;pop=20

for gen in range(1,61):
    noise=np.random.randn(pop,n_dim)*sigma
    fits=np.array([eval_policy(Policy() if False else pol,50,2) for _ in range(pop)]) # wrong, fix below

# Actually let me just do it properly
print("Reducing scope...")
r=np.random.RandomState(42)
mean=pol.get_params().copy()
for gen in range(1,61):
    noise=r.randn(pop,n_dim)*sigma;fits=[]
    for i in range(pop):
        p=pol.get_params().copy();pol.set_params(mean+noise[i])
        fits.append(eval_policy(pol,60,1))
    pol.set_params(mean)
    base_fit=eval_policy(pol,60,2)
    ft=np.array(fits);ft=(ft-ft.mean())/(ft.std()+1e-8)
    mean+=lr*(noise.T@ft)/pop;pol.set_params(mean)
    if gen%10==0:print(f"  Gen{gen:3d} base={base_fit:.0f} mean_fit={ft.mean():+.3f}")

print("\nFinal eval (500 steps):")
succ=0;steps=[]
for t in range(20):
    np.random.seed(42+t*100);th=np.random.uniform(-.05,.05)
    s=np.array([[0.,0.,th/THS,0.]],dtype=np.float32)
    for st in range(500):
        a=pol.forward(s[0],S_CP[0].numpy())
        sr=s*np.array([[XS,XDS,THS,THDS]])
        sn=cp_step(sr,np.array([a]));s=sn/np.array([XS,XDS,THS,THDS])
        if abs(s[0,2]*THS)>.21 or abs(s[0,0]*XS)>2.4:break
    steps.append(st+1)
    if st+1>=500:succ+=1
print(f"CartPole ES: {succ}/20 ({succ*5}%) steps={np.mean(steps):.0f}")