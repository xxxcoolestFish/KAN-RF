# CDPN v6: Reference Prediction Architecture
import torch,torch.nn as nn,numpy as np,sys,os,time
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
DEVICE="cpu";torch.manual_seed(42);np.random.seed(42);mse=nn.MSELoss()
L1=L2=1.;LC1=LC2=.5;I1=I2=1.;DT=.05;MAX_V1=6.;MAX_V2=8.;TARGET_H=1.0;G0=14.7

def acrobot_step(sr,a,g=G0):
    c1,s1,c2,s2=sr[:,0],sr[:,1],sr[:,2],sr[:,3];t1=torch.atan2(s1,c1);t2=torch.atan2(s2,c2)
    st1,st2=torch.sin(t1),torch.sin(t2);ct1,ct2=torch.cos(t1),torch.cos(t2);st12=torch.sin(t1+t2)
    d11=1.*LC1**2+1.*(L1**2+LC2**2+2*L1*LC2*ct2)+I1+I2;d22=1.*LC2**2+I2;d12=1.*(LC2**2+L1*LC2*ct2)+I2
    p2=1.*g*LC2*st12;p1=-(1.*LC1+1.*L1)*g*st1-p2;c1h=2*1.*L1*LC2*st2;h=c1h*(sr[:,4]*MAX_V1)
    d1=sr[:,4]*MAX_V1;d2=sr[:,5]*MAX_V2
    t2a=(a[...,0]+d12/d11*p1-h*d2-c1h*d1*d2-p2)/(d22-d12**2/d11+1e-6);t1a=-(d12*t2a+p1)/d11
    d1=d1+t1a*DT;d2=d2+t2a*DT;d1=d1.clamp(-MAX_V1,MAX_V1);d2=d2.clamp(-MAX_V2,MAX_V2)
    t1=t1+d1*DT;t2=t2+d2*DT
    return torch.stack([torch.cos(t1),torch.sin(t1),torch.cos(t2),torch.sin(t2),d1/MAX_V1,d2/MAX_V2],-1)

def tip_h(s):
    t1=torch.atan2(s[:,1],s[:,0]);t12=torch.atan2(s[:,3],s[:,2])+t1
    return -L1*torch.cos(t1)-L2*torch.cos(t12)

def gen_data(n=5000,g=9.8):
    xs,ys=[],[]
    for _ in range(n):
        th1=np.random.uniform(-np.pi,np.pi);th2=np.random.uniform(-np.pi,np.pi)
        d1=np.random.uniform(-MAX_V1,MAX_V1)/MAX_V1;d2=np.random.uniform(-MAX_V2,MAX_V2)/MAX_V2
        a=np.random.uniform(-1.,1.)
        s=torch.tensor([[np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),d1,d2]],dtype=torch.float32)
        sn=acrobot_step(s,torch.tensor([[a]]),g)
        xs.append(torch.cat([s,torch.tensor([[a]])],-1).float());ys.append(sn.float())
    return torch.cat(xs),torch.cat(ys)

def random_states(batch=128):
    th1=np.random.uniform(-np.pi,np.pi,batch);th2=np.random.uniform(-np.pi,np.pi,batch)
    d1=np.random.uniform(-1.,1.,batch);d2=np.random.uniform(-1.,1.,batch)
    return torch.stack([torch.cos(torch.tensor(th1)),torch.sin(torch.tensor(th1)),
        torch.cos(torch.tensor(th2)),torch.sin(torch.tensor(th2)),
        torch.tensor(d1),torch.tensor(d2)],dim=-1).float()

A_REF=torch.tensor([-1.,-0.5,0.,0.5,1.]).unsqueeze(-1)

class RefPolicy(nn.Module):
    """Policy that takes [s(6) + predictions(Sref)(30)] = 36-dim input."""
    def __init__(self):
        super().__init__()
        self.pk=ProtoKAN([36,32,1],n_prototypes=16)
    def forward(self,s,preds):
        x=torch.cat([s,preds.view(s.shape[0],-1)],dim=-1)
        return torch.tanh(self.pk(x))

def make_preds(wm,s):
    """Compute WM(s, a_ref) for all reference actions."""
    batch=s.shape[0];n_ref=5
    s_exp=s.unsqueeze(1).expand(-1,n_ref,-1)
    a_exp=A_REF.unsqueeze(0).expand(batch,-1,-1)
    sa=torch.cat([s_exp,a_exp],dim=-1)
    with torch.no_grad():preds=wm(sa.view(-1,7)).view(batch,n_ref,6)
    return preds

def train_policy_ref(policy,wm,epochs=20,H=3,batch=128,lr=1e-3,label=""):
    opt=torch.optim.Adam(policy.parameters(),lr=lr)
    for p in wm.parameters():p.requires_grad=False
    S_G=torch.tensor([[-1.,0.,1.,0.,0.,0.]]).to(DEVICE)
    for ep in range(1,epochs+1):
        tl=0.0
        for _ in range(100):
            s=random_states(batch)
            preds=make_preds(wm,s)
            opt.zero_grad();loss=torch.tensor(0.,device=DEVICE);sc=s.clone()
            for t in range(H):
                a=policy(sc,make_preds(wm,sc));sc=wm(torch.cat([sc,a],-1))
                loss=loss+(0.9**t)*(sc-S_G.expand(batch,-1)).pow(2).sum(dim=-1).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),10.)
            opt.step();tl+=loss.item()
        if ep%10==0:print(f"    {label} ep{ep:3d} loss={tl/100:.4f}")
    return policy

def evaluate_ref(policy,wm,g=G0,n=20):
    succ=0
    for t in range(n):
        th1=np.random.uniform(-.1,.1);th2=np.random.uniform(-.1,.1)
        s=torch.tensor([[np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),0.,0.]],dtype=torch.float32)
        for _ in range(500):
            with torch.no_grad():a=policy(s,make_preds(wm,s)).item()
            s=acrobot_step(s,torch.tensor([[a]]),g)
            if tip_h(s).item()>TARGET_H:succ+=1;break
    return succ

def evaluate_old(policy,single_wm,g=G0,n=20):
    """Backward-compatible eval for standard policies (no ref preds)."""
    succ=0
    for t in range(n):
        th1=np.random.uniform(-.1,.1);th2=np.random.uniform(-.1,.1)
        s=torch.tensor([[np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),0.,0.]],dtype=torch.float32)
        for _ in range(500):
            with torch.no_grad():a=torch.tanh(single_wm.pk(s)).item()
            s=acrobot_step(s,torch.tensor([[a]]),g)
            if tip_h(s).item()>TARGET_H:succ+=1;break
    return succ

def train_wm_lbfgs(x,y,n_epochs=80):
    wm=ProtoKAN([7,16,6],n_prototypes=16).to(DEVICE)
    for l in wm.layers:l.log_sigma.data.fill_(-1.5)
    opt=torch.optim.LBFGS(wm.parameters(),lr=1.,max_iter=20,history_size=50,line_search_fn="strong_wolfe")
    nt=int(len(x)*.85);bv=float("inf");bs=None
    def c():opt.zero_grad();l=mse(wm(x[:nt]),y[:nt]);l.backward();return l
    for _ in range(n_epochs):
        opt.step(c);v=mse(wm(x[nt:]),y[nt:]).item()
        if v<bv:bv=v;bs={k:v.clone() for k,v in wm.state_dict().items()}
    wm.load_state_dict(bs);wm.eval();return wm,bv

def fine_tune_wm(wm,x,y,steps=500,lr=1e-4):
    wm.train();opt=torch.optim.Adam(wm.parameters(),lr=lr)
    for _ in range(steps):
        idx=torch.randint(0,len(x),(256,));l=mse(wm(x[idx].to(DEVICE)),y[idx].to(DEVICE))
        opt.zero_grad();l.backward();opt.step()
    wm.eval();return wm

print("="*70)
print("Reference Prediction Architecture")
print("="*70);t0=time.time()

print("\n[1] Train WM0 on g=9.8")
X9,Y9=gen_data(5000,g=9.8);X15,Y15=gen_data(5000,g=G0);X20,Y20=gen_data(5000,g=19.6)
wm0,bv=train_wm_lbfgs(X9,Y9);print(f"  WM0 val_mse={bv:.6f}")

print("\n[2] Train RefPolicy on g=9.8 (BPTT H=3, 20 ep)")
policy=RefPolicy()
policy=train_policy_ref(policy,wm0,epochs=20,label="Ref")
succ9=evaluate_ref(policy,wm0,g=9.8);succ15=evaluate_ref(policy,wm0,g=G0)
print(f"  g=9.8={succ9}/20  g=14.7(zero-shot W0)={succ15}/20")

print("\n[3] Fine-tune WM0 -> WM1 on g=14.7")
wm1=fine_tune_wm(ProtoKAN([7,16,6],n_prototypes=16),X15,Y15,steps=500)
for l in wm1.layers:l.log_sigma.data.fill_(-1.5)
for p in wm1.parameters():p.requires_grad=False;wm1.eval()
print(f"  WM1 ready")

print("\n[4] Zero-shot: RefPolicy with WM1 predictions")
succ15_zs=evaluate_ref(policy,wm1,g=G0)
succ20_zs=evaluate_ref(policy,wm1,g=19.6)
print(f"  g=14.7(WM1 zero-shot)={succ15_zs}/20")
print(f"  g=19.6(WM1 zero-shot)={succ20_zs}/20")

print("\n[5] Fine-tune RefPolicy on g=14.7 (10 ep consistency+BPTT)")
opt=torch.optim.Adam(policy.parameters(),lr=1e-4)
S_G=torch.tensor([[-1.,0.,1.,0.,0.,0.]]).to(DEVICE)
for ep in range(1,11):
    tl=0.0
    for _ in range(100):
        s=random_states(128)
        preds=make_preds(wm1,s)
        opt.zero_grad();loss=torch.tensor(0.,device=DEVICE);sc=s.clone()
        for t in range(3):
            a=policy(sc,make_preds(wm1,sc));sc=wm1(torch.cat([sc,a],-1))
            loss=loss+(0.9**t)*(sc-S_G.expand(128,-1)).pow(2).sum(dim=-1).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),10.)
        opt.step();tl+=loss.item()
    if ep%5==0:
        g15=evaluate_ref(policy,wm1,g=G0);g9=evaluate_ref(policy,wm1,g=9.8)
        print(f"    ep{ep:2d} g={G0}={g15}/20 g=9.8={g9}/20 loss={tl/100:.4f}")
succ15_ft=evaluate_ref(policy,wm1,g=G0);succ9_ft=evaluate_ref(policy,wm1,g=9.8)

print(f"\n{'='*70}\nRESULTS\n{'='*70}")
print(f"  g=9.8 trained:          {succ9}/20")
print(f"  g=14.7 zero-shot (W0):  {succ15}/20  (using OLD predictions)")
print(f"  g=14.7 zero-shot (W1):  {succ15_zs}/20  (using NEW predictions)")
print(f"  g=14.7 fine-tuned:      {succ15_ft}/20")
print(f"  g=19.6 zero-shot (W1):  {succ20_zs}/20")
print(f"  Time: {time.time()-t0:.0f}s")
