# CDPN v6: Adaptive hybrid loss (energy + goal distance)
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

def energy(s):
    t1=torch.atan2(s[:,1],s[:,0]);t2=torch.atan2(s[:,3],s[:,2]);t12=t1+t2
    w1=s[:,4]*MAX_V1;w2=s[:,5]*MAX_V2;w12=w1+w2
    KE=0.5*(1.*LC1**2*w1**2+I1*w1**2)+0.5*(1.*(L1**2*w1**2+LC2**2*w12**2+2*L1*LC2*w1*w12*torch.cos(t2))+I2*w12**2)
    PE=1.*G0*LC1*torch.sin(t1)+1.*G0*(L1*torch.sin(t1)+LC2*torch.sin(t12))
    return KE+PE

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

class ProtoPolicy(nn.Module):
    def __init__(self):super().__init__();self.pk=ProtoKAN([6,16,1],n_prototypes=16)
    def forward(self,s):return torch.tanh(self.pk(s))

def evaluate(policy,g=G0,n=20):
    succ=0
    for t in range(n):
        th1=np.random.uniform(-.1,.1);th2=np.random.uniform(-.1,.1)
        s=torch.tensor([[np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),0.,0.]],dtype=torch.float32)
        for _ in range(500):
            with torch.no_grad():a=policy(s).item()
            s=acrobot_step(s,torch.tensor([[a]]),g)
            if tip_h(s).item()>TARGET_H:succ+=1;break
    return succ

def train_policy_bptt(policy,wm,epochs=20,H=3,batch=128,lr=1e-3,label=""):
    opt=torch.optim.Adam(policy.parameters(),lr=lr);wm.eval()
    for p in wm.parameters():p.requires_grad=False
    S_G=torch.tensor([[-1.,0.,1.,0.,0.,0.]]).to(DEVICE)
    for ep in range(1,epochs+1):
        tl=0.0
        for _ in range(100):
            s=random_states(batch)
            opt.zero_grad();loss=torch.tensor(0.,device=DEVICE);sc=s.clone()
            for t in range(H):
                a=policy(sc);sc=wm(torch.cat([sc,a],-1))
                loss=loss+(0.9**t)*(sc-S_G.expand(batch,-1)).pow(2).sum(dim=-1).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),10.)
            opt.step();tl+=loss.item()
        if ep%10==0:print(f"    {label} ep{ep:3d} loss={tl/100:.4f}")
    return policy

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

def hybrid_loss(s_pred, s, s_goal, E_crit=20, alpha=1.0):
    """L = -w_E*E(s_pred) + w_G*||s_pred - s_goal||^2
    w_E = sigmoid(alpha*(E_crit - E(s))), w_G = 1 - w_E
    """
    E_s=energy(s).detach()  # weight depends on CURRENT state, no grad needed
    w_E=torch.sigmoid(alpha*(E_crit-E_s))
    w_G=1-w_E
    e_pred=energy(s_pred)
    d_pred=(s_pred-s_goal).pow(2).sum(dim=-1)
    loss=(-w_E*e_pred + w_G*d_pred).mean()
    return loss,w_E.mean().item()

print("="*70)
print("Adaptive Hybrid Loss: Energy + Goal Distance")
print("="*70);t0=time.time()

print("\n[1] Train")
X9,Y9=gen_data(5000,g=9.8);X15,Y15=gen_data(5000,g=G0)
wm0,_=train_wm_lbfgs(X9,Y9,80)
policy0=ProtoPolicy();policy0=train_policy_bptt(policy0,wm0,epochs=20,label="pi0")
succ0_9=evaluate(policy0,g=9.8);succ0_15=evaluate(policy0,g=G0)
print(f"  pi0: g=9.8={succ0_9}/20  g={G0}={succ0_15}/20")
wm1=ProtoKAN([7,16,6],n_prototypes=16).to(DEVICE)
for l in wm1.layers:l.log_sigma.data.fill_(-1.5)
wm1.load_state_dict(wm0.state_dict());wm1=fine_tune_wm(wm1,X15,Y15,steps=500)
for p in wm1.parameters():p.requires_grad=False;wm1.eval()

# Try different E_crit values
print("\n[2] Hybrid loss training (20 ep)")
S_G=torch.tensor([[-1.,0.,1.,0.,0.,0.]]).to(DEVICE)
for E_crit in [5,10,15,20,30]:
    pi=ProtoPolicy();pi.load_state_dict(policy0.state_dict())
    op=torch.optim.Adam(pi.parameters(),lr=1e-3)
    for ep in range(1,21):
        tl=0.0
        for _ in range(100):
            s=random_states(128)
            op.zero_grad()
            a=pi(s);s_pred=wm1(torch.cat([s,a],-1))
            loss,we=hybrid_loss(s_pred,s,S_G,E_crit=E_crit)
            loss.backward();torch.nn.utils.clip_grad_norm_(pi.parameters(),10.)
            op.step();tl+=loss.item()
        if ep%10==0:
            g15=evaluate(pi,g=G0);g9=evaluate(pi,g=9.8)
            print(f"    E_crit={E_crit:2d} ep{ep:2d} g={G0}={g15}/20 g=9.8={g9}/20 w_E={we:.3f}")
    g15=evaluate(pi,g=G0);g9=evaluate(pi,g=9.8)
    print(f"  E_crit={E_crit:2d}: FINAL g={G0}={g15}/20  g=9.8={g9}/20")

# Compare: pure consistency baseline
print("\n[3] Pure consistency (baseline, 10 ep)")
pi_c=ProtoPolicy();pi_c.load_state_dict(policy0.state_dict())
oc=torch.optim.Adam(pi_c.parameters(),lr=1e-3)
for ep in range(1,11):
    tl=0.0
    for _ in range(100):
        s=random_states(128);oc.zero_grad()
        with torch.no_grad():a_old=policy0(s);s_t=wm0(torch.cat([s,a_old],-1))
        a=pi_c(s);s_pred=wm1(torch.cat([s,a],-1))
        l=mse(s_pred,s_t);l.backward();torch.nn.utils.clip_grad_norm_(pi_c.parameters(),10.)
        oc.step();tl+=l.item()
g15_c=evaluate(pi_c,g=G0);g9_c=evaluate(pi_c,g=9.8)
print(f"  Consistency: g={G0}={g15_c}/20  g=9.8={g9_c}/20")

print(f"\nTime: {time.time()-t0:.0f}s")
