"""CDPN v6 + BSRM on Acrobot. Only BSRM on first layer."""
import torch,torch.nn as nn,numpy as np,sys,os,time
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
DEVICE="cpu";torch.manual_seed(42);np.random.seed(42);mse=nn.MSELoss()
L1=L2=1.;LC1=LC2=.5;I1=I2=1.;DT=.05;MAX_V1=6.;MAX_V2=8.;TARGET_H=1.0

def acrobot_step(sr,a,g=9.8):
    c1,s1,c2,s2=sr[:,0],sr[:,1],sr[:,2],sr[:,3];t1=torch.atan2(s1,c1);t2=torch.atan2(s2,c2)
    st1,st2=torch.sin(t1),torch.sin(t2);ct1,ct2=torch.cos(t1),torch.cos(t2);st12=torch.sin(t1+t2)
    d11=1.*LC1**2+1.*(L1**2+LC2**2+2*L1*LC2*ct2)+I1+I2
    d22=1.*LC2**2+I2;d12=1.*(LC2**2+L1*LC2*ct2)+I2
    p2=1.*g*LC2*st12;p1=-(1.*LC1+1.*L1)*g*st1-p2
    c1h=2*1.*L1*LC2*st2;h=c1h*(sr[:,4]*MAX_V1);d1=sr[:,4]*MAX_V1;d2=sr[:,5]*MAX_V2
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

class ProtoPolicy(nn.Module):
    def __init__(self):super().__init__();self.pk=ProtoKAN([6,16,1],n_prototypes=16)
    def forward(self,s):return torch.tanh(self.pk(s))

def train_policy(policy,wm,epochs=100,H=3,batch=128,lr=1e-3,label=""):
    opt=torch.optim.Adam(policy.parameters(),lr=lr);wm.eval()
    for p in wm.parameters():p.requires_grad=False
    S_G=torch.tensor([[-1.,0.,1.,0.,0.,0.]]).to(DEVICE);t0=time.time()
    for ep in range(1,epochs+1):
        tl=0.0
        for _ in range(100):
            th1=np.random.uniform(-np.pi,np.pi,batch);th2=np.random.uniform(-np.pi,np.pi,batch)
            d1=np.random.uniform(-1.,1.,batch);d2=np.random.uniform(-1.,1.,batch)
            s=torch.stack([torch.cos(torch.tensor(th1)),torch.sin(torch.tensor(th1)),
                torch.cos(torch.tensor(th2)),torch.sin(torch.tensor(th2)),
                torch.tensor(d1),torch.tensor(d2)],dim=-1).float()
            opt.zero_grad();loss=torch.tensor(0.,device=DEVICE);sc=s.clone()
            for t in range(H):
                a=policy(sc);sc=wm(torch.cat([sc,a],dim=-1))
                loss=loss+(0.9**t)*(sc-S_G.expand(batch,-1)).pow(2).sum(dim=-1).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),10.);opt.step();tl+=loss.item()
        if ep%50==0:print(f"    {label} ep{ep:3d} loss={tl/100:.4f} time={time.time()-t0:.0f}s")
    return policy

def evaluate(policy,g=9.8,n=20):
    succ=0
    for t in range(n):
        th1=np.random.uniform(-.1,.1);th2=np.random.uniform(-.1,.1)
        s=torch.tensor([[np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),0.,0.]],dtype=torch.float32)
        for _ in range(500):
            with torch.no_grad():a=policy(s).item()
            s=acrobot_step(s,torch.tensor([[a]]),g)
            if tip_h(s).item()>TARGET_H:succ+=1;break
    return succ

def compute_bsrm(policy,wm_old,wm_new,g_new=19.6,trials=5):
    """BSRM: only on first layer (maps state->hidden)."""
    s_list,a_list=[],[]
    for _ in range(trials):
        th1=np.random.uniform(-.3,.3);th2=np.random.uniform(-.3,.3)
        s=torch.tensor([[np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),0.,0.]],dtype=torch.float32)
        for _ in range(100):
            with torch.no_grad():a=policy(s)+torch.randn(1,1)*0.1
            sn=acrobot_step(s,torch.tensor([[a.item()]]),g_new)
            s_list.append(s);a_list.append(a);s=sn
    S=torch.cat(s_list);A=torch.cat(a_list)
    with torch.no_grad():delta=(wm_new(torch.cat([S,A],-1))-wm_old(torch.cat([S,A],-1))).norm(dim=-1)
    masks={}
    # Only layer 0 (state -> hidden)
    layer=policy.pk.layers[0]
    sigma=torch.exp(layer.log_sigma).item();pos=layer.proto_pos.detach()
    relevance=torch.zeros(layer.in_dim,layer.n_prototypes)
    for i in range(layer.in_dim):
        si=S[:,i].unsqueeze(-1)  # (N,1)
        diff=si-pos.unsqueeze(0)  # (N,1)-(1,P)->(N,P)
        weight=torch.exp(-diff.pow(2)/(2*sigma**2))
        relevance[i]=(delta.unsqueeze(-1)*weight).sum(dim=0)
    thr=relevance.mean()+2*relevance.std()
    mask=(relevance>thr).float().to(DEVICE)
    mask_val=mask.unsqueeze(0).expand(layer.out_dim,-1,-1)
    mask_der=mask_val.clone()
    masks[0]={'proto_val':mask_val,'proto_der':mask_der}
    n_sel=mask_val.sum().item();n_tot=mask_val.numel()
    print(f"  BSRM: {int(n_sel)}/{n_tot} ({100*n_sel/n_tot:.0f}%) coefficients selected")
    return masks

def adapt_bsrm(policy,wm,masks,epochs=30,H=3,batch=128,lr=1e-4,label=""):
    opt=torch.optim.Adam(policy.parameters(),lr=lr);wm.eval()
    for p in wm.parameters():p.requires_grad=False
    S_G=torch.tensor([[-1.,0.,1.,0.,0.,0.]]).to(DEVICE)
    for ep in range(1,epochs+1):
        tl=0.0
        for _ in range(100):
            th1=np.random.uniform(-np.pi,np.pi,batch);th2=np.random.uniform(-np.pi,np.pi,batch)
            d1=np.random.uniform(-1.,1.,batch);d2=np.random.uniform(-1.,1.,batch)
            s=torch.stack([torch.cos(torch.tensor(th1)),torch.sin(torch.tensor(th1)),
                torch.cos(torch.tensor(th2)),torch.sin(torch.tensor(th2)),
                torch.tensor(d1),torch.tensor(d2)],dim=-1).float()
            opt.zero_grad();loss=torch.tensor(0.,device=DEVICE);sc=s.clone()
            for t in range(H):
                a=policy(sc);sc=wm(torch.cat([sc,a],dim=-1))
                loss=loss+(0.9**t)*(sc-S_G.expand(batch,-1)).pow(2).sum(dim=-1).mean()
            loss.backward()
            if masks:
                policy.pk.layers[0].proto_val.grad*=masks[0]['proto_val']
                policy.pk.layers[0].proto_der.grad*=masks[0]['proto_der']
            opt.step();tl+=loss.item()
        if ep%15==0:print(f"    {label} ep{ep:3d} loss={tl/100:.4f}")
    return policy

print("="*60);print("CDPN v6 + BSRM on Acrobot");print("="*60)
print("\n[1] Train WM on g=9.8")
X9,Y9=gen_data(5000,g=9.8);X19,Y19=gen_data(5000,g=19.6)
wm_old=ProtoKAN([7,16,6],n_prototypes=16).to(DEVICE)
for l in wm_old.layers:l.log_sigma.data.fill_(-1.5)
opt=torch.optim.LBFGS(wm_old.parameters(),lr=1.,max_iter=20,history_size=50,line_search_fn="strong_wolfe")
nt=int(len(X9)*.85);bv=float("inf");bs=None
def c():opt.zero_grad();l=mse(wm_old(X9[:nt]),Y9[:nt]);l.backward();return l
for _ in range(80):
    opt.step(c);v=mse(wm_old(X9[nt:]),Y9[nt:]).item()
    if v<bv:bv=v;bs={k:v.clone() for k,v in wm_old.state_dict().items()}
wm_old.load_state_dict(bs);wm_old.eval();print(f"  WM val_mse={bv:.6f}")

print("\n[2] Train ProtoKAN Policy")
policy=ProtoPolicy();policy=train_policy(policy,wm_old,label="Initial")
succ=evaluate(policy,g=9.8);print(f"  g=9.8 baseline: {succ}/20")

print("\n[3] Fine-tune WM on g=19.6")
wm_new=ProtoKAN([7,16,6],n_prototypes=16).to(DEVICE)
for l in wm_new.layers:l.log_sigma.data.fill_(-1.5)
wm_new.load_state_dict(wm_old.state_dict())
opt_wm=torch.optim.Adam(wm_new.parameters(),lr=1e-4);wm_new.train()
for _ in range(200):
    idx=torch.randint(0,len(X19),(256,))
    l=mse(wm_new(X19[idx].to(DEVICE)),Y19[idx].to(DEVICE))
    opt_wm.zero_grad();l.backward();opt_wm.step()
wm_new.eval()

print("\n[4] BSRM (first layer only)")
masks=compute_bsrm(policy,wm_old,wm_new)

print("\n[5] Adapt with BSRM")
p_bsrm=ProtoPolicy();p_bsrm.load_state_dict(policy.state_dict())
p_bsrm=adapt_bsrm(p_bsrm,wm_new,masks,label="BSRM")
g19b=evaluate(p_bsrm,g=19.6);g9b=evaluate(p_bsrm,g=9.8)
print(f"  BSRM: g=19.6={g19b}/20  g=9.8={g9b}/20")

print("\n[6] Full fine-tune")
p_full=ProtoPolicy();p_full.load_state_dict(policy.state_dict())
p_full=adapt_bsrm(p_full,wm_new,{},label="FULL")
g19f=evaluate(p_full,g=19.6);g9f=evaluate(p_full,g=9.8)
print(f"  FULL: g=19.6={g19f}/20  g=9.8={g9f}/20")

print(f"\nSUMMARY: BSRM {g19b}/20 FULL {g19f}/20  Forgetting: BSRM={g9b}/20 FULL={g9f}/20")
