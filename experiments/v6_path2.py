# Path 2: Dynamics change signature encoding + conditional policy
import torch,torch.nn as nn,numpy as np,sys,os,time
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
from sklearn.decomposition import PCA
DEVICE="cpu";torch.manual_seed(42);np.random.seed(42);mse=nn.MSELoss()
L1=L2=1.;LC1=LC2=.5;I1=I2=1.;DT=.05;MAX_V1=6.;MAX_V2=8.;TARGET_H=1.0

def acrobot_step(sr,a,g=9.8):
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

# -- Define probe points --
PROBE_ANGLES=[0,np.pi/6,np.pi/3,np.pi/2,2*np.pi/3,np.pi]
PROBE_VELS=[-0.3,0.,0.3]
PROBE_ACTS=[-1.,0.,1.]
def build_probes():
    pts=[]
    for a1 in PROBE_ANGLES:
        for v1 in PROBE_VELS:
            for a2 in [0.]:
                for act in PROBE_ACTS:
                    s=torch.tensor([[np.cos(a1),np.sin(a1),np.cos(a2),np.sin(a2),v1/6.,0.]],dtype=torch.float32)
                    at=torch.tensor([[act]],dtype=torch.float32)
                    pts.append(torch.cat([s,at],dim=-1))
    return torch.cat(pts)
PROBES=build_probes()
N_PROBE=PROBES.shape[0]
print(f"  Probe points: {N_PROBE}  (states{len(PROBE_ANGLES)}x{len(PROBE_VELS)}vels x {len(PROBE_ACTS)}acts")

# -- Conditional policy --
class CondPolicy(nn.Module):
    def __init__(self,z_dim=2):
        super().__init__()
        self.pk=ProtoKAN([6+z_dim,32,1],n_prototypes=16)
        self.z_dim=z_dim
    def forward(self,s,z):
        x=torch.cat([s,z],dim=-1)
        return torch.tanh(self.pk(x))

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

def compute_d_vec(wm,wm0,probes):
    """Compute d = wm(probes) - wm0(probes)."""
    with torch.no_grad():
        p0=wm0(probes).numpy()
        p1=wm(probes).numpy()
    return (p1-p0).ravel()

def compute_z(d_vec,pca):
    """Project d through PCA to get z."""
    return pca.transform(d_vec.reshape(1,-1))[0]

def evaluate_cond(policy,wm0,wm1,pca,g=14.7,n=20):
    succ=0;d=compute_d_vec(wm1,wm0,PROBES);z=compute_z(d,pca)
    z_t=torch.tensor(z,dtype=torch.float32).unsqueeze(0)
    for t in range(n):
        th1=np.random.uniform(-.1,.1);th2=np.random.uniform(-.1,.1)
        s=torch.tensor([[np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),0.,0.]],dtype=torch.float32)
        for _ in range(500):
            with torch.no_grad():
                a=policy(s,z_t.expand(s.shape[0],-1)).item()
            s=acrobot_step(s,torch.tensor([[a]]),g)
            if tip_h(s).item()>TARGET_H:succ+=1;break
    return succ

print("="*70)
print("Path 2: Dynamics Change Signature + Conditional Policy")
print("="*70);t0=time.time()

TRAIN_GRAVS=[9.8,10.,12.,17.,19.6]
TEST_GRAV=14.7  # held out
ALL_GRAVS=TRAIN_GRAVS+[TEST_GRAV]

print("\n[1] Generate data and train WM0")
data={}
for g in ALL_GRAVS:
    x,y=gen_data(5000,g)
    data[g]=(x,y)

wm0,bv=train_wm_lbfgs(data[9.8][0],data[9.8][1])
print(f"  WM0 val_mse={bv:.6f}")

print("\n[2] Fine-tune WM for each gravity")
wms={9.8:wm0}
for g in ALL_GRAVS:
    if g==9.8:continue
    wm_ft=fine_tune_wm(ProtoKAN([7,16,6],n_prototypes=16),data[g][0],data[g][1])
    for l in wm_ft.layers:l.log_sigma.data.fill_(-1.5)
    wms[g]=wm_ft
    with torch.no_grad():e=mse(wm_ft(data[g][0][:1000]),data[g][1][:1000]).item()
    print(f"  g={g:4.1f}: FT MSE={e:.6f}")

print("\n[3] Compute d vectors and train PCA encoder")
d_vecs={g:compute_d_vec(wms[g],wm0,PROBES) for g in ALL_GRAVS}
D_train=np.array([d_vecs[g] for g in TRAIN_GRAVS])
pca=PCA(n_components=2)
pca.fit(D_train)
for g in ALL_GRAVS:
    z=compute_z(d_vecs[g],pca)
    print(f"  g={g:4.1f}: d_norm={np.linalg.norm(d_vecs[g]):.6f}  z=[{z[0]:+.4f},{z[1]:+.4f}]")
print(f"  PCA explained variance: {pca.explained_variance_ratio_}")
print(f"  Unseen g={TEST_GRAV}: z projection in PCA space")

print("\n[4] Train CondPolicy(s,z) on training gravities")
policy=CondPolicy(z_dim=2)
opt=torch.optim.Adam(policy.parameters(),lr=1e-3)
S_G=torch.tensor([[-1.,0.,1.,0.,0.,0.]]).to(DEVICE)
for ep in range(1,31):
    tl=0.0
    for _ in range(80):
        g=np.random.choice(TRAIN_GRAVS)
        wm=wms[g]
        z=torch.tensor(compute_z(d_vecs[g],pca),dtype=torch.float32).unsqueeze(0)
        s=random_states(128)
        opt.zero_grad();loss=torch.tensor(0.,device=DEVICE);sc=s.clone()
        for t in range(3):
            a=policy(sc,z.expand(sc.shape[0],-1));sc=wm(torch.cat([sc,a],-1))
            loss=loss+(0.9**t)*(sc-S_G.expand(128,-1)).pow(2).sum(dim=-1).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),10.)
        opt.step();tl+=loss.item()
    if ep%10==0:
        # Quick eval on training gravities
        for g in [9.8,12.,19.6]:
            gr=evaluate_cond(policy,wm0,wms[g],pca,g=g,n=5)
            if g==9.8:g9r=gr
        print(f"    ep{ep:2d} loss={tl/80:.4f} g=9.8={g9r}/5 g=12={gr}/5")

print("\n[5] ZERO-SHOT: test on unseen g=14.7")
g15_zs=evaluate_cond(policy,wm0,wms[TEST_GRAV],pca,g=TEST_GRAV,n=20)
g19_zs=evaluate_cond(policy,wm0,wms[19.6],pca,g=19.6,n=20)
g9_zs=evaluate_cond(policy,wm0,wms[9.8],pca,g=9.8,n=20)
print(f"  g=9.8 (trained):    {g9_zs}/20")
print(f"  g=14.7 (UNSEEN):    {g15_zs}/20  <-- zero-shot!")
print(f"  g=19.6 (trained):   {g19_zs}/20")

print(f"\nTotal time: {time.time()-t0:.0f}s")
