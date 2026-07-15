# Path 2 v2: Fixed WM fine-tuning (start from WM0 weights)
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

PROBE_ANGLES=[0,np.pi/6,np.pi/3,np.pi/2,2*np.pi/3,np.pi]
PROBE_VELS=[-0.3,0.,0.3];PROBE_ACTS=[-1.,0.,1.]
def build_probes():
    pts=[]
    for a1 in PROBE_ANGLES:
        for v1 in PROBE_VELS:
            for act in PROBE_ACTS:
                s=torch.tensor([[np.cos(a1),np.sin(a1),np.cos(0.),np.sin(0.),v1/6.,0.]],dtype=torch.float32)
                at=torch.tensor([[act]],dtype=torch.float32)
                pts.append(torch.cat([s,at],dim=-1))
    return torch.cat(pts)
PROBES=build_probes();N_PROBE=PROBES.shape[0]
print(f"  Probes: {N_PROBE}pts")

class CondPolicy(nn.Module):
    def __init__(self,z_dim=2):
        super().__init__();self.pk=ProtoKAN([6+z_dim,32,1],n_prototypes=16);self.zd=z_dim
    def forward(self,s,z):
        sz=torch.cat([s,z.expand(s.shape[0],-1)],dim=-1);return torch.tanh(self.pk(sz))

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

def fine_tune_wm_from(wm_src,x,y,steps=500,lr=1e-4):
    wm=ProtoKAN([7,16,6],n_prototypes=16).to(DEVICE)
    for l in wm.layers:l.log_sigma.data.fill_(-1.5)
    wm.load_state_dict(wm_src.state_dict())
    wm.train();opt=torch.optim.Adam(wm.parameters(),lr=lr)
    for _ in range(steps):
        idx=torch.randint(0,len(x),(256,));l=mse(wm(x[idx].to(DEVICE)),y[idx].to(DEVICE))
        opt.zero_grad();l.backward();opt.step()
    wm.eval();return wm

def d_vec(wm,wm0):
    with torch.no_grad():return (wm(PROBES)-wm0(PROBES)).numpy().ravel()
def get_z(wm,wm0,pca):
    return pca.transform(d_vec(wm,wm0).reshape(1,-1))[0]

def eval_cond(policy,wm0,wm1,pca,g=14.7,n=20):
    z=get_z(wm1,wm0,pca);zt=torch.tensor(z,dtype=torch.float32).unsqueeze(0);sc=0
    for t in range(n):
        th1=np.random.uniform(-.1,.1);th2=np.random.uniform(-.1,.1)
        s=torch.tensor([[np.cos(th1),np.sin(th1),np.cos(th2),np.sin(th2),0.,0.]],dtype=torch.float32)
        for _ in range(500):
            with torch.no_grad():a=policy(s,zt.expand(s.shape[0],-1)).item()
            s=acrobot_step(s,torch.tensor([[a]]),g)
            if tip_h(s).item()>TARGET_H:sc+=1;break
    return sc

print("="*70)
print("Path 2 v2: D signature + CondPolicy (fixed FT)")
print("="*70);t0=time.time()
TRAIN_G=[9.8,10.,12.,17.,19.6];TEST_G=14.7

print("\n[1] Data + WM0")
data={g:gen_data(5000,g) for g in TRAIN_G+[TEST_G]}
wm0,bv=train_wm_lbfgs(data[9.8][0],data[9.8][1])
print(f"  WM0 val_mse={bv:.6f}")

print("\n[2] Fine-tune WMs (from WM0 weights)")
wms={9.8:wm0}
for g in TRAIN_G+[TEST_G]:
    if g==9.8:continue
    x,y=data[g];wm=fine_tune_wm_from(wm0,x,y)
    wms[g]=wm
    e=mse(wm(x[:1000]),y[:1000]).item()
    print(f"  g={g:4.1f}: FT MSE={e:.6f}")

print("\n[3] PCA encoder on d vectors")
d_all=np.array([d_vec(wms[g],wm0) for g in TRAIN_G])
pca=PCA(n_components=2).fit(d_all)
for g in TRAIN_G+[TEST_G]:
    z=get_z(wms[g],wm0,pca)
    print(f"  g={g:4.1f}: z=[{z[0]:+.4f},{z[1]:+.4f}]")
print(f"  EVR: {pca.explained_variance_ratio_}")

print("\n[4] Train CondPolicy(s,z) (30 ep)")
policy=CondPolicy();opt=torch.optim.Adam(policy.parameters(),lr=3e-4)
S_G=torch.tensor([[-1.,0.,1.,0.,0.,0.]]).to(DEVICE)
for ep in range(1,31):
    tl=0.0
    for _ in range(60):
        g=np.random.choice(TRAIN_G);wm=wms[g]
        zt=torch.tensor(get_z(wms[g],wm0,pca),dtype=torch.float32).unsqueeze(0)
        s=random_states(96);opt.zero_grad();loss=torch.tensor(0.,device=DEVICE);sc=s.clone()
        for t in range(3):
            a=policy(sc,zt.expand(sc.shape[0],-1));sc=wm(torch.cat([sc,a],-1))
            loss=loss+(0.9**t)*(sc-S_G.expand(96,-1)).pow(2).sum(dim=-1).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),10.)
        opt.step();tl+=loss.item()
    if ep%10==0:
        g9=eval_cond(policy,wm0,wms[9.8],pca,g=9.8,n=5)
        gt=eval_cond(policy,wm0,wms[12.0],pca,g=12.0,n=5)
        print(f"    ep{ep:2d} loss={tl/60:.4f} g=9.8={g9}/5 g=12={gt}/5")

print("\n[5] Zero-shot test")
for g,label in [(9.8,"trained"),(TEST_G,"UNSEEN"),(19.6,"trained")]:
    r=eval_cond(policy,wm0,wms[g],pca,g=g,n=20)
    print(f"  g={g:4.1f} ({label}): {r}/20")
print(f"Time: {time.time()-t0:.0f}s")
