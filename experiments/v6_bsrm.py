"""CDPN v6 + BSRM: selective coefficient update during adaptation."""
import torch,torch.nn as nn,numpy as np,gymnasium as gym,time,sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
from experiments.baseline_sweep import generate_pendulum_data,train_wm
DEVICE="cpu";PI_2=np.pi/2;S_T=torch.tensor([[0.,1.,0.]]).to(DEVICE)
torch.manual_seed(42);np.random.seed(42);mse=nn.MSELoss()

class ProtoPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.pk=ProtoKAN([3,16,1],n_prototypes=16)
    def forward(self,s):return torch.tanh(self.pk(s))

def train_policy(policy,wm,epochs=100,H=3,batch=128,lr=1e-3,label=""):
    opt=torch.optim.Adam(policy.parameters(),lr=lr)
    wm.eval()
    for p in wm.parameters():p.requires_grad=False
    t0=time.time()
    for ep in range(1,epochs+1):
        total_loss=0.0
        for _ in range(100):
            th=np.random.uniform(-np.pi,np.pi,batch)
            td=np.random.uniform(-8.,8.,batch)
            s=torch.stack([torch.cos(torch.tensor(th)),torch.sin(torch.tensor(th)),torch.tensor(td)/8.],dim=-1).float()
            opt.zero_grad();loss=torch.tensor(0.,device=DEVICE);s_cur=s.clone()
            for t in range(H):
                a=policy(s_cur);s_cur=wm(torch.cat([s_cur,a],dim=-1))
                loss=loss+(0.9**t)*(s_cur-S_T.expand(batch,-1)).pow(2).sum(dim=-1).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),10.);opt.step()
            total_loss+=loss.item()
        if ep%50==0:print(f"    {label} ep{ep:3d} loss={total_loss/100:.4f}")
    return policy

def evaluate(policy,env,n=20):
    succ=0;steps=[]
    for t in range(n):
        obs,_=env.reset(seed=42+t*100)
        for st in range(200):
            s=torch.tensor([[obs[0],obs[1],obs[2]/8.]],dtype=torch.float32)
            with torch.no_grad():a=policy(s).item()
            obs,_,_,_,_=env.step([a*2.])
            err=min(abs(np.arctan2(obs[1],obs[0])-PI_2),2*np.pi-abs(np.arctan2(obs[1],obs[0])-PI_2))
            if err<.2:succ+=1;steps.append(st+1);break
        else:steps.append(300)
    return succ,steps

def compute_bsrm(policy,wm_old,wm_new,env15,policy_noise=0.1,trials=5):
    """BSRM: compute relevance masks for ProtoKAN coefficients.
    
    Args:
        policy: trained policy (to collect data in new env)
        wm_old: WM trained on g=10 (frozen)
        wm_new: WM fine-tuned on g=15 (frozen)
        env15: g=15 environment
        policy_noise: exploration noise for data collection
        trials: number of episodes to collect
    """
    # Step 1: Collect data D' by running old policy in new env
    s_list,a_list=[],[]
    for _ in range(trials):
        obs,_=env15.reset(seed=np.random.randint(10000))
        for _ in range(200):
            s=torch.tensor([[obs[0],obs[1],obs[2]/8.]],dtype=torch.float32)
            with torch.no_grad():
                a=policy(s)+torch.randn(1,1)*policy_noise
            obs,_,tr,tr2,_=env15.step([a.item()*2.])
            s_list.append(s);a_list.append(a)
            if tr or tr2:break
    S=torch.cat(s_list);A=torch.cat(a_list)
    
    # Step 2: Compute delta(s) = ||WM'(s,a) - WM_0(s,a)||
    with torch.no_grad():
        pred_old=wm_old(torch.cat([S,A],dim=-1))
        pred_new=wm_new(torch.cat([S,A],dim=-1))
        delta=(pred_new-pred_old).norm(dim=-1)  # (N,)
    
    # Step 3: Map delta to each prototype via Gaussian weight
    masks={}
    for l_idx,layer in enumerate(policy.pk.layers):
        sigma=torch.exp(layer.log_sigma).item()
        pos=layer.proto_pos.detach()  # (n_protos,)
        relevance=torch.zeros(layer.in_dim,layer.n_prototypes)
        for i in range(layer.in_dim):
            si=S[:,i:i+1]  # (N,1)
            diff=si.unsqueeze(-1)-pos.unsqueeze(0)  # (N,1,n_protos)
            weight=torch.exp(-diff.pow(2)/(2*sigma**2))  # (N,1,n_protos)
            relevance[i]=(delta.unsqueeze(-1)*weight.squeeze(1)).sum(dim=0)
        
        # Select: relevance > mean+2*std (statistically significant)
        thr=relevance.mean()+2*relevance.std()
        mask=(relevance>thr).float().to(DEVICE)
        # Expand to match proto_val shape: (out_dim,in_dim,n_protos)
        mask_val=mask.unsqueeze(0).expand(layer.out_dim,-1,-1)
        mask_der=mask_val.clone()
        masks[l_idx]={'proto_val':mask_val,'proto_der':mask_der}
        
        n_selected=mask_val.sum().item()
        n_total=mask_val.numel()
        print(f"    BSRM: Layer {l_idx}: {int(n_selected)}/{n_total} coefficients selected")
    return masks

def adapt_with_bsrm(policy,wm,masks,epochs=30,H=3,batch=128,lr=1e-4,label=""):
    """Adapt policy: only update BSRM-selected coefficients."""
    opt=torch.optim.Adam(policy.parameters(),lr=lr)
    wm.eval()
    for p in wm.parameters():p.requires_grad=False
    t0=time.time()
    for ep in range(1,epochs+1):
        total_loss=0.0
        for _ in range(100):
            th=np.random.uniform(-np.pi,np.pi,batch)
            td=np.random.uniform(-8.,8.,batch)
            s=torch.stack([torch.cos(torch.tensor(th)),torch.sin(torch.tensor(th)),torch.tensor(td)/8.],dim=-1).float()
            opt.zero_grad();loss=torch.tensor(0.,device=DEVICE);s_cur=s.clone()
            for t in range(H):
                a=policy(s_cur);s_cur=wm(torch.cat([s_cur,a],dim=-1))
                loss=loss+(0.9**t)*(s_cur-S_T.expand(batch,-1)).pow(2).sum(dim=-1).mean()
            loss.backward()
            # Apply BSRM mask: zero out gradients for non-selected coefficients
            for l_idx,layer in enumerate(policy.pk.layers):
                layer.proto_val.grad*=masks[l_idx]['proto_val']
                layer.proto_der.grad*=masks[l_idx]['proto_der']
            opt.step()
            total_loss+=loss.item()
        if ep%15==0:print(f"    {label} ep{ep:3d} loss={total_loss/100:.4f} time={time.time()-t0:.0f}s")
    return policy

# ========== Main ==========
print("="*60)
print("CDPN v6 + BSRM: Selective Adaptation")
print("="*60)

# Phase 1: Train WM on g=10
print("\n[1] Train WM on g=10")
X,Y=generate_pendulum_data(5000,seed=42)
X15,Y15=generate_pendulum_data(5000,seed=43)
wm_old,_=train_wm(X.to(DEVICE),Y.to(DEVICE))

# Phase 2: Train policy through differentiable WM
print("\n[2] Train ProtoKAN Policy (H=3, epochs=100)")
policy=ProtoPolicy()
policy=train_policy(policy,wm_old,epochs=100,H=3,batch=128,label="Initial")
env=gym.make("Pendulum-v1")
succ,_=evaluate(policy,env);print(f"  g=10 baseline: {succ}/20")

# Phase 3: Fine-tune WM on g=15
print("\n[3] Fine-tune WM on g=15")
wm_new=ProtoKAN([4,12,3],n_prototypes=16).to(DEVICE)
for l in wm_new.layers:l.log_sigma.data.fill_(-1.5)
wm_new.load_state_dict(wm_old.state_dict())
opt_wm=torch.optim.Adam(wm_new.parameters(),lr=1e-4);wm_new.train()
for _ in range(200):
    idx=torch.randint(0,len(X15),(256,));l=mse(wm_new(X15[idx].to(DEVICE)),Y15[idx].to(DEVICE))
    opt_wm.zero_grad();l.backward();opt_wm.step()
wm_new.eval();print(f"  WM fine-tuned")

# Phase 4: BSRM
print("\n[4] BSRM: compute relevance masks")
env15=gym.make("Pendulum-v1")
masks=compute_bsrm(policy,wm_old,wm_new,env15,policy_noise=0.1,trials=5)

# Phase 5: Adapt with BSRM
print("\n[5] Adapt with BSRM (30 epochs)")
policy_bsrm=ProtoPolicy()
policy_bsrm.load_state_dict(policy.state_dict())
policy_bsrm=adapt_with_bsrm(policy_bsrm,wm_new,masks,epochs=30,label="BSRM")

g15_bsrm,_=evaluate(policy_bsrm,env15)
g10_bsrm,_=evaluate(policy_bsrm,env)
print(f"  BSRM: g=15={g15_bsrm}/20  g=10(forgetting)={g10_bsrm}/20")

# Phase 6: Full fine-tune (baseline for comparison)
print("\n[6] Full fine-tune (baseline, 30 epochs)")
policy_full=ProtoPolicy()
policy_full.load_state_dict(policy.state_dict())
_=adapt_with_bsrm(policy_full,wm_new,{},epochs=30,label="FULL")
# No masks -> all gradients flow

g15_full,_=evaluate(policy_full,env15)
g10_full,_=evaluate(policy_full,env)
print(f"  FULL: g=15={g15_full}/20  g=10(forgetting)={g10_full}/20")

print("\n"+"="*60);print("SUMMARY");print("="*60)
print(f"  Baseline:          g=10={succ}/20")
print(f"  BSRM adaptation:   g=15={g15_bsrm}/20  g=10={g10_bsrm}/20")
print(f"  Full fine-tune:    g=15={g15_full}/20  g=10={g10_full}/20")
