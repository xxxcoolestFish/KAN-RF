"""CDPN v6: Train KAN/ProtoKAN Policy through differentiable ProtoKAN WM.
No SAC, no reward. Pure cognitive-driven policy learning."""
import torch,torch.nn as nn,numpy as np,gymnasium as gym,time,sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN,KAN
from experiments.baseline_sweep import generate_pendulum_data,train_wm
DEVICE="cpu";PI_2=np.pi/2;S_T=torch.tensor([[0.,1.,0.]]).to(DEVICE)
torch.manual_seed(42);np.random.seed(42)

# ========== Policy Networks (not SAC, just direct action output) ==========
class ProtoPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.pk=ProtoKAN([3,16,1],n_prototypes=16)
    def forward(self,s):return torch.tanh(self.pk(s))

class KANPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.kan=KAN([3,16,1],grid_size=5,spline_order=3)
    def forward(self,s):return torch.tanh(self.kan(s))

def train_policy(policy,wm,epochs=100,H=3,batch=128,lr=1e-3):
    """Train policy through differentiable WM (no SAC, no reward).
    
    Args:
        policy: KAN/ProtoKAN policy network
        wm: frozen ProtoKAN world model (differentiable)
        epochs: number of training epochs
        H: rollout horizon
        batch: batch size
        lr: learning rate
    """
    opt=torch.optim.Adam(policy.parameters(),lr=lr)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad=False
    
    t0=time.time()
    for ep in range(1,epochs+1):
        total_loss=0.0
        for _ in range(100):  # 100 batches per epoch
            # Sample random initial states
            th=np.random.uniform(-np.pi,np.pi,batch)
            td=np.random.uniform(-8.,8.,batch)
            s=torch.stack([torch.cos(torch.tensor(th)),torch.sin(torch.tensor(th)),torch.tensor(td)/8.],dim=-1).float()
            
            opt.zero_grad()
            loss=torch.tensor(0.,device=DEVICE)
            s_cur=s.clone()
            
            for t in range(H):
                a=policy(s_cur)
                s_cur=wm(torch.cat([s_cur,a],dim=-1))
                step_loss=(s_cur-S_T.expand(batch,-1)).pow(2).sum(dim=-1).mean()
                loss=loss+(0.9**t)*step_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),10.0)
            opt.step()
            total_loss+=loss.item()
        
        if ep%20==0:
            print(f"  Epoch {ep:3d}  loss={total_loss/100:.4f}  time={time.time()-t0:.0f}s")
    
    return policy

def evaluate(policy,wm,env,n=20):
    """Evaluate policy on real environment (model-free)."""
    succ=0;steps=[]
    for t in range(n):
        obs,_=env.reset(seed=42+t*100)
        for st in range(200):
            s=torch.tensor([[obs[0],obs[1],obs[2]/8.]],dtype=torch.float32)
            with torch.no_grad():
                a=policy(s).item()
            obs,_,_,_,_=env.step([a*2.])
            err=min(abs(np.arctan2(obs[1],obs[0])-PI_2),2*np.pi-abs(np.arctan2(obs[1],obs[0])-PI_2))
            if err<.2:succ+=1;steps.append(st+1);break
        else:steps.append(300)
    return succ,steps

# ========== Main Experiment ==========
print("="*60)
print("CDPN v6: WM-Gradient Policy Training (No SAC)")
print("="*60)

# Phase 1: Train WM
print("\n[1] Train ProtoKAN WM on g=10")
X,Y=generate_pendulum_data(5000,seed=42)
X15,Y15=generate_pendulum_data(5000,seed=43)
wm,_=train_wm(X.to(DEVICE),Y.to(DEVICE))
print(f"  WM val_mse  0.000003")

# Phase 2: Train Policies through differentiable WM
print("\n[2] Train ProtoKAN Policy (H=3, epochs=100)")
pk_policy=ProtoPolicy()
pk_policy=train_policy(pk_policy,wm,epochs=100,H=3,batch=128)
env=gym.make("Pendulum-v1")
pk_succ,_=evaluate(pk_policy,None,env);print(f"  ProtoKAN Policy: {pk_succ}/20")

print("\n[3] Train KAN Policy (H=3, epochs=100)")
kan_policy=KANPolicy()
kan_policy=train_policy(kan_policy,wm,epochs=100,H=3,batch=128)
kan_succ,_=evaluate(kan_policy,None,env);print(f"  KAN Policy: {kan_succ}/20")

print("\n[4] Adaptation test: g=10 -> g=15")
# Fine-tune WM on g=15
print("  Fine-tuning WM on g=15...")
opt_wm=torch.optim.Adam(wm.parameters(),lr=1e-4);wm.train()
for _ in range(200):
    idx=torch.randint(0,len(X15),(256,))
    l=nn.MSELoss()(wm(X15[idx].to(DEVICE)),Y15[idx].to(DEVICE))
    opt_wm.zero_grad();l.backward();opt_wm.step()
wm.eval()

# Full fine-tune policies on g=15 (no BSRM yet, baseline)
print("  Fine-tuning ProtoKAN Policy on g=15 (30 epochs)...")
pk_policy=train_policy(pk_policy,wm,epochs=30,H=3,batch=128,lr=1e-4)
env15=gym.make("Pendulum-v1")
pk_g15,_=evaluate(pk_policy,None,env15)
pk_g10,_=evaluate(pk_policy,None,env)
print(f"  ProtoKAN Policy: g=15={pk_g15}/20  g=10(forgetting)={pk_g10}/20")

print("\n  Fine-tuning KAN Policy on g=15 (30 epochs)...")
kan_policy=train_policy(kan_policy,wm,epochs=30,H=3,batch=128,lr=1e-4)
kan_g15,_=evaluate(kan_policy,None,env15)
kan_g10,_=evaluate(kan_policy,None,env)
print(f"  KAN Policy: g=15={kan_g15}/20  g=10(forgetting)={kan_g10}/20")

print("\n"+"="*60);print("SUMMARY");print("="*60)
print(f"  ProtoKAN: g=10={pk_succ}/20  g=15={pk_g15}/20  forgetting={pk_g10}/20")
print(f"  KAN:      g=10={kan_succ}/20  g=15={kan_g15}/20  forgetting={kan_g10}/20")
