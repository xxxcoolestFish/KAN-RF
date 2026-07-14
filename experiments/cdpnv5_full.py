"""CDPNv5 full validation: SAC + WM on Pendulum (adaptation) + CartPole."""
import torch,torch.nn as nn,numpy as np,gymnasium as gym,time,sys,os;from collections import deque
from kanrf import ProtoKAN
from experiments.baseline_sweep import generate_pendulum_data,train_wm as train_wm_pen
DEVICE="cpu";PI_2=np.pi/2;torch.manual_seed(42);np.random.seed(42)

class Buf:

LOG_MIN=-5;LOG_MAX=2
class SAC:

class ActorNet(nn.Module):

class QNet(nn.Module):

def eval_pendulum(ag,env,n=20):











# ===== PART 2: CARTIPOLE SAC + WM =====
print("\n"+ "="*60);print("PART 2: CARTIPOLE SAC+WM");print("="*60)

# CartPole dynamics (for WM training and evaluation)
G=9.8;MC=1.;MP=.1;L=.5;DT=.02;FM=10.;XS=2.5;XDS=3.;THS=.3;THDS=3.
def cp_step(sr,a,g=G,mp=MP,l=L):
    if a.dim()>1:a=a.squeeze(-1)
    x,xd,th,thd=sr[:,0],sr[:,1],sr[:,2],sr[:,3]
    f=a*FM;ct,st=torch.cos(th),torch.sin(th);tm_=MC+mp;pml_=mp*l
    tmp=(f+pml_*thd**2*st)/tm_;denom=.5*(4./3.-mp*ct**2/tm_)+1e-8
    th_a=(g*st-ct*tmp)/denom;x_a=tmp-pml_*th_a*ct/tm_
    return torch.stack([x+xd*DT+x_a*DT**2/2,xd+x_a*DT,th+thd*DT+th_a*DT**2/2,thd+th_a*DT],-1)

print("\n[2.1] Train CartPole WM")
cpx,cpy=[],[]
for _ in range(5000):
    x=np.random.uniform(-2.4,2.4);xd=np.random.uniform(-3.,3.);th=np.random.uniform(-.3,.3);thd=np.random.uniform(-3.,3.);a=np.random.uniform(-1.,1.)
    sr=torch.tensor([[x,xd,th,thd]],dtype=torch.float32)
    sn=cp_step(sr,torch.tensor([a]))
    sn[:,0]/=XS;sn[:,1]/=XDS;sn[:,2]/=THS;sn[:,3]/=THDS
    sr[:,0]/=XS;sr[:,1]/=XDS;sr[:,2]/=THS;sr[:,3]/=THDS
    cpx.append(torch.cat([sr,torch.tensor([[a]])],-1).float());cpy.append(sn.float())
cpx=torch.cat(cpx);cpy=torch.cat(cpy)
cp_wm=ProtoKAN([5,16,4],n_prototypes=16).to(DEVICE)
for l in cp_wm.layers:l.log_sigma.data.fill_(-1.5)
nt_=int(len(cpx)*.85)
optc=torch.optim.LBFGS(cp_wm.parameters(),lr=1.,max_iter=20,history_size=50,line_search_fn="strong_wolfe")
bv=float("inf");bs=None
def cc():optc.zero_grad();l=mse(cp_wm(cpx[:nt_]),cpy[:nt_]);l.backward();return l
for _ in range(80):
    optc.step(cc);v=mse(cp_wm(cpx[nt_:]),cpy[nt_:]).item()
    if v<bv:bv=v;bs={k:v.clone() for k,v in cp_wm.state_dict().items()}
cp_wm.load_state_dict(bs);cp_wm.eval()
print(f"  WM val_mse={bv:.6f}")

print("\n[2.2] SAC on CartPole")
ag2=SAC(4);bu2=Buf()
rng_cp=np.random.RandomState(42)
o2=torch.tensor([[0.,0.,rng_cp.uniform(-.05,.05)/THS,0.]],dtype=torch.float32)
for st in range(1,20001):
    a=ag2.act(o2.squeeze(0))
    sr=o2.clone();sr[:,0]*=XS;sr[:,1]*=XDS;sr[:,2]*=THS;sr[:,3]*=THDS
    sn=cp_step(sr,torch.tensor([a]));sn[:,0]/=XS;sn[:,1]/=XDS;sn[:,2]/=THS;sn[:,3]/=THDS
    r=-(sn[0,2]**2+.1*sn[0,0]**2)
    done=1. if abs(sn[0,2].item()*THS)>.21 or abs(sn[0,0].item()*XS)>2.4 else 0.
    bu2.push(o2.squeeze(0).numpy(),a,r,sn.squeeze(0).numpy(),done)
    if st%2==0:ag2.upd(bu2)
    o2=sn
    if done:o2=torch.tensor([[0.,0.,rng_cp.uniform(-.05,.05)/THS,0.]],dtype=torch.float32)
    if st%5000==0:
        with torch.no_grad():
            s,a,_,s_,_=bu2.sample(128);wp=cp_wm(torch.cat([s,a],-1));we=mse(wp,s_).item()
        succ=0
        for _ in range(10):
            th=rng_cp.uniform(-.05,.05)
            s=torch.tensor([[0.,0.,th/THS,0.]],dtype=torch.float32)
            for _s in range(500):
                a2=ag2.act(s.squeeze(0),dt=True)
                sr2=s.clone();sr2[:,0]*=XS;sr2[:,1]*=XDS;sr2[:,2]*=THS;sr2[:,3]*=THDS
                sn2=cp_step(sr2,torch.tensor([a2]));sn2[:,0]/=XS;sn2[:,1]/=XDS;sn2[:,2]/=THS;sn2[:,3]/=THDS;s=sn2
                if abs(s[0,2].item()*THS)>.21 or abs(s[0,0].item()*XS)>2.4:break
            if _s>=499:succ+=1
        print(f"  Step{st:6d} wm_err={we:.6f} eval={succ}/10")

print(f"\nDone. Total time: {time.time()-t0:.0f}s")