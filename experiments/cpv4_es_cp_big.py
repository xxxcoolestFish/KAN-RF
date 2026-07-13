import numpy as np,sys,os,time;sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G=9.8;MC=1.;MP=.1;L=.5;DT=.02;FM=10.;XS=2.5;XDS=3.;THS=.3;THDS=3.

def cp_step_np(sr,a):
    x,xd,th,thd=sr[:,0],sr[:,1],sr[:,2],sr[:,3];f=a*FM;ct,st=np.cos(th),np.sin(th)
    tm_=MC+MP;pml_=MP*L;tmp=(f+pml_*thd**2*st)/tm_
    denom=.5*(4./3.-MP*ct**2/tm_)+1e-8;th_a=(G*st-ct*tmp)/denom;x_a=tmp-pml_*th_a*ct/tm_
    return np.column_stack([x+xd*DT+x_a*DT**2/2,xd+x_a*DT,th+thd*DT+th_a*DT**2/2,thd+th_a*DT])

def eval_one(params,s0,H=200):
    w1=params[:512].reshape(8,64);b1=params[512:576];w2=params[576:3648].reshape(64,48)
    b2=params[3648:3696];w3=params[3696:3744].reshape(48,1);b3=params[3744:3745]
    s=s0.copy()
    for st in range(H):
        x=np.concatenate([s[0],np.array([0.,0.,0.,0.])]);x=np.tanh(x@w1+b1)
        x=np.tanh(x@w2+b2);a=np.tanh((x@w3+b3)[0])
        sr=s*np.array([[XS,XDS,THS,THDS]])
        s_next=cp_step_np(sr,np.array([a]))/np.array([XS,XDS,THS,THDS])
        if abs(s_next[0,2]*THS)>.21 or abs(s_next[0,0]*XS)>2.4:return st
        s=s_next
    return H

rng=np.random.RandomState(42);d=3745;m=np.random.randn(d)*.01
sigma=.3;lr=.05;pop=200

print(f"ES pop={pop}, gens=300, H=200, numpy")
t0=time.time()
for gen in range(1,301):
    noise=rng.randn(pop,d)*sigma
    th0=np.random.uniform(-.05,.05)
    s0=np.array([[0.,0.,th0/THS,0.]],dtype=np.float32)
    fits=np.zeros(pop)
    for i in range(pop):
        fits[i]=eval_one(m+noise[i],s0,200)
    ft=fits.copy();ft=(ft-ft.mean())/(ft.std()+1e-8)
    m+=lr*(noise.T@ft)/pop
    if gen%50==0:print(f"  Gen{gen:4d} best={fits.max():.0f} median={np.median(fits):.0f} mean={fits.mean():.0f}")

print(f"Time: {time.time()-t0:.0f}s")
print("\nFinal eval:")
succ=0;steps=[]
for t in range(20):
    np.random.seed(42+t*100);th=np.random.uniform(-.05,.05)
    s=np.array([[0.,0.,th/THS,0.]],dtype=np.float32)
    for st in range(500):
        x=np.concatenate([s[0],np.array([0.,0.,0.,0.])])
        w1=m[:512].reshape(8,64);b1=m[512:576];w2=m[576:3648].reshape(64,48)
        b2=m[3648:3696];w3=m[3696:3744].reshape(48,1);b3=m[3744:3745]
        x=np.tanh(x@w1+b1);x=np.tanh(x@w2+b2);a=np.tanh((x@w3+b3)[0])
        sr=s*np.array([[XS,XDS,THS,THDS]]);sn=cp_step_np(sr,np.array([a]))/np.array([XS,XDS,THS,THDS]);s=sn
        if abs(s[0,2]*THS)>.21 or abs(s[0,0]*XS)>2.4:break
    steps.append(st+1)
    if st+1>=500:succ+=1
print(f"CartPole: {succ}/20 ({succ*5}%) steps={np.mean(steps):.0f} max={max(steps)}")