import numpy as np,sys,os;sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G=9.8;MC=1.;MP=.1;L=.5;DT=.02;FM=10.;XS=2.5;XDS=3.;THS=.3;THDS=3.
def cps(sr,a):
    x,xd,th,thd=sr[:,0],sr[:,1],sr[:,2],sr[:,3];f=a*FM;ct,st=np.cos(th),np.sin(th)
    tm_=MC+MP;pml_=MP*L;tmp=(f+pml_*thd**2*st)/tm_
    d_=.5*(4./3.-MP*ct**2/tm_)+1e-8;th_a=(G*st-ct*tmp)/d_;x_a=tmp-pml_*th_a*ct/tm_
    return np.c_[x+xd*DT+x_a*DT**2/2,xd+x_a*DT,th+thd*DT+th_a*DT**2/2,thd+th_a*DT]
def ev(p,s,H=200):
    for st in range(H):
        w1=p[:512].reshape(8,64);b1=p[512:576];w2=p[576:3648].reshape(64,48)
        b2=p[3648:3696];w3=p[3696:3744].reshape(48,1);b3=p[3744:3745]
        x=np.r_[s[0],[0,0,0,0]];x=np.tanh(x@w1+b1);x=np.tanh(x@w2+b2);a=np.tanh((x@w3+b3)[0])
        sn=cps(s*np.array([[XS,XDS,THS,THDS]]),np.array([a]))/np.array([XS,XDS,THS,THDS]);s=sn
        if abs(s[0,2]*THS)>.21 or abs(s[0,0]*XS)>2.4:return st
    return H

d=3745;rng=np.random.RandomState(42);m=rng.randn(d)*.01;sig=.3;lr=.05;pop=200
print(f"ES pop={pop}, gens=200, nr=3 fixed seeds, rank-based")
import time as tm;t0=tm.time()
for g in range(1,201):
    nse=rng.randn(pop,d)*sig;seeds=rng.randint(0,10000,3);fits=np.zeros(pop)
    for i in range(pop):
        t=0
        for sd in seeds:
            rs=np.random.RandomState(sd);th=rs.uniform(-.05,.05)
            t+=ev(m+nse[i],np.array([[0.,0.,th/THS,0.]]))
        fits[i]=t/3
    rk=np.argsort(np.argsort(fits));ft=(rk-rk.mean())/(rk.std()+1e-8)
    m+=lr*(nse.T@ft)/pop
    if g%40==0:print(f"  Gen{g:4d} best={fits.max():.0f} median={np.median(fits):.0f} mean={fits.mean():.0f}")
print(f"Time: {tm.time()-t0:.0f}s")
print("\nEval (20 trials):")
succ,stp=0,[]
for t in range(20):
    th=np.random.uniform(-.05,.05);s=np.array([[0.,0.,th/THS,0.]])
    for st in range(500):
        w1=m[:512].reshape(8,64);b1=m[512:576];w2=m[576:3648].reshape(64,48)
        b2=m[3648:3696];w3=m[3696:3744].reshape(48,1);b3=m[3744:3745]
        x=np.r_[s[0],[0,0,0,0]];x=np.tanh(x@w1+b1);x=np.tanh(x@w2+b2);a=np.tanh((x@w3+b3)[0])
        sn=cps(s*np.array([[XS,XDS,THS,THDS]]),np.array([a]))/np.array([XS,XDS,THS,THDS]);s=sn
        if abs(s[0,2]*THS)>.21 or abs(s[0,0]*XS)>2.4:break
    stp.append(st+1)
    if st+1>=500:succ+=1
print(f"CartPole: {succ}/20 ({succ*5}%) steps={np.mean(stp):.0f} max={max(stp)}")