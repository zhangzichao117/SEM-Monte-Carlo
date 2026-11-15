import os,sys,time
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,scipy.optimize as so
from scipy.stats import norm
from fit_sem import sigma,objective,edges
fit=np.load('/mnt/data/sem_fit.npz'); truth=fit['params']; M=fit['model_cov']
labels=['MAE -> AUT','MAE -> COM','MAE -> REL','MAE -> BE','AUT -> BE','COM -> BE','REL -> BE','MAE -> SV','AUT -> SV','COM -> SV','REL -> SV','BE -> SV','MAE -> AUT -> SV','MAE -> COM -> SV','MAE -> REL -> SV','MAE -> BE -> SV','MAE -> AUT -> BE -> SV','MAE -> COM -> BE -> SV','MAE -> REL -> BE -> SV','MAE -> AUT -> BE','MAE -> COM -> BE','MAE -> REL -> BE']
def effects(par):
 V,L,B,C=sigma(par);sd=np.sqrt(np.diag(C));bb=B*sd[None,:]/sd[:,None]
 direct=[bb[a,b] for a,b in edges]; a=bb[1:4,0];s=bb[5,1:4];t=bb[4,1:4]
 ind=[a[i]*s[i] for i in range(3)]+[bb[4,0]*bb[5,4]]+[a[i]*t[i]*bb[5,4] for i in range(3)]+[a[i]*t[i] for i in range(3)]
 return np.r_[direct,ind]
def deriv(fun,x,step=1e-5):
 base=fun(x); arr=np.empty((np.size(base),len(x)))
 for k in range(len(x)):
  z=x.copy();z[k]+=step;v=fun(z);z[k]-=2*step;w=fun(z);arr[:,k]=((v-w)/(2*step)).ravel()
 return arr
def calc_se(par,n):
 V=sigma(par)[0];inv=np.linalg.inv(V);J=deriv(lambda x:sigma(x)[0],par).reshape(26,26,64)
 G=np.einsum('ij,jka,kl->ila',inv,J,inv,optimize=True)
 I=n/2*np.einsum('ija,jib->ab',G,J,optimize=True)
 d=deriv(effects,par)
 try:cp=np.linalg.inv(I)
 except np.linalg.LinAlgError:cp=np.linalg.pinv(I)
 return np.sqrt(np.maximum(np.einsum('ia,ab,ib->i',d,cp,d,optimize=True),0))
N,reps,seed=map(int,sys.argv[1:]);rng=np.random.default_rng(seed);start=time.time();tru=effects(truth)
est=[];se=[];status=[];boundary=[]
bounds=[(.05,3)]*20+[(-3,3)]*32+[(-2,2)]*12
for j in range(reps):
 x=rng.multivariate_normal(np.zeros(26),M,size=N,method='cholesky');s=np.cov(x,rowvar=False,bias=True)
 r=so.minimize(lambda par:objective(par,s),truth,method='L-BFGS-B',bounds=bounds,options={'maxiter':150,'maxfun':12000,'ftol':1e-9})
 valid=np.all(np.isfinite(r.x)); bnd=bool(any(abs(v-lo)<1e-5 or abs(v-hi)<1e-5 for v,(lo,hi) in zip(r.x,bounds)))
 status.append(bool(r.success));boundary.append(bnd)
 if valid:
  est.append(effects(r.x));se.append(calc_se(r.x,N))
 else:
  est.append(np.full(22,np.nan));se.append(np.full(22,np.nan))
 if (j+1)%200==0:print('progress',N,j+1,'/',reps,'elapsed',round(time.time()-start,1),flush=True)
est=np.array(est);se=np.array(se);status=np.array(status);boundary=np.array(boundary)
np.savez_compressed(f'/mnt/data/final_mc_{N}_{seed}.npz',n=N,reps=reps,seed=seed,est=est,se=se,status=status,boundary=boundary,true=tru,labels=np.array(labels))
print('DONE',N,reps,'success',status.sum(),'boundary',boundary.sum(),'time',round(time.time()-start,1),flush=True)
