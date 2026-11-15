import os,sys
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.optimize import minimize
from fit_sem_readonly import sigma,objective
from effects_readonly import effects
nrep=int(sys.argv[1]);seed=int(sys.argv[2]);D=np.load('/mnt/data/sav_extracted.npz');X=D['data'][:,5:31];fit=np.load('/mnt/data/sem_fit.npz');p0=fit['params'];rng=np.random.default_rng(seed)
bounds=[(.05,3)]*20+[(-3,3)]*32+[(-2,2)]*12
vals=[];ok=[]
for i in range(nrep):
 z=X[rng.integers(0,len(X),len(X))];S=np.cov(z,rowvar=False,bias=True)
 res=minimize(lambda p:objective(p,S),p0,method='L-BFGS-B',bounds=bounds,options={'maxiter':150,'ftol':1e-9})
 vals.append(effects(res.x));ok.append(res.success)
np.savez_compressed('/mnt/data/bootstrap_observed_'+str(seed)+'.npz',est=np.array(vals),status=np.array(ok),seed=seed)
print(nrep,sum(ok))
