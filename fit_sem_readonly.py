import numpy as np, scipy.optimize as so
from scipy.linalg import solve_triangular
D=np.load('/mnt/data/sav_extracted.npz');X=D['data'][:,5:31];n,p=X.shape
S=np.cov(X,rowvar=False,bias=True); groups=[6,3,3,3,5,6];fac=np.repeat(np.arange(6),groups);first=np.cumsum([0]+groups[:-1]);free=np.array([i for i in range(p) if i not in first]);edges=[(1,0),(2,0),(3,0),(4,0),(4,1),(4,2),(4,3),(5,0),(5,1),(5,2),(5,3),(5,4)];
# covariance from structural disturbances; add residual covariance needs (optional)
def sigma(par, corr=False):
 lam=np.ones(p);lam[free]=par[:20];L=np.zeros((26,6));L[np.arange(p),fac]=lam
 err=np.exp(par[20:46]);v=np.exp(par[46:52]); B=np.zeros((6,6))
 for i,(a,b) in enumerate(edges):B[a,b]=par[52+i]
 Psi=np.diag(v)
 if corr:
  for i,(a,b) in enumerate([(1,2),(1,3),(2,3)]):Psi[a,b]=Psi[b,a]=np.tanh(par[64+i])*np.sqrt(v[a]*v[b])*0.5
 A=np.linalg.inv(np.eye(6)-B);C=A@Psi@A.T
 return L@C@L.T+np.diag(err),L,B,C

def objective(par,S=S):
 try:
  M=sigma(par)[0];sg,ld=np.linalg.slogdet(M)
  if sg<=0:return 1e8
  return ld+np.trace(np.linalg.solve(M,S))
 except: return 1e8
