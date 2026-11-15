import numpy as np
from fit_sem_readonly import sigma,edges
labels=['MAE -> AUT','MAE -> COM','MAE -> REL','MAE -> BE','AUT -> BE','COM -> BE','REL -> BE','MAE -> SV','AUT -> SV','COM -> SV','REL -> SV','BE -> SV','MAE -> AUT -> SV','MAE -> COM -> SV','MAE -> REL -> SV','MAE -> BE -> SV','MAE -> AUT -> BE -> SV','MAE -> COM -> BE -> SV','MAE -> REL -> BE -> SV','MAE -> AUT -> BE','MAE -> COM -> BE','MAE -> REL -> BE']
def effects(par):
 V,L,B,C=sigma(par);sd=np.sqrt(np.diag(C));bb=B*sd[None,:]/sd[:,None]
 direct=[bb[a,b] for a,b in edges]; a=bb[1:4,0];s=bb[5,1:4];t=bb[4,1:4]
 ind=[a[i]*s[i] for i in range(3)]+[bb[4,0]*bb[5,4]]+[a[i]*t[i]*bb[5,4] for i in range(3)]+[a[i]*t[i] for i in range(3)]
 return np.r_[direct,ind]
