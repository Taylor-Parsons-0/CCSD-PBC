import numpy as np

def t1Eq():
  if T==1:
    t1_f=np.zeros((V,O))
    for a in range(V):
      for i in range(O):
        for ka in range():
          for ki in range():
            t1[a,i,ka,ki]+=Fock[a,i,ka,ki]
            for k in range(O):
              t1[a,i,ka,ki]-=kapoo[k,i,kk,ki]*t1[a,k,ka,kk]
              for l in range(O):
                for c in range(V):
                  for kk in range():
                    for kl in range():
                      t1[a,i,ka,ki]-=w[k,l,i,c,kk,kl,ki,kc]*t2[a,c,k,l,ka,kc,kk,kl]
                  for kl in range():
                    t1[a,i,ka,ki]-=w[k,l,i,c,kk,kl,ki,kc]*t1[a,k,ka,kk]*t1[c,l,kc,kl]
              for c in range(V):
                t1[a,i,ka,ki]-=2*Fock[k,c,kk,kc]*t1[a,k,ka,kk]*t1[c,i,kc,ki]
                t1[a,i,ka,ki]+=kapov[k,c,kk,kc]*t1[c,i,kc,ki]*t1[a,k,ka,kk]
                for kk in range():
                  t1[a,i,ka,ki]+=kap[k,c,kk,kc]*(2*t2[c,a,k,i,kc,ka,kk,ki]-t2[c,a,i,k,kc,ka,ki,kk])
                  t1[a,i,ka,ki]+=w[a,k,i,c,ka,kk,ki,kc]*t1[c,k,kc,kk]
                for d in range(V):
                  for kk in range():
                    t1[a,i,ka,ki]+=w[a,k,c,d,ka,kk,kc,kd]*t1[c,i,kc,ki]*t1[d,k,kd,kk]
                    for kc in range():
                      t1[a,i,ka,ki]+=w[a,k,c,d,ka,kk,kc,kd]*t2[c,d,i,k,kc,kd,ki,kk]
            for c in range(V):
              t1[a,i,ka,ki]+=kapvv[a,c,ka,kc]*t1[c,i,kc,ki]
