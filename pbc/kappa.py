import numpy as np

def kapEqs():
  if T==1:
    kapoo=np.zeros((O,O,K,K))
    kapvv=np.zeros((V,V,K,K))
    kapov=np.zeros((O,V,K,K))
    #kap OO
    for k in range(O):
      for i in range(O):
        for kk in range():
          for ki in range():
            if ki==kk:
              kapoo[k,i,kk,ki]+=Fock[k,i,kk,ki]
              for l in range(O):
                for c in range(V):
                  for d in range(V):
                    for kl in range():
                      kapoo[k,i,kk,ki]+=w[k,l,c,d,kk,kl,kc,kd]*t1[c,i,kc,ki]*t1[d,l,kd,kl]
                      for kc in range():
                        if kc==kl:
                          kapoo[k,i,kk,ki]+=w[k,l,c,d,kk,kl,kc,kd]*t2[c,d,i,l,kc,kd,ki,kl]
    #kap VV
    for a in range(V):
      for c in range(V):
        for ka in range():
          for kc in range():
            if kc==ka:
              kapvv[a,c,ka,kc]+=Fock[a,c,ka,kc]
              for k in range(O):
                for l in range(O):
                  for d in range(V):
                   for kk in range():
                     for kl in range():
                       if kl==kk:
                         kapvv[a,c,ka,kc]-=w[k,l,c,d,kk,kl,kc,kd]*t2[a,d,k,l,ka,kd,kk,kl]
                   for kl in range():
                     kapvv[a,c,ka,kc]-=w[k,l,c,d,kk,kl,kc,kd]*t1[a,k,ka,kk]*t1[d,l,kd,kl]
    #kap OV
    for k in range(O):
      for c in range(V):
        for kk in range():
          for kc in range():
            if kc==kk:
              kapov[k,c,kk,kc]+=Fock[k,c,kk,kc]
              for l in range(O):
                for d in range(V):
                  for kl in range():
                    kapov[k,c,kk,kc]+=w[k,l,c,d,kk,kl,kc,kd]*t1[d,l,kd,kl]
  return kapoo,kapvv,kapov
