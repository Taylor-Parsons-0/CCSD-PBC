import numpy as np

def lamEqs():
  if T==1:
    lamoo=np.zeros((O,O,K,K))
    lamvv=np.zeros((V,V,K,K))
    #lam OO
    for k in range(O):
      for i in range(O):
        for kk in range():
          for ki in range():
            if ki==kk:
              lamoo[k,i,kk,ki]+=kapoo[k,i,kk,ki]
              for c in range(V):
                lamoo[k,i,kk,ki]+=Fock[k,c,kk,kc]*t1[c,i,kc,ki]
              for l in range(O):
                for c in range(V):
                  for kl in range():
                    lamoo[k,i,kk,ki]+=w[k,l,i,c,kk,kl,ki,kc]*t1[c,l,kc,kl]
    #lam VV
    for a in range(V):
      for c in range(V):
        for ka in range():
          for kc in range():
            if kc==ka:
              lamvv[a,c,ka,kc]+=kapvv[a,c,ka,kc]
              for k in range(O):
                lamvv[a,c,ka,kc]+=Fock[k,c,kk,kc]*t1[a,k,ka,kk]
                for d in range(V):
                  for kk in range():
                    lamvv[a,c,ka,kc]+=w[a,k,c,d,ka,kk,kc,kd]*t1[d,k,kd,kk]
  return lamoo,lamvv
