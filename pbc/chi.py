import numpy as np

def chiEqs():
  if T==1:
    chioooo=np.zeros((O,O,O,O,K,K,K,K))
    chivvvv=np.zeros((V,V,V,V,K,K,K,K))
    chivoov=np.zeros((V,O,O,V,K,K,K,K))
    chivovo=np.zeros((V,O,V,O,K,K,K,K))
    #chi OOOO
    for k in range(O):
      for l in range(O):
        for i in range(O):
          for j in range(O):
            for kk in range():
              for kl in range():
                for ki in range():
                  for kj in range():
                    if ki==kk and kj===kl:
                      chioooo[k,l,i,j,kk,kl,ki,kj]+=nu[k,l,i,j,kk,kl,ki,kj]
                      for c in range(V):
                        chioooo[k,l,i,j,kk,kl,ki,kj]+=nu[k,l,i,c,kk,kl,ki,kc]*t1[c,j,kc,kj]
                        chioooo[k,l,i,j,kk,kl,ki,kj]+=nu[k,l,c,j,kk,kl,kc,kj]*t1[c,i,kc,ki]
                        for d in range(V):
                          chioooo[k,l,i,j,kk,kl,ki,kj]+=nu[k,l,c,d,kk,kl,kc,kd]*t1[c,i,kc,ki]*t1[d,j,kd,kj]
                          for kc in range():
                            chioooo[k,l,i,j,kk,kl,ki,kj]+=nu[k,l,c,d,kk,kl,kc,kd]*t2[c,d,i,j,kc,kd,ki,kj]
    #chi VVVV
    for a in range(V):
      for b in range(V):
        for c in range(V):
          for d in range(V):
            for ka in range():
              for kb in range():
                for kc in range():
                  for kd in range():
                    if kc==ka and kd==kb:
                      chivvvv[a,b,c,d,ka,kb,kc,kd]+=nu[a,b,c,d,ka,kb,kc,kd]
                      for k in range(O):
                        chivvvv[a,b,c,d,ka,kb,kc,kd]-=nu[a,k,c,d,ka,kk,kc,kd]*t1[b,k,kb,kk]
                        chivvvv[a,b,c,d,ka,kb,kc,kd]-=nu[k,b,c,d,kk,kb,kc,kd]*t1[a,k,ka,kk]
    #chi VOOV
    for a in range(V):
      for k in range(O):
        for i in range(O):
          for c in range(V):
            for ka in range():
              for kk in range():
                for ki in range():
                  for kc in range():
                    if ki==ka and kc==kk:
                      chivoov[a,k,i,c,ka,kk,ki,kc]+=nu[a,k,i,c,ka,kk,ki,kc]
                      for l in range(O):
                        chivoov[a,k,i,c,ka,kk,ki,kc]-=nu[l,k,i,c,kl,kk,ki,kc]*t1[a,l,ka,kl]
                        for d in range(V):
                         chivoov[a,k,i,c,ka,kk,ki,kc]-=nu[l,k,d,c,kl,kk,kd,kc]*t1[d,i,kd,ki]*t1[a,l,ka,kl]
                         for kl in range():
                           chivoov[a,k,i,c,ka,kk,ki,kc]-=(1/2)*nu[l,k,d,c,kl,kk,kd,kc]*t2[d,a,i,l,kd,ka,ki,kl]
                           chivoov[a,k,i,c,ka,kk,ki,kc]+=(1/2)*w[l,k,d,c,kl,kk,kd,kc]*t2[a,d,i,l,ka,kd,ki,kl]
                      for d in range(V):
                        chivoov[a,k,i,c,ka,kk,ki,kc]+=nu[a,k,d,c,ka,kk,kd,kc]*t1[d,i,kd,ki]
    #chi VOVO
    for a in range(V):
      for k in range(O):
        for c in range(V):
          for i in range(O):
            for ka inr ange()
              for kk in range():
                for kc in range():
                  for ki in range():
                    if kc==ka and ki==kk:
                      chivovo[a,k,c,i,ka,kk,kc,ki]+=nu[a,k,c,i,ka,kk,kc,ki]
                      for l in range(O):
                        chivovo[a,k,c,i,ka,kk,kc,ki]-=nu[l,k,c,i,kl,kk,kc,ki]*t1[a,l,ka,kl]
                        for d in range(V):
                          chivovo[a,k,c,i,ka,kk,kc,ki]-=nu[l,k,c,d,kl,kk,kc,kd]*t1[d,i,kd,ki]*t1[a,l,ka,kl]
                          for kl in range():
                            chivovo[a,k,c,i,ka,kk,kc,ki]-=(1/2)*nu[l,k,c,d,kl,kk,kc,kd]*t2[d,a,i,l,kd,ka,ki,kl]
                      for d in range(V):
                        chivovo[a,k,c,i,ka,kk,kc,ki]+=nu[a,k,c,d,ka,kk,kc,kd]*t1[d,i,kd,ki]
  return chioooo,chivvvv,chivoov,chivovo
