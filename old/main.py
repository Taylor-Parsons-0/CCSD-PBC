import numpy as np
import os
import sys
import re
import time
from stuff import getFort, get2e, conMO
from ein_ccsdAmps import tau_tildeEq, tauEq, intermediateEqs, t1Eq, t2Eq, CCSD 
#from lam import lamInts, lam1Eq, lam2Eq 
#from lam_l930 import zInts, Z1eq, Z2eq, Zeq 

#Define molecule
if len(sys.argv)==2:
  molecule=sys.argv[1]
else:
  print("MISSING MOLECULE NAME")
  exit()
log=f"{molecule}.log"

#Occupied orbitals
O, NB, scfE, Fock, Coeff=getFort(molecule, log)
#Initialize arrays
coul=np.zeros((NB,NB,NB,NB))
exc=np.zeros((NB,NB,NB,NB))
OE=np.zeros((NB))
AOInt=np.zeros((NB, NB, NB, NB))
MO=np.zeros((NB*2, NB*2, NB*2, NB*2))
delta=np.identity(NB*2)
twoE=np.zeros((NB, NB, NB, NB))

#Get 2e integrals
AOInt=get2e(AOInt, log)

#Change to spin orbital form
MO=conMO(twoE, MO, Coeff, NB, AOInt)

#Initialize T1 and T2
O=O*2
NB=NB*2
t1=np.zeros((NB, NB))
t2=np.zeros((NB, NB, NB, NB))

#Initial T2 Guess
MP2=np.zeros((NB,NB,NB,NB))
for i in range(O):
  for j in range(O):
    for a in range(O,NB):
      for b in range(O,NB):
        t2[a,b,i,j]+=MO[i,j,a,b]/(Fock[i,i]+Fock[j,j]-Fock[a,a]-Fock[b,b])
#        print(MO[i,j,a,b])
#print("guess",np.sum(abs(t2)))
#Define Denominator Arrays
D1=np.zeros((NB, NB))
D2=np.zeros((NB, NB, NB, NB))

#Equation (13)
for a in range(O, NB):
  for i in range(O):
    D1[a,i]=Fock[i,i]-Fock[a,a]
#Equation (14)
for a in range(O, NB):
  for b in range(O, NB):
    for i in range(O):
      for j in range(O):
        D2[a,b,i,j]=Fock[i,i]+Fock[j,j]-Fock[a,a]-Fock[b,b]


#CCSD Convergence Loop
#Iterate until converged

E_Corr2=0
DiffE=1
DiffT1=1
DiffT2=1
t1RMSE=1
t2RMSE=1
N=0

tau=tauEq(O, NB, Fock, t1, t2, MO, 1)

#Clean pervious outputs
os.system("rm out.txt")

#CCSD T and E Loop
with open("out.txt","a") as writer:
  writer.write("*******SOLVING CCSD T AMPLITUDE AND ENERGY*******\n___________________________________________________________________\n___________________________________________________________________\n\n")
while DiffE>0.000000000000001 or DiffT1>0.00000000001 or DiffT2>0.00000000001 or t1RMSE>0.00000000001 or t2RMSE>0.00000000001:
  E_Corr1=E_Corr2
  tau_tilde=tau_tildeEq(O, NB, Fock, t1, t2, MO, 1)
  tau=tauEq(O, NB, Fock, t1, t2, MO, 1)
  F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej=intermediateEqs(O, NB, Fock, t1, t2, MO, 1, tau_tilde, tau)
  t1_f=t1Eq(O, NB, Fock, t1, t2, MO, 1, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D1)
  start=time.time()
  t2_f=t2Eq(O, NB, Fock, t1, t2, MO, 1, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D2)
#  with open("out.txt","a") as writer:
#    writer.write(f"Time for Step {N}: {time.time()-start}\n")
  DiffT1=abs(np.max(t1_f-t1))
  DiffT2=abs(np.max(t2_f-t2))
  t1RMSE=(np.sum(((t1_f-t1)**(2))/(np.size(t1))))**(1/2)
  t2RMSE=(np.sum(((t2_f-t2)**(2))/(np.size(t2))))**(1/2)
  t1=np.copy(t1_f)
  t2=np.copy(t2_f)
  E_Corr2=CCSD(O, NB, Fock, t1, t2, MO, 1, tau)
  DiffE=abs(E_Corr2-E_Corr1)
  N+=1
  with open("out.txt","a") as writer:
    writer.write(f"Iteration {N}: CCSD {E_Corr2}\n")
    writer.write(f"Total Energy: {scfE+E_Corr2}\n")
    writer.write(f"Time: {time.time()-start}\n")
DiffL1=1
DiffL2=1
lam1RMSE=1
lam2RMSE=1
N=0

#CCSD Lambda and Energy Grad Loop
#with open("out.txt","a") as writer:
#  writer.write("\n\n\n\n*******SOLVING CCSD LAMBDA AMPLITUDE*******\n___________________________________________________________________\n___________________________________________________________________\n\n")
#while DiffL1>0.00000001 or DiffL2>0.00000001 or lam1RMSE>0.0000000001 or lam2RMSE>0.0000000001:
#  if N==0:
#    lam1=np.zeros((NB, NB))
#    lam2=np.zeros((NB, NB, NB, NB))
#  G_ae, G_mi, Wtt_mbej, Ft_ea, Ft_im, Wt_efab, Wt_ijmn, Wt_ejmb, Wt_iemn, Wt_mnie, Wt_efam=lamInts(O, NB, Fock, t1, t2, MO, 1, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, lam1, lam2)
#  lam1_f=lam1Eq(O, NB, Fock, t1, t2, MO, 1, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, G_ae, G_mi, Wtt_mbej, Ft_ea, Ft_im, Wt_efab, Wt_ijmn, Wt_ejmb, Wt_iemn, Wt_mnie, Wt_efam, D1, lam1, lam2)
#  lam2_f=lam2Eq(O, NB, Fock, t1, t2, MO, 1, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, G_ae, G_mi, Wtt_mbej, Ft_ea, Ft_im, Wt_efab, Wt_ijmn, Wt_ejmb, Wt_iemn, Wt_mnie, Wt_efam, D2, lam1, lam2)
#  DiffL1=abs(np.max(lam1_f-lam1))
#  DiffL2=abs(np.max(lam2_f-lam2))
#  lam1RMSE=(np.sum(((lam1_f-lam1)**(2))/(np.size(lam1))))**(1/2)
#  lam2RMSE=(np.sum(((lam2_f-lam2)**(2))/(np.size(lam2))))**(1/2)
#  lam1=np.copy(lam1_f)
#  lam2=np.copy(lam2_f)
#  N+=1
#  print(f"Iteration {N}\n")
#  print(f"Time: {time.time()-start}")
#  print(f"lambda_1 RMSE: {lam1RMSE}")
#  print(f"lambda_2 RMSE: {lam2RMSE}")
#  with open("out.txt","a") as writer:
#    writer.write(f"Iteration {N}\n")
#    writer.write(f"Time: {time.time()-start}\n")
##CHECK AMPLITUDES
#with open("lam_amps.txt","a") as writer:
#  writer.write("SINGLES\n__________________________________________\n")
#  for p in range(len(lam1)):
#    for q in range(len(lam1)):
#      if abs(lam1[p,q])>1e-6:
#        one=f"{lam1[p,q], (p,q)}\n"
#        writer.write(one)
#  writer.write("\n\n________________________________________\nDOUBLES\n")
#  for p in range(len(lam2)):
#    for q in range(len(lam2)):
#      for r in range(len(lam2)):
#        for s in range(len(lam2)):
#          if abs(lam2[p,q,r,s])>1e-6:
#            two=f"{lam2[p,q,r,s], (p,q,r,s)}\n"
#            writer.write(two)
#  

#Initialize Z1 and Z2
Z1=np.zeros((NB, NB))
Z2=np.zeros((NB, NB, NB, NB))
W_ia=np.zeros((NB, NB))
W_ijab=np.zeros((NB, NB, NB, NB))

Z1RMSE=1
Z2RMSE=2
#CCSD Lambda Equations from l930
#with open("Z_out.txt","a") as writer:
#  writer.write("\n\n\n\n*******SOLVING CCSD LAMBDA AMPLITUDE*******\n___________________________________________________________________\n___________________________________________________________________\n\n")
#while Z1RMSE>0.0000000001 or Z2RMSE>0.0000000001:
##  if N==0:
##    W_ia=np.zeros((NB, NB))
##    Z1=np.zeros((NB, NB))
##    W_ijab=np.zeros((NB, NB, NB, NB))
##    Z2=np.zeros((NB, NB, NB, NB))
#
#  G_ij, G_ab, H_il, H_ijkl, H_ad, Y_jkbc, S_ijkl, Z_bc, T_jk, W_cdaj=zInts(O, NB, Fock, t1, t2, MO, 1, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D1, D2, W_ia, W_ijab, Z1, Z2)   
#  W1=Z1eq(O, NB, Fock, t1, t2, MO, 1, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, W_ia, W_ijab, D1, D2, G_ij, G_ab, H_il, H_ijkl, H_ad, Y_jkbc, S_ijkl, Z_bc, T_jk, W_cdaj, Z1, Z2)
#  W2=Z2eq(O, NB, Fock, t1, t2, MO, 1, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, W_ia, W_ijab, D1, D2, G_ij, G_ab, H_il, H_ijkl, H_ad, Y_jkbc, S_ijkl, Z_bc, T_jk, W_cdaj, Z1, Z2)
#  Z_ia, Z_ijab=Zeq(O, NB, Fock, t1, t2, MO, 1, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, W1, W2, D1, D2, G_ij, G_ab, H_il, H_ijkl, H_ad, Y_jkbc, S_ijkl, Z_bc, T_jk, W_cdaj, Z1, Z2)
#  
#  for i in range(O):
#    for j in range(O):
#      for a in range(O, NB):
#        for b in range(O, NB):
#          if abs(W1[i,a])>1e-16:
#            print(f"{(i,a), Z_ia[i,a]}")
#          if abs(W2[i,j,a,b])>1e-16:
#            print(f"{(i,j,a,b), Z_ijab[i,j,a,b]}")
#
#  Z1RMSE=(np.sum(((Z_ia-Z1)**(2))/(np.size(Z1))))**(1/2)
#  Z2RMSE=(np.sum(((Z_ijab-Z2)**(2))/(np.size(Z2))))**(1/2)
#  
#  W_ia=np.copy(W1)
#  W_ijab=np.copy(W2)
#  Z1=np.copy(Z_ia)
#  Z2=np.copy(Z_ijab)
#  N+=1
#  print(f"Iteration {N}\n")
#  print(f"Time: {time.time()-start}")
#  print(f"lambda_1 RMSE: {Z1RMSE}")
#  print(f"lambda_2 RMSE: {Z2RMSE}")
#
##Check scalar products of arrays
#def scalp(X):
#  Xf=np.ndarray.flatten(X)
#  val=np.dot(Xf,Xf)
#  return val
#
#print(f"Scalar product of G_ij: {scalp(G_ij)}\n")
#print(f"Scalar product of G_ab: {scalp(G_ab)}")
#print(f"Scalar product of H_il: {scalp(H_il)}")
#print(f"Scalar product of H_ijkl: {scalp(H_ijkl)}")
#print(f"Scalar product of H_ad: {scalp(H_ad)}")
#print(f"Scalar product of Y_jkbc: {scalp(Y_jkbc)}")
#print(f"Scalar product of S_ijkl: {scalp(S_ijkl)}")
#print(f"Scalar product of Z_bc: {scalp(Z_bc)}")
#print(f"Scalar product of T_jk: {scalp(T_jk)}")
#print(f"Scalar product of W_cdaj: {scalp(W_cdaj)}")
