import numpy as np
import os
import sys
import re
import time
from read import getFort, get2e, conMO
from ein_ccsdAmps import tau_tildeEq, tauEq, intermediateEqs, t1Eq, t2Eq, E_CCSD 
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
O, V, NB, scfE, Fock, Coeff=getFort(molecule, log)
print(f"Info molecule, {O},{V},{NB},{scfE}")
#Initialize arrays
coul=np.zeros((NB,NB,NB,NB))
exc=np.zeros((NB,NB,NB,NB))
OE=np.zeros((NB))
AOInt=np.zeros((NB, NB, NB, NB))
twoE=np.zeros((NB, NB, NB, NB))
O2 = O*2
V2 = V*2
#NB=NB*2
IJKL=np.zeros((O2,O2,O2,O2))
ABCD=np.zeros((V2,V2,V2,V2))
IABC=np.zeros((O2,V2,V2,V2))
IJAB=np.zeros((O2,O2,V2,V2))
IJKA=np.zeros((O2,O2,O2,V2))
IAJB=np.zeros((O2,V2,O2,V2))
#IJKL=np.zeros((2*O,2*O,2*O,2*O))
#ABCD=np.zeros((2*V,2*V,2*V,2*V))
#IABC=np.zeros((2*O,2*V,2*V,2*V))
#IJAB=np.zeros((2*O,2*O,2*V,2*V))
#IJKA=np.zeros((2*O,2*O,2*O,2*V))
#AIBC=np.zeros((2*V,2*O,2*V,2*V))
#IABJ=np.zeros((2*O,2*V,2*V,2*O))
#IJAK=np.zeros((2*O,2*O,2*V,2*O))
#IAJB=np.zeros((2*O,2*V,2*O,2*V))
#ABCI=np.zeros((2*V,2*V,2*V,2*O))
#IAJK=np.zeros((2*O,2*V,2*O,2*O))
#MO=np.zeros((2*NB, 2*NB, 2*NB, 2*NB))
delta=np.identity(2*NB)
#twoE=np.zeros((2*NB, 2*NB, 2*NB, 2*NB))

#Get 2e integrals
AOInt=get2e(AOInt, log)

#Change to spin orbital form
IJKL, ABCD, IABC, IJAB, IJKA, IAJB=conMO(O, V, NB, Fock, Coeff, AOInt, IJKL, ABCD, IABC, IJAB, IJKA, IAJB)

#Initialize T1 and T2
#O=O*2
#V=V*2
#NB=NB*2
# t1=np.zeros((V, O))
# t2=np.zeros((V, V, O, O))
#Define Denominator Arrays
# D1=np.zeros((V, O))
D2 = np.zeros((O2 O2, V2, V2))
t2 = np.zeros((O2, O2, V2, V2))
#Initial T2 Guess
for i in range(O2):
  for j in range(O2):
    den = Fock[i,i]+Fock[j,j]
    for a in range(V2):
      for b in range(V2):
        D2[i,j,a,b] = den-Fock[a+O2,a+O2]-Fock[b+O2,b+O2]
t2 = IJAB/D2        
MP2 = 0.25*np.einsum('ijab,ijab',IJAB,t2,optimize=True)
OV = O + V
#for i in range(O):
#  for j in range(O):
#    for a in range(V):
#      for b in range(V):
#        D2[i,j,a,b]=Fock[i,i]+Fock[j,j]-Fock[a+O,a+O]-Fock[b+O,b+O]
#        D2[i+O,j+O,a+V,b+V]=Fock[i+OV,i+OV]+Fock[j+OV,j+OV]-Fock[a+O+OV,a+O+OV]-Fock[b+O+OV,b+O+OV]
#        D2[i,j+O,a,b+V]=Fock[i,i]+Fock[j+OV,j+OV]-Fock[a+O,a+O]-Fock[b+O+OV,b+O+OV]
#        D2[i,j+O,a+V,b]=Fock[i,i]+Fock[j+OV,j+OV]-Fock[a+O+OV,a+O+OV]-Fock[b+O,b+O]
#        D2[i+O,j,a+V,b]=Fock[i+OV,i+OV]+Fock[j,j]-Fock[a+O+OV,a+O+OV]-Fock[b+O,b+O]
#        D2[i+O,j,a,b+V]=Fock[i+OV,i+OV]+Fock[j,j]-Fock[a+O,a+O]-Fock[b+O+OV,b+O+OV]
#        t2[i,j,a,b]=IJAB[i,j,a,b]/D2[i,j,a,b]
#        t2[i+O,j+O,a+V,b+V] = IJAB[i+O,j+O,a+V,b+V]/D2[i+O,j+O,a+V,b+V]
#        t2[i,j+O,a,b+V]     = IJAB[i,j+O,a,b+V]    /D2[i,j+O,a,b+V]
#        t2[i,j+O,a+V,b]     = IJAB[i,j+O,a+V,b]    /D2[i,j+O,a+V,b]
#        t2[i+O,j,a+V,b]     = IJAB[i+O,j,a+V,b]    /D2[i+O,j,a+V,b]
#        t2[i+O,j,a,b+V]     = IJAB[i+O,j,a,b+V]    /D2[i+O,j,a,b+V]             
MP2 = 0.25*np.einsum('ijab,ijab',IJAB,t2,optimize=True)
# print(f"{O},{V},{NB},{Fock[0,0]},{Fock[O,O]},{2*Fock[0,0]-Fock[O,O]}")
# for i in range(O*2):
#   for j in range(O*2):
#     for a in range(V*2):
#       for b in range(V*2):
#         print(f"{i+1,j+1,a+1,b+1}: {IJAB[i,j,a,b]},{D2[i,j,a,b]},{t2[i,j,a,b]}")
#print("t2\n",t2)
print("MP2",MP2)
print("guess",np.einsum('ijab,ijab',t2,t2,optimize=True))
stop

#Equation (13)
for a in range(V2):
  for i in range(O2):
    D1[i,a]=Fock[i,i]-Fock[a+O2,a+O2]
#    D1[i+O,a+V]=Fock[i+OV,i+OV]-Fock[a+O+OV,a+O+OV]
#Equation (14)
#for a in range(V):
#  for b in range(V):
#      for j in range(O):
#        D2[a,b,i,j]=Fock[i,i]+Fock[j,j]-Fock[a+O,a+O]-Fock[b+O,b+O]


#CCSD Convergence Loop
#Iterate until converged

E_Corr2=0
DiffE=1
#DiffT1=1
#DiffT2=1
#t1RMSE=1
#t2RMSE=1
N=0

#tau=tauEq(O, V, t1, t2, 1)#NB, Fock, t1, t2, MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, 1)

#Clean pervious outputs
os.system(f"rm {molecule}.txt")

#CCSD T and E Loop
with open(f"{molecule}.txt","a") as writer:
  writer.write("*******SOLVING CCSD T AMPLITUDE AND ENERGY*******\n___________________________________________________________________\n___________________________________________________________________\n\n")
while DiffE>1e-9 or DiffT1>1e-7 or DiffT2>1e-7 or t1RMSE>1e-7 or t2RMSE>1e-7:
  E_Corr1=E_Corr2
#Calculate intermediates
#  tau_tilde=tau_tildeEq(O, V, t1, t2, 1) #NB, t1, t2, 1)#Fock, t1, t2, #MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, 1)
#  tau=tauEq(O, V, t1, t2, 1)#NB, Fock, t1, t2, MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, 1)
#  F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej=intermediateEqs(O, V, NB, Fock, t1, t2, MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, 1, tau_tilde, tau)
##Do t1 step
#  t1_f=t1Eq(O, V, NB, Fock, t1, t2, MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, 1, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D1)
#  start=time.time()
##Do t2 step
#  t2_f=t2Eq(O, V, NB, Fock, t1, t2, MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, 1, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D2)
##Check for convergence
#  DiffT1=abs(np.max(t1_f-t1))
#  DiffT2=abs(np.max(t2_f-t2))
#  t1RMSE=(np.sum(((t1_f-t1)**(2))/(np.size(t1))))**(1/2)
#  t2RMSE=(np.sum(((t2_f-t2)**(2))/(np.size(t2))))**(1/2)
#  t1=np.copy(t1_f)
#  t2=np.copy(t2_f)
#  E_Corr2=CCSD(O, V, NB, Fock, t1, IJAB, 1, tau)#t2, IJAB, 1, tau)#MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, 1, tau)
#  DiffE=abs(E_Corr2-E_Corr1)
#  N+=1
#Calculate intermediates
  tau_tilde = tau_tildeEq(1, O, V, t1, t2)
  tau = tauEq(1, O, V, t1, t2)
  F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej = intermediateEqs(1, O, V, Fock, t1, t2, IJKL, ABCD, IABC, IJAB, IJKA, tau_tilde, tau)
  #Do t1 step
  t1_f = t1Eq(1, O, Fock, t1, t2, IABC, IJKA, IAJB, F_ae, F_mi, F_me, D1)
  start=time.time()
  #Do t2 step
  t2_f = t2Eq(1, t1, t2, IABC, IJAB, IJKA, IAJB, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D2)
#Check for convergence
  DiffT1 = abs(np.max(t1_f-t1))
  DiffT2 = abs(np.max(t2_f-t2))
  t1RMSE = (np.sum(((t1_f-t1)**(2))/(np.size(t1))))**(1/2)
  t2RMSE = (np.sum(((t2_f-t2)**(2))/(np.size(t2))))**(1/2)
  t1 = np.copy(t1_f)
  t2 = np.copy(t2_f)
  E_Corr2 = E_CCSD(O, Fock, t1, IJAB, tau)
  DiffE = abs(E_Corr2-E_Corr1)
  N +=1
  with open(f"{molecule}.txt","a") as writer:
    writer.write(f"Iteration {N}: CCSD {E_Corr2}\n")
    writer.write(f"Total Energy: {scfE+E_Corr2}\n")
    writer.write(f"Time: {time.time()-start}\n")
#  shape=np.shape(t2)
#  with open("ijab.txt","a") as writer:
#    writer.write(f"Iteration {N}\n---------------------------------------------------\n")
#    for i in range(O):
#      for j in range(O):
#        for a in range(V):
#          for b in range(V):
#            if IJAB[i,j,a,b]>1e-6:
#              writer.write(f"ijab({i+1, j+1, a+1, b+1}): {IJAB[i,j,a,b]}\n")
#  with open("t2s.txt","a") as writer:
#    writer.write(f"Iteration {N}\n---------------------------------------------------\n")
#    for i in range(shape[0]):
#      for j in range(shape[1]):
#        for k in range(shape[2]):
#          for l in range(shape[3]):
#            if t2[i,j,k,l]>1e-6:
#              writer.write(f"t2({i+1,j+1,k+1,l+1}): {t2[i,j,k,l]}\n\n")
DiffL1=1
DiffL2=1
lam1RMSE=1
lam2RMSE=1
N=0

shape=np.shape(t2)

#with open("t2s.txt","w") as writer:
#  for i in range(shape[0]):
#    for j in range(shape[1]):
#      for k in range(shape[2]):
#        for l in range(shape[3]):
#          if t2[i,j,k,l]>1e-6:
#            writer.write(f"t2({i,j,k,l}): {t2[i,j,k,l]}\n\n")

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
