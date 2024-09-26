import numpy as np
import os
import sys
import re
import time
from read import getFort, get2e, conMO
from ein_ccsdAmps import tau_tildeEq, tauEq, intermediateEqs, t1Eq, t2Eq, E_CCSD, L_intermediate, L_intermediate_const, l1Eq, l2Eq 
#from lam import lamInts, lam1Eq, lam2Eq 
#from lam_l930 import zInts, Z1eq, Z2eq, Zeq 

#Define molecule
if len(sys.argv)==2:
  molecule=sys.argv[1]
else:
  print("MISSING MOLECULE NAME")
  exit()
log=f"{molecule}.log"
#Clean pervious outputs
os.system(f"rm {molecule}.txt")

#Occupied orbitals
O, V, NB, scfE, Fock, Coeff=getFort(molecule, log)
#Initialize arrays
coul=np.zeros((NB,NB,NB,NB))
exc=np.zeros((NB,NB,NB,NB))
OE=np.zeros((NB))
AOInt=np.zeros((NB, NB, NB, NB))
twoE=np.zeros((NB, NB, NB, NB))
O2 = O*2
V2 = V*2
IJKL=np.zeros((O2,O2,O2,O2))
ABCD=np.zeros((V2,V2,V2,V2))
IABC=np.zeros((O2,V2,V2,V2))
IJAB=np.zeros((O2,O2,V2,V2))
IJKA=np.zeros((O2,O2,O2,V2))
IAJB=np.zeros((O2,V2,O2,V2))

#Get 2e integrals
start=time.time()
AOInt=get2e(AOInt, log)
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Read 2ERI, Time: {time.time()-start}\n")

#Change to spin orbital form
start=time.time()
IJKL, ABCD, IABC, IJAB, IJKA, IAJB=conMO(O, V, NB, Coeff, AOInt, IJKL, ABCD, IABC, IJAB, IJKA, IAJB)
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"2ERI AO->MO, Time: {time.time()-start}\n")

#Initialize T1 and T2
start=time.time()
t1=np.zeros((O2, V2))
t2 = np.zeros((O2, O2, V2, V2))
#Define Denominator Arrays and compute E(SCF)
D1 = np.zeros((O2, V2))
D2 = np.zeros((O2, O2, V2, V2))
#Initial T2 Guess
for i in range(O2):
  for j in range(O2):
    den = Fock[i,i]+Fock[j,j]
    for a in range(V2):
      for b in range(V2):
        D2[i,j,a,b] = den-Fock[a+O2,a+O2]-Fock[b+O2,b+O2]
t2 = IJAB/D2
# D1 denominator
for a in range(V2):
  for i in range(O2):
    D1[i,a]=Fock[i,i]-Fock[a+O2,a+O2]
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Compute energy denominators, Time: {time.time()-start}\n")

#CCSD Convergence Loop
#Iterate until converged
E_Corr2=0
DiffE=1
DiffT1=1
DiffT2=1
t1RMSE=1
t2RMSE=1
N=0
MaxIt = 100
#CCSD T and E Loop
start0=time.time()
with open(f"{molecule}.txt","a") as writer:
  writer.write("___________________________________________________________________\n\n*******SOLVING CCSD T AMPLITUDE AND ENERGY EQS.*******\n___________________________________________________________________\n\n")
while DiffE>1e-9 or DiffT1>1e-7 or DiffT2>1e-7 or t1RMSE>1e-7 or t2RMSE>1e-7 and N< MaxIt:
  E_Corr1=E_Corr2
  # Calculate intermediates
  start=time.time()
  tau_tilde = tau_tildeEq(1, O, V, t1, t2)
  tau = tauEq(1, O, V, t1, t2)
  F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej = intermediateEqs(1, O, V, Fock, t1, t2, IJKL, ABCD, IABC, IJAB, IAJB, IJKA, tau_tilde, tau)
  # Do t1 step
  t1_f = t1Eq(1, O, Fock, t1, t2, IABC, IJKA, IAJB, F_ae, F_mi, F_me, D1)
  # t1_f = np.zeros((O2, V2))
  # Do t2 step
  t2_f = t2Eq(1, t1, t2, IABC, IJAB, IJKA, IAJB, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D2)
  # t2_f = IJAB/D2
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
    writer.write(f"Iteration {N}: E_corr(CCSD) {E_Corr2}, ")
    writer.write(f"E(CCSD): {scfE+E_Corr2}, ")
    writer.write(f"Time: {time.time()-start}\n")
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Total Time: {time.time()-start0}\n")
DiffL1=1
DiffL2=1
lam1RMSE=1
lam2RMSE=1
N=0

shape=np.shape(t2)

#CCSD Lambda equations loop
#Iterate until converged
EL_Corr2=0
DiffE=1
DiffT1=1
DiffT2=1
t1RMSE=1
t2RMSE=1
l1 = np.zeros((O2, V2))
l2 = np.zeros((O2, O2, V2, V2))
l1 = np.copy(t1)
l2 = np.copy(t2)
N=0
#CCSD Lambda Loop
with open(f"{molecule}.txt","a") as writer:
  writer.write("\n\n___________________________________________________________________\n\n*******SOLVING CCSD Lambda AMPLITUDE EQS.*******\n___________________________________________________________________\n\n")
# Compute constant intermediates
start=time.time()
tau_tilde = tau_tildeEq(1, O, V, t1, t2)
tau = tauEq(1,O,V,t1,t2)
F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej = intermediateEqs(1, O, V, Fock, t1, t2, IJKL, ABCD, IABC, IJAB, IAJB, IJKA, tau_tilde, tau)
W_efam, W_iemn = L_intermediate_const(1,t1,t2,tau,IJAB,IAJB,IJKA,IABC,F_ae,F_mi,F_me,W_mnij,W_abef,W_mbej)
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Compute constant intermediates, Time: {time.time()-start}\n")
# Loop
start0=time.time()
while DiffE>1e-9 or DiffT1>1e-7 or DiffT2>1e-7 or t1RMSE>1e-7 or t2RMSE>1e-7 and N< MaxIt:
  EL_Corr1 = EL_Corr2
  # Calculate intermediates
  start=time.time()
  G_ae, G_mi = L_intermediate(1,t2,l2)
  # Do l1 step
  l1_f = l1Eq(1,t1,l1,l2,IJAB,IABC,IJKA,W_efam,W_iemn,W_mbej,F_ae,F_mi,F_me,G_ae,G_mi,D1)
  # l1_f = np.zeros((O2, V2))
  # Do l2 step
  l2_f = l2Eq(1,t1,l1,l2,IABC,IJAB,IJKA,F_ae,F_mi,F_me,G_ae,G_mi,W_mnij,W_abef,W_mbej,D2)
  # l2_f = IJAB/D2
  DiffT1 = abs(np.max(l1_f-l1))
  DiffT2 = abs(np.max(l2_f-l2))
  t1RMSE = (np.sum(((l1_f-l1)**(2))/(np.size(l1))))**(1/2)
  t2RMSE = (np.sum(((l2_f-l2)**(2))/(np.size(l2))))**(1/2)
  l1 = np.copy(l1_f)
  l2 = np.copy(l2_f)
  # Compute the a fake tau to check the energy, consistently with
  # Gaussian, but let's use the tau_tilde array.
  tau_tilde = tauEq(1, O, V, l1, l2)
  EL_Corr2 = E_CCSD(O,Fock,l1,IJAB,tau_tilde)
  DiffE = abs(EL_Corr2-EL_Corr1)
  N +=1
  with open(f"{molecule}.txt","a") as writer:
    writer.write(f"Iteration {N}: DE(L-CCSD) {EL_Corr2}, ")
    writer.write(f"E(L-CCSD): {scfE+EL_Corr2}, ")
    writer.write(f"Time: {time.time()-start}\n")
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Total Time: {time.time()-start0}\n")
DiffL1=1
DiffL2=1
lam1RMSE=1
lam2RMSE=1
N=0
