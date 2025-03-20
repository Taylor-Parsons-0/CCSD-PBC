import numpy as np
import os
import sys
import re
import time
from read import getFort, get2e, conMO, getpert
from ein_ccsdAmps import denom, AmpIt, tau_tildeEq, tauEq, intermediateEqs, t1Eq, t2Eq, E_CCSD, L_intermediate, L_intermediate_const, l1Eq, l2Eq, pert_rhs, tx1Eq, tx2Eq, Xi, TrDen1
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
NB2 = NB*2
IJKL=np.zeros((O2,O2,O2,O2))
ABCD=np.zeros((V2,V2,V2,V2))
IABC=np.zeros((O2,V2,V2,V2))
IJAB=np.zeros((O2,O2,V2,V2))
IJKA=np.zeros((O2,O2,O2,V2))
IAJB=np.zeros((O2,V2,O2,V2))

##########################################################################  
# Get AO 2e integrals and transform in MO basis
##########################################################################  
start=time.time()
AOInt=get2e(AOInt, log)
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Read 2ERI, Time: {time.time()-start:.2f}s\n")
#Change to spin orbital form
start=time.time()
IJKL, ABCD, IABC, IJAB, IJKA, IAJB=conMO(O, V, NB, Coeff, AOInt, IJKL, ABCD, IABC, IJAB, IJKA, IAJB)
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"2ERI AO->MO, Time: {time.time()-start:.2f}s\n")

start=time.time()
# Convergence thresholds on energy and amplitudes
ThrE = 1e-10
ThrA = ThrE*100
# Maximum number of iterations allowed
MaxIt = 100
# Define denominator arrays
W = 0
D1, D2 =  denom(1, O2, V2, Fock, W)
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Compute energy denominators, Time: {time.time()-start:.2f}s\n")
  
##########################################################################  
# CCSD Energy and Amplitudes
##########################################################################
start=time.time()
# Initialize T1 and T2
t1 = np.zeros((O2, V2))
t2 = np.zeros((O2, O2, V2, V2))
t2 = IJAB/D2
# Solve amplitude equations
with open(f"{molecule}.txt","a") as writer:
  writer.write("****************************************************\n")
  writer.write("*          SOLVING CCSD T AMPLITUDE EQS.           *\n")
  writer.write("****************************************************\n")
tau = np.zeros((O2, O2, V2, V2))
W_efam = np.zeros((V2, V2, V2, O2))
W_iemn = np.zeros((O2, V2, O2, O2))
W_mbej = np.zeros((O2, V2, V2, O2))
W_mnij = np.zeros((O2, O2, O2, O2))
W_abef = np.zeros((V2, V2, V2, V2))
F_ae = np.zeros((V2, V2))
F_mi = np.zeros((O2, O2))
F_me = np.zeros((O2, V2))
t1, t2 = AmpIt("T",molecule,O,V,MaxIt,ThrE,ThrA,scfE,Fock,IJKL,ABCD,IABC,IJAB,IAJB,IJKA,tau,W_efam,W_iemn,W_mbej,W_mnij,W_abef,F_ae,F_mi,F_me,D1,D2,D1,D2,t1,t2,t1,t2,t1,t2)

##########################################################################  
# Compute constant intermediates
##########################################################################  
start=time.time()
tau_tilde = tau_tildeEq(1, t1, t2)
tau = tauEq(1, t1, t2)
F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej = intermediateEqs(1, O, V, Fock, t1, t2, IJKL, ABCD, IABC, IJAB, IAJB, IJKA, tau_tilde, tau)
del ABCD
W_efam, W_iemn = L_intermediate_const(1,t1,t2,tau,IJAB,IAJB,IJKA,IABC,F_ae,F_mi,F_me,W_mnij,W_abef,W_mbej)
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Compute constant intermediates, Time: {time.time()-start:.2f}s\n")
  
##########################################################################  
# CCSD Lambda Amplitudes
##########################################################################
l1 = np.zeros((O2, V2))
l2 = np.zeros((O2, O2, V2, V2))
l1 = np.copy(t1)
l2 = np.copy(t2)
start0=time.time()
with open(f"{molecule}.txt","a") as writer:
  writer.write("****************************************************\n")
  writer.write("*        SOLVING CCSD Lambda AMPLITUDE EQS.        *\n")
  writer.write("****************************************************\n")
l1, l2 = AmpIt("L",molecule,O,V,MaxIt,ThrE,ThrA,scfE,Fock,IJKL,W_abef,IABC,IJAB,IAJB,IJKA,tau,W_efam,W_iemn,W_mbej,W_mnij,W_abef,F_ae,F_mi,F_me,D1,D2,D1,D2,t1,t2,l1,l2,t1,t2)

##########################################################################  
# CCSD LR equations
##########################################################################
#
# NPert = number of perturbations (3 for dipoles and 6 for quadrupoles)
# WPert = frequency of perturbation
# if WPErt != 0, there two sets of amplitudes per perturbation Tx(+w) and Tx(-w)
# Use same intermediates as in Lambda equations
with open(f"{molecule}.txt","a") as writer:
  writer.write("****************************************************\n")
  writer.write("*           COMPUTING CCSD LR FUNCTION             *\n")
  writer.write("****************************************************\n")
PertType = "DipE"
NP, X_ij, X_ia, X_ab = getpert(O,V,NB,Coeff,PertType,molecule)
# For now, hardwire frequency of 300 nm
Wlist = []
Wlist.append(0.15187784178412805)
tx1 = np.zeros((len(Wlist), NP, 2, O2, V2))
tx2 = np.zeros((len(Wlist), NP, 2, O2, O2, V2, V2))
tensor = np.zeros((len(Wlist), NP, NP))
for iw in range(len(Wlist)):
  # Loop over frequencies    
  W = Wlist[iw]
  NW = 2
  if (W==0): NW = 1 
  for ip in range(NP):
    # Loop over number of pertubations
    rhs1, rhs2 = pert_rhs(1, t1, t2, X_ij[ip,:,:], X_ia[ip,:,:], X_ab[ip,:,:])
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n Perturbation {PertType}-{ip+1}\n")
    for ipmw in range(NW):
      # Loop over +/-omega
      PMW = W
      if (ipmw==1): PMW = -W 
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f"\n Frequency {PMW:+f}\n")
      # Reset denominators including frequency term
      D1, D2 =  denom(1, O2, V2, Fock, PMW)
      # Initialize amplitudes
      tx1[iw,ip,ipmw,:,:] = -rhs1/D1
      tx2[iw,ip,ipmw,:,:,:,:] = -rhs2/D2
      # Amplitudes loop
      tx1[iw,ip,ipmw,:,:], tx2[iw,ip,ipmw,:,:,:,:] = AmpIt("Tx",molecule,O,V,MaxIt,ThrE,ThrA,scfE,Fock,IJKL,W_abef,IABC,IJAB,IAJB,IJKA,tau,W_efam,W_iemn,W_mbej,W_mnij,W_abef,F_ae,F_mi,F_me,rhs1,rhs2,D1,D2,t1,t2,l1,l2,tx1[iw,ip,ipmw,:,:],tx2[iw,ip,ipmw,:,:,:,:])
  #
  # Now that we have all the Tx amplitudes for this W, we can compute
  # the corresponding Xi amplitudes and contract with all other Tx
  # amplitudes, and the transition 1PDM-like rho1 and contract with
  # the perturbation integrals
  #
  start0=time.time()
  # Reset denominators
  D1, D2 =  denom(1, O2, V2, Fock, 0)
  for ip in range(NP):
    # Evaluate Xi amplitudes 
    Xi1, Xi2 = Xi(1,tx1[iw,ip,0,:,:],tx2[iw,ip,0,:,:,:,:],l1,l2,t1,IABC,IJAB,IJKA,F_ae,F_mi,F_me,W_mbej,D2)
    for ipa in range(NP):
      # Contract Xi(ip) with Tx(ipa)
      tensor[iw,ip,ipa] -= np.einsum('ia,ia->',Xi1,tx1[iw,ipa,1,:,:],optimize=True) 
      tensor[iw,ip,ipa] -= 0.25*np.einsum('ijab,ijab->',Xi2,tx2[iw,ipa,1,:,:,:,:],optimize=True)
    del Xi1, Xi2
    for ipmw in range(NW):
      # Loop over +/-omega
      PMW = W
      if (ipmw==1): PMW = -W
      # Evaluate 1PDM
      rho1 = TrDen1(1,O2,NB2,tx1[iw,ip,ipmw,:,:],tx2[iw,ip,ipmw,:,:,:,:],l1,l2,t1,t2)
      for ipa in range(NP):
        # Contract 1PDM(ip) with Pert(ipa)
        tensor[iw,ip,ipa] += np.einsum('ij,ij->',X_ij[ipa,:,:],rho1[:O2,:O2],optimize=True) 
        tensor[iw,ip,ipa] += np.einsum('ia,ia->',X_ia[ipa,:,:],rho1[:O2,O2:],optimize=True)   
        tensor[iw,ip,ipa] += np.einsum('ab,ab->',X_ab[ipa,:,:],rho1[O2:,O2:],optimize=True)   
  # Print the tensor for frequency W
  with open(f"{molecule}.txt","a") as writer:
    writer.write(f"\n DipE(LG)-DipE(LG) Polarizability in a.u. for W = {W:.6f} a.u.\n")
  for ip in range(NP):
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f" {ip+1} {tensor[iw,ip,0]:+.6f} {tensor[iw,ip,1]:+.6f} {tensor[iw,ip,2]:+.6f}\n")
  with open(f"{molecule}.txt","a") as writer:
    writer.write(f"Time: {time.time()-start:.2f}\n")
               
