import numpy as np
import os
import sys
import re
import time
from read import getFort, get2e, conMO, getpert
from ein_ccsdAmps import denom, AmpIt, tau_tildeEq, tauEq, T_interm, t1Eq, t2Eq, E_CCSD, fill_kl, L_Interm, Const_Interm, l1Eq, l2Eq#, pert_rhs, tx1Eq, tx2Eq, Xi, TrDen1

#Define molecule
if len(sys.argv)<2:
  print("MISSING MOLECULE NAME")
  exit()
else:
  molecule=sys.argv[1]
#Clean pervious outputs
os.system(f"rm {molecule}.txt")

#Occupied orbitals
O, V, NB, scfE, Fock, MOCoef, ipbc, k_weights, Core=getFort(molecule)
O2 = O*2
V2 = V*2
NB2 = NB*2
# # Initialize arrays
# coul=np.zeros((NB,NB,NB,NB))
# exc=np.zeros((NB,NB,NB,NB))
# OE=np.zeros((NB))
#AOInt=np.zeros((NBX, NBX, NBX, NBX))
# twoE=np.zeros((NB, NB, NB, NB))
# IJKL=np.zeros((O2,O2,O2,O2))
# ABCD=np.zeros((V2,V2,V2,V2))
# IABC=np.zeros((O2,V2,V2,V2))
# IJAB=np.zeros((O2,O2,V2,V2))
# IJKA=np.zeros((O2,O2,O2,V2))
# IABJ=np.zeros((O2,V2,V2,O2))

##########################################################################  
# Get AO 2e integrals and transform in MO basis
##########################################################################  
start=time.time()
#AOInt=get2e(NB,ipbc,AOInt)
AOInt = get2e(NB,ipbc)
print(f"AOInt-3 {AOInt.shape} {len(AOInt)}")

with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Read 2ERI, Time: {time.time()-start:.2f}s\n")
#Change to spin orbital form
start=time.time()
IJKL,ABCD,IABC,IJAB,IJKA,IABJ = conMO(O,V,NB,ipbc,MOCoef,AOInt)
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"2ERI AO->MO, Time: {time.time()-start:.2f}s\n")

start=time.time()
# PBC Info
nmtpbc = 1
Nkp = 1
kp = []
Ok = O
Vk = V
O2k = O2
V2k = V2
NBk = NB
NB2k = NB2
if(ipbc):
  nmtpbc = ipbc[1]
  kp, l_list = fill_kl(ipbc)
  Nkp = len(kp)
  O2k = O2*Nkp
  V2k = V2*Nkp
  Ok = O*Nkp
  Vk = V*Nkp
  NBk = NB*Nkp
  NB2k = NB2*Nkp
NkpC = Nkp*Nkp*Nkp
# Convergence thresholds on energy and amplitudes
ThrE = 1e-8
ThrA = ThrE*100
# Maximum number of iterations allowed
MaxIt = 100
# Define denominator arrays
W = 0
D1, D2 =  denom(1,O2,V2,kp,Fock,W)
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Compute energy denominators, Time: {time.time()-start:.2f}s\n")
  
##########################################################################  
# CCSD Energy and Amplitudes
##########################################################################
start=time.time()
# Initialize T1 and T2
t1 = np.zeros((O2k,V2k),dtype=Fock.dtype)
# t2 = np.zeros((O2, O2, V2, V2))
t2 = np.conjugate(IJAB)/D2.real
EMP2 = 0.25*np.einsum('ijab,ijab',IJAB,t2,optimize=True)/NkpC
# t2 = IJAB/D2.real
# EMP2 = 0.25*np.einsum('ijab,ijab',np.conjugate(IJAB),t2,optimize=True)/NkpC
# t2 = t2.reshape((4,O2,4,O2,4,V2,4,V2))
# t2 = np.transpose(t2,axes=(0,2,4,6,1,3,5,7))
# print(f"t2--0: \n ")
# pi2 = round(2*np.pi,10)
# for n in range(Nkp):
#   for h in range(Nkp):
#     for k in range(Nkp):
#       for g in range(Nkp):
#         kn = kp[n]
#         kh = kp[h]
#         kk = kp[k]
#         kg = kp[g]
#         ktot = round(kn-kk+kh-kg,10)
#         if(abs(ktot) > 1e-8 and abs(ktot%pi2) > 1e-8): 
#           print(f"K values {kn} {kh} {kk} {kg} {ktot} {abs(ktot % pi2)}")
#           print(f"Ktot {ktot} {abs(ktot % pi2)}")
#           print(f"K indexes {n} {h} {k} {g} ")
#           print(f"{t2[n,h,k,g,:,:,:,:]}")

# Solve amplitude equations
with open(f"{molecule}.txt","a") as writer:
  writer.write("****************************************************\n")
  writer.write("*          SOLVING CCSD T AMPLITUDE EQS.           *\n")
  writer.write("****************************************************\n")
  writer.write(f"E(MP2) = {EMP2.real:.10f}\n")
tau = np.zeros((O2k,O2k,V2k,V2k),dtype=Fock.dtype)
W_efam = np.zeros((V2k,V2k,V2k,O2k),dtype=Fock.dtype)
W_iemn = np.zeros((O2k,V2k,O2k,O2k),dtype=Fock.dtype)
W_mbej = np.zeros((O2k,V2k,V2k,O2k),dtype=Fock.dtype)
W_mnij = np.zeros((O2k,O2k,O2k,O2k),dtype=Fock.dtype)
W_abef = np.zeros((V2k,V2k,V2k,V2k),dtype=Fock.dtype)
F_ae = np.zeros((V2k,V2k),dtype=Fock.dtype)
F_mi = np.zeros((O2k,O2k),dtype=Fock.dtype)
F_me = np.zeros((O2k,V2k),dtype=Fock.dtype)
t1, t2 = AmpIt("T",molecule,Ok,Vk,Nkp,MaxIt,ThrE,ThrA,scfE,Fock,IJKL,ABCD,
               IABC,IJAB,IABJ,IJKA,tau,W_efam,W_iemn,W_mbej,W_mnij,W_abef,
               F_ae,F_mi,F_me,D1,D2,D1,D2,t1,t2,t1,t2,t1,t2,ipbc)
#exit()

##########################################################################  
# Compute constant intermediates
##########################################################################  
start=time.time()
tau_tilde = tau_tildeEq(1, Nkp, t1, t2)
tau = tauEq(1, Nkp, t1, t2)
F_ae,F_mi,F_me,W_mnij,W_abef,W_mbej = T_interm(1,Ok,Vk,Nkp,Fock,t1,t2,IJKL,
                                               ABCD,IABC,IJAB,IABJ,IJKA,
                                               tau_tilde,tau)
del ABCD
fae_prod = np.einsum('ia,ia->',np.conjugate(F_ae),F_ae,optimize=True)/Nkp 
fmi_prod = np.einsum('ia,ia->',np.conjugate(F_mi),F_mi,optimize=True)/Nkp
fme_prod = np.einsum('ia,ia->',np.conjugate(F_me),F_me,optimize=True)/Nkp
wabef_prod = np.einsum('ijab,ijab->',np.conjugate(W_abef),W_abef,optimize=True)/(Nkp*Nkp*Nkp)
wmbej_prod = np.einsum('ijab,ijab->',np.conjugate(W_mbej),W_mbej,optimize=True)/(Nkp*Nkp*Nkp)
wmnij_prod = np.einsum('ijab,ijab->',np.conjugate(W_mnij),W_mnij,optimize=True)/(Nkp*Nkp*Nkp)
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Products before: Fae Fmi Fme Wabef Wmbej Wmnij\n {fae_prod.real} {fmi_prod.real} {fme_prod.real} {wabef_prod.real} {wmbej_prod.real} {wmnij_prod.real}\n")
F_ae,F_mi,W_abef,W_mbej,W_efam,W_iemn = Const_Interm(1,Nkp,t1,t2,tau,IJAB,
                                                     IABJ,IJKA,IABC,F_ae,
                                                     F_mi,F_me,W_mnij,
                                                     W_abef,W_mbej)
fae_prod = np.einsum('ia,ia->',np.conjugate(F_ae),F_ae,optimize=True)/Nkp 
fmi_prod = np.einsum('ia,ia->',np.conjugate(F_mi),F_mi,optimize=True)/Nkp
wabef_prod = np.einsum('ijab,ijab->',np.conjugate(W_abef),W_abef,optimize=True)/(Nkp*Nkp*Nkp)
wmbej_prod = np.einsum('ijab,ijab->',np.conjugate(W_mbej),W_mbej,optimize=True)/(Nkp*Nkp*Nkp)
wefam_prod = np.einsum('ijab,ijab->',np.conjugate(W_efam),W_efam,optimize=True)/(Nkp*Nkp*Nkp)
wiemn_prod = np.einsum('ijab,ijab->',np.conjugate(W_iemn),W_iemn,optimize=True)/(Nkp*Nkp*Nkp)
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Compute constant intermediates, Time: {time.time()-start:.2f}s\n")
  writer.write(f"Products: Fae Fmi Wabef Wmbej Wefam Wiemn\n {fae_prod.real} {fmi_prod.real} {wabef_prod.real} {wmbej_prod.real} {wefam_prod.real} {wiemn_prod.real}\n")
# exit()
  
##########################################################################  
# CCSD Lambda Amplitudes
##########################################################################
# l1 = np.zeros((O2, V2))
# l2 = np.zeros((O2, O2, V2, V2))
l1 = np.copy(np.conjugate(t1))
l2 = np.copy(np.conjugate(t2))
# l1 = np.copy(t1)
# l2 = np.copy(t2)
start0=time.time()
with open(f"{molecule}.txt","a") as writer:
  writer.write("****************************************************\n")
  writer.write("*        SOLVING CCSD Lambda AMPLITUDE EQS.        *\n")
  writer.write("****************************************************\n")
l1, l2 = AmpIt("L",molecule,Ok,Vk,Nkp,MaxIt,ThrE,ThrA,scfE,Fock,IJKL,W_abef,
               IABC,IJAB,IABJ,IJKA,tau,W_efam,W_iemn,W_mbej,W_mnij,W_abef,
               F_ae,F_mi,F_me,D1,D2,D1,D2,t1,t2,l1,l2,t1,t2,ipbc)
exit()

# ##########################################################################  
# # CCSD LR equations
# ##########################################################################
# #
# # NPert = number of perturbations (3 for dipoles and 6 for quadrupoles)
# # WPert = frequency of perturbation
# # if WPErt != 0, there two sets of amplitudes per perturbation Tx(+w) and Tx(-w)
# # Use same intermediates as in Lambda equations
# with open(f"{molecule}.txt","a") as writer:
#   writer.write("****************************************************\n")
#   writer.write("*           COMPUTING CCSD LR FUNCTION             *\n")
#   writer.write("****************************************************\n")
# PertType = "DipE"
# NP, X_ij, X_ia, X_ab = getpert(O,V,NB,MOCoef,PertType,molecule)
# # For now, hardwire frequency of 300 nm
# Wlist = []
# Wlist.append(0.15187784178412805)
# tx1 = np.zeros((len(Wlist), NP, 2, O2, V2))
# tx2 = np.zeros((len(Wlist), NP, 2, O2, O2, V2, V2))
# tensor = np.zeros((len(Wlist), NP, NP))
# for iw in range(len(Wlist)):
#   # Loop over frequencies    
#   W = Wlist[iw]
#   NW = 2
#   if (W==0): NW = 1 
#   for ip in range(NP):
#     # Loop over number of pertubations
#     rhs1, rhs2 = pert_rhs(1, t1, t2, X_ij[ip,:,:], X_ia[ip,:,:], X_ab[ip,:,:])
#     with open(f"{molecule}.txt","a") as writer:
#       writer.write(f"\n Perturbation {PertType}-{ip+1}\n")
#     for ipmw in range(NW):
#       # Loop over +/-omega
#       PMW = W
#       if (ipmw==1): PMW = -W 
#       with open(f"{molecule}.txt","a") as writer:
#         writer.write(f"\n Frequency {PMW:+f}\n")
#       # Reset denominators including frequency term and initialize amplitudes
#       D1, D2 =  denom(1, O2, V2, Fock, PMW)
#       tx1[iw,ip,ipmw,:,:] -= rhs1/D1
#       tx2[iw,ip,ipmw,:,:,:,:] -= rhs2/D2
#       # Amplitudes loop
#       tx1[iw,ip,ipmw,:,:], tx2[iw,ip,ipmw,:,:,:,:] = AmpIt("Tx",molecule,O,V,MaxIt,ThrE,ThrA,
#                                                            scfE,Fock,IJKL,W_abef,IABC,IJAB,
#                                                            IABJ,IJKA,tau,W_efam,W_iemn,W_mbej,
#                                                            W_mnij,W_abef,F_ae,F_mi,F_me,rhs1,
#                                                            rhs2,D1,D2,t1,t2,l1,l2,
#                                                            tx1[iw,ip,ipmw,:,:],
#                                                            tx2[iw,ip,ipmw,:,:,:,:])
#   #
#   # Now that we have all the Tx amplitudes for this W, we can compute
#   # the corresponding Xi amplitudes and contract with all other Tx
#   # amplitudes, and the transition 1PDM-like rho1 and contract with
#   # the perturbation integrals
#   #
#   start0=time.time()
#   # Reset denominators
#   D1, D2 =  denom(1, O2, V2, Fock, 0)
#   for ip in range(NP):
#     # Evaluate Xi amplitudes 
#     Xi1, Xi2 = Xi(1,tx1[iw,ip,0,:,:],tx2[iw,ip,0,:,:,:,:],
#                   l1,l2,t1,IABC,IJAB,IJKA,F_ae,F_mi,F_me,W_mbej,D2)
#     for ipa in range(NP):
#       # Contract Xi(ip) with Tx(ipa)
#       tensor[iw,ip,ipa] -= np.einsum('ia,ia->',Xi1,tx1[iw,ipa,1,:,:],optimize=True) 
#       tensor[iw,ip,ipa] -= 0.25*np.einsum('ijab,ijab->',Xi2,tx2[iw,ipa,1,:,:,:,:],optimize=True)
#     del Xi1, Xi2
#     for ipmw in range(NW):
#       # Loop over +/-omega
#       # Evaluate 1PDM
#       rho1 = TrDen1(1,O2,NB2,tx1[iw,ip,ipmw,:,:],tx2[iw,ip,ipmw,:,:,:,:],l1,l2,t1,t2)
#       for ipa in range(NP):
#         # Contract 1PDM(ip) with Pert(ipa)
#         tensor[iw,ip,ipa] += np.einsum('ij,ij->',X_ij[ipa,:,:],rho1[:O2,:O2],optimize=True) 
#         tensor[iw,ip,ipa] += np.einsum('ia,ia->',X_ia[ipa,:,:],rho1[:O2,O2:],optimize=True)   
#         tensor[iw,ip,ipa] += np.einsum('ab,ab->',X_ab[ipa,:,:],rho1[O2:,O2:],optimize=True)   
#   # Print the tensor for frequency W
#   with open(f"{molecule}.txt","a") as writer:
#     writer.write(f"\n DipE(LG)-DipE(LG) Polarizability in a.u. for W = {W:.6f} a.u.\n")
#   for ip in range(NP):
#     with open(f"{molecule}.txt","a") as writer:
#       writer.write(f" {ip+1} {tensor[iw,ip,0]:+.6f} {tensor[iw,ip,1]:+.6f} {tensor[iw,ip,2]:+.6f}\n")
#   with open(f"{molecule}.txt","a") as writer:
#     writer.write(f"Time: {time.time()-start:.2f}\n")
               
