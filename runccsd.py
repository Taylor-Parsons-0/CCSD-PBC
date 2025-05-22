import numpy as np
import os
import sys
import re
import time

from read import getFort, getFock, get2e, conMO, getPert
from ein_ccsdAmps import mem_check, denom, AmpIt, tau_tildeEq, tauEq, T_interm, t1Eq, t2Eq, E_CCSD, fill_kl, L_Interm, Const_Interm, l1Eq, l2Eq, pert_rhs, tx1Eq, tx2Eq, Xi, TrDen1

#Define molecule
if len(sys.argv)<2:
  print("MISSING MOLECULE NAME")
  exit()
else:
  molecule=sys.argv[1]
scratch = "/Users/marco/scratch"
#Clean pervious outputs
os.system(f"rm {molecule}.txt")
start0=time.time()
tot_mem, avlb_mem = mem_check()
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Total Memory: {tot_mem:.2f}GB, Available Memory: {avlb_mem:.2f}GB \n")

# Retrieve various quantities
O, V, NB, scfE, MOCoef, ipbc, k_weights = getFort(molecule)
Fock = getFock(molecule,O,V,NB,ipbc,"MO",False,MOCoef)
#O, V, NB, scfE, Fock, MOCoef, ipbc, k_weights, Core=getFort(molecule)
tot_mem, avlb_mem = mem_check()
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Read MO Coeff and Fock Matrix, Time: {time.time()-start0:.2f}s, AvlMem: {avlb_mem:.2f}GB \n")
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
# np.save(f"{molecule}_txts/AOInt",AOInt)
# AOInt2 = np.load(f"{molecule}_txts/AOInt.npy")
tot_mem, avlb_mem = mem_check()
print(f"AOInt-3 {AOInt.shape} {np.size(AOInt)}")

with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Read AO 2ERI, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB \n")
#Change to spin orbital form
start=time.time()
IJKL,ABCD,IABC,IJAB,IJKA,IABJ = conMO(molecule,scratch,O,V,NB,ipbc,MOCoef,AOInt)
tot_mem, avlb_mem = mem_check()
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"2ERI AO->MO, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB \n")

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
tot_mem, avlb_mem = mem_check()
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Compute energy denominators, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
  
##########################################################################  
# CCSD Energy and Amplitudes
##########################################################################
start=time.time()
# Initialize T1 and T2
t1 = np.zeros((O2k,V2k),dtype=Fock.dtype)
# t2 = np.zeros((O2, O2, V2, V2))
t2 = np.conjugate(IJAB)/D2.real
EMP2 = 0.25*np.einsum('ijab,ijab',IJAB,t2,optimize=True)/NkpC
tot_mem, avlb_mem = mem_check()
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"T guess, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
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
tau = []
W_efam = []
W_iemn = []
W_mbej = []
W_mnij = []
W_abef = []
F_ae = []
F_mi = []
F_me = []
# tau = np.zeros((O2k,O2k,V2k,V2k),dtype=Fock.dtype)
# W_efam = np.zeros((V2k,V2k,V2k,O2k),dtype=Fock.dtype)
# W_iemn = np.zeros((O2k,V2k,O2k,O2k),dtype=Fock.dtype)
# W_mbej = np.zeros((O2k,V2k,V2k,O2k),dtype=Fock.dtype)
# W_mnij = np.zeros((O2k,O2k,O2k,O2k),dtype=Fock.dtype)
# W_abef = np.zeros((V2k,V2k,V2k,V2k),dtype=Fock.dtype)
# F_ae = np.zeros((V2k,V2k),dtype=Fock.dtype)
# F_mi = np.zeros((O2k,O2k),dtype=Fock.dtype)
# F_me = np.zeros((O2k,V2k),dtype=Fock.dtype)
t1, t2 = AmpIt("T",molecule,scratch,Ok,Vk,Nkp,MaxIt,ThrE,ThrA,scfE,Fock,
               IJKL,ABCD,IABC,IJAB,IABJ,IJKA,tau,W_efam,W_iemn,W_mbej,
               W_mnij,W_abef,F_ae,F_mi,F_me,D1,D2,D1,D2,t1,t2,t1,t2,t1,
               t2,ipbc)

##########################################################################  
# Compute constant intermediates
##########################################################################  
start=time.time()
tau_tilde = tau_tildeEq(1, Nkp, t1, t2)
tau = tauEq(1, Nkp, t1, t2)
F_ae,F_mi,F_me,W_mnij,W_abef,W_mbej = T_interm(1,Ok,Vk,Nkp,Fock,t1,t2,IJKL,
                                               ABCD,IABC,IJAB,IABJ,IJKA,
                                               tau_tilde,tau)
#del ABCD
if(f"{scratch}/{molecule}-ABCD.npy"): 
  os.system(f"mv {scratch}/{molecule}-ABCD.npy {scratch}/{molecule}-Wabef.npy")
else:
  W_abef = ABCD
# fae_prod = np.einsum('ia,ia->',np.conjugate(F_ae),F_ae,optimize=True)/Nkp 
# fmi_prod = np.einsum('ia,ia->',np.conjugate(F_mi),F_mi,optimize=True)/Nkp
# fme_prod = np.einsum('ia,ia->',np.conjugate(F_me),F_me,optimize=True)/Nkp
# wabef_prod = np.einsum('ijab,ijab->',np.conjugate(W_abef),W_abef,optimize=True)/(Nkp*Nkp*Nkp)
# wmbej_prod = np.einsum('ijab,ijab->',np.conjugate(W_mbej),W_mbej,optimize=True)/(Nkp*Nkp*Nkp)
# wmnij_prod = np.einsum('ijab,ijab->',np.conjugate(W_mnij),W_mnij,optimize=True)/(Nkp*Nkp*Nkp)
# with open(f"{molecule}.txt","a") as writer:
#   writer.write(f"Products before: Fae Fmi Fme Wabef Wmbej Wmnij\n {fae_prod.real} {fmi_prod.real} {fme_prod.real} {wabef_prod.real} {wmbej_prod.real} {wmnij_prod.real}\n")
F_ae,F_mi,W_abef,W_mbej,W_efam,W_iemn = Const_Interm(1,molecule,scratch,Nkp,
                                                     t1,t2,tau,IJAB,
                                                     IABJ,IJKA,IABC,F_ae,
                                                     F_mi,F_me,W_mnij,
                                                     W_abef,W_mbej)
tot_mem, avlb_mem = mem_check()
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Compute constant intermediates, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
# fae_prod = np.einsum('ia,ia->',np.conjugate(F_ae),F_ae,optimize=True)/Nkp 
# fmi_prod = np.einsum('ia,ia->',np.conjugate(F_mi),F_mi,optimize=True)/Nkp
# wabef_prod = np.einsum('ijab,ijab->',np.conjugate(W_abef),W_abef,optimize=True)/(Nkp*Nkp*Nkp)
# wmbej_prod = np.einsum('ijab,ijab->',np.conjugate(W_mbej),W_mbej,optimize=True)/(Nkp*Nkp*Nkp)
# wefam_prod = np.einsum('ijab,ijab->',np.conjugate(W_efam),W_efam,optimize=True)/(Nkp*Nkp*Nkp)
# wiemn_prod = np.einsum('ijab,ijab->',np.conjugate(W_iemn),W_iemn,optimize=True)/(Nkp*Nkp*Nkp)
# with open(f"{molecule}.txt","a") as writer:
#   writer.write(f"Compute constant intermediates, Time: {time.time()-start:.2f}s\n")
#   writer.write(f"Products: Fae Fmi Wabef Wmbej Wefam Wiemn\n {fae_prod.real} {fmi_prod.real} {wabef_prod.real} {wmbej_prod.real} {wefam_prod.real} {wiemn_prod.real}\n")
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
with open(f"{molecule}.txt","a") as writer:
  writer.write("****************************************************\n")
  writer.write("*        SOLVING CCSD Lambda AMPLITUDE EQS.        *\n")
  writer.write("****************************************************\n")
l1, l2 = AmpIt("L",molecule,scratch,Ok,Vk,Nkp,MaxIt,ThrE,ThrA,scfE,Fock,
               IJKL,W_abef,IABC,IJAB,IABJ,IJKA,tau,W_efam,W_iemn,W_mbej,
               W_mnij,W_abef,F_ae,F_mi,F_me,D1,D2,D1,D2,t1,t2,l1,l2,t1,
               t2,ipbc)

##########################################################################  
# CCSD LR equations
##########################################################################
#
# NPert = number of perturbations (3 for dipoles and 6 for quadrupoles)
# WPert = frequency of perturbation
# if WPErt != 0, there two sets of amplitudes per perturbation Tx(+w) and Tx(-w)
# Use same intermediates as in Lambda equations
start=time.time()
with open(f"{molecule}.txt","a") as writer:
  writer.write("****************************************************\n")
  writer.write("*           COMPUTING CCSD LR FUNCTION             *\n")
  writer.write("****************************************************\n")
PertType = "DipE"
NP, X_ij, X_ia, X_ab = getPert(O,V,NB,ipbc,MOCoef,Fock,PertType,molecule)
tot_mem, avlb_mem = mem_check()
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Read perturbation integrals, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
xij_prod = np.einsum('ia,ia->',np.conjugate(X_ij[0,:,:]),X_ij[0,:,:],optimize=True)/Nkp 
xia_prod = np.einsum('ia,ia->',np.conjugate(X_ia[0,:,:]),X_ia[0,:,:],optimize=True)/Nkp 
xab_prod = np.einsum('ia,ia->',np.conjugate(X_ab[0,:,:]),X_ab[0,:,:],optimize=True)/Nkp 
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Compute constant intermediates, Time: {time.time()-start:.2f}s\n")
  writer.write(f"Products Pert-x: {xij_prod.real} {xia_prod.real} {xab_prod.real}\n")
xij_prod = np.einsum('ia,ia->',np.conjugate(X_ij[1,:,:]),X_ij[1,:,:],optimize=True)/Nkp 
xia_prod = np.einsum('ia,ia->',np.conjugate(X_ia[1,:,:]),X_ia[1,:,:],optimize=True)/Nkp 
xab_prod = np.einsum('ia,ia->',np.conjugate(X_ab[1,:,:]),X_ab[1,:,:],optimize=True)/Nkp 
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Compute constant intermediates, Time: {time.time()-start:.2f}s\n")
  writer.write(f"Products Pert-y: {xij_prod.real} {xia_prod.real} {xab_prod.real}\n")
xij_prod = np.einsum('ia,ia->',np.conjugate(X_ij[2,:,:]),X_ij[2,:,:],optimize=True)/Nkp 
xia_prod = np.einsum('ia,ia->',np.conjugate(X_ia[2,:,:]),X_ia[2,:,:],optimize=True)/Nkp 
xab_prod = np.einsum('ia,ia->',np.conjugate(X_ab[2,:,:]),X_ab[2,:,:],optimize=True)/Nkp 
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Compute constant intermediates, Time: {time.time()-start:.2f}s\n")
  writer.write(f"Products Pert-z: {xij_prod.real} {xia_prod.real} {xab_prod.real}\n")
#exit()
# For now, hardwire frequency of 300 nm
Wlist = []
#Wlist.append(0.0)
Wlist.append(0.15187784178412805)
tensor = np.zeros((len(Wlist), NP, NP),dtype=Fock.dtype)
for iw in range(len(Wlist)):
  # Loop over frequencies    
  W = Wlist[iw]
  NW = 2
  tx1 = np.zeros((NP,2,O2k,V2k),dtype=Fock.dtype)
  tx2 = np.zeros((NP,2,O2k,O2k,V2k,V2k),dtype=Fock.dtype)
  MaxX = np.zeros((NP))
  if (W==0): NW = 1 
#  for ip in range(2):
  for ip in range(NP):
    # Loop over number of non-zero pertubations
    MaxIJr = np.max(abs(X_ij[ip,:,:].real))
    MaxIJi = np.max(abs(X_ij[ip,:,:].imag))
    MaxIAr = np.max(abs(X_ia[ip,:,:].real))
    MaxIAi = np.max(abs(X_ia[ip,:,:].imag))
    MaxABr = np.max(abs(X_ab[ip,:,:].real))
    MaxABi = np.max(abs(X_ab[ip,:,:].imag))
    MaxX[ip] = max(MaxIJr,MaxIJi,MaxIAr,MaxIAi,MaxABr,MaxABi)
    if(MaxX[ip] > 1e-15):
      start=time.time()
      rhs1, rhs2, rhs1a, rhs1b, rhs1c = pert_rhs(1, Nkp, O2k, V2k, t1, t2, X_ij[ip,:,:], X_ia[ip,:,:], X_ab[ip,:,:])
      tot_mem, avlb_mem = mem_check()
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f"Form right hand side, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
      rhs1_prod = np.einsum('ia,ia->',np.conjugate(rhs1),rhs1,optimize=True)/Nkp 
      rhs2_prod = np.einsum('ijab,ijab->',np.conjugate(rhs2),rhs2,optimize=True)/(Nkp*Nkp*Nkp)
      rhs1a_prod = np.einsum('ia,ia->',np.conjugate(rhs1a),rhs1a,optimize=True)/Nkp 
      rhs1b_prod = np.einsum('ia,ia->',np.conjugate(rhs1b),rhs1b,optimize=True)/Nkp 
      rhs1c_prod = np.einsum('ia,ia->',np.conjugate(rhs1c),rhs1c,optimize=True)/Nkp
      rhs1d = rhs1a - rhs1b
      rhs1e = rhs1a - rhs1c
      rhs1d_prod = np.einsum('ia,ia->',np.conjugate(rhs1d),rhs1d,optimize=True)/Nkp 
      rhs1e_prod = np.einsum('ia,ia->',np.conjugate(rhs1e),rhs1e,optimize=True)/Nkp
    
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f"Compute constant intermediates, Time: {time.time()-start:.2f}s\n")
        writer.write(f"Products Rhs: {rhs1_prod.real} {rhs2_prod.real} {rhs1a_prod.real} {rhs1b_prod.real}  {rhs1c_prod.real} {rhs1d_prod.real}  {rhs1e_prod.real}\n")
      print(f"rhs1a rhs1b\n")
      #     if(ipbc):
      #       rhs1a = rhs1a.reshape(Nkp,O2,Nkp,V2)
      #       rhs1b = rhs1b.reshape(Nkp,O2,Nkp,V2)
      #       rhs1c = rhs1c.reshape(Nkp,O2,Nkp,V2)
      #       sum1a = 0
      #       sum1b = 0
      #       sum1c = 0
      #       for k in range(Nkp):
      #         for h in range(Nkp):
      #           for i in range(O2):
      #             for a in range (V2):
      #               if(abs(rhs1a[k,i,h,a].real) > 1e-12 or abs(rhs1b[k,i,h,a].real) > 1e-12 or abs(rhs1a[k,i,h,a].imag) > 1e-12 or abs(rhs1b[k,i,h,a].imag) > 1e-12 ):
      #                 print(f"{k+1},{i+1},{h+1},{a+1} {rhs1a[k,i,h,a]:.6e} {rhs1b[k,i,h,a]:.6e} {rhs1c[k,i,h,a]:.6e} ")
      #               if(k == h and i ==1 and a == 1):
      #                 sum1a += rhs1a[k,i,h,a]
      #                 sum1b += rhs1b[k,i,h,a]
      #                 sum1c += rhs1c[k,i,h,a]
      #         print(f"sum1a={sum1a/Nkp} sum1b={sum1b/Nkp} sum1c={sum1c/Nkp} ")
      #     else:
      #       for i in range(O2k):
      #         for a in range (V2k):
      #           if(abs(rhs1a[i,a]) > 1e-12 or abs(rhs1b[i,a]) > 1e-12 ):
      #             print(f"{i+1},{a+1} {rhs1a[i,a]:.6e} {rhs1b[i,a]:.6e} {rhs1c[i,a]:.6e} ")
      # #    exit()
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f"\n Perturbation {PertType}-{ip+1}\n")
      #    for ipmw in range(1):
      for ipmw in range(NW):
        # Loop over +/-omega
        PMW = W
        if (ipmw==1): PMW = -W 
        with open(f"{molecule}.txt","a") as writer:
          writer.write(f"\n Frequency {PMW:+f}\n")
        # Reset denominators including frequency term and initialize amplitudes
        D1, D2 =  denom(1, O2, V2, kp, Fock, PMW)
        # tx1[ip,ipmw,:,:] = np.copy(t1)
        tx1[ip,ipmw,:,:] -= rhs1/D1.real
        #      tx1[ip,ipmw,:,:] -= rhs1
        # tx2[ip,ipmw,:,:,:,:] = np.copy(t2)
        tx2[ip,ipmw,:,:,:,:] -= rhs2/D2.real
        t1_prod = np.einsum('ia,ia->',np.conjugate(t1),t1,optimize=True)/Nkp 
        rhs1_prod = np.einsum('ia,ia->',np.conjugate(rhs1),tx1[ip,ipmw,:,:],optimize=True)/Nkp 
        rhs2_prod = np.einsum('ijab,ijab->',np.conjugate(rhs2),tx2[ip,ipmw,:,:,:,:],optimize=True)/(Nkp*Nkp*Nkp)
        with open(f"{molecule}.txt","a") as writer:
          writer.write(f"Compute constant intermediates, Time: {time.time()-start:.2f}s\n")
          writer.write(f"Products Tx: {t1_prod.real} {rhs1_prod.real} {rhs2_prod.real}\n")
    
        # Amplitudes loop
        tx1[ip,ipmw,:,:], tx2[ip,ipmw,:,:,:,:] = AmpIt("Tx",molecule,scratch,Ok,Vk,Nkp,
                                                       MaxIt,ThrE,ThrA,scfE,Fock,IJKL,
                                                       W_abef,IABC,IJAB,IABJ,IJKA,tau,
                                                       W_efam,W_iemn,W_mbej,W_mnij,W_abef,
                                                       F_ae,F_mi,F_me,rhs1,rhs2,D1,D2,t1,t2,
                                                       l1,l2,tx1[ip,ipmw,:,:],
                                                       tx2[ip,ipmw,:,:,:,:],ipbc)
  #
  # Now that we have all the Tx amplitudes for this W, we can compute
  # the corresponding Xi amplitudes and contract with all other Tx
  # amplitudes, and the transition 1PDM-like rho1 and contract with
  # the perturbation integrals
  #
  # Reset denominators
  D1, D2 =  denom(1, O2, V2, kp, Fock, 0)
#  for ip in range(2):
  for ip in range(NP):
    if(MaxX[ip] > 1e-15):
      # Evaluate Xi amplitudes 
      start=time.time()
      Xi1, Xi2 = Xi(1,Nkp,tx1[ip,0,:,:],tx2[ip,0,:,:,:,:],l1,l2,t1,IABC,IJAB,IJKA,F_ae,F_mi,
                    F_me,W_mbej,D2)
      tot_mem, avlb_mem = mem_check()
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f"Form Xi terms, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
      Xi1_prod = np.einsum('ia,ia->',np.conjugate(Xi1),Xi1,optimize=True)/Nkp 
      Xi2_prod = np.einsum('ijab,ijab->',np.conjugate(Xi2),Xi2,optimize=True)/(Nkp*Nkp*Nkp)
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f"Xi Products: {Xi1_prod.real} {Xi2_prod.real}\n")
      #    for ipa in range(2):
      for ipa in range(NP):
        # Contract Xi(ip) with Tx(ipa)
        tensor[iw,ip,ipa] -= np.einsum('ia,ia->',Xi1,np.conjugate(tx1[ipa,1,:,:]),optimize=True)/Nkp 
        tensor[iw,ip,ipa] -= 0.25*np.einsum('ijab,ijab->',Xi2,np.conjugate(tx2[ipa,1,:,:,:,:]),optimize=True)/NkpC
        ten_prod1 = -np.einsum('ia,ia->',Xi1,np.conjugate(tx1[ipa,1,:,:]),optimize=True)/Nkp
        ten_prod2 = -0.25*np.einsum('ijab,ijab->',Xi2,np.conjugate(tx2[ipa,1,:,:,:,:]),optimize=True)/(Nkp*Nkp*Nkp)
        with open(f"{molecule}.txt","a") as writer:
          writer.write(f"Tensor-Xi Products: {ip+1} {ipa+1} {ten_prod1} {ten_prod2} {ten_prod1+ten_prod2} \n")
      del Xi1, Xi2
      for ipmw in range(NW):
        # Loop over +/-omega
        # Evaluate 1PDM
        start=time.time()
        rho1 = TrDen1(1,O2k,NB2k,Nkp,tx1[ip,ipmw,:,:],tx2[ip,ipmw,:,:,:,:],l1,l2,t1,t2)
        tot_mem, avlb_mem = mem_check()
        with open(f"{molecule}.txt","a") as writer:
          writer.write(f"Form Rho, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
        rho1ij_prod = np.einsum('ij,ij->',np.conjugate(rho1[:O2k,:O2k]),rho1[:O2k,:O2k],optimize=True)/Nkp 
        rho1ia_prod = np.einsum('ij,ij->',np.conjugate(rho1[:O2k,O2k:]),rho1[:O2k,O2k:],optimize=True)/Nkp 
        rho1ab_prod = np.einsum('ij,ij->',np.conjugate(rho1[O2k:,O2k:]),rho1[O2k:,O2k:],optimize=True)/Nkp 
        tp1 = np.einsum('ij,ij->',X_ij[0,:,:],rho1[:O2k,:O2k],optimize=True)/Nkp 
        tp2 = np.einsum('ia,ia->',X_ia[0,:,:],rho1[:O2k,O2k:],optimize=True)/Nkp   
        tp3 = np.einsum('ab,ab->',X_ab[0,:,:],rho1[O2k:,O2k:],optimize=True)/Nkp   
        tp4 = np.einsum('ij,ij->',np.conjugate(X_ij[0,:,:]),rho1[:O2k,:O2k],optimize=True)/Nkp 
        tp5 = np.einsum('ia,ia->',np.conjugate(X_ia[0,:,:]),rho1[:O2k,O2k:],optimize=True)/Nkp   
        tp6 = np.einsum('ab,ab->',np.conjugate(X_ab[0,:,:]),rho1[O2k:,O2k:],optimize=True)/Nkp   
        tp7 = np.einsum('ij,ij->',X_ij[0,:,:],np.conjugate(rho1[:O2k,:O2k]),optimize=True)/Nkp 
        tp8 = np.einsum('ia,ia->',X_ia[0,:,:],np.conjugate(rho1[:O2k,O2k:]),optimize=True)/Nkp   
        tp9 = np.einsum('ab,ab->',X_ab[0,:,:],np.conjugate(rho1[O2k:,O2k:]),optimize=True)/Nkp   
        with open(f"{molecule}.txt","a") as writer:
          writer.write(f"Compute constant intermediates, Time: {time.time()-start:.2f}s\n")
          writer.write(f"Products Rho: {rho1ij_prod.real} {rho1ia_prod.real} {rho1ab_prod.real}\n")
          writer.write(f"Products t1-3: {tp1.real} {tp2.real} {tp3.real} {(tp1+tp3).real} \n")
          writer.write(f"Products t4-6: {tp4.real} {tp5.real} {tp6.real} {(tp4+tp6).real} \n")
          writer.write(f"Products t7-9: {tp7.real} {tp8.real} {tp9.real} {(tp7+tp9).real} \n")
        #      exit()
        #      for ipa in range(2):
        for ipa in range(NP):
          # Contract 1PDM(ip) with Pert(ipa)
          tensor[iw,ip,ipa] += np.einsum('ij,ij->',np.conjugate(X_ij[ipa,:,:]),rho1[:O2k,:O2k],optimize=True)/Nkp 
          tensor[iw,ip,ipa] += np.einsum('ia,ia->',np.conjugate(X_ia[ipa,:,:]),rho1[:O2k,O2k:],optimize=True)/Nkp   
          tensor[iw,ip,ipa] += np.einsum('ab,ab->',np.conjugate(X_ab[ipa,:,:]),rho1[O2k:,O2k:],optimize=True)/Nkp   
          ten_prod1 = np.einsum('ij,ij->',np.conjugate(X_ij[ipa,:,:]),rho1[:O2k,:O2k],optimize=True)/Nkp 
          ten_prod2 = np.einsum('ia,ia->',np.conjugate(X_ia[ipa,:,:]),rho1[:O2k,O2k:],optimize=True)/Nkp   
          ten_prod3 = np.einsum('ab,ab->',np.conjugate(X_ab[ipa,:,:]),rho1[O2k:,O2k:],optimize=True)/Nkp   
          with open(f"{molecule}.txt","a") as writer:
            writer.write(f"Tensor-Rho Products: {ip+1} {ipa+1} {ten_prod1} {ten_prod2}  {ten_prod3} {ten_prod1+ten_prod2+ten_prod3}\n")
  # Print the tensor for frequency W
  with open(f"{molecule}.txt","a") as writer:
    writer.write(f"\n DipE(LG)-DipE(LG) Polarizability in a.u. for W = {W:.6f} a.u.\n")
  for ip in range(NP):
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f" {ip+1} {tensor[iw,ip,0].real} {tensor[iw,ip,1].real} {tensor[iw,ip,2].real}\n")
      # writer.write(f" {ip+1} {tensor[iw,ip,0]:+.6f} {tensor[iw,ip,1]:+.6f} {tensor[iw,ip,2]:+.6f}\n")
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Total Calculation Time: {time.time()-start0:.2f}s\n")
# Delete scratch files
os.system(f"rm {scratch}/*.npy")
               
