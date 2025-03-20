import numpy as np
import os
import sys
import re
import time
from read import getFort

##########################################################################
# Compute E and intermediates for CCSD equations
##########################################################################

#Initialize arrays
#Define molecule
if len(sys.argv)==2:
  molecule=sys.argv[1]
else:
  print("MISSING MOLECULE NAME AS FIRST ARG")
  exit()
log=f"{molecule}.log"
O, V, NB, scfE, Fock, Coeff=getFort(molecule, log)

##########################################################################
# Compute energy denominators
##########################################################################
def denom(T, O2, V2, Fock, W):
  if T==1:
    D1 = np.zeros((O2, V2))
    D2 = np.zeros((O2, O2, V2, V2))
    # D1 denominator
    for a in range(V2):
      for i in range(O2):
        D1[i,a]=Fock[i,i]-Fock[a+O2,a+O2] - W
    # D1 denominator
    for i in range(O2):
      for j in range(O2):
        den = Fock[i,i]+Fock[j,j]
        for a in range(V2):
          for b in range(V2):
            D2[i,j,a,b] = den-Fock[a+O2,a+O2]-Fock[b+O2,b+O2] - W
  return D1, D2

##########################################################################
# Wrapper routine for iterative solution of CCSD amplitude equations
##########################################################################
def AmpIt(AmpType,molecule,O,V,MaxIt,ThrE,ThrA,scfE,Fock,IJKL,ABCD,IABC,IJAB,IAJB,IJKA,tau,W_efam,W_iemn,W_mbej,W_mnij,W_abef,F_ae,F_mi,F_me,rhs1,rhs2,D1,D2,t1,t2,l1,l2,tx1,tx2):
  E_Corr2 = 0
  N = 0
  not_conver = True
  start0=time.time()
  while not_conver and N< MaxIt:
    start = time.time()
    N +=1
    E_Corr1 = E_Corr2
    if(AmpType == "T"):
      # Ground state T amplitudes
      # Calculate intermediates
      tau_tilde = tau_tildeEq(1, t1, t2)
      tau = tauEq(1, t1, t2)
      F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej = intermediateEqs(1, O, V, Fock, t1, t2, IJKL, ABCD, IABC, IJAB, IAJB, IJKA, tau_tilde, tau)
      # Amplitude iteration
      t1_f = t1Eq(1,O,Fock,t1,t2,IABC,IJKA,IAJB,F_ae,F_mi,F_me,D1)
      t2_f = t2Eq(1,t1,t2,IABC,IJAB,IJKA,IAJB,tau,F_ae,F_mi,F_me,W_mnij,W_abef,W_mbej,D2)
      tau = tauEq(1, t1_f, t2_f)
      # Evaluate convergence
      not_conver,E_Corr2,t1,t2 = AmpConv(AmpType,O,t1,t2,t1_f,t2_f,tau,Fock,D1,IJAB,ThrE,ThrA,E_Corr1)
      del t1_f, t2_f
      textA = f"Iter. {N}: E_corr(CCSD) {E_Corr2:.10f}, E(CCSD): {scfE+E_Corr1:.10f}"
      a1 = t1
      a2 = t2
    elif (AmpType == "L"):
      # Ground state Lambda (or Z) amplitudes
      # Calculate intermediates
      G_ae, G_mi = L_intermediate(1,t2,l2)
      # Amplitude iteration
      l1_f = l1Eq(1,t1,l1,l2,IJAB,IABC,IJKA,W_efam,W_iemn,W_mbej,F_ae,F_mi,F_me,G_ae,G_mi,D1)
      l2_f = l2Eq(1,t1,l1,l2,IABC,IJAB,IJKA,F_ae,F_mi,F_me,G_ae,G_mi,W_mnij,W_abef,W_mbej,D2)
      tau_tilde = tauEq(1, l1_f, l2_f)
      E_Corr2 = E_CCSD(O, Fock, l1_f, IJAB, tau_tilde)
      # Evaluate convergence
      not_conver, E_Corr2, l1, l2 = AmpConv(AmpType,O,l1,l2,l1_f,l2_f,tau_tilde,Fock,D1,IJAB,ThrE,ThrA,E_Corr1)
      del l1_f, l2_f, G_ae, G_mi 
      textA = f"Iter. {N}: DE(L-CCSD) {E_Corr2:.10f}, E(L-CCSD): {scfE+E_Corr1:.10f}"
      a1 = l1
      a2 = l2
    elif (AmpType == "Tx"):
      # Perturbed T amplitudes
      # Calculate intermediates
      G_ae, G_mi = L_intermediate(1,IJAB,tx2)
      # Amplitude iteration
      tx1_f = tx1Eq(1,tx1,tx2,t1,IABC,IJKA,W_mbej,F_ae,F_mi,F_me,G_ae,G_mi,D1)
      tx1_f -= rhs1/D1
      tx2_f = tx2Eq(1,tx1,tx2,t1,t2,IABC,IJAB,IJKA,F_ae,F_mi,F_me,G_ae,G_mi,W_mnij,W_abef,W_efam,W_iemn,W_mbej,D2)
      tx2_f -= rhs2/D2
      # Evaluate convergence
      not_conver, E_Corr2, tx1, tx2 = AmpConv(AmpType,O,tx1,tx2,tx1_f,tx2_f,tau,Fock,rhs1,rhs2,ThrE,ThrA,E_Corr1)
      del tx1_f, tx2_f, G_ae, G_mi 
      textA = f"Iter. {N}: DE(Tx-CCSD) {E_Corr2:.10f}, E(Tx-CCSD): {scfE+E_Corr1:.10f}"
      a1 = tx1
      a2 = tx2
    else :
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f"Amplitude type {AmpType} is not implemented. ")
      exit()
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"{textA}, Time: {time.time()-start:.2f}s\n")
  if(not_conver):
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"{AmpType} amplitude equations convergence failure\n")
    exit()
  else:
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"{AmpType} amplitude equations converged in {time.time()-start0:.2f}s\n")
  return a1, a2

##########################################################################
# Evaluate convergence criteria and update amplitudes for amplitude
# iterations
##########################################################################
def AmpConv(AmpType,O,a1,a2,a1_f,a2_f,tau,Fock,I1Int,I2Int,ThrE,ThrA,E_Corr1):
  DiffA1 = abs(np.max(a1_f-a1))
  DiffA2 = abs(np.max(a2_f-a2))
  a1RMSE = (np.sum(((a1_f-a1)**(2))/(np.size(a1))))**(1/2)
  a2RMSE = (np.sum(((a2_f-a2)**(2))/(np.size(a2))))**(1/2)
  a1 = np.copy(a1_f)
  a2 = np.copy(a2_f)
  if(AmpType == "T" or AmpType == "L"):
    # Here I2int should be the IJAB integrals
    E_Corr2 = E_CCSD(O, Fock, a1, I2Int, tau)
  elif (AmpType == "Tx"):
    # Here I1/2int should be the right hand side perturbations
    E_Corr2 = -0.25*np.einsum('ijab,ijab->',I2Int,a2,optimize=True)
    E_Corr2 -= np.einsum('ia,ia->',I1Int,a1,optimize=True)
  DiffE = abs(E_Corr2-E_Corr1)
  not_conver = DiffE> ThrE or DiffA1> ThrA or DiffA2> ThrA or a1RMSE> ThrA or a2RMSE> ThrA
  E_Corr1 = E_Corr2
  return not_conver, E_Corr2, a1, a2

##########################################################################
# tau_tilde intermediate for CCSD T equations
##########################################################################
def tau_tildeEq(T, t1, t2):
  if T==1:
    tau_tilde = np.copy(t2) 
    tau_tilde += 0.5*np.einsum('ia,jb->ijab',t1,t1,optimize=True) 
    tau_tilde -= 0.5*np.einsum('ib,ja->ijab',t1,t1,optimize=True)
  return tau_tilde

##########################################################################
# tau intermediate for CCSD T equations
##########################################################################
def tauEq(T, t1, t2):
  if T==1:
    tau = np.copy(t2)
    tau += np.einsum('ia,jb->ijab',t1,t1,optimize=True)
    tau -= np.einsum('ib,ja->ijab',t1,t1,optimize=True)
  return tau

##########################################################################
# F and W intermediates for CCSD T equations
##########################################################################
def intermediateEqs(T, O, V, Fock, t1, t2, IJKL, ABCD, IABC, IJAB, IAJB, IJKA, tau_tilde, tau):
  O2=2*O
  V2=2*V
  if T==1:
    # F_ae
    F_ae = np.zeros((V2, V2))
    F_ae += (1 - np.eye(V2)) * Fock[O2:, O2:] #Add flag, function to set diagonal elements to zero
    F_ae = -0.5 * np.einsum('me,ma->ae', Fock[:O2, O2:], t1, optimize=True)
    F_ae += np.einsum('mf,mafe->ae', t1, IABC, optimize=True)
    F_ae -= 0.5 * np.einsum('mnaf,mnef->ae', tau_tilde, IJAB, optimize=True)    
    # F_mi
    F_mi=np.zeros((O2, O2))
    F_mi += (1 - np.eye(O2)) * Fock[:O2, :O2]
    F_mi += 0.5 * np.einsum('ie,me->mi', t1, Fock[:O2, O2:], optimize=True)
    F_mi += np.einsum('ne,mnie->mi', t1, IJKA, optimize=True)
    F_mi += 0.5 * np.einsum('inef,mnef->mi', tau_tilde, IJAB, optimize=True)
    # F_me
    F_me = np.zeros((O2, V2))
    F_me = np.copy(Fock[:O2, O2:])
    F_me += np.einsum('nf,mnef->me', t1, IJAB, optimize=True)
    # W_mnij
    W_mnij = np.copy(IJKL)
    W_mnij += np.einsum('je,mnie->mnij', t1, IJKA, optimize=True)
    W_mnij -= np.einsum('ie,mnje->mnij', t1, IJKA, optimize=True)
    W_mnij += 0.5 * np.einsum('mnef,ijef->mnij', IJAB, tau, optimize=True)
    # W_abef
    W_abef = np.copy(ABCD)
    W_abef += np.einsum('mb,maef->abef',t1,IABC,optimize=True)
    W_abef -= np.einsum('ma,mbef->abef',t1,IABC,optimize=True)
    # W_mbej
    W_mbej = np.copy(-np.transpose(IAJB, axes=(0,1,3,2)))
    W_mbej += np.einsum('jf,mbef->mbej', t1, IABC, optimize=True)
    W_mbej += np.einsum('nb,mnje->mbej', t1, IJKA, optimize=True)
    W_mbej -= 0.5 * np.einsum('jnfb,mnef->mbej', t2, IJAB, optimize=True)
    W_mbej -= np.einsum('jf,nb,mnef->mbej', t1, t1, IJAB, optimize=True)
  return F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej

#########################################################################
# CCSD T1 amplitude equation
#########################################################################
def t1Eq(T, O, Fock, t1, t2, IABC, IJKA, IAJB, F_ae, F_mi, F_me, D1):
  O2=2*O
  if T==1:
    t1_f = np.copy(Fock[:O2, O2:])  
    t1_f += np.einsum('ie,ae->ia', t1, F_ae, optimize=True)
    t1_f -= np.einsum('ma,mi->ia', t1, F_mi, optimize=True)
    t1_f += np.einsum('imae,me->ia', t2, F_me, optimize=True)
    t1_f -= 0.5 * np.einsum('imef,maef->ia', t2, IABC,optimize=True)
    t1_f += 0.5 * np.einsum('mnae,nmie->ia', t2, IJKA,optimize=True)
    t1_f -= np.einsum('nf,naif->ia', t1, IAJB,optimize=True)
    t1_f /= D1
  return t1_f

#########################################################################
# CCSD T2 amplitude equation
#########################################################################
def t2Eq(T, t1, t2, IABC, IJAB, IJKA, IAJB, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D2):
  if T==1:
    # P(ab) terms
    X1 = F_ae - 0.5*np.einsum('mb,me->be',t1,F_me,optimize=True)
    X2 = np.einsum('ijae,be->ijab',t2,X1,optimize=True)
    X2 -= np.einsum('ma,ijmb->ijab',t1,IJKA,optimize=True)
    t2_f = IJAB + X2 - np.transpose(X2,axes=(0,1,3,2))
    del X1, X2
    # t2_f = np.copy(IJAB)
    # P(ij) terms
    X1 = F_mi + 0.5*np.einsum('je,me->mj',t1,F_me,optimize=True)
    X2 = -np.einsum('imab,mj->ijab',t2,X1,optimize=True)
    X2 -= np.einsum('ie,jeab->ijab',t1,IABC,optimize=True)
    t2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
    del X1, X2
    # P(ij,ab) terms
    X1 = np.einsum('ie,mbje->mbij',t1,IAJB,optimize=True)
    X2 = np.einsum('imae,mbej->ijab',t2,W_mbej,optimize=True)
    X2 += np.einsum('ma,mbij->ijab',t1,X1,optimize=True)
    t2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
    t2_f -= np.transpose(X2,axes=(0,1,3,2))
    t2_f += np.transpose(X2,axes=(1,0,3,2))
    del X1, X2
    # tau terms
    t2_f += 0.5*np.einsum('ijef,abef->ijab',tau,W_abef,optimize=True)
    t2_f += 0.5*np.einsum('mnab,mnij->ijab',tau,W_mnij,optimize=True)
    # Divide by energy denominator
    t2_f /= D2    
  return t2_f

#########################################################################
# CCSD energy
#########################################################################
def E_CCSD(O, Fock, t1, IJAB, tau):
  O2 = 2*O
  E_Corr2_1 = np.einsum('ia,ia->', t1, Fock[:O2, O2:])
  E_Corr2_2 = 0.25 * np.einsum('ijab,ijab->', tau, IJAB)
  E_Corr2 = E_Corr2_1 + E_Corr2_2
  return E_Corr2

#########################################################################
# Define constant intermediates for CCSD Lambda and response equations
#########################################################################
def L_intermediate_const(T, t1, t2, tau, IJAB, IAJB, IJKA, IABC, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej):
  if T==1:
    # Remember that the contraction for Lambda is over the opposite one or two indices (same for W_mnij)
    F_ae -= 0.5*np.einsum('ma,me->ae',t1,F_me,optimize=True)
    # The sign of this terms is wrong in the paper
    F_mi += 0.5*np.einsum('me,ie->mi',F_me,t1,optimize=True)
    # Here we are forming the tilde-W_abef intermediate as in the
    # paper, at the cost of doing a o2v4 contraction once. The
    # tilde-W_nmij is already as in the paper, as we already doubled
    # the IJAB contribution for the t2 equations.
    W_abef += 0.5*np.einsum('mnab,mnef->abef',tau,IJAB,optimize=True)
    W_mbej += 0.5*np.einsum('nmfe,jnbf->mbej',IJAB,t2,optimize=True)
    # These intermediates are new
    W_efam = np.einsum('mnef,na->efam',t2,F_me,optimize=True)
    W_efam -= np.transpose(IABC,axes=(2,3,1,0)) 
    W_efam += np.einsum('efag,mg->efam',W_abef,t1,optimize=True)
    W_efam -= 0.5*np.einsum('noef,noma->efam',tau,IJKA,optimize=True)
    W_iemn = -np.einsum('mnef,if->iemn',t2,F_me,optimize=True)
    W_iemn += np.transpose(IJKA,axes=(2,3,0,1)) 
    W_iemn -= np.einsum('iomn,oe->iemn',W_mnij,t1,optimize=True)
    W_iemn += 0.5*np.einsum('iefg,mnfg->iemn',IABC,tau,optimize=True)
    # Create a temp intermediates
    WW_mbej = -np.einsum('mnef,njbf->mbej',IJAB,t2,optimize=True) 
    WW_mbej -= np.transpose(IAJB,axes=(0,1,3,2)) 
    X1 = - np.einsum('ne,nfam->efam',t1,WW_mbej,optimize=True)
    X1 += np.einsum('nega,mnfg->efam',IABC,t2,optimize=True)
    X2 = X1 - np.transpose(X1,axes=(1,0,2,3))
    W_efam += X2
    del X1,X2
    X1 = np.einsum('mf,iefn->iemn',t1,WW_mbej,optimize=True)
    X1 += np.einsum('iomf,noef->iemn',IJKA,t2,optimize=True)
    X2 = X1 - np.transpose(X1,axes=(0,1,3,2))
    W_iemn += X2
    del X1,X2,WW_mbej
    # Done and return
  return W_efam, W_iemn

#########################################################################
# Define changing intermediates for CCSD Lambda equations
#########################################################################
def L_intermediate(T, t2, l2):
  if T==1:
    G_ae = -0.5*np.einsum('mnaf,mnef->ae',l2,t2,optimize=True)
    G_mi = 0.5*np.einsum('mnef,inef->mi',t2,l2,optimize=True)
  return G_ae, G_mi

#########################################################################
# CCSD Lambda1 amplitude equation
#########################################################################
def l1Eq(T, t1, l1, l2, IJAB, IABC, IJKA, W_efam, W_iemn, W_mbej, F_ae, F_mi, F_me, G_ae, G_mi, D1):
  if T==1:
    l1_f = np.copy(F_me)  
    l1_f += np.einsum('ie,ea->ia',l1,F_ae,optimize=True)
    l1_f -= np.einsum('im,ma->ia',F_mi,l1,optimize=True)
    l1_f += np.einsum('me,ieam->ia',l1,W_mbej,optimize=True)
    l1_f += 0.5*np.einsum('imef,efam->ia',l2,W_efam,optimize=True)
    l1_f -= 0.5*np.einsum('iemn,mnae->ia',W_iemn,l2,optimize=True)
    l1_f += np.einsum('ef,iefa->ia',G_ae,IABC,optimize=True)
    l1_f -= np.einsum('mn,mina->ia',G_mi,IJKA,optimize=True)
    X1 = np.einsum('mf,fe->me',t1,G_ae,optimize=True)
    X1 -= np.einsum('mn,ne->me',G_mi,t1,optimize=True)
    l1_f += np.einsum('me,imae->ia',X1,IJAB,optimize=True)
    del X1
    l1_f /= D1
  return l1_f

#########################################################################
# CCSD Lambda2 amplitude equation
#########################################################################
def l2Eq(T, t1, l1, l2, IABC, IJAB, IJKA, F_ae, F_mi, F_me, G_ae, G_mi, W_mnij, W_abef, W_mbej, D2):
  if T==1:
    l2_f = np.copy(IJAB)
    l2_f += 0.5*np.einsum('ijef,efab->ijab',l2,W_abef,optimize=True)
    l2_f += 0.5*np.einsum('ijmn,mnab->ijab',W_mnij,l2,optimize=True)
    # P(ab) terms
    X1 = G_ae - np.einsum('mb,me->be',l1,t1,optimize=True)
    X2 = np.einsum('ijae,be->ijab',IJAB,X1,optimize=True)
    X2 -= np.einsum('ma,ijmb->ijab',l1,IJKA,optimize=True)
    X2 += np.einsum('ijae,eb->ijab',l2,F_ae,optimize=True) 
    l2_f += X2 - np.transpose(X2,axes=(0,1,3,2))
    del X1, X2
    # P(ij) terms
    X1 = G_mi + np.einsum('me,je->mj',t1,l1,optimize=True)
    X2 = np.einsum('imab,mj->ijab',IJAB,X1,optimize=True)
    X2 += np.einsum('ie,jeab->ijab',l1,IABC,optimize=True)
    X2 += np.einsum('imab,jm->ijab',l2,F_mi,optimize=True) 
    l2_f += np.transpose(X2,axes=(1,0,2,3)) - X2 
    del X1, X2
    # P(ij,ab) terms
    X2 = np.einsum('imae,jebm->ijab',l2,W_mbej,optimize=True)
    X2 += np.einsum('ia,jb->ijab',l1,F_me,optimize=True)
    l2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
    l2_f -= np.transpose(X2,axes=(0,1,3,2))
    l2_f += np.transpose(X2,axes=(1,0,3,2))
    del X2
    # Divide by energy denominator
    l2_f /= D2    
  return l2_f

#########################################################################
# Form constant terms based on 1e perturbation X:
# <S|e^{-T}Xe^{T}|0> and <D|e^{-T}Xe^{T}|0>.
#########################################################################
def pert_rhs(T, t1, t2, X_ij, X_ia, X_ab):
  if T==1:
  # # X is supposed to be in MO basis and already divided in oo, ov, and vv blocks
  #   X_ij, X_ia, X_ab = getpert(Coeff,pert_type,i_pert)
    # Singles
    rhs1 = np.copy(X_ia) 
    rhs1 += np.einsum('kc,ikac->ia',X_ia,t2,optimize=True)
    rhs1 -= np.einsum('kc,ic,ka->ia',X_ia,t1,t1,optimize=True)
    rhs1 += np.einsum('ic,ac->ia',t1,X_ab,optimize=True)
    rhs1 -= np.einsum('ki,ka->ia',X_ij,t1,optimize=True)
    # Doubles
    # P(ij) terms: -P(ij) t(kjab)(X(ik)+X(kc)t(ic))
    X1 = np.copy(X_ij) + np.einsum('ic,kc->ik',t1,X_ia,optimize=True)
    X2 = -np.einsum('ik,kjab->ijab',X1,t2,optimize=True)
    rhs2 = X2 - np.transpose(X2,axes=(1,0,2,3))
    # P(ab) terms: P(ab) t(ijac)(X(cb)-X(kc)t(kb))
    X1 = np.copy(X_ab) - np.einsum('kc,kb->cb',X_ia,t1,optimize=True)
    X2 = np.einsum('ijac,cb->ijab',t2,X1,optimize=True)
    rhs2 += X2 - np.transpose(X2,axes=(0,1,3,2))
    del X1, X2
  return rhs1, rhs2

#########################################################################
# CCSD Tx1 (or EOM R1) amplitude equation
#########################################################################
def tx1Eq(T, tx1, tx2, t1, IABC, IJKA, W_mbej, F_ae, F_mi, F_me, G_ae, G_mi, D1):
  # Constant term needs to be added outside (as it's not in the EOM eqs.)
  # It requires getting G_ae, G_mi = L_intermediate(T, IJAB, tx2)
  if T==1:
    # tx1_f = np.copy(F_me) #this one needs to be checked! 
    tx1_f = np.einsum('ie,ae->ia',tx1,F_ae,optimize=True)
    tx1_f -= np.einsum('mi,ma->ia',F_mi,tx1,optimize=True)
    tx1_f += np.einsum('me,maei->ia',tx1,W_mbej,optimize=True)
    tx1_f += np.einsum('imae,me->ia', tx2, F_me, optimize=True)
    tx1_f -= 0.5 * np.einsum('imef,maef->ia', tx2, IABC,optimize=True)
    tx1_f += 0.5 * np.einsum('nmea,nmie->ia', tx2, IJKA,optimize=True)
    tx1_f += np.einsum('ib,ab->ia',t1,G_ae,optimize=True)
    tx1_f -= np.einsum('ji,ja->ia',G_mi,t1,optimize=True)
    tx1_f /= D1
  return tx1_f

#########################################################################
# CCSD Tx2 (or EOM R2) amplitude equation
#########################################################################
def tx2Eq(T, tx1, tx2, t1, t2, IABC, IJAB, IJKA, F_ae, F_mi, F_me, G_ae, G_mi, W_mnij, W_abef, W_efam, W_iemn, W_mbej, D2):
  # Constant term needs to be added outside (as it's not in the EOM eqs.)
  # It requires getting G_ae, G_mi = L_intermediate(T, IJAB, tx2)
  if T==1:
    #tx2_f = np.copy(IJAB) #this one needs to be checked! 
    tx2_f = 0.5*np.einsum('ijef,abef->ijab',tx2,W_abef,optimize=True)
    tx2_f += 0.5*np.einsum('mnij,mnab->ijab',W_mnij,tx2,optimize=True)
    # P(ij) terms
    X0 = np.einsum('kc,kmcd->md',tx1,IJAB,optimize=True)
    X1 = G_mi + np.einsum('md,jd->mj',X0,t1,optimize=True)
    X1 += np.einsum('kc,mkjc->mj',tx1,IJKA,optimize=True)
    X2 = -np.einsum('imab,mj->ijab',t2,X1,optimize=True)
    X2 += np.einsum('ic,abcj->ijab',tx1,W_efam,optimize=True)
    X2 -= np.einsum('imab,mj->ijab',tx2,F_mi,optimize=True)
    tx2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
    del X1,X2
    # P(ab) terms
    X1 = G_ae - np.einsum('mb,md->bd',t1,X0,optimize=True)
    X1 += np.einsum('kc,kbcd->bd',tx1,IABC,optimize=True)
    X2 = np.einsum('ijae,be->ijab',t2,X1,optimize=True)
    X2 -= np.einsum('ka,kbij->ijab',tx1,W_iemn,optimize=True)
    X2 += np.einsum('ijae,be->ijab',tx2,F_ae,optimize=True)
    tx2_f += X2 - np.transpose(X2,axes=(0,1,3,2))
    del X0,X1,X2
    # P(ij,ab) terms
    X2 = np.einsum('imae,mbej->ijab',tx2,W_mbej,optimize=True)
    #+ np.einsum('ia,jb->ijab',tx1,F_me,optimize=True) # not sure about this one because it's disconnected
    tx2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
    tx2_f -= np.transpose(X2,axes=(0,1,3,2))
    tx2_f += np.transpose(X2,axes=(1,0,3,2))
    del X2
    # Divide by energy denominator
    tx2_f /= D2    
  return tx2_f

#########################################################################
# CCSD Xi amplitudes for LR and EOM gradients
#########################################################################
def Xi(T, tx1, tx2, l1, l2, t1, IABC, IJAB, IJKA, F_ae, F_mi, F_me, W_mbej, D2):
  # L can be the ground or excited state Lambda amplitudes
  # Tx can be the LR Tx or the EOM R amplitudes
  if T==1:
    # Term 1
    # Xi1 : R1*Lg*H
    #       -R(jb)*(Lg(ja)*X(ib) + Lg(ib)*X(ja))
    #             _                  _                 _
    #       <0|L[[H,O1],R]|0> = <0|L(HR)conn|1> - <0|L(HR)disc|1>
    # P(ab) terms
    X2 = np.einsum('ijae,eb->ijab',l2,F_ae,optimize=True) 
    X1 = X2 - np.transpose(X2,axes=(0,1,3,2))
    del X2
    # P(ij) terms
    X2 = np.einsum('imab,jm->ijab',l2,F_mi,optimize=True) 
    X1 += np.transpose(X2,axes=(1,0,2,3)) - X2 
    del X2
    # P(ij,ab)-like terms
    X2 = np.einsum('imae,jebm->ijab',l2,W_mbej,optimize=True)
    X2 += np.einsum('ia,jb->ijab',l1,F_me,optimize=True)
    X1 += X2 + np.transpose(X2,axes=(1,0,3,2))
    del X2
    X1 -= l2*D2
    Xi1 = -np.einsum('ijab,jb->ia',X1,tx1,optimize=True)
    del X1
    # Term 2
    # Xi1 : -Lg(ijdb)t(md)R(kjcb)<mk||ac>
    X1 = np.einsum('kjcb,mkac->jmab',tx2,IJAB,optimize=True)
    X2 = np.einsum('ijdb,md->ijmb',l2,t1,optimize=True)
    Xi1 -= np.einsum('ijmb,jmab->ia',X2,X1,optimize=True)
    del X1, X2
    # Term 3
    # Xi1 : Lg(jibd)R(jkbc)<kd||ca>
    #     : Lg(jmba)R(jkbc)[<ki||mc>+t(md)<ki||dc>]
    # Xi2 : P(ij,ab)Lg(kica)[R(kmcd)-R(mc)t(kd)-t(mc)R(kd)]<mj||db>
    X1 = np.einsum('jibd,jkbc->ikdc',l2,tx2,optimize=True)
    Xi1 += np.einsum('ikdc,kdca->ia',X1,IABC,optimize=True)
    X2 = IJKA + np.einsum('md,kidc->kimc',t1,IJAB,optimize=True)
    Xi1 += np.einsum('mkac,kimc->ia',X1,X2,optimize=True)
    del X2
    X2 = np.einsum('kica,mc->kima',l2,tx1,optimize=True)
    X1 -= np.einsum('kima,kd->imad',X2,t1,optimize=True)
    X2 = np.einsum('kica,mc->kima',l2,t1,optimize=True)
    X1 -= np.einsum('kima,kd->imad',X2,tx1,optimize=True)
    del X2
    X2 = np.einsum('imad,mjdb->ijab',X1,IJAB,optimize=True)
    Xi2 = X2 - np.transpose(X2,axes=(0,1,3,2))
    Xi2 -= np.transpose(X2,axes=(1,0,2,3))
    Xi2 += np.transpose(X2,axes=(1,0,3,2))
    del X1, X2
    # Term 4
    # Xi1 :  1/2R(jkbc)Lg(jkbd)<di||ca>
    #     :  1/2R(jmbd)Lg(jmba)t(kc)<ik||cd>
    #     : -1/2R(jmbd)Lg(jmbc)t(kc)<ik||ad>
    # Xi2 : -1/2P(ab)R(kmcd)Lg(kmbd)<ij||ac>
    X1 = 0.5*np.einsum('jkbc,jkbd->cd',tx2,l2,optimize=True)
    Xi1 -= np.einsum('cd,idca->ia',X1,IABC,optimize=True)
    X2 = np.einsum('kc,ikcd->id',t1,IJAB,optimize=True)
    Xi1 += np.einsum('da,id->ia',X1,X2,optimize=True)
    del X2
    X2 = np.einsum('dc,kc->kd',X1,t1,optimize=True)
    Xi1 -= np.einsum('kd,ikad->ia',X2,IJAB,optimize=True)
    del X2
    X2 = -np.einsum('cb,ijac->ijab',X1,IJAB,optimize=True)
    Xi2 += X2 - np.transpose(X2,axes=(0,1,3,2))
    del X1, X2
    # Term 5
    # Xi1 : -1/2Lg(id)R(kmcd)<km||ca>
    # Xi2 : -1/2P(ab)Lg(ijad)R(kmcd)<km||cb>
    #     :    -P(ab)Lg(ijad)R(md)t(kc)<km||cb>
    #     :    -P(ab)Lg(ijad)t(md)R(kc)<km||cb>
    #     :     P(ab)Lg(ijad)R(kc)<kd||cb>
    X1 = -0.5*np.einsum('kmcd,kmca->da',tx2,IJAB,optimize=True)
    Xi1 += np.einsum('id,da->ia',l1,X1,optimize=True)
    X2 = np.einsum('kc,kmcb->mb',t1,IJAB,optimize=True)
    X1 -= np.einsum('md,mb->db',tx1,X2,optimize=True)
    del X2
    X2 = np.einsum('kc,kmcb->mb',tx1,IJAB,optimize=True)
    X1 -= np.einsum('md,mb->db',t1,X2,optimize=True) 
    del X2
    X1 += np.einsum('kc,kdcb->db',tx1,IABC,optimize=True) 
    X2 = np.einsum('ijad,db->ijab',l2,X1,optimize=True)
    Xi2 += X2 - np.transpose(X2,axes=(0,1,3,2))
    del X1, X2
    # Term 6
    # Xi1 :  1/2Lg(jkbc)R(jmbc)<im||ka>
    #     : -1/2Lg(jibc)R(jmbc)t(kd)<mk||ad>
    #     : -1/2Lg(jkbc)R(jmbc)t(kd)<im||ad>
    #     :     Lg(jb)R(jmbd)<im||ad>
    # Xi2 : -1/2P(ij)Lg(jmcd)R(kmcd)<ik||ab>
    X1 = 0.5*np.einsum('jkbc,jmbc->km',l2,tx2,optimize=True)
    Xi1 += np.einsum('km,imka->ia',X1,IJKA,optimize=True)
    X2 = np.einsum('ik,jkab->ijab',X1,IJAB,optimize=True)
    Xi2 += X2 - np.transpose(X2,axes=(1,0,2,3))
    del X2
    X2 = np.einsum('kd,mkad->ma',t1,IJAB,optimize=True)
    Xi1 -= np.einsum('im,ma->ia',X1,X2,optimize=True)
    del X2
    X2 = -np.einsum('km,kd->md',X1,t1,optimize=True)
    X2 += np.einsum('jb,jmbd->md',l1,tx2,optimize=True)
    Xi1 += np.einsum('md,imad->ia',X2,IJAB,optimize=True)
    del X1, X2
    # Term 7
    # Xi1 : -1/2Lg(ka)R(kmcd)<im||cd>
    # Xi2 : -1/2P(ij)Lg(kjab)R(kmcd)<im||cd>
    #     :    -P(ij)Lg(kjab)R(kc)t(md)<im||cd>
    #     :    -P(ij)Lg(kjab)t(kc)R(md)<im||cd>
    #     :    -P(ij)Lg(kjab)R(mc)<im||kc>
    X1 = -0.5*np.einsum('imcd,kmcd->ik',IJAB,tx2,optimize=True)
    Xi1 += np.einsum('ik,ka->ia',X1,l1,optimize=True)
    X2 = np.einsum('md,imcd->ic',t1,IJAB,optimize=True)
    X1 -= np.einsum('ic,kc->ik',X2,tx1,optimize=True)
    del X2
    X2 = np.einsum('md,imcd->ic',tx1,IJAB,optimize=True)
    X1 -= np.einsum('ic,kc->ik',X2,t1,optimize=True)
    del X2
    X1 -= np.einsum('imkc,mc->ik',IJKA,tx1,optimize=True)
    X2 = np.einsum('ik,kjab->ijab',X1,l2,optimize=True)
    Xi2 += X2 - np.transpose(X2,axes=(1,0,2,3))
    del X1, X2
    # Term 8
    # Xi1 : -1/4Lg(jkac)R(jkbd)<ic||bd>
    #     :  1/4Lg(jkac)t(mc)R(jkbd)<im||bd>
    # Xi2 :  1/4Lg(kmab)R(kmcd)<ij||cd>
    #     :     Lg(kmab)R(kc)t(md)<ij||cd>
    #     :     Lg(kmab)R(kc)<ij||cm>
    X1 = 0.25*np.einsum('ijcd,kmcd->ijkm',IJAB,tx2,optimize=True)
    X2 = np.einsum('imjk,mc->icjk',X1,t1,optimize=True)
    X2 -= 0.25*np.einsum('icbd,jkbd->icjk',IABC,tx2,optimize=True)
    Xi1 += np.einsum('icjk,jkac->ia',X2,l2,optimize=True)
    del X2
    X2 = IJKA + np.einsum('ijdc,md->ijmc',IJAB,t1,optimize=True)
    X1 -= np.einsum('ijmc,kc->ijkm',X2,tx1,optimize=True)
    Xi2 += np.einsum('ijkm,kmab->ijab',X1,l2,optimize=True)
    del X1, X2
    # Term 9
    # Xi1 : 1/4Lg(kicd)R(jmcd)<jm||ka>
    #     : 1/4Lg(kicd)R(jmcd)t(kb)<jm||ba>
    # Xi2 : 1/4Lg(ijcd)R(kmcd)<km||ab>
    #     :    Lg(ijcd)R(kc)t(md)<km||ab>
    X1 = 0.25*np.einsum('ijcd,kmcd->ijkm',l2,tx2,optimize=True)
    X2 = IJKA + np.einsum('jmba,kb->jmka',IJAB,t1,optimize=True)
    Xi1 += np.einsum('kijm,jmka->ia',X1,X2,optimize=True)
    del X2
    X2 = np.einsum('ijcd,kc->ijkd',l2,tx1,optimize=True)
    X1 += np.einsum('ijkd,md->ijkm',X2,t1,optimize=True)
    Xi2 += np.einsum('ijkm,kmab->ijab',X1,IJAB,optimize=True)
    del X1, X2
    # Term 10
    # Xi2 :  P(ij,ab)Lg(ia)R(kc)<kj||cb>
    #     :    -P(ab)Lg(ka)R(kc)<ij||cb>
    #     :    -P(ij)Lg(ic)R(kc)<kj||ab>
    X1 = np.einsum('kc,kjcb->jb',tx1,IJAB,optimize=True)
    X2 = np.einsum('ia,jb->ijab',l1,X1,optimize=True)
    Xi2 += X2 - np.transpose(X2,axes=(0,1,3,2))
    Xi2 -= np.transpose(X2,axes=(1,0,2,3))
    Xi2 += np.transpose(X2,axes=(1,0,3,2))
    del X1, X2
    X1 = np.einsum('ka,kc->ac',l1,tx1,optimize=True)
    X2 = -np.einsum('ac,ijcb->ijab',X1,IJAB,optimize=True)
    Xi2 += X2 - np.transpose(X2,axes=(0,1,3,2))
    del X1, X2
    X1 = np.einsum('ic,kc->ik',l1,tx1,optimize=True)
    X2 = -np.einsum('ik,kjab->ijab',X1,IJAB,optimize=True)
    Xi2 += X2 - np.transpose(X2,axes=(1,0,2,3))
    del X1, X2
    # Term 11
    # Xi2 :  P(ij,ab)Lg(ikac)R(mc)<jm||kb>
    #     : -P(ij,ab)Lg(ikac)R(kd)<jc||db>
    X1 = np.einsum('mc,jmkb->jbkc',tx1,IJKA,optimize=True)
    X1 -= np.einsum('kd,jcdb->jbkc',tx1,IABC,optimize=True)
    X2 = np.einsum('ikac,jbkc->ijab',l2,X1,optimize=True)
    Xi2 += X2 - np.transpose(X2,axes=(0,1,3,2))
    Xi2 -= np.transpose(X2,axes=(1,0,2,3))
    Xi2 += np.transpose(X2,axes=(1,0,3,2))
    del X1, X2
    # Term 12
    # Xi2 : -Lg(ijcd)R(kc)<kd||ab>
    X1 = np.einsum('ijcd,kc->ijkd',l2,tx1,optimize=True)
    Xi2 -= np.einsum('ijkd,kdab->ijab',X1,IABC,optimize=True)
  return Xi1, Xi2

#########################################################################
# CCSD rho1 transition density for LR and EOM 
#########################################################################
def TrDen1(T, O2, NB2, tx1, tx2, l1, l2, t1, t2):
  # For now, implement only the LR term: <0|(1+Lg)[e^{-T}{p^{+}q}e^{T},X^{B}]|0>
  # Density is returned in MO basis
  if T==1:
    rho1 = np.zeros((NB2,NB2))
    # AI block
    # There are no contributions for the LR function
    #
    # IJ block
    # - R(ic)L(jc) -1/2Rx(ikcd)Lg(jkcd)
    X1 = -np.einsum('ic,jc->ij',tx1,l1,optimize=True)
    X1 -= 0.5*np.einsum('ikcd,jkcd->ij',tx2,l2,optimize=True)
    rho1[:O2,:O2] = np.copy(X1)
    # AB block
    # L(ka)R(kb) +1/2Lg(kmca)Rx(kmcb)
    X2 = np.einsum('ka,kb->ab',l1,tx1,optimize=True)
    X2 += 0.5*np.einsum('kmca,kmcb->ab',l2,tx2,optimize=True)
    rho1[O2:,O2:] = np.copy(X2)
    # IA block
    # Rx(ia) -t(ka)[Rx(ic)Lg(kc) + 1/2Rx(imcd)Lg(kmcd)]
    # -t(ic)[Rx(ka)Lg(ck) + 1/2Rx(kmad)Lg(cdkm)]
    rho1[:O2,O2:] = np.copy(tx1)
    rho1[:O2,O2:] += np.einsum('ik,ka->ia',X1,t1,optimize=True)
    rho1[:O2,O2:] -= np.einsum('ic,ca->ia',t1,X2,optimize=True)
    del X1, X2
    # + Rx(ikac)Lg(ck)
    rho1[:O2,O2:] += np.einsum('ikac,kc->ia',tx2,l1,optimize=True)
    # -1/2t(kmad)Rx(ic)Lg(kmcd)
    X2 = 0.5*np.einsum('kmca,kmcd->ad',l2,t2,optimize=True)
    rho1[:O2,O2:] -= np.einsum('id,ad->ia',tx1,X2,optimize=True)
    del X2
    # -1/2 t(imcd)Rx(ka)Lg(kmcd)
    X1 = 0.5*np.einsum('imcd,kmcd->ik',t2,l2,optimize=True)
    rho1[:O2,O2:] -= np.einsum('ik,ka->ia',X1,tx1,optimize=True)
  return rho1
