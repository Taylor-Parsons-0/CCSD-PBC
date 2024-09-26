import numpy as np
import os
import sys
import re
import time
from read import getFort

##########################################################################
#Compute E and intermediates for CCSD T equations         ################
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

#####################################################################
def tau_tildeEq(T, O, V, t1, t2):
  # tau_tilde intermediate for CCSD T equations
  if T==1:
    tau_tilde = t2 + 0.5*(np.einsum('ia,jb->ijab',t1,t1,optimize=True) - np.einsum('ib,ja->ijab',t1,t1,optimize=True))
  return tau_tilde

########################################################################
def tauEq(T, O, V, t1, t2):
  # tau intermediate for CCSD T equations
  if T==1:
    tau = t2 + np.einsum('ia,jb->ijab',t1,t1,optimize=True) - np.einsum('ib,ja->ijab',t1,t1,optimize=True)
  return tau

########################################################################
def intermediateEqs(T, O, V, Fock, t1, t2, IJKL, ABCD, IABC, IJAB, IAJB, IJKA, tau_tilde, tau):
  # F and W intermediates for CCSD T equations
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
    W_abef =np.copy(ABCD)
    W_abef += np.einsum('mb,maef->abef',t1,IABC,optimize=True)
    W_abef -= np.einsum('ma,mbef->abef',t1,IABC,optimize=True)
    # W_mbej
    W_mbej = np.copy(-np.transpose(IAJB, axes=(0,1,3,2)))
    W_mbej += np.einsum('jf,mbef->mbej', t1, IABC, optimize=True)
    W_mbej += np.einsum('nb,mnje->mbej', t1, IJKA, optimize=True)
    W_mbej -= 0.5 * np.einsum('jnfb,mnef->mbej', t2, IJAB, optimize=True)
    W_mbej -= np.einsum('jf,nb,mnef->mbej', t1, t1, IJAB, optimize=True)
  return F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej

###########################################################################
def t1Eq(T, O, Fock, t1, t2, IABC, IJKA, IAJB, F_ae, F_mi, F_me, D1):
  O2=2*O
  # CCSD T1 amplitude equation
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

###########################################################################
def t2Eq(T, t1, t2, IABC, IJAB, IJKA, IAJB, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D2):
  if T==1:
  # CCSD T2 amplitude equation
    # P(ab) terms
    X1 = F_ae - 0.5*np.einsum('mb,me->be',t1,F_me,optimize=True)
    X2 = np.einsum('ijae,be->ijab',t2,X1,optimize=True) - np.einsum('ma,ijmb->ijab',t1,IJKA,optimize=True)
    t2_f = IJAB + X2 - np.transpose(X2,axes=(0,1,3,2))
    del X1, X2
    # t2_f = np.copy(IJAB)
    # P(ij) terms
    X1 = F_mi + 0.5*np.einsum('je,me->mj',t1,F_me,optimize=True)
    X2 = - np.einsum('imab,mj->ijab',t2,X1,optimize=True) - np.einsum('ie,jeab->ijab',t1,IABC,optimize=True)
    t2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
    del X1, X2
    # P(ij,ab) terms
    X1 = np.einsum('ie,mbje->mbij',t1,IAJB,optimize=True)
    X2 = np.einsum('imae,mbej->ijab',t2,W_mbej,optimize=True) + np.einsum('ma,mbij->ijab',t1,X1,optimize=True)
    t2_f += X2 - np.transpose(X2,axes=(1,0,2,3)) - np.transpose(X2,axes=(0,1,3,2))  + np.transpose(X2,axes=(1,0,3,2))
    del X1, X2
    # tau terms
    t2_f += 0.5*np.einsum('ijef,abef->ijab',tau,W_abef,optimize=True) + 0.5*np.einsum('mnab,mnij->ijab',tau,W_mnij,optimize=True)
    # Divide by energy denominator
    t2_f /= D2    
  return t2_f

###########################################################################
def E_CCSD(O, Fock, t1, IJAB, tau):
  # CCSD energy
  O2 = 2*O
  E_Corr2_1 = np.einsum('ia,ia->', t1, Fock[:O2, O2:])
  E_Corr2_2 = 0.25 * np.einsum('ijab,ijab->', tau, IJAB)
  E_Corr2 = E_Corr2_1 + E_Corr2_2
  
  return E_Corr2

###########################################################################
def L_intermediate_const(T, t1, t2, tau, IJAB, IAJB, IJKA, IABC, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej):
  # Define constant intermediates for CCSD Lambda equations
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
    #MC check the sign of the IABC transposition
    W_efam = -np.transpose(IABC,axes=(2,3,1,0)) + np.einsum('mnef,na->efam',t2,F_me,optimize=True) + np.einsum('efag,mg->efam',W_abef,t1,optimize=True) - 0.5*np.einsum('noef,noma->efam',tau,IJKA,optimize=True)
    W_iemn = np.transpose(IJKA,axes=(2,3,0,1)) - np.einsum('mnef,if->iemn',t2,F_me,optimize=True) - np.einsum('iomn,oe->iemn',W_mnij,t1,optimize=True) + 0.5*np.einsum('iefg,mnfg->iemn',IABC,tau,optimize=True)
    # Create a temp intermediates
    WW_mbej = -np.transpose(IAJB,axes=(0,1,3,2)) - np.einsum('mnef,njbf->mbej',IJAB,t2,optimize=True) 
    X1 = - np.einsum('ne,nfam->efam',t1,WW_mbej,optimize=True) + np.einsum('nega,mnfg->efam',IABC,t2,optimize=True)
    X2 = X1 - np.transpose(X1,axes=(1,0,2,3))
    W_efam += X2
    del X1,X2
    X1 = np.einsum('mf,iefn->iemn',t1,WW_mbej,optimize=True) + np.einsum('iomf,noef->iemn',IJKA,t2,optimize=True)
    X2 = X1 - np.transpose(X1,axes=(0,1,3,2))
    W_iemn += X2
    del X1,X2,WW_mbej
    # Done and return
  return W_efam, W_iemn

###########################################################################
def L_intermediate(T, t2, l2):
  # Define intermediates for CCSD Lambda equations
  if T==1:
    G_ae = -0.5*np.einsum('mnaf,mnef->ae',l2,t2,optimize=True)
    G_mi = 0.5*np.einsum('mnef,inef->mi',t2,l2,optimize=True)
  return G_ae, G_mi

###########################################################################
def l1Eq(T, t1, l1, l2, IJAB, IABC, IJKA, W_efam, W_iemn, W_mbej, F_ae, F_mi, F_me, G_ae, G_mi, D1):
  # CCSD Lambda1 amplitude equation
  if T==1:
    l1_f = np.copy(F_me)  
    l1_f += np.einsum('ie,ea->ia',l1,F_ae,optimize=True) - np.einsum('im,ma->ia',F_mi,l1,optimize=True) + np.einsum('me,ieam->ia',l1,W_mbej,optimize=True)
    l1_f += 0.5*np.einsum('imef,efam->ia',l2,W_efam,optimize=True) - 0.5*np.einsum('iemn,mnae->ia',W_iemn,l2,optimize=True)
    l1_f += np.einsum('ef,iefa->ia',G_ae,IABC,optimize=True) - np.einsum('mn,mina->ia',G_mi,IJKA,optimize=True)
    X1 = np.einsum('mf,fe->me',t1,G_ae,optimize=True) - np.einsum('mn,ne->me',G_mi,t1,optimize=True)
    l1_f += np.einsum('me,imae->ia',X1,IJAB,optimize=True)
    del X1
    l1_f /= D1
  return l1_f

###########################################################################
def l2Eq(T, t1, l1, l2, IABC, IJAB, IJKA, F_ae, F_mi, F_me, G_ae, G_mi, W_mnij, W_abef, W_mbej, D2):
  if T==1:
  # CCSD Lambda2 amplitude equation
    l2_f = np.copy(IJAB)
    l2_f += 0.5*np.einsum('ijef,efab->ijab',l2,W_abef,optimize=True) + 0.5*np.einsum('ijmn,mnab->ijab',W_mnij,l2,optimize=True)
    # P(ab) terms
    X1 = G_ae - np.einsum('mb,me->be',l1,t1,optimize=True)
    X2 = np.einsum('ijae,be->ijab',IJAB,X1,optimize=True) - np.einsum('ma,ijmb->ijab',l1,IJKA,optimize=True) + np.einsum('ijae,eb->ijab',l2,F_ae,optimize=True) 
    l2_f += X2 - np.transpose(X2,axes=(0,1,3,2))
    del X1, X2
    # P(ij) terms
    X1 = G_mi + np.einsum('me,je->mj',t1,l1,optimize=True)
    X2 = np.einsum('imab,mj->ijab',IJAB,X1,optimize=True) + np.einsum('ie,jeab->ijab',l1,IABC,optimize=True) + np.einsum('imab,jm->ijab',l2,F_mi,optimize=True) 
    l2_f += np.transpose(X2,axes=(1,0,2,3)) - X2 
    del X1, X2
    # P(ij,ab) terms
    X2 = np.einsum('imae,jebm->ijab',l2,W_mbej,optimize=True) + np.einsum('ia,jb->ijab',l1,F_me,optimize=True)
    l2_f += X2 - np.transpose(X2,axes=(1,0,2,3)) - np.transpose(X2,axes=(0,1,3,2))  + np.transpose(X2,axes=(1,0,3,2))
    del X2
    # Divide by energy denominator
    l2_f /= D2    
  return l2_f
