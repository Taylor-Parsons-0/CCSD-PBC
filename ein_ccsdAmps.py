import numpy as np
import os
import sys
import re
import time
from read import getFort, get2e, conMO

#Initialize arrays
#Define molecule
if len(sys.argv)==2:
  molecule=sys.argv[1]
else:
  print("MISSING MOLECULE NAME AS FIRST ARG")
  exit()
log=f"{molecule}.log"
O, V, NB, scfE, Fock, Coeff=getFort(molecule, log)
V=NB-O
coul=np.zeros((NB, NB, NB, NB))
exc=np.zeros((NB, NB, NB, NB))
OE=np.zeros((NB))
AOInt=np.zeros((NB, NB, NB, NB))
MO=np.zeros((NB*2, NB*2, NB*2, NB*2))
#IJKL=np.zeros((O,O,O,O))
#IJAB=np.zeros((O,O,V,V))
#IABJ=np.zeros((O,V,V,O))
#IABC=np.zeros((O,V,V,V))
#IJKA=np.zeros((O,O,O,V))
#AIBC=np.zeros((V,O,V,V))
delta=np.identity(NB*2)
twoE=np.zeros((NB, NB, NB, NB))

#####################################################################
#Tau equations ######################################################
#####################################################################
#def tau_tildeEq(O, V, t1, t2, T):#NB, Fock, t1, t2, MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, T):
#  if T==1:
#    O2=2*O
#    V2=2*V
#    tau_tilde=np.zeros((O2, O2, V2, V2))
#    #Equation (10)
#    for i in range(O2):
#      for j in range(O2):
#        for a in range(V2):
#          for b in range(V2):
#            tau_tilde[i,j,a,b] = t2[i,j,a,b] + (1/2)*(t1[i,a] * t1[j,b] - t1[i,b] * t1[j,a])
def tau_tildeEq(T, O, V, t1, t2):
  if T==1:
    tau_tilde = t2 + (1/2)*np.einsum('ia,jb->ijab',t1,t1,optimize=True) - np.einsum('ib,ja->ijab',t1,t1,optimize=True)
  return tau_tilde

#def tauEq(O, V, t1, t2, T):#NB, Fock, t1, t2, MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, T):
#  if T==1:
#    O2=2*O
#    V2=2*V
#    tau=np.zeros((O2, O2, V2, V2))
#    #Equation (11)
#    for i in range(O2):
#      for j in range(O2):
#        for a in range(V2):
#          for b in range(V2):
#            tau[i,j,a,b] = t2[i,j,a,b] + t1[i,a] * t1[j,b] - t1[i,b] * t1[j,a]
def tauEq(T, O, V, t1, t2):
  if T==1:
    tau = t2 + np.einsum('ia,jb->ijab',t1,t1,optimiza=True) - np.einsum('ib,ja->ijab',t1,t1,optimiza=True)
  return tau

def intermediateEqs(T, O, V, Fock, t1, t2, IJKL, ABCD, IABC, IJAB, IJKA, tau_tilde, tau):
########################################################################
### Begin F and W intermediates ########################################
########################################################################
#  print("O:",O)
#  print("V:",V)
#  print("NB:",NB)
#  print("delta:",np.shape(delta))
#  print("Fock:",np.shape(Fock))
#  print("t1:",np.shape(t1))
#  print("MO:",np.shape(MO))
#  print("tau_tilde:",np.shape(tau_tilde))
  O2=2*O
  V2=2*V
  if T==1:
    #Equation (4)
#    F_ae=np.zeros((V, V))
#    for a in range(V):
#      for e in range(V):
#        F_ae[a,e]=(1-delta[a+O,e])*Fock[a+O,e]
#        for m in range(O):
#          F_ae[a,e]-= (1/2)*Fock[m,e+O]*t1[a,m]
#          for f in range(V):
#            F_ae[a,e]+=t1[f,m]*IABC[m,a,f,e] #MO[m,a+O,f,e+O]
#            for n in range(O):
#              F_ae[a,e]-= (1/2)*tau_tilde[a,f,m,n]*IJAB[m,n,e,f] #MO[m,n,e+O,f+O]
#    V = NB - O
#NEW
    F_ae = np.zeros((V2, V2))
#    F_ae += (1 - np.eye(V)) * Fock[O:, O:] #Add flag, function to set diagonal elements to zero
    F_ae += (1 - np.eye(V2)) * Fock[O2:, O2:] #Add flag, function to set diagonal elements to zero

#    F_ae -= 0.5 * np.einsum('me,am->ae', Fock[:O, O:], t1,optimize=True)
    F_ae = -0.5 * np.einsum('me,am->ae', Fock[:O2, O2:], t1, optimize=True])

#    F_ae += np.einsum('fm,mafe->ae', t1, IABC) #Nest these two
    F_ae += np.einsum('mf,mafe->ae', t1, IABC, optimize=True)

#    F_ae -= 0.5 * np.einsum('afmn,mnef->ae', tau_tilde, IJAB)
    F_ae -= 0.5 * np.einsum('mnaf,mnef->ae', tau_tilde, IJAB, optimize=True)    
#___
    #Equation (5) error
    F_mi=np.zeros((O, O))
 #   for m in range(O):
 #     for i in range(O):
 #       F_mi[m,i]=(1-delta[m,i])*Fock[m,i]
 #       for e in range(V):
 #         F_mi[m,i]+=(1/2)*t1[e,i]*Fock[m,e+O]
 #         for n in range(O):
 #           F_mi[m,i]+=t1[e,n]*IJKA[m,n,i,e]
 #           for f in range(V):
 #             F_mi[m,i]+=(1/2)*tau_tilde[e,f,i,n]*IJAB[m,n,e,f]
 
    #F_mi = (1 - delta[:O, :O]) * Fock[:O, :O] #Add flag function to set diagonal elements to zero
#NEW
#    F_mi = (1 - delta[:O, :O]) * Fock[:O, :O]
    F_mi = (1 - delta[:O2, :O2]) * Fock[:O2, :O2]

#    F_mi = (1 - delta[:O, :O]) * Fock[O+V:O2+V, O+V:O2+V]   

#    F_mi += 0.5 * np.einsum('ei,me->mi', t1, Fock[:O, O:])
    F_mi += 0.5 * np.einsum('ie,me->mi', t1, Fock[:O2, :O2], optimize=True)

#    F_mi += np.einsum('en,mnie->mi', t1, IJKA) #Nest these two
    F_mi += np.einsum('ne,mnie->mi', t1, IJKA, optimize=True)

#    F_mi += 0.5 * np.einsum('efin,mnef->mi', tau_tilde, IJAB)
    F_mi += 0.5 * np.einsum('inef,mnef->mi', tau_tilde, IJAB, optimize=True)
#___    
    #Equation (6)
#    F_me=np.zeros((O, V))
#    for m in range(O):
#      for e in range(V):
#        F_me[m,e]=Fock[m,e+O]
#        for n in range(O):
#          for f in range(V):
#            F_me[m,e]+=t1[f,n]*IJAB[m,n,e,f] #MO[m,n,e+O,f+O]
#    F_me = np.zeros((O, V))
#NEW
    F_me = Fock[:O2, O2:]
    #F_me += np.einsum('fn,mnef->me', t1, IJAB) #Nest this into last line of Fmi
    F_me += np.einsum('nf,mnef->me', t1, IJAB, optimize=True)
#___
    #Equation (7)
#    W_mnij=np.zeros((O, O, O, O))
#    for m in range(O):
#      for n in range(O):
#        for i in range(O):
#          for j in range(O):
#            W_mnij[m,n,i,j]=IJKL[m,n,i,j] #MO[m,n,i,j]
#            for e in range(V):
#              W_mnij[m,n,i,j]+=t1[e,j]*IJKA[m,n,i,e]-t1[e,i]*IJKA[m,n,j,e]
#                                      #MO[m,n,i,e+O]-t1[e,i]*MO[m,n,j,e+O]
#              for f in range(V):
#                W_mnij[m,n,i,j]+=(1/4)*tau[e,f,i,j]*IJAB[m,n,e,f] #MO[m,n,e+O,f+O]
#    W_mnij = np.zeros((O, O, O, O))
#NEW
    W_mnij = np.copy(IJKL)

#    W_mnij += np.einsum('ej,mnie->mnij', t1, IJKA)#Nest these three
    W_mnij += np.einsum('je,mnie->mnij', t1, IJKA, optimize=True)

#    W_mnij -= np.einsum('ei,mnje->mnij', t1, IJKA)
    W_mnij -= np.einsum('ie,mnje->mnij', t1, IJKA, optimize=True)

#    W_mnij += 0.25 * np.einsum('efij,mnef->mnij', tau, IJAB)
    W_mnij += 0.25 * np.einsum('ijef,mnef->mnij', tau, IJAB, optimize=True)
#___
    #Equation (8)
#    W_abef=np.zeros((V, V, V, V))
#    for a in range(V):
#      for b in range(V):
#        for e in range(V):
#          for f in range(V):
#            W_abef[a,b,e,f]=ABCD[a,b,e,f] #MO[a+O,b+O,e+O,f+O]
#            for m in range(O):
#              W_abef[a,b,e,f]-= t1[b,m]*AIBC[a,m,e,f]-t1[a,m]*AIBC[b,m,e,f]
#                                       #MO[a+O,m,e+O,f+O]-t1[a,m]*MO[b+O,m,e+O,f+O]
#              for n in range(O):
#                W_abef[a,b,e,f]+=(1/4)*tau[a,b,m,n]*IJAB[m,n,e,f] #MO[m,n,e+O,f+O]
    #W_abef = np.zeros((V, V, V, V))
#NEW
    W_abef =np.copy(ABCD)

#    W_abef -= np.einsum('bm,mafe->abef', t1, IABC) #Nest these three
    W_abef -= np.einsum('mb,mafe->abef', t1, IABC, optimize=True)

#    W_abef += np.einsum('am,mbfe->abef', t1, IABC)
    W_abef += np.einsum('ma,mbfe->abef', t1, IABC, optimize=True)

#    W_abef += 0.25 * np.einsum('abmn,mnef->abef', tau, IJAB)
    W_abef += 0.25 * np.einsum('mnab,mnef->abef', tau, IJAB, optimize=True)
#___
    #Equation (9)
#    W_mbej=np.zeros((O, V, V, O))
#    for m in range(O):
#      for b in range(V):
#        for e in range(V):
#          for j in range(O):
#            W_mbej[m,b,e,j]=IABJ[m,b,e,j] #MO[m,b+O,e+O,j]
#            for f in range(V):
#              W_mbej[m,b,e,j]+=t1[f,j]*IABC[m,b,e,f] #MO[m,b+O,e+O,f+O]
#            for n in range(O):
#              W_mbej[m,b,e,j]-=t1[b,n]*IJAK[m,n,e,j] #MO[m,n,e+O,j]
#              for f in range(V):
#                W_mbej[m,b,e,j]-= ((1/2)*t2[f,b,j,n]+t1[f,j]*t1[b,n])*IJAB[m,n,e,f] #MO[m,n,e+O,f+O]
    #W_mbej = np.zeros((O, V, V, O))
#NEW
    W_mbej = np.copy(IABJ)
#    W_mbej += np.einsum('fj,mbef->mbej', t1, IABC)
    W_mbej += np.einsum('jf,mbej->mbej', t1, IABC, optimize=True)

#    W_mbej += np.einsum('bn,mnje->mbej', t1, IJKA)
    W_mbej += np.einsum('nb,mnje->mbej', t1, IJKA, optimize=True)

#    W_mbej -= 0.5 * np.einsum('fbjn,mnef->mbej', t2, IJAB)
    W_mbej -= 0.5 * np.einsum('jnfb,mnef->mbej', t2, IJAB, optimize=True)

#    W_mbej -= np.einsum('fj,bn,mnef->mbej', t1, t1, IJAB)
    W_mbej -= np.einsum('jf,nb,mnef->mbej', t1, t1, IJAB, optimize=True)

#___
  return F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej

##################################################################################
#Solve for T equations ###########################################################
##################################################################################

#Final t1
#t1_f=np.zeros((NB, NB))
#def t1Eq(O, V, NB, Fock, t1, t2, MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, T, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D1):
def t1Eq(T, O, Fock, t1, t2, IABC, IJKA, IAJB, F_ae, F_mi, F_me, D1):
  O2=2*O 
  if T==1:
#    t1_f = np.copy(Fock[:O2, O2:])
#    t1_f += np.einsum('ei,ae->ai', t1, F_ae)
#    t1_f -= np.einsum('am,mi->ai', t1, F_mi)
#    t1_f += np.einsum('aeim,me->ai', t2, F_me)
#    t1_f -= 0.5 * np.einsum('efim,maef->ai', t2, IABC)#MO[:O, O:, O:, O:])
#    t1_f += 0.5 * np.einsum('aemn,nmie->ai', t2, IJKA)#MO[:O, :O, O:, :O])
#    t1_f -= np.einsum('fn,naif->ai', t1, IAJB)#MO[:O, O:, :O, O:])
#    t1_f /= D1
    t1_f = np.copy(Fock[:O2, O2:])  
    t1_f += np.einsum('ie,ae->ia', t1, F_ae,optimize=True)
    t1_f -= np.einsum('ma,mi->ia', t1, F_mi,optimize=True)
    t1_f += np.einsum('imae,me->ia', t2, F_me,optimize=True)
    t1_f -= 0.5 * np.einsum('imef,maef->ia', t2, IABC,optimize=True)#MO[:O, O:, O:, O:])
    t1_f += 0.5 * np.einsum('mnae,nmie->ia', t2, IJKA,optimize=True)#MO[:O, :O, O:, :O])
    t1_f -= np.einsum('nf,naif->ia', t1, IAJB,optimize=True)#MO[:O, O:, :O, O:])
    t1_f /= D1
  return t1_f

#Final t2
#def t2Eq(O, V, NB, Fock, t1, t2, MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, T, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D2):
def t2Eq(T, t1, t2, IABC, IJAB, IJKA, IAJB, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D2):
  if T==1:
#Initialize T2
#    #Form I1
#    X1=np.einsum("me,aeij->amij",F_me,t2)
#    I1=np.einsum("bm,amij->abij",t1,X1)
#    #Form I3
#    X3=np.einsum("me,abim->abie",F_me,t2)
#    I3=np.einsum("ej,abie->abij",t1,X3)
#    #Form I5
#    X5=np.einsum("ei,mbej->mbij",t1,IABJ)
#    I5=np.einsum("am,mbij->abij",t1,X5)
#    #Form I7
#    X7=np.einsum("ej,mbei->mbji",t1,IABJ)
#    I7=np.einsum("am,mbji->abij",t1,X7)
#    I8=I7
#    #Equation (3)
#    t2_f = np.transpose(IJAB, axes=(2,3,0,1)) - (1/2)*I1 + (1/2)*np.transpose(I1,axes=(1,0,2,3)) - (1/2)*( I3 - np.transpose(I3, axes=(0,1,3,2)) ) - (I5 - np.transpose(I5,axes=(1,0,2,3))) + I7 - np.transpose(I7, axes=(1,0,2,3))
#    t2_f += np.einsum("be,aeij->abij",F_ae,t2,optimize=True) - np.einsum("ae,beij->abij",F_ae,t2,optimize=True) + np.einsum("ei,jeba->abij",t1,IABC,optimize=True) - np.einsum("ej,ieba->abij",t1,IABC,optimize=True)
#    t2_f += (1/2)*np.einsum("efij,abef->abij",tau,W_abef,optimize=True)  
#    t2_f -= np.einsum("mj,abim->abij",F_mi,t2,optimize=True) - np.einsum("mi,abjm->abij",F_mi,t2,optimize=True) + np.einsum("am,mbij->abij",t1,IAJK,optimize=True) - np.einsum("bm,maij->abij",t1,IAJK,optimize=True)
#    t2_f += np.einsum("aeim,mbej->abij",t2,W_mbej,optimize=True) - np.einsum("beim,maej->abij",t2,W_mbej,optimize=True) - np.einsum("aejm,mbei->abij",t2,W_mbej,optimize=True) + np.einsum("bejm,maei->abij",t2,W_mbej,optimize=True)
#    t2_f += (1/2)*np.einsum("abmn,mnij->abij",tau,W_mnij,optimize=True)
#    t2_f /= D2    
    #Form I1
    X1=np.einsum("me,ijae->amij",F_me,t2)
    I1=np.einsum("mb,amij->ijab",t1,X1)
    #Form I3 
    X3=np.einsum("me,imab->abie",F_me,t2)
    I3=np.einsum("je,abie->ijab",t1,X3)
    #Form I5
    #MC - changed sign to us IAJB instead of IABJ
    X5=-np.einsum("ie,mbje->mbij",t1,IAJB) + np.einsum("je,mbie->mbij",t1,IAJB)
    I5=np.einsum("ma,mbij->ijab",t1,X5)
    # #Form I7
    # #MC - changed sign to us IAJB instead of IABJ
    # X7=
    # I7=np.einsum("ma,mbji->abij",t1,X7)
    #Equation (3)
    t2_f = IJAB - (1/2)*I1 + (1/2)*np.transpose(I1,axes=(0,1,3,2)) - (1/2)*(I3-np.transpose(I3, axes=(1,0,2,3))) - (I5 - np.transpose(I5,axes=(0,1,3,2)))
    #+ I7 - np.transpose(I7, axes=(1,0,2,3))
    t2_f += np.einsum("be,ijae->ijab",F_ae,t2,optimize=True) - np.einsum("ae,ijbe->ijab",F_ae,t2,optimize=True) + np.einsum("ie,jeba->ijab",t1,IABC,optimize=True) - np.einsum("je,ieba->ijab",t1,IABC,optimize=True)
    t2_f += (1/2)*np.einsum("ijef,abef->ijab",tau,W_abef,optimize=True)  
    t2_f -= np.einsum("mj,imab->ijab",F_mi,t2,optimize=True) - np.einsum("mi,jmab->ijab",F_mi,t2,optimize=True) + np.einsum("ma,ijmb->ijab",t1,IJKA,optimize=True) - np.einsum("mb,ijma->ijab",t1,IJKA,optimize=True)
    t2_f += np.einsum("imae,mbej->ijab",t2,W_mbej,optimize=True) - np.einsum("imbe,maej->ijab",t2,W_mbej,optimize=True) - np.einsum("jmae,mbei->ijab",t2,W_mbej,optimize=True) + np.einsum("jmbe,maei->ijab",t2,W_mbej,optimize=True)
    t2_f += (1/2)*np.einsum("mnab,mnij->ijab",tau,W_mnij,optimize=True)
    t2_f /= D2    
  return t2_f

#########################################################
#Solve for CCSD energy###################################
#########################################################
#Equation 1
#def CCSD(O, V, NB, Fock, t1, t2, MO,  IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, T, tau):
#  E_Corr2=0
#  for i in range(O):
#    for a in range(V):
#      E_Corr2+=t1[a,i]*Fock[i,a+O]
#      for j in range(O):
#        for b in range(V):
#          E_Corr2+=(1/4)*tau[a,b,i,j]*IJAB[i,j,a,b]
#  return E_Corr2
#def CCSD(O, V, NB, Fock, t1, IJAB, T, tau):#MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, T, tau):
#  E_Corr2_1 = np.einsum('ai,ia->', t1, Fock[:O, O:])
#  E_Corr2_2 = 0.25 * np.einsum('abij,ijab->', tau, IJAB)
#  E_Corr2 = E_Corr2_1 + E_Corr2_2

###########################################################################
def E_CCSD(O, Fock, t1, IJAB, tau):
#Solve for CCSD energy###################################
  O2 = 2*O
  E_Corr2_1 = np.einsum('ia,ia->', t1, Fock[:O2, O2:])
  E_Corr2_2 = 0.25 * np.einsum('ijab,ijab->', tau, IJAB)
  E_Corr2 = E_Corr2_1 + E_Corr2_2
  
  return E_Corr2
