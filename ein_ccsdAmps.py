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
def tau_tildeEq(T, O, V, t1, t2):
  if T==1:
    # tau_tilde=np.zeros((O, O, V, V))
    # #Equation (10)
    # for a in range(V):
    #   for b in range(V):
    #     for i in range(O):
    #       for j in range(O):
    #         tau_tilde[i,j,a,b]=t2[i,j,a,b]+(1/2)*(t1[i,a]*t1[j,b]-t1[i,b]*t1[j,a])
    tau_tilde = t2 + (1/2)*(np.einsum('ia,jb->ijab',t1,t1,optimiza=True) - np.einsum('ib,ja->ijab',t1,t1,optimiza=True))
  return tau_tilde

def tauEq(T, O, V, t1, t2):
  if T==1:
    # tau=np.zeros((V, V, O, O))
    # #Equation (11)
    # for a in range(V):
    #   for b in range(V):
    #     for i in range(O):
    #       for j in range(O):
    #         tau[a,b,i,j]=t2[a,b,i,j]+t1[a,i]*t1[b,j]-t1[b,i]*t1[a,j]
    tau = t2 + np.einsum('ia,jb->ijab',t1,t1,optimiza=True) - np.einsum('ib,ja->ijab',t1,t1,optimiza=True)
  return tau

########################################################################
### Begin F and W intermediates ########################################
########################################################################
def intermediateEqs(O, V, NB, Fock, t1, t2, MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, T, tau_tilde, tau):
#  print("O:",O)
#  print("V:",V)
#  print("NB:",NB)
#  print("delta:",np.shape(delta))
#  print("Fock:",np.shape(Fock))
#  print("t1:",np.shape(t1))
#  print("MO:",np.shape(MO))
#  print("tau_tilde:",np.shape(tau_tilde))
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
    V = NB - O
    F_ae = np.zeros((V, V))
    F_ae += (1 - np.eye(V)) * Fock[O:, O:] #Add flag, function to set diagonal elements to zero
    F_ae -= 0.5 * np.einsum('me,am->ae', Fock[:O, O:], t1)
    F_ae += np.einsum('fm,mafe->ae', t1, IABC) #Nest these two
    F_ae -= 0.5 * np.einsum('afmn,mnef->ae', tau_tilde, IJAB)
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
    F_mi = (1 - delta[:O, :O]) * Fock[:O, :O] #Add flag function to set diagonal elements to zero
    F_mi += 0.5 * np.einsum('ei,me->mi', t1, Fock[:O, O:])
    F_mi += np.einsum('en,mnie->mi', t1, IJKA) #Nest these two
    F_mi += 0.5 * np.einsum('efin,mnef->mi', tau_tilde, IJAB)
    
    #Equation (6)
#    F_me=np.zeros((O, V))
#    for m in range(O):
#      for e in range(V):
#        F_me[m,e]=Fock[m,e+O]
#        for n in range(O):
#          for f in range(V):
#            F_me[m,e]+=t1[f,n]*IJAB[m,n,e,f] #MO[m,n,e+O,f+O]
    F_me = np.zeros((O, V))
    F_me += Fock[:O, O:]
    F_me += np.einsum('fn,mnef->me', t1, IJAB) #Nest this into last line of Fmi

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
    W_mnij = np.copy(IJKL)
#    W_mnij += IJKL
    W_mnij += np.einsum('ej,mnie->mnij', t1, IJKA)#Nest these three
    W_mnij -= np.einsum('ei,mnje->mnij', t1, IJKA)
    W_mnij += 0.25 * np.einsum('efij,mnef->mnij', tau, IJAB)

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
    W_abef =np.copy(ABCD)
#    W_abef -= np.einsum('bm,amef->abef', t1, AIBC) #Nest these three
#    W_abef += np.einsum('am,bmef->abef', t1, AIBC)
    W_abef -= np.einsum('bm,mafe->abef', t1, IABC) #Nest these three
    W_abef += np.einsum('am,mbfe->abef', t1, IABC)
    W_abef += 0.25 * np.einsum('abmn,mnef->abef', tau, IJAB)

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
    W_mbej = np.copy(IABJ)
    W_mbej += np.einsum('fj,mbef->mbej', t1, IABC)
    W_mbej += np.einsum('bn,mnje->mbej', t1, IJKA)
    W_mbej -= 0.5 * np.einsum('fbjn,mnef->mbej', t2, IJAB)
    W_mbej -= np.einsum('fj,bn,mnef->mbej', t1, t1, IJAB)

  return F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej

##################################################################################
#Solve for T equations ###########################################################
##################################################################################

#Final t1
def t1Eq(T, O, Fock, t1, t2, IABC, IJKA, IAJB, F_ae, F_mi, F_me, D1):
  O2 = 2*O
  if T==1:
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
def t2Eq(T, t1, t2, IABC, IJAB, IJKA, IAJB, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D2):
  if T==1:
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

###########################################################################
def E_CCSD(O, Fock, t1, IJAB, tau):
#Solve for CCSD energy###################################
  O2 = 2*O
  E_Corr2_1 = np.einsum('ia,ia->', t1, Fock[:O2, O2:])
  E_Corr2_2 = 0.25 * np.einsum('ijab,ijab->', tau, IJAB)
  E_Corr2 = E_Corr2_1 + E_Corr2_2
  
  return E_Corr2
