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
def tau_tildeEq(O, V, t1, t2, T):#NB, Fock, t1, t2, MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, T):
  if T==1:
    tau_tilde=np.zeros((V, V, O, O))
    #Equation (10)
    for a in range(V):
      for b in range(V):
        for i in range(O):
          for j in range(O):
            tau_tilde[a,b,i,j]=t2[a,b,i,j]+(1/2)*(t1[a,i]*t1[b,j]-t1[b,i]*t1[a,j])
  return tau_tilde

def tauEq(O, V, t1, t2, T):#NB, Fock, t1, t2, MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, T):
  if T==1:
    tau=np.zeros((V, V, O, O))
    #Equation (11)
    for a in range(V):
      for b in range(V):
        for i in range(O):
          for j in range(O):
            tau[a,b,i,j]=t2[a,b,i,j]+t1[a,i]*t1[b,j]-t1[b,i]*t1[a,j]
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
#t1_f=np.zeros((NB, NB))
def t1Eq(O, V, NB, Fock, t1, t2, MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, T, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D1):
  if T==1:
#    t1_f=np.zeros((V, O))
#  #Equation (2)
#    for a in range(V):
#      for i in range(O):
#        t1_f[a,i]=Fock[i,a+O]
#        for e in range(V):
#          t1_f[a,i]+=t1[e,i]*F_ae[a,e]
#        for m in range(O):
#          t1_f[a,i]-= t1[a,m]*F_mi[m,i]
#          for e in range(V):
#            t1_f[a,i]+=t2[a,e,i,m]*F_me[m,e]
#            for f in range(V):
#              t1_f[a,i]-= (1/2)*t2[e,f,i,m]*IABC[m,a,e,f] #MO[m,a+O,e+O,f+O]
#            for n in range(O):
#              t1_f[a,i]-= (1/2)*t2[a,e,m,n]*IJAK[n,m,e,i] #MO[n,m,e+O,i]
#        for n in range(O):
#          for f in range(V):
#            t1_f[a,i]-= t1[f,n]*IAJB[n,a,i,f] #MO[n,a+O,i,f+O]
#        t1_f[a,i]=t1_f[a,i]/D1[a,i]
#    t1_f = np.zeros((V, O))
    t1_f = np.copy(Fock[O:, :O])
    t1_f += np.einsum('ei,ae->ai', t1, F_ae)
    t1_f -= np.einsum('am,mi->ai', t1, F_mi)
    t1_f += np.einsum('aeim,me->ai', t2, F_me)
    t1_f -= 0.5 * np.einsum('efim,maef->ai', t2, IABC)#MO[:O, O:, O:, O:])
    t1_f += 0.5 * np.einsum('aemn,nmie->ai', t2, IJKA)#MO[:O, :O, O:, :O])
    t1_f -= np.einsum('fn,naif->ai', t1, IAJB)#MO[:O, O:, :O, O:])
    t1_f /= D1
  return t1_f

#Final t2
def t2Eq(O, V, NB, Fock, t1, t2, MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, T, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D2):
  if T==1:
#Initialize T2
    t2_f=np.zeros((V, V, O, O))
#Form 8 N^5 Intermediates
#    X1=np.zeros((V, O, O, O))
#    X2=np.zeros((V, O, O, O))
#    X3=np.zeros((NB, NB, NB, NB))
#    X4=np.zeros((V, V, O, V))
#    X5=np.zeros((NB, NB, NB, NB))
#    X6=np.zeros((O,V,O,O))
#    X7=np.zeros((NB,NB,NB,NB))
#    X8=np.zeros((O, V, O, O))
#
#    I1=np.zeros((V, V, O, O))
#    I2=np.zeros((V, V, O, O))
#    I3=np.zeros((NB, NB, NB, NB))
#    I4=np.zeros((V, V, O, O))
#    I5=np.zeros((NB, NB, NB, NB))
#    I6=np.zeros((V, V, O, O))
#    I7=np.zeros((NB, NB, NB, NB))
#    I8=np.zeros((V, V, O, O))

#Form I1
#    for a in range(O,NB):
#      for i in range(O):
#        for j in range(O,NB):
#          for e in range(O,NB):
#            for m in range(O):
#              X1[a,m,i,j]+=F_me[m,e]*t2[a,e,i,j]
#    for a in range(O,NB):
#      for b in range(O,NB):
#        for i in range(O):
#          for j in range(O):
#            for m in range(O):
#              I1[a,b,i,j]+=t1[b,m]*X1[a,m,i,j]
#    print(np.shape(t2))
    X1=np.einsum("me,aeij->amij",F_me,t2)
    I1=np.einsum("bm,amij->abij",t1,X1)
#    print(np.sum(abs(I1)))
#    print(np.sum(abs(I1))) 
#Form I2
#    for b in range(V):
#      for i in range(O):
#        for j in range(O):
#          for e in range(V):
#            for m in range(O):
#              X2[b,m,i,j]+=F_me[m,e]*t2[b,e,i,j]
#
#
#    for a in range(V):
#      for b in range(V):
#        for i in range(O):
#          for j in range(O):
#            for m in range(O):
#              I2[b,a,i,j]+=t1[a,m]*X2[b,m,i,j]#
#    X2=np.einsum("me,beij->bmij",F_me,t2)
#    I2=np.einsum("am,bmij->baij",t1,X2)
#    I2=I1
#    print(np.sum(abs(I2)))
#Form I3 
#    for a in range(O,NB):
#      for b in range(O,NB):
#        for i in range(O):
#          for m in range(O):
#            for e in range(O,NB):
#              X3[a,b,i,e]+=F_me[m,e]*t2[a,b,i,m]
#
#
#    for a in range(O,NB):
#      for b in range(O,NB):
#        for i in range(O):
#          for j in range(O):
#            for e in range(O,NB):
#              I3[a,b,i,j]+=t1[e,j]*X3[a,b,i,e]
    X3=np.einsum("me,abim->abie",F_me,t2)
    I3=np.einsum("ej,abie->abij",t1,X3)
#Form I4
#    for a in range(V):
#      for b in range(V):
#        for j in range(O):
#          for m in range(O):
#            for e in range(V):
#              X4[a,b,j,e]+=F_me[m,e]*t2[a,b,j,m]
#
#
#    for a in range(V):
#      for b in range(V):
#        for i in range(O):
#          for j in range(O):
#            for e in range(V):
#              I4[a,b,j,i]+=t1[e,i]*X4[a,b,j,e]
    I4=I3
#Form I5
#    for b in range(O,NB):
#      for i in range(O):
#        for j in range(O):
#          for m in range(O):
#            for e in range(O,NB):
#              X5[m,b,i,j]+=t1[e,i]*IABJ[m,b,e,j] #MO[m,b,e,j]
#
#
#    for a in range(O,NB):
#      for b in range(O,NB):
#        for i in range(O):
#          for j in range(O):
#            for m in range(O):
#              I5[a,b,i,j]+=t1[a,m]*X5[m,b,i,j]
    X5=np.einsum("ei,mbej->mbij",t1,IABJ)
    I5=np.einsum("am,mbij->abij",t1,X5)
#Form I6
#    for a in range(V):
#      for i in range(O):
#        for j in range(O):
#          for m in range(O):
#            for e in range(V):
#              X6[m,a,i,j]+=t1[e,i]*IABJ[m,a,e,j]
#
#
#    for a in range(V):
#      for b in range(V):
#        for i in range(O):
#          for j in range(O):
#            for m in range(O):
#              I6[b,a,i,j]+=t1[b,m]*X6[m,a,i,j]
    I6=I5
#Form I7
#    for b in range(O,NB):
#      for i in range(O):
#        for j in range(O):
#          for m in range(O):
#            for e in range(O,NB):
#              X7[m,b,j,i]+=t1[e,j]*IABJ[m,b,e,i] #MO[m,b,e,i]
#
#
#    for a in range(O,NB):
#      for b in range(O,NB):
#        for i in range(O):
#          for j in range(O):
#            for m in range(O):
#              I7[a,b,j,i]+=t1[a,m]*X7[m,b,j,i]
    X7=np.einsum("ej,mbei->mbji",t1,IABJ)
    I7=np.einsum("am,mbji->abij",t1,X7)
#Form I8
#    for a in range(V):
#      for i in range(O):
#        for j in range(O):
#          for m in range(O):
#            for e in range(V):
#              X8[m,a,j,i]+=t1[e,j]*IABJ[m,a,e,i]
#
#
#    for a in range(V):
#      for b in range(V):
#        for i in range(O):
#          for j in range(O):
#            for m in range(O):
#              I8[b,a,j,i]+=t1[b,m]*X8[m,a,j,i]
    I8=I7
    #Equation (3)
    t2_f = np.transpose(IJAB, axes=(2,3,0,1)) - (1/2)*I1 + (1/2)*np.transpose(I1,axes=(1,0,2,3)) - (1/2)*( I3 - np.transpose(I3, axes=(0,1,3,2)) ) - (I5 - np.transpose(I5,axes=(1,0,2,3))) + I7 - np.transpose(I7, axes=(1,0,2,3))
    t2_f += np.einsum("be,aeij->abij",F_ae,t2,optimize=True) - np.einsum("ae,beij->abij",F_ae,t2,optimize=True) + np.einsum("ei,jeba->abij",t1,IABC,optimize=True) - np.einsum("ej,ieba->abij",t1,IABC,optimize=True)
    t2_f += (1/2)*np.einsum("efij,abef->abij",tau,W_abef,optimize=True)  
    t2_f -= np.einsum("mj,abim->abij",F_mi,t2,optimize=True) - np.einsum("mi,abjm->abij",F_mi,t2,optimize=True) + np.einsum("am,mbij->abij",t1,IAJK,optimize=True) - np.einsum("bm,maij->abij",t1,IAJK,optimize=True)
    t2_f += np.einsum("aeim,mbej->abij",t2,W_mbej,optimize=True) - np.einsum("beim,maej->abij",t2,W_mbej,optimize=True) - np.einsum("aejm,mbei->abij",t2,W_mbej,optimize=True) + np.einsum("bejm,maei->abij",t2,W_mbej,optimize=True)
    t2_f += (1/2)*np.einsum("abmn,mnij->abij",tau,W_mnij,optimize=True)
#    for a in range(V):
#      for b in range(V):
#        for i in range(O):
#          for j in range(O):
#            t2_f[a,b,i,j]=IJAB[i,j,a,b] #MO[i,j,a,b]
#            t2_f[a,b,i,j]+=(-1/2)*I1[a,b,i,j]+(1/2)*I1[b,a,i,j]-(1/2)*(I3[a,b,i,j]-I4[a,b,j,i]) - (I5[a,b,i,j]-I6[b,a,i,j]) + I7[a,b,j,i]-I8[b,a,j,i]
#            t2_f[a,b,i,j]+=  I7[a,b,j,i]-I8[b,a,j,i]
#            for e in range(V):
#             t2_f[a,b,i,j]+=t2[a,e,i,j]*F_ae[b,e]-t2[b,e,i,j]*F_ae[a,e] +t1[e,i]*ABCI[a,b,e,j]-t1[e,j]*ABCI[a,b,e,i]
#              for f in range(V):
#                              t2_f[a,b,i,j]+=(1/2)*tau[e,f,i,j]*W_abef[a,b,e,f]
#            for m in range(O):
#              t2_f[a,b,i,j]-= t2[a,b,i,m]*F_mi[m,j]-t2[a,b,j,m]*F_mi[m,i] +t1[a,m]*IAJK[m,b,i,j]-t1[b,m]*IAJK[m,a,i,j]
                                                                                  
#              for e in range(V):
#                  t2_f[a,b,i,j]+=t2[a,e,i,m]*W_mbej[m,b,e,j]-t2[b,e,i,m]*W_mbej[m,a,e,j] +(-1)*t2[a,e,j,m]*W_mbej[m,b,e,i] + t2[b,e,j,m]*W_mbej[m,a,e,i]
#              for n in range(O):
#                t2_f[a,b,i,j]+=(1/2)*tau[a,b,m,n]*W_mnij[m,n,i,j]
#            t2_f[a,b,i,j]=t2_f[a,b,i,j]/D2[a,b,i,j]
#    print("I7",np.shape(I1))
#    print(np.shape(np.transpose(I1,axes=(0,1,3,2))))
#    print("I1",np.shape(t2))
    #t2 = np.transpose(IJAB, axes=(2,3,0,1)) + (1/2)*I1 - (1/2)*np.transpose(I1,axes=(1,0,2,3)) - (1/2)*( I3 - np.transpose(I3, axes=(0,1,3,2))  ) - (I5 - np.transpose(I5,axes=(1,0,2,3))) + I7 - np.transpose(I7, axes=(0,1,3,2))
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
def CCSD(O, V, NB, Fock, t1, IJAB, T, tau):#MO, IJKL, ABCD, IABC, IJAB, IJKA, AIBC, IABJ, IJAK, IAJB, ABCI, IAJK, T, tau):
  E_Corr2_1 = np.einsum('ai,ia->', t1, Fock[:O, O:])
  E_Corr2_2 = 0.25 * np.einsum('abij,ijab->', tau, IJAB)
  E_Corr2 = E_Corr2_1 + E_Corr2_2
  
  return E_Corr2
