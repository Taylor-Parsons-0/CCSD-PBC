import numpy as np
import os
import sys
import re
import time
from stuff import getFort, get2e, conMO

#Initialize arrays
#Define molecule
if len(sys.argv)==2:
  molecule=sys.argv[1]
else:
  print("MISSING MOLECULE NAME")
  exit()
log=f"{molecule}.log"
O, NB, scfE, Fock, Coeff=getFort(molecule, log)
coul=np.zeros((NB,NB,NB,NB))
exc=np.zeros((NB,NB,NB,NB))
OE=np.zeros((NB))
AOInt=np.zeros((NB, NB, NB, NB))
MO=np.zeros((NB*2, NB*2, NB*2, NB*2))
delta=np.identity(NB*2)
twoE=np.zeros((NB, NB, NB, NB))

#####################################################################
#Tau equations ######################################################
#####################################################################
def tau_tildeEq(O, NB, Fock, t1, t2, MO, T):
  if T==1:
    tau_tilde=np.zeros((NB, NB, NB, NB))
    #Equation (10)
    for a in range(O, NB):
      for b in range(O, NB):
        for i in range(O):
          for j in range(O):
            tau_tilde[a,b,i,j]=t2[a,b,i,j]+(1/2)*(t1[a,i]*t1[b,j]-t1[b,i]*t1[a,j])
  return tau_tilde

def tauEq(O, NB, Fock, t1, t2, MO, T):
  if T==1:
    tau=np.zeros((NB, NB, NB, NB))
    #Equation (11)
    for a in range(O, NB):
      for b in range(O, NB):
        for i in range(O):
          for j in range(O):
            tau[a,b,i,j]=t2[a,b,i,j]+t1[a,i]*t1[b,j]-t1[b,i]*t1[a,j]
  return tau







########################################################################
### Begin F and W intermediates ########################################
########################################################################
def intermediateEqs(O, NB, Fock, t1, t2, MO, T, tau_tilde, tau):
  print("O:",O)
  print("NB:",NB)
  print("delta:",np.shape(delta))
  print("Fock:",np.shape(Fock))
  print("t1:",np.shape(t1))
  print("MO:",np.shape(MO))
  print("tau_tilde:",np.shape(tau_tilde))

  if T==1:
    #Equation (4)
    F_ae=np.zeros((NB, NB))
    for a in range(O, NB):
      for e in range(O, NB):
        F_ae[a,e]=(1-delta[a,e])*Fock[a,e]
        for m in range(O):
          F_ae[a,e]-= (1/2)*Fock[m,e]*t1[a,m]
          for f in range(O, NB):
            F_ae[a,e]+=t1[f,m]*MO[m,a,f,e]
            for n in range(O):
              F_ae[a,e]-= (1/2)*tau_tilde[a,f,m,n]*MO[m,n,e,f]

    #Equation (5)
    F_mi=np.zeros((NB, NB))
    for m in range(O):
      for i in range(O):
        F_mi[m,i]=(1-delta[m,i])*Fock[m,i]
        for e in range(O, NB):
          F_mi[m,i]+=(1/2)*t1[e,i]*Fock[m,e]
          for n in range(O):
            F_mi[m,i]+=t1[e,n]*MO[m,n,i,e]
            for f in range(O, NB):
              F_mi[m,i]+=(1/2)*tau_tilde[e,f,i,n]*MO[m,n,e,f]
    #Equation (6)
    F_me=np.zeros((NB, NB))
    for m in range(O):
      for e in range(O, NB):
        F_me[m,e]=Fock[m,e]
        for n in range(O):
          for f in range(O, NB):
            F_me[m,e]+=t1[f,n]*MO[m,n,e,f]

    #Equation (7)
    W_mnij=np.zeros((NB, NB, NB, NB))
    for m in range(O):
      for n in range(O):
        for i in range(O):
          for j in range(O):
            W_mnij[m,n,i,j]=MO[m,n,i,j]
            for e in range(O, NB):
              W_mnij[m,n,i,j]+=t1[e,j]*MO[m,n,i,e]-t1[e,i]*MO[m,n,j,e]
              for f in range(O, NB):
                W_mnij[m,n,i,j]+=(1/4)*tau[e,f,i,j]*MO[m,n,e,f]

    #Equation (8)
    W_abef=np.zeros((NB, NB, NB, NB))
    for a in range(O, NB):
      for b in range(O, NB):
        for e in range(O, NB):
          for f in range(O, NB):
            W_abef[a,b,e,f]=MO[a,b,e,f]
            for m in range(O):
              W_abef[a,b,e,f]-= t1[b,m]*MO[a,m,e,f]-t1[a,m]*MO[b,m,e,f]
              for n in range(O):
                W_abef[a,b,e,f]+=(1/4)*tau[a,b,m,n]*MO[m,n,e,f]
    
    #Equation (9)
    W_mbej=np.zeros((NB, NB, NB, NB))
    for m in range(O):
      for b in range(O, NB):
        for e in range(O, NB):
          for j in range(O):
            W_mbej[m,b,e,j]=MO[m,b,e,j]
            for f in range(O, NB):
              W_mbej[m,b,e,j]+=t1[f,j]*MO[m,b,e,f]
            for n in range(O):
              W_mbej[m,b,e,j]-=t1[b,n]*MO[m,n,e,j]
              for f in range(O, NB):
                W_mbej[m,b,e,j]-= ((1/2)*t2[f,b,j,n]+t1[f,j]*t1[b,n])*MO[m,n,e,f]
  return F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej



##################################################################################
#Solve for T equations ###########################################################
##################################################################################

#Final t1
#t1_f=np.zeros((NB, NB))
def t1Eq(O, NB, Fock, t1, t2, MO, T, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D1):
  if T==1:
    t1_f=np.zeros((NB, NB))
  #Equation (2)
    for a in range(O, NB):
      for i in range(O):
        t1_f[a,i]=Fock[i,a]
        for e in range(O, NB):
          t1_f[a,i]+=t1[e,i]*F_ae[a,e]
        for m in range(O):
          t1_f[a,i]-= t1[a,m]*F_mi[m,i]
          for e in range(O, NB):
            t1_f[a,i]+=t2[a,e,i,m]*F_me[m,e]
            for f in range(O, NB):
              t1_f[a,i]-= (1/2)*t2[e,f,i,m]*MO[m,a,e,f]
            for n in range(O):
              t1_f[a,i]-= (1/2)*t2[a,e,m,n]*MO[n,m,e,i]
        for n in range(O):
          for f in range(O, NB):
            t1_f[a,i]-= t1[f,n]*MO[n,a,i,f]
        t1_f[a,i]=t1_f[a,i]/D1[a,i]
    #print(t1)
  return t1_f

#Final t2
def t2Eq(O, NB, Fock, t1, t2, MO, T, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, D2):
  if T==1:
#Initialize T2
    t2_f=np.zeros((NB, NB, NB, NB))
#Form 8 N^5 Intermediates
    X1=np.zeros((NB, NB, NB, NB))
    X2=np.zeros((NB, NB, NB, NB))
    X3=np.zeros((NB, NB, NB, NB))
    X4=np.zeros((NB, NB, NB, NB))
    X5=np.zeros((NB, NB, NB, NB))
    X6=np.zeros((NB,NB,NB,NB))
    X7=np.zeros((NB,NB,NB,NB))
    X8=np.zeros((NB, NB, NB, NB))

    I1=np.zeros((NB, NB, NB, NB))
    I2=np.zeros((NB, NB, NB, NB))
    I3=np.zeros((NB, NB, NB, NB))
    I4=np.zeros((NB, NB, NB, NB))
    I5=np.zeros((NB, NB, NB, NB))
    I6=np.zeros((NB, NB, NB, NB))
    I7=np.zeros((NB, NB, NB, NB))
    I8=np.zeros((NB, NB, NB, NB))

#Form I1
    for a in range(O,NB):
      for i in range(O):
        for j in range(O,NB):
          for e in range(O,NB):
            for m in range(O):
              X1[a,m,i,j]+=F_me[m,e]*t2[a,e,i,j]
    for a in range(O,NB):
      for b in range(O,NB):
        for i in range(O):
          for j in range(O):
            for m in range(O):
              I1[a,b,i,j]+=t1[b,m]*X1[a,m,i,j]
    print(np.sum(abs(F_me)))
#Form I2
    for b in range(O,NB):
      for i in range(O):
        for j in range(O):
          for e in range(O,NB):
            for m in range(O):
              X2[b,m,i,j]+=F_me[m,e]*t2[b,e,i,j]
 

    for a in range(O,NB):
      for b in range(O,NB):
        for i in range(O):
          for j in range(O):
            for m in range(O):
              I2[b,a,i,j]+=t1[a,m]*X2[b,m,i,j]
#Form I3 
    for a in range(O,NB):
      for b in range(O,NB):
        for i in range(O):
          for m in range(O):
            for e in range(O,NB):
              X3[a,b,i,e]+=F_me[m,e]*t2[a,b,i,m]
 
 
    for a in range(O,NB):
      for b in range(O,NB):
        for i in range(O):
          for j in range(O):
            for e in range(O,NB):
              I3[a,b,i,j]+=t1[e,j]*X3[a,b,i,e]
#Form I4
    for a in range(O,NB):
      for b in range(O,NB):
        for j in range(O):
          for m in range(O):
            for e in range(O,NB):
              X4[a,b,j,e]+=F_me[m,e]*t2[a,b,j,m]
 
 
    for a in range(O,NB):
      for b in range(O,NB):
        for i in range(O):
          for j in range(O):
            for e in range(O,NB):
              I4[a,b,j,i]+=t1[e,i]*X4[a,b,j,e]


#Form I5
    for b in range(O,NB):
      for i in range(O):
        for j in range(O):
          for m in range(O):
            for e in range(O,NB):
              X5[m,b,i,j]+=t1[e,i]*MO[m,b,e,j]
 
 
    for a in range(O,NB):
      for b in range(O,NB):
        for i in range(O):
          for j in range(O):
            for m in range(O):
              I5[a,b,i,j]+=t1[a,m]*X5[m,b,i,j]
#Form I6
    for a in range(O,NB):
      for i in range(O):
        for j in range(O):
          for m in range(O):
            for e in range(O,NB):
              X6[m,a,i,j]+=t1[e,i]*MO[m,a,e,j]


    for a in range(O,NB):
      for b in range(O,NB):
        for i in range(O):
          for j in range(O):
            for m in range(O):
              I6[b,a,i,j]+=t1[b,m]*X6[m,a,i,j]
#Form I7
    for b in range(O,NB):
      for i in range(O):
        for j in range(O):
          for m in range(O):
            for e in range(O,NB):
              X7[m,b,j,i]+=t1[e,j]*MO[m,b,e,i]


    for a in range(O,NB):
      for b in range(O,NB):
        for i in range(O):
          for j in range(O):
            for m in range(O):
              I7[a,b,j,i]+=t1[a,m]*X7[m,b,j,i]
#Form I8
    for a in range(O,NB):
      for i in range(O):
        for j in range(O):
          for m in range(O):
            for e in range(O,NB):
              X8[m,a,j,i]+=t1[e,j]*MO[m,a,e,i]


    for a in range(O,NB):
      for b in range(O,NB):
        for i in range(O):
          for j in range(O):
            for m in range(O):
              I8[b,a,j,i]+=t1[b,m]*X8[m,a,j,i]
    #print(np.sum(abs(I1)))
    #print(np.sum(abs(I2)))
    #Equation (3)
    for a in range(O, NB):
      for b in range(O, NB):
        for i in range(O):
          for j in range(O):
            t2_f[a,b,i,j]=MO[i,j,a,b]
            #print(np.sum(abs(t2_f)))
            t2_f[a,b,i,j]+=(-1/2)*I1[a,b,i,j]+(1/2)*I2[b,a,i,j]-(1/2)*(I3[a,b,i,j]-I4[a,b,j,i]) - (I5[a,b,i,j]-I6[b,a,i,j]) + I7[a,b,j,i]-I8[b,a,j,i]
            for e in range(O, NB):
              t2_f[a,b,i,j]+=t2[a,e,i,j]*F_ae[b,e]-t2[b,e,i,j]*F_ae[a,e] +t1[e,i]*MO[a,b,e,j]-t1[e,j]*MO[a,b,e,i]
              for f in range(O, NB):
                              t2_f[a,b,i,j]+=(1/2)*tau[e,f,i,j]*W_abef[a,b,e,f]
            for m in range(O):
              t2_f[a,b,i,j]-= t2[a,b,i,m]*F_mi[m,j]-t2[a,b,j,m]*F_mi[m,i] +t1[a,m]*MO[m,b,i,j]-t1[b,m]*MO[m,a,i,j]
              for e in range(O, NB):
                  t2_f[a,b,i,j]+=t2[a,e,i,m]*W_mbej[m,b,e,j]-t2[b,e,i,m]*W_mbej[m,a,e,j] +(-1)*t2[a,e,j,m]*W_mbej[m,b,e,i] + t2[b,e,j,m]*W_mbej[m,a,e,i]
              for n in range(O):
                t2_f[a,b,i,j]+=(1/2)*tau[a,b,m,n]*W_mnij[m,n,i,j]
            t2_f[a,b,i,j]=t2_f[a,b,i,j]/D2[a,b,i,j]
  return t2_f





#########################################################
#Solve for CCSD energy###################################
#########################################################
#Equation 1
def CCSD(O, NB, Fock, t1, t2, MO, T, tau):
  E_Corr2=0
  for i in range(O):
    for a in range(O,NB):
      E_Corr2+=t1[a,i]*Fock[i,a]
      #print(E_Corr2)
      for j in range(O):
        for b in range(O,NB):
          E_Corr2+=(1/4)*tau[a,b,i,j]*MO[i,j,a,b]
          #print(NB-O)
  #print(E_Corr2)
  return E_Corr2
