import numpy as np
import os
import sys
import re
#from main import F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, t1, t2


#Intermediates Needed for lambda equation solution
#MAYBE NEED TO SWAP INTERMEDIATE INDICES
def lamInts(O, NB, Fock, t1, t2, MO, T, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, lam1, lam2):
  if T==1:
#DEFINE G Intermediates
    G_ae=np.zeros((NB, NB))
    for a in range(O, NB):
      for e in range(O, NB):
        for m in range(O):
          for n in range(O):
            for f in range(O, NB):
              G_ae[a,e]-=(1/2)*t2[e,f,m,n]*lam2[m,n,a,f]

    G_mi=np.zeros((NB, NB))
    for m in range(O):
      for i in range(O):
        for m in range(O):
          for e in range(O, NB):
            for f in range(O, NB):
              G_mi[m,i]+=(1/2)*t2[e,f,m,n]*lam2[i,n,e,f]

#DEFINE Double tilde terms
    Wtt_iefn=np.zeros((NB, NB, NB, NB))
    for i in range(O):
      for e in range(O, NB):
        for f in range(O, NB):
          for n in range(O):
            Wtt_iefn[i,e,f,n]+=MO[i,e,f,n]
            for j in range(O):
              for g in range(O, NB):
                Wtt_iefn[i,e,f,n]-=t2[e,g,j,n]*MO[i,j,f,g]

    Wtt_nfam=np.zeros((NB, NB, NB, NB))
    for n in range(O):
      for f in range(O, NB):
        for a in range(O, NB):
          for m in range(O):
            Wtt_nfam+=MO[n,f,a,m]
            for o in range(O):
              for b in range(O, NB):
                Wtt_nfam[n,f,a,m]-=t2[b,f,o,m]*MO[n,o,a,b]

#EQ 17
    Ft_ea=np.zeros((NB,NB))
    for e in range(NB,O):
      for a in range(NB,O):
        Ft_ea[e,a]+=F_ae[e,a]
        for m in range(O):
          Ft_ea[e,a]-=t1[e,m]*F_me[m,a]
#EQ 18
    Ft_im=np.zeros((NB,NB))
    for i in range(O):
      for m in range(O):
        Ft_im[i,m]+=F_mi[i,m]
        for e in range(O,NB):
          Ft_im[i,m]-=(1/2)*t1[e,m]*F_me[i,e]
#EQ 24
    Wtt_mbej=np.zeros((NB, NB, NB, NB))
    for m in range(O):
      for b in range(O, NB):
        for e in range(O, NB):
          for j in range(O):
            Wtt_mbej[m,b,e,j]+=MO[m,b,e,j]
            for n in range(O):
              for f in range(O, NB):
                Wtt_mbej[m,b,e,j]+=t2[b,f,n,j]*MO[m,n,e,f] 
#EQ 19
    Wt_efab=np.zeros((NB, NB, NB, NB))
    for e in range(O,NB):
      for f in range(O,NB):
        for a in range(O,NB):
          for b in range(O,NB):
            Wt_efab[e,f,a,b]+=MO[e,f,a,b]
            for m in range(O):
              Wt_efab[e,f,a,b]-=t1[e,m]*MO[m,f,a,b] - t1[f,m]*MO[m,e,a,b]
              for n in range(O):
                Wt_efab[e,f,a,b]+=(1/2)*tau[e,f,m,n]*MO[m,n,a,b]
#EQ 20
    Wt_ijmn=np.zeros((NB, NB, NB, NB))
    for i in range(O):
      for j in range(O):
        for m in range(O):
          for n in range(O):
            Wt_ijmn[i,j,m,n]+=MO[i,j,m,n]
            for e in range(O,NB):
              Wt_ijmn[i,j,m,n]+=t1[e,m]*MO[i,j,e,n] - t1[e,n]*MO[i,j,e,m]
              for f in range(O,NB):
                Wt_ijmn[i,j,m,n]+=(1/2)*tau[e,f,m,n]*MO[e,f,i,j]
#EQ 21
    Wt_ejmb=np.zeros((NB, NB, NB, NB))
    for e in range(O, NB):
      for j in range(O):
        for m in range(O):
          for b in range(O, NB):
            Wt_ejmb[e,j,m,b]+=MO[e,j,m,b]
            for f in range(O, NB):
              Wt_ejmb[e,j,m,b]+=t1[f,m]*MO[e,j,f,b]
            for n in range(O):
              Wt_ejmb[e,j,m,b]+=t1[e,n]*MO[n,j,m,b]
              for f in range(O, NB):
                Wt_ejmb[e,j,m,b]+=(t2[e,f,m,n]-t1[f,m]*t1[e,n])*MO[n,j,f,b]
#EQ 22
    Wt_iemn=np.zeros((NB, NB, NB, NB))
    for i in range(O):
      for e in range(O, NB):
        for m in range(O):
          for n in range(O):
            Wt_iemn[i,e,m,n]+=MO[i,e,m,n]
            for f in range(O, NB):
              Wt_iemn[i,e,m,n]+=F_me[i,f]*t2[e,f,m,n]
              Wt_iemn[i,e,m,n]+=t1[f,m]*Wtt_iefn[i,e,f,n] - t1[f,n]*Wtt_iefn[i,e,f,m] #Check Wtt
              for g in range(O, NB):
                Wt_iemn[i,e,m,n]+=(1/2)*MO[i,e,f,g]*tau[f,g,m,n]
            for o in range(O):
              Wt_iemn[i,e,m,n]+=t1[e,o]*Wt_ijmn[i,o,m,n] 
              for f in range(O, NB):
                Wt_iemn[i,e,m,n]+=MO[i,o,m,f]*t2[e,f,n,o] - MO[i,o,n,f]*t2[e,f,m,o]
#???
    Wt_mnie=np.zeros((NB, NB, NB, NB))
    for m in range(O):
      for n in range(O):
        for i in range(O):
          for e in range(O, NB):
            Wt_mnie[m,n,i,e]=Wt_iemn[i,e,m,n]
#EQ 23
    Wt_efam=np.zeros((NB, NB, NB, NB))
    for e in range(O, NB):
      for f in range(O, NB):
        for a in range(O, NB):
          for m in range(O):
            Wt_efam[e,f,a,m]+=MO[e,f,a,m]
            for n in range(O):
              Wt_efam[e,f,a,m]+=F_me[n,a]*t2[e,f,m,n]
              Wt_efam[e,f,a,m]+=t1[e,n]*Wtt_nfam[n,f,a,m] - t1[f,n]*Wtt_nfam[n,e,a,m]
              for g in range(O, NB):
                Wt_efam[e,f,a,m]+=MO[e,n,a,g]*t2[f,g,m,n] - MO[f,n,a,g]*t2[e,g,m,n]
              for o in range(O):
                Wt_efam[e,f,a,m]+=(1/2)*MO[a,m,n,o]*tau[e,f,n,o]
            for g in range(O, NB):
              Wt_efam[e,f,a,m]+=t1[g,m]*Wt_efab[e,f,a,g]
  return G_ae, G_mi, Ft_ea, Ft_im, Wt_efab, Wt_ijmn, Wt_ejmb, Wt_iemn, Wt_mnie, Wt_efam      

#Lambda Amplitude equations 
#from Gauss et al, J. Chem. Phys. 95, 2623–2638 (1991) https://doi.org/10.1063/1.460915
#EQ 15
def lam1Eq(O, NB, Fock, t1, t2, MO, T, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, G_ae, G_mi, Ft_ea, Ft_im, Wt_efab, Wt_ijmn, Wt_ejmb, Wt_iemn, Wt_mnie, Wt_efam, D1, lam1, lam2):
  if T==1:
    lam1_f=np.zeros((NB, NB))
    for a in range(O, NB):
      for i in range(O):
        lam1_f[a,i]+=F_me[i,a]
        for e in range(O, NB):
          lam1_f[a,i]+=lam1[i,e]*Ft_ea[e,a]
          for f in range(O, NB):
            lam1_f[a,i]-=G_ae[e,f]*MO[e,i,f,a]
        for m in range(O):
          lam1_f[a,i]+=lam1[m,a]*Ft_im[i,m]
          for e in range(O, NB):
            lam1_f[a,i]+=lam1[m,e]*Wt_ejmb[e,i,m,a]
            for f in range(O, NB):
              lam1_f[a,i]+=lam2[i,m,e,f]*Wt_efam[e,f,a,m]
              lam1_f[a,i]+=G_ae[f,e]*t1[f,m]*MO[i,m,a,e]
            for n in range(O):
              lam1_f[a,i]+=G_mi[m,n]*t1[e,m]*MO[i,m,a,e]
          for n in range(O):
            lam1_f[a,i]-=G_mi[m,n]*MO[m,i,n,a]
            for e in range(O, NB):
              lam1_f[a,i]+=lam2[m,n,a,e]*Wt_iemn[m,n,i,e]#Need Wt_mnie ?????
        lam1_f[a,i]=lam1_f[a,i]/D1[a,i]
  return lam1_f
#EQ 16
def lam2Eq(O, NB, Fock, t1, t2, MO, T, tau_tilde, tau, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, G_ae, G_mi, Ft_ea, Ft_im, Wt_efab, Wt_ijmn, Wt_ejmb, Wt_iemn, Wt_mnie, Wt_efam, D2, lam1, lam2):
  if T==1:
    lam2_f=np.zeros((NB, NB, NB, NB))
    for a in range(O, NB):
      for b in range(O, NB):
        for i in range(O):
          for j in range(O):
            lam2_f[a,b,i,j]+=MO[i,j,a,b]
            for e in range(O, NB):
              lam2_f[a,b,i,j]+=lam2[a,e,i,j]*Ft_ea[e,b] + lam2[b,e,i,j]*Ft_ea[e,a]
              lam2_f[a,b,i,j]+=MO[i,j,a,e]*G_ae[b,e] - MO[i,j,b,e]*G_ae[a,e]
              for m in range(O):
                lam2_f[a,b,i,j]+= (-1)*MO[i,j,a,e]*lam1[m,b]*t1[e,m] + MO[i,j,b,e]*lam1[m,a]*t1[e,m]
              for f in range(O, NB):
                lam2_f[a,b,i,j]+=(1/2)*lam2[i,j,e,f]*Wt_efab[e,f,a,b]
            for m in range(O):
              lam2_f[a,b,i,j]-=lam2[i,m,a,b]*Ft_im[j,m] - lam2[j,m,a,b]*Ft_im[i,m]
              for n in range(O):
                lam2_f[a,b,i,j]+=(1/2)*lam2[m,n,a,b]*Wt_ijmn[i,j,m,n]
              for e in range(O, NB):
                lam2_f[a,b,i,j]+=lam2[i,m,a,e]*Wt_ejmb[e,j,m,b]-lam2[j,m,a,e]*Wt_ejmb[e,i,m,b]-lam2[i,m,b,e]*Wt_ejmb[e,j,m,a]+lam2[j,m,b,e]*Wt_ejmb[e,i,m,a]
  return lam2_f 
