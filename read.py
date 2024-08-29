import numpy as np
import os
import sys
import re
import time

##########################################################################
#Get O, NB, SCF energy, MO coefficients, Orbital energies ################
##########################################################################

def getFort(molecule, log):
  mol=sys.argv[1]
#O, V, NB
  O=0
  V=0
  orbs=[]
  with open(f"{mol}_txts/occ.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split())
  del text[0]
  ind=0
  for i in range(len(text)):
    for j in range(len(text[i])):
      orbs.append(text[i][j])
      if int(float(orbs[ind]))==2:
        O+=1
      if int(float(orbs[ind]))==0:
        V+=1 
      ind +=1 
  NB=O+V
#SCF Energy
  os.system(f"grep -i 'scf done' {log} > scf.txt")
  with open(f"{mol}_txts/scf.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split())
 
  scfE=float(text[0][4])
  os.system("rm scf.txt")
#Fock
  OE=[]
  with open(f"{mol}_txts/orb.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split())
  del text[0]
  for i in range(len(text)):
    for j in range(len(text[i])):
      OE.append(float(text[i][j]))
  OE=np.array(OE)
  Fock=np.zeros((NB*2))
  # for i in range(NB*2):
  #   Fock[i]=OE[i//2]
#  Fock[:NB] = OE[:NB]
#  Fock[NB:] = OE[:NB]
#  Fock=np.diag(Fock)

  Fock[:O] = OE[:O]
  Fock[O:2*O] = OE[:O]
  Fock[2*O:2*O+V] = OE[O:NB]
  Fock[2*O+V:] = OE[O:NB] 
  Fock=np.diag(Fock)

#MO Coefficients
#  Coeff=np.zeros((NB,NB))
#  with open(f"{mol}_txts/mocoef.txt","r") as reader:
#    text=[]
#    for line in reader:
#      text.append(line.split())
#    for i in range(len(Coeff)):
#      for j in range(len(Coeff)):
#        Coeff[i][j]=float(text[i+2][j+1].replace("D","E"))
#  Coeff=np.transpose(Coeff)
  #print(Coeff)
  Coeff=[[] for _ in range(NB)]
  with open(f"{mol}_txts/mocoef.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split())
  del text[0]
  bad=[]
  for i in range(len(text)):
    n=0
    for j in range(len(text[i])):
      if "D" not in text[i][j]:
        n+=1
    if n==len(text[i]):
      bad.append(i)
  for i in range(len(text)):
    if i not in bad:
      for j in range(len(text[i])-1):
        Coeff[int(text[i][0])-1].append(float(text[i][j+1].replace("D","E")))
  Coeff=np.transpose(np.array(Coeff))
#  
  return O, V, NB, scfE, Fock, Coeff




########################################################
#####Get 2e integrals###################################
########################################################
def get2e(AOInt, log):
  mol=sys.argv[1]
  with open(f"{mol}_txts/twoeint.txt", "r") as reader:
    for line in reader:
      text=line.split()
      if "I=" and "J=" and "K=" and "L=" in text:
        I=int(text[1])-1
        J=int(text[3])-1
        K=int(text[5])-1
        L=int(text[7])-1
        AOInt[I,J,K,L]=float(text[9].replace("D", "E"))
        AOInt[J,I,K,L]=AOInt[I,J,K,L]
        AOInt[I,J,L,K]=AOInt[I,J,K,L]
        AOInt[J,I,L,K]=AOInt[I,J,K,L]

        AOInt[K,L,I,J]=AOInt[I,J,K,L]
        AOInt[L,K,I,J]=AOInt[I,J,K,L]
        AOInt[K,L,J,I]=AOInt[I,J,K,L]
        AOInt[L,K,J,I]=AOInt[I,J,K,L]
  return AOInt



#########################################################
####### Change to MO Basis ##############################
#########################################################
def conMO(O, V, NB, Fock, Coeff, AOInt, IJKL, ABCD, IABC, IJAB, IJKA, IAJB):
#  O, V, NB, scfE, Fock, Coeff=getFort(molecule, log)

#  #Initialize temporary arrs
  # temp = np.zeros((NB,NB,NB,NB))
  # temp2 = np.zeros((NB,NB,NB,NB))
  # temp3= np.zeros((NB,NB,NB,NB))
  #Transform AO to MO
  # print(f"AOInt \n")
  # for i in range(NB):
  #   for j in range(NB):
  #     for k in range(NB):
  #       for l in range(NB):
  #         print(f"{i,j,k,l,AOInt[i,j,k,l]}")
  temp = np.einsum('im,mjkl->ijkl',Coeff,AOInt,optimize=True)
  temp2 = np.einsum('jm,imkl->ijkl',Coeff,temp,optimize=True)
  temp = np.einsum('km,ijml->ijkl',Coeff,temp2,optimize=True)
  twoE = np.einsum('lm,ijkm->ijkl',Coeff,temp,optimize=True)
  # print(f"twoE \n")
  # for i in range(NB):
  #   for j in range(NB):
  #     for k in range(NB):
  #       for l in range(NB):
  #         print(f"{i,j,k,l,twoE[i,j,k,l]}")
  # for i in range(NB):
  #   for m in range(NB):
  #     temp[i,:,:,:] += Coeff[i,m]*AOInt[m,:,:,:]
  #   for j in range(NB):
  #     for n in range(NB):
  #       temp2[i,j,:,:] += Coeff[j,n]*temp[i,n,:,:]
  #     for k in range(NB):
  #       for o in range(NB):
  #         temp3[i,j,k,:] += Coeff[k,o]*temp2[i,j,o,:]
  #       for l in range(NB):
  #         for p in range(NB):
  #           twoE[i,j,k,l] += round(Coeff[l,p]*temp3[i,j,k,p],16)

  # #Change to spin-orbital form
  # for p in range(NB*2):
  #   for q in range(NB*2):
  #     for r in range(NB*2):
  #       for s in range(NB*2):
  #         coulomb=twoE[p//2,r//2,q//2,s//2]*(r%2==p%2)*(q%2==s%2)
  #         exchange=twoE[p//2,s//2,q//2,r//2]*(p%2==s%2)*(q%2==r%2)
  #         MO[p,q,r,s]=round(coulomb-exchange,16)

  MO = np.zeros((2*NB,2*NB,2*NB,2*NB))
  for p in range(NB):
    for q in range(NB):
      for r in range(NB):
        for s in range(NB):
          coulomb = twoE[p,r,q,s]
          exchange = twoE[p,s,q,r]
          MO[p,q,r,s] = round(coulomb-exchange,16)
          MO[p+NB,q+NB,r+NB,s+NB] = round(coulomb-exchange,16)
          MO[p,q+NB,r,s+NB] = round(coulomb,16)
          MO[p,q+NB,r+NB,s] = round(-exchange,16)
          MO[p+NB,q,r+NB,s] = round(coulomb,16)
          MO[p+NB,q,r,s+NB] = round(-exchange,16)
          
# IJKL  
  for i in range(O):
    for j in range(O):
      for k in range(O):
        for l in range(O):
          IJKL[i,j,k,l] = MO[i,j,k,l]
          IJKL[i+O,j+O,k+O,l+O] = MO[i+NB, j+NB, k+NB, l+NB]
          IJKL[i+O, j, k+O, l] = MO[i+NB, j, k+NB, l]
          IJKL[i, j+O, a, l+O] = MO[i,j+NB,k,l+NB]
          IJKL[i+O,j,k,l+O] = MO[i+NB,j,k,l+NB]
          IJKL[i,j+O,k+O,l] = MO[i,j+NB,k+NB,l]
# ABCD          
  for a in range(V):
    for b in range(V):
      for c in range(V):
        for d in range(V):
          ABCD[a,b,c,d] = MO[a+O,b+O,c+O,d+O]
          ABCD[a+V,b+V,c+V,d+V] = MO[a+O+NB,b+O+NB,c+O+NB,d+O+NB]
          ABCD[a+V,b,c+V,d] = MO[a+O+NB,b+O,c+O+NB,d+O]
          ABCD[a,b+V,c,d+V] = MO[a+O,b+O+NB,c+O,d+O+NB]
          ABCD[a+V,b,c,d+V] = MO[a+O+NB,b+O,c+O,d+O+NB]
          ABCD[a,b+V,c+V,d] = MO[a+O,b+O+NB,c+O+NB,d+O]
# IABC          
  for i in range(O):
    for a in range(V):
      for b in range(V):
        for c in range(V):
          IABC[i,a,b,c] = MO[i,a+O,b+O,c+O]
          IABC[i+O,a+V,b+V,c+V] = MO[i+NB,a+O+NB,b+O+NB,c+O+NB]
          IABC[i+O,a,b+V,c] = MO[i+NB,a+O,b+O+NB,c+O]
          IABC[i,a+V,b,c+V] = MO[i,a+O+NB,b,c+O+NB]
          IABC[i+O,a,b,c+V] = MO[i+NB,a,b,c+O+NB]
          IABC[i,a+V,b+V,c] = MO[i,a+O+NB,b+O+NB,c]
# IJAB          
  for i in range(O):
    for j in range(O):
      for a in range(V):
        for b in range(V):
          IJAB[i,j,a,b] = MO[i,j,a+O,b+O]
          IJAB[i+O,j+O,a+V,b+V] = MO[i+NB,j+NB,a+O+NB,b+O+NB]
          IJAB[i+O,j,a+V,b] = MO[i+NB,j,a+O+NB,b+O]
          IJAB[i,j+O,a,b+V] = MO[i,j+NB,a+O,b+O+NB]
          IJAB[i+O,j,a,b+V] = MO[i+NB,j,a+O,b+O+NB]
          IJAB[i,j+O,a+V,b] = MO[i,j+NB,a+O+NB,b+O]
# IJKA
  for i in range(O):
    for j in range(O):
      for k in range(O):
        for a in range(V):
          IJKA[i,j,k,a] = MO[i,j,k,a+O]
          IJKA[i+O,j+O,k+O,a+V] = MO[i+NB,j+NB,k+NB,a+O+NB]
          IJKA[i+O,j,k+O,a] = MO[i+NB,j,k+NB,a]
          IJKA[i,j+O,k,a+V] = MO[i,j+NB,k,a+O+NB]
          IJKA[i+O,j,k,a+V] = MO[i+NB,j,k,a+O+NB]
          IJKA[i,j+O,k+O,a] = MO[i,j+NB,k+NB,a]
  # for a in range(V):
  #   for i in range(O):
  #     for b in range(V):
  #       for c in range(V):
  #         AIBC[a,i,b,c]=MO[a+O,i,b+O,c+O]
  # for i in range(O):
  #   for a in range(V):
  #     for b in range(V):
  #       for j in range(O):
  #         IABJ[i,a,b,j]=MO[i,a+O,b+O,j]
  # for i in range(O):
  #   for j in range(O):
  #     for a in range(V):
  #       for k in range(O):
  #         IJAK[i,j,a,k]=MO[i,j,a+O,k]
# IAJB  
  for i in range(O):
    for a in range(V):
      for j in range(O):
        for b in range(V):
          IAJB[i,a,j,b] = MO[i,a+O,j,b+O]
          IAJB[i+O,a+V,j+O,b+V] = MO[i+NB,a+O+NB,j+NB,b+O+NB]
          IAJB[i+O,a,j+O,b] = MO[i+NB,a,j+NB,b]
          IAJB[i,a+V,j,b+V] = MO[i,a+O+NB,j,b+O+NB]
          IAJB[i+O,a,j,b+V] = MO[i+NB,a,j,b+O+NB]
          IAJB[i,a+V,j+O,b] = MO[i,a+O+NB,j+NB,b]
  # for a in range(V):
  #   for b in range(V):
  #     for c in range(V):
  #       for i in range(O):
  #         ABCI[a,b,c,i]=MO[a+O,b+O,c+O,i]
  # for i in range(O):
  #   for a in range(V):
  #     for j in range(O):
  #       for k in range(O):
  #         IAJK[i,a,j,k]=MO[i,a+O,j,k]
  del MO, AOInt, twoE, temp, temp2
  return IJKL, ABCD, IABC, IJAB, IJKA, IAJB
