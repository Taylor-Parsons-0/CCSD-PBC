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
  for i in range(len(text)):
    for j in range(len(text[i])):
      orbs.append(text[i][j])
      if int(float(orbs[j]))==2:
        O+=1
      if int(float(orbs[j]))==0:
        V+=1
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
  for i in range(NB*2):
    Fock[i]=OE[i//2]
  Fock=np.diag(Fock)
#MO Coefficients
  Coeff=np.zeros((NB,NB))
  with open(f"{mol}_txts/mocoef.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split())
    for i in range(len(Coeff)):
      for j in range(len(Coeff)):
        Coeff[i][j]=float(text[i+2][j+1].replace("D","E"))
  Coeff=np.transpose(Coeff)
  #print(Coeff)
  

#  os.system(f"grep -i 'occ. eigenvalue' {molecule}.log > OV.txt")
#  with open("OV.txt","r") as reader:
#    ov=[]
#    for line in reader:
#      ov.append(line.split())
#  ov=ov[0][4:]
#  O=len(ov)
#
#  #Setting search routines
#  getNB=re.compile("NBsUse=")
#  get2e=re.compile("Dumping Two-electron integrals")
#  end=re.compile("Leave Link  316")
# 
#  #Getting number of basis functions
#  with open(f"{log}", "r") as reader:
#    for line in reader:
#      if getNB.search(line):
#        NB=(int(line.split()[1]))
# 
#  #Creating new file from fort.7
#  os.system(f"cp fort.7 fort7{molecule}.txt")
# 
#  #Reformatting file for use with code
#  with open(f"fort7{molecule}.txt", "r") as reader:
#    replaced1=reader.read().replace("D", "E")
#    replaced2=replaced1.replace("-", " -")
#    replaced3=replaced2.replace("E -", "E-")
# 
#  with open(f"fort7{molecule}.txt", "w") as writer:
#    writer.write(replaced3)
# 
#  #Get SCF Energy
#  os.system(f"grep -i 'scf done' {log} > scf.txt")
#  with open("scf.txt","r") as reader:
#    text=[]
#    for line in reader:
#      text.append(line.split())
# 
#  scfE=float(text[0][4])
# 
#  #Get OE Values
#  os.system(f"grep -i 'alpha' fort7{molecule}.txt > OEs.txt")
#  with open("OEs.txt", "r") as reader:
#    OE_list=[]
#    for line in reader:
#      text=line.split()
#      OE_list.append(text)
# 
#  OE=np.array([float(OE_list[i][4]) for i in range(len(OE_list))])
#  #Fock matrix elements from orbital energies
#  Fock=np.zeros((NB*2))
#  for i in range(NB*2):
#    Fock[i]=OE[i//2]
#  Fock=np.diag(Fock)
# 
#  #Get MO Coefficients
#  C_list=[]
#  with open(f"fort7{molecule}.txt", "r") as reader:
#    for line in reader:
#      text=line.split()
#      if len(text)>2 and text[2]=="MO":
#        pass
#      else:
#        for i in range(len(text)):
#          C_list.append((text[i]))
#  C_list.pop(0)
#  C_list=[float(C_list[i]) for i in range(len(C_list))]
# 
#  Coeff=np.array([C_list[i:i+NB] for i in range(0, NB*NB, NB)])
#  os.system("rm    OEs.txt     OV.txt      fort7H2.txt     scf.txt")
  return O, NB, scfE, Fock, Coeff




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
def conMO(twoE, MO, Coeff, NB, AOInt):
  #Initialize temporary arrs
  temp = np.zeros((NB,NB,NB,NB))
  temp2 = np.zeros((NB,NB,NB,NB))
  temp3= np.zeros((NB,NB,NB,NB))
  #Transform AO to MO
  for i in range(NB):
    for m in range(NB):
      temp[i,:,:,:] += Coeff[i,m]*AOInt[m,:,:,:]
    for j in range(NB):
      for n in range(NB):
        temp2[i,j,:,:] += Coeff[j,n]*temp[i,n,:,:]
      for k in range(NB):
        for o in range(NB):
          temp3[i,j,k,:] += Coeff[k,o]*temp2[i,j,o,:]
        for l in range(NB):
          for p in range(NB):
            twoE[i,j,k,l] += round(Coeff[l,p]*temp3[i,j,k,p],16)

  #Change to spin-orbital form
  for p in range(NB*2):
    for q in range(NB*2):
      for r in range(NB*2):
        for s in range(NB*2):
          coulomb=twoE[p//2,r//2,q//2,s//2]*(r%2==p%2)*(q%2==s%2)
          exchange=twoE[p//2,s//2,q//2,r//2]*(p%2==s%2)*(q%2==r%2)
          MO[p,q,r,s]=round(coulomb-exchange,16)
  #print(np.sum(abs(MO)))
  O=1
  V=3
  IJAB=np.zeros((O,O,V,V))
#  for i in range(O):
#    for j in range(O):
#      for a in range(V):
#        for b in range(V):
#          print(MO[i,j,a+O,b+O])
  return MO
