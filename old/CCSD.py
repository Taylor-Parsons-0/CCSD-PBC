import numpy as np
import os
import sys
import re
import time
#from funcs import tau_tildeEq, tauEq 

#Define arrays to be used in routine
coul=np.zeros((NB,NB,NB,NB))
exc=np.zeros((NB,NB,NB,NB))
OE=np.zeros((NB))
Fock=np.zeros((NB*2))
AOInt=np.zeros((NB, NB, NB, NB))
MO=np.zeros((NB*2, NB*2, NB*2, NB*2))
delta=np.identity(NB*2)
twoE=np.zeros((NB, NB, NB, NB))

E_lis=[]  
t2_flat_ls=[]

#Molecule
if len(sys.argv)==2:
  name=molecule
else:
  print("MISSING MOLECULE NAME")
  exit()


#Occupied
def getOcc():
  os.system(f"grep -i 'occ. eigenvalue' {molecule}.log > OV.txt")
  with open("OV.txt","r") as reader:
    ov=[]
    for line in reader:
      ov.append(line.split())
  ov=ov[0][4:]
  O=len(ov)
  return O

#Get NB, SCF energy, MO coefficients, Orbital energies 
def getFort():
  #Setting search routines
  getNB=re.compile("NBsUse=")
  get2e=re.compile("Dumping Two-electron integrals")
  end=re.compile("Leave Link  316")
  
  #Getting number of basis functions
  with open(f"{log}", "r") as reader:
    for line in reader:
      if getNB.search(line):
        NB=(int(line.split()[1]))
  
  #Creating new file from fort.7
  os.system(f"cp fort.7 fort7{molecule}.txt")
  
  #Reformatting file for use with code
  with open(f"fort7{molecule}.txt", "r") as reader:
    replaced1=reader.read().replace("D", "E")
    replaced2=replaced1.replace("-", " -")
    replaced3=replaced2.replace("E -", "E-")
  
  with open(f"fort7{molecule}.txt", "w") as writer:
    writer.write(replaced3)
  
  #Get SCF Energy
  os.system(f"grep -i 'scf done' {log} > scf.txt")
  with open("scf.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split())
  
  scfE=float(text[0][4])
  
  #Get OE Values
  os.system(f"grep -i 'alpha' fort7{molecule}.txt > OEs.txt")
  with open("OEs.txt", "r") as reader:
    OE_list=[]
    for line in reader:
      text=line.split()
      OE_list.append(text)
  
  OE=np.array([float(OE_list[i][4]) for i in range(len(OE_list))])
  
  #Get MO Coefficients
  C_list=[]
  with open(f"fort7{molecule}.txt", "r") as reader:
    for line in reader:
      text=line.split()
      if len(text)>2 and text[2]=="MO":
        pass
      else:
        for i in range(len(text)):
          C_list.append((text[i]))
  C_list.pop(0)
  C_list=[float(C_list[i]) for i in range(len(C_list))]
  
  Coeff=np.array([C_list[i:i+NB] for i in range(0, NB*NB, NB)])
  return NB, scfE, OE, Coeff 


#Get 2e- integrals
def get2e(log):
  with open(f"{log}", "r") as reader:
    for line in reader:
      get2e.search(line)
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
  
      if end.search(line):
        continue
  return AOInt

MO2 = np.zeros((NB,NB,NB,NB))

temp = np.zeros((NB,NB,NB,NB))  
temp2 = np.zeros((NB,NB,NB,NB))  
temp3= np.zeros((NB,NB,NB,NB))

def conMO():
  #Initialize temporary arrs
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
  return MO

AA=coul-exc
BB=coul-exc
AB=coul
twoE_array=np.array(twoE)
twoE_flat=twoE_array.flatten()
twoE_scalp=np.dot(twoE_flat,twoE_flat)

#Change to spin-orbital form
for p in range(NB*2):
  for q in range(NB*2):
    for r in range(NB*2):
      for s in range(NB*2): 
        coulomb=twoE[p//2,r//2,q//2,s//2]*(r%2==p%2)*(q%2==s%2)
        exchange=twoE[p//2,s//2,q//2,r//2]*(p%2==s%2)*(q%2==r%2)
        MO[p,q,r,s]=round(coulomb-exchange,16)

IJKL=np.zeros((O*2,O*2,O*2,O*2))
IJKA=np.zeros((O*2,O*2,O*2,NB*2-O*2))
IJAB=np.zeros((O*2,O*2,NB*2-O*2,NB*2-O*2))
IABC=np.zeros((O*2,NB*2,NB*2,NB*2))
IAJB=np.zeros((O*2,NB*2,O*2,NB*2))
NB=NB*2
O=O*2

#Fock matrix elements from orbital energies
for i in range(NB):
  Fock[i]=OE[i//2]
Fock=np.diag(Fock)

#T1 and T2 equations
t1=np.zeros((NB, NB))
t2=np.zeros((NB, NB, NB, NB))

#Initial T2 Guess
MP2=np.zeros((NB,NB,NB,NB))
for i in range(O):
  for j in range(O):
    for a in range(O,NB):
      for b in range(O,NB):
        t2[a,b,i,j]+=MO[i,j,a,b]/(Fock[i,i]+Fock[j,j]-Fock[a,a]-Fock[b,b])

#Tau equations
def tau_tildeEq(T, t1, t2):
  if T==1:
    tau_tilde=np.zeros((NB, NB, NB, NB))
    #Equation (10)
    for a in range(O, NB):
      for b in range(O, NB):
        for i in range(O):
          for j in range(O):
            tau_tilde[a,b,i,j]=t2[a,b,i,j]+(1/2)*(t1[a,i]*t1[b,j]-t1[b,i]*t1[a,j])
  return tau_tilde

def tauEq(T):
  if T==1:
    tau=np.zeros((NB, NB, NB, NB))
    #Equation (11)
    for a in range(O, NB):
      for b in range(O, NB):
        for i in range(O):
          for j in range(O):
            tau[a,b,i,j]=t2[a,b,i,j]+t1[a,i]*t1[b,j]-t1[b,i]*t1[a,j]
  return tau

#Begin F and W intermediates
def intermediateEqs(T):
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



#Define Denominator Arrays
D1=np.zeros((NB, NB))
D2=np.zeros((NB, NB, NB, NB))

#Equation (13)
for a in range(O, NB):
  for i in range(O):
    D1[a,i]=Fock[i,i]-Fock[a,a]

#Equation (14)
for a in range(O, NB):
  for b in range(O, NB):
    for i in range(O):
      for j in range(O):
        D2[a,b,i,j]=Fock[i,i]+Fock[j,j]-Fock[a,a]-Fock[b,b]

 
#Solve for T equations

#Final t1
#t1_f=np.zeros((NB, NB))
def t1Eq(T, t1, t2):
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
  return t1_f

#Final t2
def t2Eq(T, t1, t2):
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

    #Equation (3)
    for a in range(O, NB):
      for b in range(O, NB):
        for i in range(O):
          for j in range(O):
            t2_f[a,b,i,j]=MO[i,j,a,b]
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

#Solve for CCSD energy
#Equation 1
def CCSD():
  E_Corr2=0
  for i in range(O):
    for a in range(O,NB):
      E_Corr2+=t1[a,i]*Fock[i,a]
      for j in range(O):
        for b in range(O,NB):
          E_Corr2+=(1/4)*tau[a,b,i,j]*MO[i,j,a,b]
  return E_Corr2

#Iterate until converged
E_Corr2=0
DiffE=1
DiffT1=1
DiffT2=1
t1RMSE=1
t2RMSE=1
N=0

tau=tauEq(1)

while DiffE>0.000000000000001 or DiffT1>0.00000000001 or DiffT2>0.00000000001 or t1RMSE>0.00000000001 or t2RMSE>0.00000000001:
  E_Corr1=E_Corr2
  tau_tilde=tau_tildeEq(1, t1, t2)
  tau=tauEq(1)
  F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej=intermediateEqs(1)
  t1_f=t1Eq(1, t1, t2)
  start=time.time()
  t2_f=t2Eq(1, t1, t2)
  print(f"Time for Step {N}: {time.time()-start}")
  DiffT1=abs(np.max(t1_f-t1))
  DiffT2=abs(np.max(t2_f-t2))
  t1RMSE=(np.sum(((t1_f-t1)**(2))/(np.size(t1))))**(1/2)
  t2RMSE=(np.sum(((t2_f-t2)**(2))/(np.size(t2))))**(1/2)
  t1=np.copy(t1_f)
  t2=np.copy(t2_f)
  E_Corr2=CCSD()
  DiffE=abs(E_Corr2-E_Corr1)
  t2_flat=t2.flatten()
  t2_flat_sprod=np.dot(t2_flat,t2_flat)
  E_lis.append(E_Corr2+scfE)
  t2_flat_ls.append(t2_flat_sprod/4)
  N+=1
  with open("new.txt","a") as writer:
    writer.write(f"Iteration {N}: CCSD {E_Corr2}\n")
    writer.write(f"Time: {time.time()-start}\n")

#Get CCSD Energy from Gaussian
os.system('grep -i "converged. e(corr" H2_ccsd.log > tmp.txt')
ls=[]
ls2=[]
with open("tmp.txt","r") as reader:
  for line in reader:
    ls.append(line.split())
for i in range(len(ls[0])):
  ls2.append(ls[0][i])
for i in range(len(ls2)):
  if "E(Corr)=" in ls2[i]:
    ccE=(ls2[i+1])

#Print energies
print("Iterations: ", N)
print("E_correlation:", E_Corr2)
print("CCSD Energy:", E_Corr2+scfE)
print("CCSD energy from Gaussian " + ccE)

os.system("rm OEs.txt fort7H2.txt scf.txt tmp.txt OV.txt")	
