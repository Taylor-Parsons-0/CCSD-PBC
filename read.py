import numpy as np
import os
import sys
import re
import time
np.set_printoptions(precision=16,threshold=sys.maxsize,floatmode='fixed')
# import readgau
# from readgau import orb
from ein_ccsdAmps import fourier, basis_tran, fill_kl, square_m
from ein_ccsdAmps import denom
sys.path.insert(0, '/Volumes/gaussian/gdv_j30p/')
np.set_printoptions(precision=16,threshold=sys.maxsize,floatmode='fixed')
from gauopen import QCBinAr as qcb


##########################################################################
#Get O, NB, SCF energy, MO coefficients, Orbital energies ################
##########################################################################

def getFort(molecule):
  mol=sys.argv[1]
  #
  #O, V, NB
  O=0
  V=0
  # orbs=[]
  with open(f"{mol}_txts/occ.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split(","))
  # Remove parentheses and empty spaces
  for j in range(len(text[0])):
    text[0][j] = text[0][j].replace("[","")
    text[0][j] = text[0][j].replace("]","")
    text[0][j] = text[0][j].replace(" ","")
  # del text[0]
  # ind=0
  # for i in range(len(text)):
  #   for j in range(len(text[i])):
  #     orbs.append(text[i][j])
  #     if int(float(orbs[ind]))==2:
  #       O+=1
  #     if int(float(orbs[ind]))==0:
  #       V+=1 
  #     ind +=1
  noa = int(text[0][0])
  nob = int(text[0][1])
  nva = int(text[0][2])
  nvb = int(text[0][3])
  nfc = int(text[0][4])
  nfv = int(text[0][5])
  if(noa != nob):
    print(f"Not ready for open shell yet")
    exit()
  O = noa
  V = nva
  NB = O + V + nfc + nfv
  NOrb = O + V
  #
  #SCF Energy
  with open(f"{mol}_txts/scf.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split())
  scfE=float(text[0][0])
  #
  # Read PBC info if available
  ipbc=[]
  k_weights=[]
  if(os.path.exists(f"{mol}_txts/pbc_info.txt")):
    #
    # Read PBC integers
    with open(f"{mol}_txts/pbc_info.txt","r") as reader:
      text=[]
      for line in reader:
        text.append(line.split())
    # Remove parentheses
    for i in range(len(text)):
      for j in range(len(text[i])):
        text[i][j] = text[i][j].replace("[","")
        text[i][j] = text[i][j].replace("]","")
    # Remove empty slots
    for i in range(len(text)):
      text[i][:] = [x for x in text[i] if x]
    for i in range(len(text)):
      for j in range(len(text[i])):
        ipbc.append(int(text[i][j]))
    #
    # Read k-point weigths
    with open(f"{mol}_txts/k_weights.txt","r") as reader:
      text=[]
      for line in reader:
        text.append(line.split())
    # Remove parentheses
    for i in range(len(text)):
      for j in range(len(text[i])):
        text[i][j] = text[i][j].replace("[","")
        text[i][j] = text[i][j].replace("]","")
    # Remove empty slots
    for i in range(len(text)):
      text[i][:] = [x for x in text[i] if x]
    for i in range(len(text)):
      for j in range(len(text[i])):
        k_weights.append(float(text[i][j]))
    k_weights = np.array(k_weights)
  #
  # MO Coefficients
  MOCoef=[[] for _ in range(NB)]
  with open(f"{mol}_txts/mocoef.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split())
  # Remove parentheses and empty spaces
  for i in range(len(text)):
    for j in range(len(text[i])):
      text[i][j] = text[i][j].replace("[","")
      text[i][j] = text[i][j].replace("]","")
  # Remove empty slots
  for i in range(len(text)):
    text[i][:] = [x for x in text[i] if x]
  MOCoef = []
  if(ipbc):
    # PBC 
    for i in range(len(text)):
      for j in range(len(text[i])):
        MOCoef.append(complex(text[i][j]))
    kp, l_list = fill_kl(ipbc)
    Nkp = len(kp)
    nrecip = ipbc[9]
    # MOCoef = np.array(MOCoef).reshape((Nkp,NOrb,NB))
    if(nrecip % 2 == 0):
      MOCoef = np.array(MOCoef).reshape((Nkp//2,NOrb,NB))
      MOCoef = np.append(MOCoef,np.conjugate(MOCoef))
    else:
      nMO = Nkp//2+1
      MOCoef = np.array(MOCoef).reshape((nMO,NOrb*NB))
      MOCoef = np.append(MOCoef,np.conjugate(MOCoef[1:-1,:]))
    MOCoef = np.array(MOCoef).reshape((Nkp,NOrb,NB))
    # print(f"MOCoef:  {MOCoef.shape}")
    # print(f"MOCoef: \n {MOCoef}")
    # exit()
  else:    
    # Molecular 
    for i in range(len(text)):
      for j in range(len(text[i])):
        MOCoef.append(float(text[i][j]))
    MOCoef = np.array(MOCoef).reshape((NOrb,NB))
    # MOCoef = np.array(MOCoef).reshape((1,NOrb,NB))
  # print(f" MOs {Nkp} {NOrb} {NB}, {len(MOCoef)} {MOCoef.shape}:\n {MOCoef}")
  # exit()
  # ind = 0
  # MOCoef=np.zeros((NB,NB))
  # for i in range(len(text)):
  #   for j in range(len(text[i])):
  #     ind1 = ind%NB
  #     ind2 = ind//NB
  #     ind += 1
  #     MOCoef[ind2,ind1] = float(text[i][j])
  # MOCoef = np.array(MOCoef)
  #
  #Fock
  if(ipbc):
    # PBC calculation
    with open(f"{mol}_txts/fock.txt","r") as reader:
      text=[]
      for line in reader:
        text.append(line.split())
    # Remove parentheses
    for i in range(len(text)):
      for j in range(len(text[i])):
        text[i][j] = text[i][j].replace("[","")
        text[i][j] = text[i][j].replace("]","")
    # Remove empty slots
    for i in range(len(text)):
      text[i][:] = [x for x in text[i] if x]
    fock_r = []
    for i in range(len(text)):
      for j in range(len(text[i])):
        fock_r.append(float(text[i][j]))
    nmtpbc = ipbc[1]
    ntt = (NB*(NB+1))//2
    kp, l_list = fill_kl(ipbc)
    Nkp = len(kp)
    fock_r = np.array(fock_r).reshape((nmtpbc,ntt))
    fock_k_lt = fourier("Dir",ipbc,fock_r)
    FockA = basis_tran("Dir",True,False,"Herm",NB,Nkp,MOCoef,fock_k_lt)
    Fock = np.zeros((Nkp,NB*2,Nkp,NB*2),dtype=complex)
    for k in range(Nkp):
      # Fill out the alpha and beta blocks
      Fock[k,:O,k,:O] = FockA[k,:O,:O]
      Fock[k,O:2*O,k,O:2*O] = FockA[k,:O,:O]
      Fock[k,2*O:2*O+V,k,2*O:2*O+V] = FockA[k,O:,O:]
      Fock[k,2*O+V:,k,2*O+V:] = FockA[k,O:,O:]
    Fock = Fock.reshape((Nkp*NB*2,Nkp*NB*2))
    del fock_r, fock_k_lt, FockA
    # print(f" pbc fock {Nkp} {4*NB*NB}, {len(Fock)}: \n {Fock}")
    # exit()
  else:
    # Molecular calculation
    OE=[]
    with open(f"{mol}_txts/orbE.txt","r") as reader:
      text=[]
      for line in reader:
        text.append(line.split())
    # Remove parentheses
    for i in range(len(text)):
      for j in range(len(text[i])):
        text[i][j] = text[i][j].replace("[","")
        text[i][j] = text[i][j].replace("]","")
    # Remove empty slots
    for i in range(len(text)):
      text[i][:] = [x for x in text[i] if x]
    for i in range(len(text)):
      for j in range(len(text[i])):
        OE.append(float(text[i][j]))
    OE=np.array(OE)
    FockD=np.zeros((NB*2))
    FockD[:O] = OE[:O]
    FockD[O:2*O] = OE[:O]
    FockD[2*O:2*O+V] = OE[O:NB]
    FockD[2*O+V:] = OE[O:NB] 
    Fock=np.zeros((NB*2,NB*2))
    Fock[:,:]=np.diag(FockD)
  #
  # Core Hamiltonian
  if(ipbc):
    # PBC calculation
    with open(f"{mol}_txts/core.txt","r") as reader:
      text=[]
      for line in reader:
        text.append(line.split())
    # Remove parentheses
    for i in range(len(text)):
      for j in range(len(text[i])):
        text[i][j] = text[i][j].replace("[","")
        text[i][j] = text[i][j].replace("]","")
    # Remove empty slots
    for i in range(len(text)):
      text[i][:] = [x for x in text[i] if x]
    core_r = []
    for i in range(len(text)):
      for j in range(len(text[i])):
        core_r.append(float(text[i][j]))
    nmtpbc = ipbc[1]
    ntt = (NB*(NB+1))//2
    kp, l_list = fill_kl(ipbc)
    Nkp = len(kp)
    core_r = np.array(core_r).reshape((nmtpbc,ntt))
    core_k_lt = fourier("Dir",ipbc,core_r)
    CoreA = basis_tran("Dir",True,False,"Herm",NB,Nkp,MOCoef,core_k_lt)
    Core = np.zeros((Nkp,NB*2,NB*2),dtype=complex)
    for k in range(Nkp):
      # Fill out the alpha and beta blocks
      Core[k,:O,:O] = CoreA[k,:O,:O]
      Core[k,O:2*O,O:2*O] = CoreA[k,:O,:O]
      Core[k,2*O:2*O+V,2*O:2*O+V] = CoreA[k,O:,O:]
      Core[k,2*O+V:,2*O+V:] = CoreA[k,O:,O:]
    del core_r, core_k_lt, CoreA
  else:
    # Molecular calculation
    corel=[]
    with open(f"{mol}_txts/core.txt","r") as reader:
      text=[]
      for line in reader:
        text.append(line.split())
    # Remove parentheses
    for i in range(len(text)):
      for j in range(len(text[i])):
        text[i][j] = text[i][j].replace("[","")
        text[i][j] = text[i][j].replace("]","")
    # Remove empty slots
    for i in range(len(text)):
      text[i][:] = [x for x in text[i] if x]
    for i in range(len(text)):
      for j in range(len(text[i])):
        corel.append(float(text[i][j]))
    corel=np.array(corel)
    # symmetrize and transform to MO basis
    coresq = np.zeros((NB,NB))
    coresq = square_m(NB,True,"Sym",corel,coresq)
    temp = np.einsum("in,nm->im", MOCoef, coresq, optimize=True)
    coresq = np.einsum("jm,im->ij", MOCoef, temp, optimize=True)
    # Fill out the alpha and beta blocks
    Core = np.zeros((2*NB,2*NB))
    Core[:O,:O] = coresq[:O,:O]
    Core[O:2*O,O:2*O] = coresq[:O,:O]
    Core[2*O:2*O+V,2*O:2*O+V] = coresq[O:,O:]
    Core[2*O+V:,2*O+V:] = coresq[O:,O:]
    del corel, coresq
    # print(f" pbc fock {Nkp} {4*NB*NB}, {len(Fock)}: {Fock}")
    # exit()
  # bad=[]
  # for i in range(len(text)):
  #   n=0
  #   for j in range(len(text[i])):
  #     if "D" not in text[i][j]:
  #       n+=1
  #   if n==len(text[i]):
  #     bad.append(i)
  # for i in range(len(text)):
  #   if i not in bad:
  #     for j in range(len(text[i])-1):
  #       MOCoef[int(text[i][0])-1].append(float(text[i][j+1].replace("D","E")))
  # MOCoef=np.transpose(np.array(MOCoef))
  #Dipole integrals length gauge
#  
  return O, V, NB, scfE, Fock, MOCoef, ipbc, k_weights, Core

########################################################
#####Get 2e integrals###################################
########################################################
def get2e(NB,ipbc):
#def get2e(NB,ipbc,AOInt):
  nmtpbc = 0
  NBX = NB
  mol_int = sys.argv[2]
  if(ipbc):
    nmtpbc = ipbc[1]
    NBX = NB*nmtpbc
    NCMax = (nmtpbc-1)//2
    kp, l_list = fill_kl(ipbc)
    print(f"NMtPBC = {nmtpbc}, NCMax = {NCMax}, Nkp = {len(kp)}")
  # AOInt=np.zeros((NBX, NBX, NBX, NBX))
  # mol=sys.argv[1]
  # icount = 0
  # with open(f"{mol}_txts/twoeint.txt", "r") as reader:
  #   for line in reader:
  #     text=line.split()
  #     if "I=" and "J=" and "K=" and "L=" in text:
  #       icount += 1
  #       I=int(text[1])-1
  #       J=int(text[3])-1
  #       K=int(text[5])-1
  #       L=int(text[7])-1
  #       integ = float(text[9].replace("D", "E"))
  #       # AOInt[I,J,K,L] = integ
  #       # AOInt[J,I,K,L] = integ
  #       # AOInt[I,J,L,K] = integ
  #       # AOInt[J,I,L,K] = integ
  #       # AOInt[K,L,I,J] = integ
  #       # AOInt[L,K,I,J] = integ
  #       # AOInt[K,L,J,I] = integ
  #       # AOInt[L,K,J,I] = integ
  #       if(ipbc):
  #         # Spread integral over cells
  #         iq = I//NB
  #         jq = J//NB
  #         kq = K//NB
  #         lq = L//NB
  #         # cell number for each function
  #         ic = l_list[iq]
  #         jc = l_list[jq]
  #         kc = l_list[kq]
  #         lc = l_list[lq]
  #         # function number in each cell
  #         ir = I%NB
  #         jr = J%NB
  #         kr = K%NB
  #         lr = L%NB
  #         if(icount == 17501):
  #           print(f"IJKL={I+1},{J+1},{K+1},{L+1} -- {integ}")
  #           print(f"Cells={ic},{jc},{kc},{lc}")
  #           print(f"Functions={ir+1},{jr+1},{kr+1},{lr+1}")
  #         # shift first function to cell 0
  #         iic = 0
  #         jjc = jc - ic
  #         kkc = kc - ic
  #         llc = lc - ic
  #         # make sure we are not shifting out of range
  #         if(max(abs(jjc),abs(kkc),abs(llc)) <= NCMax):
  #           II = ir
  #           JJ = jr + NB*l_list.index(jjc)
  #           KK = kr + NB*l_list.index(kkc)
  #           LL = lr + NB*l_list.index(llc)
  #           if(icount == 17501):
  #             print(f"I0JKL={II+1},{JJ+1},{KK+1},{LL+1}")
  #             print(f"Cells={iic},{jjc},{kkc},{llc}")
  #           AOInt[II,JJ,KK,LL] = integ
  #           AOInt[JJ,II,KK,LL] = integ
  #           AOInt[II,JJ,LL,KK] = integ
  #           AOInt[JJ,II,LL,KK] = integ
  #           AOInt[KK,LL,II,JJ] = integ
  #           AOInt[LL,KK,II,JJ] = integ
  #           AOInt[KK,LL,JJ,II] = integ
  #           AOInt[LL,KK,JJ,II] = integ
  #         # else:
  #         #   print(f"Cells bad={iic},{jjc},{kkc},{llc}")
  #         #   print(f"IJKL={I+1},{J+1},{K+1},{L+1} -- {integ}")
  #         #   print(f"Cells={ic},{jc},{kc},{lc}")
  #         #   print(f"Functions={ir+1},{jr+1},{kr+1},{lr+1}")
  #         #   exit()
  #         # shift second function to cell 0
  #         iic = ic - jc
  #         jjc = 0
  #         kkc = kc - jc
  #         llc = lc - jc
  #         # make sure we are not shifting out of range
  #         if(max(abs(iic),abs(kkc),abs(llc)) <= NCMax):
  #           II = ir + NB*l_list.index(iic)
  #           JJ = jr 
  #           KK = kr + NB*l_list.index(kkc)
  #           LL = lr + NB*l_list.index(llc)
  #           if(icount == 17501):
  #             print(f"IJ0KL={II+1},{JJ+1},{KK+1},{LL+1}")
  #             print(f"Cells={iic},{jjc},{kkc},{llc}")
  #           AOInt[II,JJ,KK,LL] = integ
  #           AOInt[JJ,II,KK,LL] = integ
  #           AOInt[II,JJ,LL,KK] = integ
  #           AOInt[JJ,II,LL,KK] = integ
  #           AOInt[KK,LL,II,JJ] = integ
  #           AOInt[LL,KK,II,JJ] = integ
  #           AOInt[KK,LL,JJ,II] = integ
  #           AOInt[LL,KK,JJ,II] = integ
  #         # else:
  #         #   print(f"Cells bad={iic},{jjc},{kkc},{llc}")
  #         #   print(f"IJKL={I+1},{J+1},{K+1},{L+1} -- {integ}")
  #         #   print(f"Cells={ic},{jc},{kc},{lc}")
  #         #   print(f"Functions={ir+1},{jr+1},{kr+1},{lr+1}")
  #         #   exit()
  #         # shift third function to cell 0
  #         iic = ic - kc
  #         jjc = jc - kc
  #         kkc = 0
  #         llc = lc - kc
  #         # make sure we are not shifting out of range
  #         if(max(abs(iic),abs(jjc),abs(llc)) <= NCMax):
  #           II = ir + NB*l_list.index(iic)
  #           JJ = jr + NB*l_list.index(jjc)
  #           KK = kr 
  #           LL = lr + NB*l_list.index(llc)
  #           if(icount == 17501):
  #             print(f"IJK0L={II+1},{JJ+1},{KK+1},{LL+1}")
  #             print(f"Cells={iic},{jjc},{kkc},{llc}")
  #           AOInt[II,JJ,KK,LL] = integ
  #           AOInt[JJ,II,KK,LL] = integ
  #           AOInt[II,JJ,LL,KK] = integ
  #           AOInt[JJ,II,LL,KK] = integ
  #           AOInt[KK,LL,II,JJ] = integ
  #           AOInt[LL,KK,II,JJ] = integ
  #           AOInt[KK,LL,JJ,II] = integ
  #           AOInt[LL,KK,JJ,II] = integ
  #         # else:
  #         #   print(f"Cells bad={iic},{jjc},{kkc},{llc}")
  #         #   print(f"IJKL={I+1},{J+1},{K+1},{L+1} -- {integ}")
  #         #   print(f"Cells={ic},{jc},{kc},{lc}")
  #         #   print(f"Functions={ir+1},{jr+1},{kr+1},{lr+1}")
  #         #   exit()
  #         # shift fourth function to cell 0
  #         iic = ic - lc
  #         jjc = jc - lc
  #         kkc = kc - lc
  #         llc = 0
  #         # make sure we are not shifting out of range
  #         if(max(abs(iic),abs(jjc),abs(kkc)) <= NCMax):
  #           II = ir + NB*l_list.index(iic)
  #           JJ = jr + NB*l_list.index(jjc)
  #           KK = kr + NB*l_list.index(kkc)
  #           LL = lr 
  #           if(icount == 17501):
  #             print(f"IJKL0={II+1},{JJ+1},{KK+1},{LL+1}")
  #             print(f"Cells={iic},{jjc},{kkc},{llc}")
  #           AOInt[II,JJ,KK,LL] = integ
  #           AOInt[JJ,II,KK,LL] = integ
  #           AOInt[II,JJ,LL,KK] = integ
  #           AOInt[JJ,II,LL,KK] = integ
  #           AOInt[KK,LL,II,JJ] = integ
  #           AOInt[LL,KK,II,JJ] = integ
  #           AOInt[KK,LL,JJ,II] = integ
  #           AOInt[LL,KK,JJ,II] = integ
  #       else:
  #         AOInt[I,J,K,L] = integ
  #         AOInt[J,I,K,L] = integ
  #         AOInt[I,J,L,K] = integ
  #         AOInt[J,I,L,K] = integ
  #         AOInt[K,L,I,J] = integ
  #         AOInt[L,K,I,J] = integ
  #         AOInt[K,L,J,I] = integ
  #         AOInt[L,K,J,I] = integ
  baf = qcb.QCBinAr(file=f"{mol_int}.baf")
  AOInt = baf.matlist["REGULAR 2E INTEGRALS"].expand()
  if(ipbc):
    print(f"AOInt-0 {AOInt.shape} {len(AOInt)}")
    nmtpbc = ipbc[1]
    AOInt = AOInt.reshape((nmtpbc,NB,nmtpbc,NB,nmtpbc,NB,nmtpbc,NB))
    prod1 = np.einsum('abcdijkl,abcdijkl',AOInt[:1,:,:,:,:,:,:,:],AOInt[:1,:,:,:,:,:,:,:],optimize=True)
    print(f"AOInt-1 {AOInt.shape} {len(AOInt)} {AOInt[0,0,0,0,0,0,0,0]}")
    AOInt = AOInt[0,:,:,:,:,:,:,:]
    AOInt = AOInt.reshape((NB,nmtpbc,NB,nmtpbc,NB,nmtpbc,NB))
    prod2 = np.einsum('bcdijkl,bcdijkl',AOInt,AOInt,optimize=True)
    print(f"AOInt-2 {AOInt.shape} {len(AOInt)} {prod1} {prod2}  {AOInt[0,0,0,0,0,0,0]}")
  return AOInt

#########################################################
####### AO -> MO Basis 2e Integral transformation########
#########################################################
def conMO(O, V, NB, ipbc, MOCoef, AOInt):
  # AOInt: single-bar 2ERI in AO, Mulliken notation [11|22]
  # MO: double-bar 2ERI in MO, physicist notation <12||12>
  start=time.time()
  O2 = O*2
  V2 = V*2
  if(ipbc):
    #
    # PBC Fourier transform
    nmtpbc = ipbc[1]
    kp, l_list = fill_kl(ipbc)
    Nkp = len(kp)
    print(f"NMtPBC = {nmtpbc}, Nkp = {Nkp}")
    co = np.einsum('k,l->kl',kp,l_list,optimize=True)
    cof = np.cos(co) + 1j*np.sin(co)
    start1=time.time()
    print(f"Form Fourier coefficients, time: {start1-start:.2f}s") 
    temp = np.einsum('hl,albmcnd->habmcnd',cof,AOInt,optimize=True)
    del AOInt
    temp2 = np.einsum('km,habmcnd->hkabcnd',np.conjugate(cof),temp,optimize=True)
    AOk = np.einsum('gn,hkabcnd->hkgabcd',cof,temp2,optimize=True)
    del temp, temp2
    start2=time.time()
    print(f"Fourier transform, time: {start2-start1:.2f}s") 
    #
    # AO(k,k')->MO(k,k') transformation
    start4=time.time()
    temp = np.einsum('hbm,hkgamcd->hkgabcd',MOCoef,AOk,optimize=True)
    del AOk
    temp2 = np.einsum('kcm,hkgabmd->hkgabcd',np.conjugate(MOCoef),temp,optimize=True)
    del temp
    temp = np.einsum('gdm,hkgabcm->hkgabcd',MOCoef,temp2,optimize=True)
    del temp2
    twoEk = np.einsum('nam,hkgmbcd->nkhgacbd',np.conjugate(MOCoef),temp,optimize=True)
    del temp
    start44=time.time()
    print(f"MO tranformation all, time: {start44-start4:.2f}s") 
    #
    # Form double-bar integrals in physicist notation <12||12>
    start=time.time()
    MO = np.zeros((2*NB,2*NB,2*NB,2*NB),dtype=complex)
    IJAB = np.zeros((Nkp,Nkp,Nkp,Nkp,O2,O2,V2,V2),dtype=complex)
    IJKL = np.zeros((Nkp,Nkp,Nkp,Nkp,O2,O2,O2,O2),dtype=complex)
    IJKA = np.zeros((Nkp,Nkp,Nkp,Nkp,O2,O2,O2,V2),dtype=complex)
    IABJ = np.zeros((Nkp,Nkp,Nkp,Nkp,O2,V2,V2,O2),dtype=complex)
    IABC = np.zeros((Nkp,Nkp,Nkp,Nkp,O2,V2,V2,V2),dtype=complex)
    ABCD = np.zeros((Nkp,Nkp,Nkp,Nkp,V2,V2,V2,V2),dtype=complex)
    pi2 = round(2*np.pi,10)
    O2k = O2*Nkp
    V2k = V2*Nkp
    Nksum = 0
    for n in range(Nkp):
      for k in range(Nkp):
        for h in range(Nkp):
          for g in range(Nkp):
            kn = kp[n]
            kh = kp[h]
            kk = kp[k]
            kg = kp[g]
            ktot = round(kn-kh+kk-kg,10)
            if(abs(ktot) < 1e-8 or abs(ktot%pi2) < 1e-8): 
              # print(f"K values {kn} {kh} {kk} {kg} {ktot} {abs(ktot % pi2)}")
              # print(f"K indexes {n} {h} {k} {g} ")
              #
              # Form double-bar integrals <12||12>. Spin blocks are stored as follows:
              # aaaa: Coulomb - Exchange
              # bbbb: Coulomb - Exchange
              # baba: Coulomb 
              # abab: Coulomb 
              # baab: - Exchange
              # abba: - Exchange
              #
              Nksum += 1
              MO[:NB,:NB,:NB,:NB] = np.copy(twoEk[n,k,h,g,:,:,:,:])
              MO[:NB,:NB,:NB,:NB] -= np.transpose(twoEk[n,k,g,h,:,:,:,:],axes=(0,1,3,2))
              MO[NB:,NB:,NB:,NB:] = np.copy(MO[:NB,:NB,:NB,:NB])
              MO[NB:,:NB,NB:,:NB] = np.copy(twoEk[n,k,h,g,:,:,:,:])
              MO[:NB,NB:,:NB,NB:] = np.copy(MO[NB:,:NB,NB:,:NB])
              MO[NB:,:NB,:NB,NB:] = -np.transpose(twoEk[n,k,g,h,:,:,:,:],axes=(0,1,3,2))
              MO[:NB,NB:,NB:,:NB] = np.copy(MO[NB:,:NB,:NB,NB:])
              #
              #IJAB
              IJAB[n,k,h,g,:O,:O,:V,:V] = np.copy(MO[:O,:O,O:NB,O:NB])
              IJAB[n,k,h,g,O:,O:,V:,V:] = np.copy(MO[NB:O+NB,NB:O+NB,O+NB:2*NB,O+NB:2*NB])
              IJAB[n,k,h,g,O:,:O,V:,:V] = np.copy(MO[NB:O+NB,:O,O+NB:2*NB,O:NB])
              IJAB[n,k,h,g,:O,O:,:V,V:] = np.copy(MO[:O,NB:O+NB,O:NB,O+NB:2*NB])
              IJAB[n,k,h,g,O:,:O,:V,V:] = np.copy(MO[NB:O+NB,:O,O:NB,O+NB:2*NB])
              IJAB[n,k,h,g,:O,O:,V:,:V] = np.copy(MO[:O,NB:O+NB,O+NB:2*NB,O:NB])
              #
              # IJKL
              IJKL[n,k,h,g,:O,:O,:O,:O] = np.copy(MO[:O,:O,:O,:O])
              IJKL[n,k,h,g,O:,O:,O:,O:] = np.copy(MO[NB:O+NB,NB:O+NB,NB:O+NB,NB:O+NB])
              IJKL[n,k,h,g,O:,:O,O:,:O] = np.copy(MO[NB:O+NB,:O,NB:O+NB,:O])
              IJKL[n,k,h,g,:O,O:,:O,O:] = np.copy(MO[:O,NB:O+NB,:O,NB:O+NB])
              IJKL[n,k,h,g,O:,:O,:O,O:] = np.copy(MO[NB:O+NB,:O,:O,NB:O+NB])
              IJKL[n,k,h,g,:O,O:,O:,:O] = np.copy(MO[:O,NB:O+NB,NB:O+NB,:O])
              #
              # IJKA
              IJKA[n,k,h,g,:O,:O,:O,:V] = np.copy(MO[:O,:O,:O,O:NB])
              IJKA[n,k,h,g,O:,O:,O:,V:] = np.copy(MO[NB:O+NB,NB:O+NB,NB:O+NB,O+NB:2*NB])
              IJKA[n,k,h,g,O:,:O,O:,:V] = np.copy(MO[NB:O+NB,:O,NB:O+NB,O:NB])
              IJKA[n,k,h,g,:O,O:,:O,V:] = np.copy(MO[:O,NB:O+NB,:O,O+NB:2*NB])
              IJKA[n,k,h,g,O:,:O,:O,V:] = np.copy(MO[NB:O+NB,:O,:O,O+NB:2*NB])
              IJKA[n,k,h,g,:O,O:,O:,:V] = np.copy(MO[:O,NB:O+NB,NB:O+NB,O:NB])
              #
              # IABJ
              IABJ[n,k,h,g,:O,:V,:V,:O] = np.copy(MO[:O,O:NB,O:NB,:O])
              IABJ[n,k,h,g,O:,V:,V:,O:] = np.copy(MO[NB:O+NB,O+NB:2*NB,O+NB:2*NB,NB:O+NB])
              IABJ[n,k,h,g,O:,:V,V:,:O] = np.copy(MO[NB:O+NB,O:NB,O+NB:2*NB,:O])
              IABJ[n,k,h,g,:O,V:,:V,O:] = np.copy(MO[:O,O+NB:2*NB,O:NB,NB:O+NB])
              IABJ[n,k,h,g,O:,:V,:V,O:] = np.copy(MO[NB:O+NB,O:NB,O:NB,NB:O+NB])
              IABJ[n,k,h,g,:O,V:,V:,:O] = np.copy(MO[:O,O+NB:2*NB,O+NB:2*NB,:O])
              #
              # IABC
              IABC[n,k,h,g,:O,:V,:V,:V] = np.copy(MO[:O,O:NB,O:NB,O:NB])
              IABC[n,k,h,g,O:,V:,V:,V:] = np.copy(MO[NB:O+NB,O+NB:2*NB,O+NB:2*NB,O+NB:2*NB])
              IABC[n,k,h,g,O:,:V,V:,:V] = np.copy(MO[NB:O+NB,O:NB,O+NB:2*NB,O:NB])
              IABC[n,k,h,g,:O,V:,:V,V:] = np.copy(MO[:O,O+NB:2*NB,O:NB,O+NB:2*NB])
              IABC[n,k,h,g,O:,:V,:V,V:] = np.copy(MO[NB:O+NB,O:NB,O:NB,O+NB:2*NB])
              IABC[n,k,h,g,:O,V:,V:,:V] = np.copy(MO[:O,O+NB:2*NB,O+NB:2*NB,O:NB])
              #
              # ABCD
              ABCD[n,k,h,g,:V,:V,:V,:V] = np.copy(MO[O:NB,O:NB,O:NB,O:NB])
              ABCD[n,k,h,g,V:,V:,V:,V:] = np.copy(MO[O+NB:2*NB,O+NB:2*NB,O+NB:2*NB,O+NB:2*NB])
              ABCD[n,k,h,g,V:,:V,V:,:V] = np.copy(MO[O+NB:2*NB,O:NB,O+NB:2*NB,O:NB])
              ABCD[n,k,h,g,:V,V:,:V,V:] = np.copy(MO[O:NB,O+NB:2*NB,O:NB,O+NB:2*NB])
              ABCD[n,k,h,g,V:,:V,:V,V:] = np.copy(MO[O+NB:2*NB,O:NB,O:NB,O+NB:2*NB])
              ABCD[n,k,h,g,:V,V:,V:,:V] = np.copy(MO[O:NB,O+NB:2*NB,O+NB:2*NB,O:NB])
    del MO, twoEk
    IJAB = np.transpose(IJAB,axes=(0,4,1,5,2,6,3,7))
    IJAB = IJAB.reshape((O2k,O2k,V2k,V2k))
    IJKL = np.transpose(IJKL,axes=(0,4,1,5,2,6,3,7))
    IJKL = IJKL.reshape((O2k,O2k,O2k,O2k))
    IJKA = np.transpose(IJKA,axes=(0,4,1,5,2,6,3,7))
    IJKA = IJKA.reshape((O2k,O2k,O2k,V2k))
    IABJ = np.transpose(IABJ,axes=(0,4,1,5,2,6,3,7))
    IABJ = IABJ.reshape((O2k,V2k,V2k,O2k))
    IABC = np.transpose(IABC,axes=(0,4,1,5,2,6,3,7))
    IABC = IABC.reshape((O2k,V2k,V2k,V2k))
    ABCD = np.transpose(ABCD,axes=(0,4,1,5,2,6,3,7))
    ABCD = ABCD.reshape((V2k,V2k,V2k,V2k))
    finish=time.time()
    print(f"Double bar formation all, time: {finish-start:.2f}s") 
  else:
    #
    # AO->MO transformation
    print(f"shape MO: {MOCoef.shape}, shape AO: {AOInt.shape}")
    start = time.time()
    temp = np.einsum('im,mjkl->ijkl',MOCoef,AOInt,optimize=True)
    del AOInt
    temp2 = np.einsum('jm,imkl->ijkl',MOCoef,temp,optimize=True)
    del temp
    temp = np.einsum('km,ijml->ijkl',MOCoef,temp2,optimize=True)
    del temp2
    twoE = np.einsum('lm,ijkm->ikjl',MOCoef,temp,optimize=True)
    del temp
    finish = time.time()
    print(f"MO tranformation, time: {finish-start:.2f}s") 
    #
    # Form double-bar integrals <12||12>. Spin blocks are stored as follows:
    # aaaa: Coulomb - Exchange
    # bbbb: Coulomb - Exchange
    # baba: Coulomb 
    # abab: Coulomb 
    # baab: - Exchange
    # abba: - Exchange
    #
    start = finish
    MO = np.zeros((2*NB,2*NB,2*NB,2*NB))
    MO[:NB,:NB,:NB,:NB] = np.copy(twoE)
    MO[:NB,:NB,:NB,:NB] -= np.einsum('pqsr->pqrs',twoE,optimize=True)
    MO[NB:,NB:,NB:,NB:] = np.copy(MO[:NB,:NB,:NB,:NB])
    MO[NB:,:NB,NB:,:NB] = np.copy(twoE)
    MO[:NB,NB:,:NB,NB:] = np.copy(MO[NB:,:NB,NB:,:NB])
    MO[NB:,:NB,:NB,NB:] = -np.einsum('pqsr->pqrs',twoE,optimize=True)
    MO[:NB,NB:,NB:,:NB] = np.copy(MO[NB:,:NB,:NB,NB:])
    del twoE
    finish = time.time()
    print(f"Double bar formation, time: {finish-start:.2f}s")
    #
    # IJAB
    start = finish
    IJAB = np.zeros((O2,O2,V2,V2))
    IJAB[:O,:O,:V,:V] = np.copy(MO[:O,:O,O:NB,O:NB])
    IJAB[O:,O:,V:,V:] = np.copy(MO[NB:O+NB,NB:O+NB,O+NB:2*NB,O+NB:2*NB])
    IJAB[O:,:O,V:,:V] = np.copy(MO[NB:O+NB,:O,O+NB:2*NB,O:NB])
    IJAB[:O,O:,:V,V:] = np.copy(MO[:O,NB:O+NB,O:NB,O+NB:2*NB])
    IJAB[O:,:O,:V,V:] = np.copy(MO[NB:O+NB,:O,O:NB,O+NB:2*NB])
    IJAB[:O,O:,V:,:V] = np.copy(MO[:O,NB:O+NB,O+NB:2*NB,O:NB])
    finish = time.time()
    print(f"IJAB, time: {finish-start:.2f}s") 
    #
    # IJKL
    start = finish
    IJKL = np.zeros((O2,O2,O2,O2))
    IJKL[:O,:O,:O,:O] = np.copy(MO[:O,:O,:O,:O])
    IJKL[O:,O:,O:,O:] = np.copy(MO[NB:O+NB,NB:O+NB,NB:O+NB,NB:O+NB])
    IJKL[O:,:O,O:,:O] = np.copy(MO[NB:O+NB,:O,NB:O+NB,:O])
    IJKL[:O,O:,:O,O:] = np.copy(MO[:O,NB:O+NB,:O,NB:O+NB])
    IJKL[O:,:O,:O,O:] = np.copy(MO[NB:O+NB,:O,:O,NB:O+NB])
    IJKL[:O,O:,O:,:O] = np.copy(MO[:O,NB:O+NB,NB:O+NB,:O])
    finish = time.time()
    print(f"IJKL, time: {finish-start:.2f}s") 
    #
    # IJKA
    start = finish
    IJKA = np.zeros((O2,O2,O2,V2))
    IJKA[:O,:O,:O,:V] = np.copy(MO[:O,:O,:O,O:NB])
    IJKA[O:,O:,O:,V:] = np.copy(MO[NB:O+NB,NB:O+NB,NB:O+NB,O+NB:2*NB])
    IJKA[O:,:O,O:,:V] = np.copy(MO[NB:O+NB,:O,NB:O+NB,O:NB])
    IJKA[:O,O:,:O,V:] = np.copy(MO[:O,NB:O+NB,:O,O+NB:2*NB])
    IJKA[O:,:O,:O,V:] = np.copy(MO[NB:O+NB,:O,:O,O+NB:2*NB])
    IJKA[:O,O:,O:,:V] = np.copy(MO[:O,NB:O+NB,NB:O+NB,O:NB])
    finish = time.time()
    print(f"IJKA, time: {finish-start:.2f}s") 
    #
    # IABJ
    start = finish
    IABJ = np.zeros((O2,V2,V2,O2))
    IABJ[:O,:V,:V,:O] = np.copy(MO[:O,O:NB,O:NB,:O])
    IABJ[O:,V:,V:,O:] = np.copy(MO[NB:O+NB,O+NB:2*NB,O+NB:2*NB,NB:O+NB])
    IABJ[O:,:V,V:,:O] = np.copy(MO[NB:O+NB,O:NB,O+NB:2*NB,:O])
    IABJ[:O,V:,:V,O:] = np.copy(MO[:O,O+NB:2*NB,O:NB,NB:O+NB])
    IABJ[O:,:V,:V,O:] = np.copy(MO[NB:O+NB,O:NB,O:NB,NB:O+NB])
    IABJ[:O,V:,V:,:O] = np.copy(MO[:O,O+NB:2*NB,O+NB:2*NB,:O])
    finish = time.time()
    print(f"IABJ, time: {finish-start:.2f}s") 
    #
    # IABC
    start = finish
    IABC = np.zeros((O2,V2,V2,V2))
    IABC[:O,:V,:V,:V] = np.copy(MO[:O,O:NB,O:NB,O:NB])
    IABC[O:,V:,V:,V:] = np.copy(MO[NB:O+NB,O+NB:2*NB,O+NB:2*NB,O+NB:2*NB])
    IABC[O:,:V,V:,:V] = np.copy(MO[NB:O+NB,O:NB,O+NB:2*NB,O:NB])
    IABC[:O,V:,:V,V:] = np.copy(MO[:O,O+NB:2*NB,O:NB,O+NB:2*NB])
    IABC[O:,:V,:V,V:] = np.copy(MO[NB:O+NB,O:NB,O:NB,O+NB:2*NB])
    IABC[:O,V:,V:,:V] = np.copy(MO[:O,O+NB:2*NB,O+NB:2*NB,O:NB])
    finish = time.time()
    print(f"IABC, time: {finish-start:.2f}s") 
    #
    # ABCD
    start = finish
    ABCD = np.zeros((V2,V2,V2,V2))
    ABCD[:V,:V,:V,:V] = np.copy(MO[O:NB,O:NB,O:NB,O:NB])
    ABCD[V:,V:,V:,V:] = np.copy(MO[O+NB:2*NB,O+NB:2*NB,O+NB:2*NB,O+NB:2*NB])
    ABCD[V:,:V,V:,:V] = np.copy(MO[O+NB:2*NB,O:NB,O+NB:2*NB,O:NB])
    ABCD[:V,V:,:V,V:] = np.copy(MO[O:NB,O+NB:2*NB,O:NB,O+NB:2*NB])
    ABCD[V:,:V,:V,V:] = np.copy(MO[O+NB:2*NB,O:NB,O:NB,O+NB:2*NB])
    ABCD[:V,V:,V:,:V] = np.copy(MO[O:NB,O+NB:2*NB,O+NB:2*NB,O:NB])
    finish = time.time()
    print(f"ABCD, time: {finish-start:.2f}s") 
    del MO
  return IJKL, ABCD, IABC, IJAB, IJKA, IABJ

#########################################################
# Get perturbation integrals and return them in MO basis
#########################################################
def getpert(O, V, NB, MOCoef, pert_type, mol):
  # print(f"{mol}_txts/dipole_r.txt and {pert_type}")
  with open(f"{mol}.txt","a") as writer:
    writer.write(f"Reading perturbation {pert_type}\n")
  if(pert_type == "DipE"):
    if(f"{mol}_txts/dipole_r.txt"):
      with open(f"{mol}_txts/dipole_r.txt","r") as reader:
        text=[]
        for line in reader:
          text.append(line.split())
      # Remove parentheses
      for i in range(len(text)):
        for j in range(len(text[i])):
          text[i][j] = text[i][j].replace("[","")
          text[i][j] = text[i][j].replace("]","")
      # Remove empty slots
      for i in range(len(text)):
        text[i][:] = [x for x in text[i] if x]
      ind = 0
      AOPert=np.zeros((3*NB*NB))
      for i in range(len(text)):
        for j in range(len(text[i])):
          AOPert[ind] = float(text[i][j])
          ind += 1
      NP = 3
      AOPert = AOPert.reshape(NP,NB,NB)
    else:
      print(f" No electric dipole integrals found\n")
      exit()
  else:
    print(f" Perturbation ",pert_type," is not available")
    exit()
  # print (f"AOPert\n",AOPert)
  temp = np.einsum('im,kml,jl->kij',MOCoef,AOPert,MOCoef,optimize=True)
  O2 = 2*O
  V2 = 2*V
  X_ij = np.zeros((NP,O2,O2))
  X_ia = np.zeros((NP,O2,V2))
  X_ab = np.zeros((NP,V2,V2))
  for n in range(NP):
    for i in range(O):
      for j in range(O):
        X_ij[n,i,j] = temp[n,i,j]
        X_ij[n,i+O,j+O] = temp[n,i,j]
    for i in range(O):
      for a in range(V):
        X_ia[n,i,a] = temp[n,i,a+O]
        X_ia[n,i+O,a+V] = temp[n,i,a+O]
    for a in range(V):
      for b in range(V):
        X_ab[n,a,b] = temp[n,a+O,b+O]
        X_ab[n,a+V,b+V] = temp[n,a+O,b+O]
  del temp, AOPert
  # with open(f"{mol}.txt","a") as writer:
  #   writer.write(f"MOPert\n {MOPert}\n")
  # print (f"MOPert\n",MOPert)
  return NP, X_ij, X_ia, X_ab

