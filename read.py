import numpy as np
import os
import sys
import re
import time
from scipy.constants import angstrom, physical_constants
np.set_printoptions(precision=16,threshold=sys.maxsize,floatmode='fixed')
# import readgau
# from readgau import orb
from ein_ccsdAmps import fourier, basis_tran, fill_kl, square_m
from ein_ccsdAmps import denom, DEk, mem_check
sys.path.insert(0, '/Volumes/gaussian/gdv_j30p/')
np.set_printoptions(precision=16,threshold=sys.maxsize,floatmode='fixed')
from gauopen import QCBinAr as qcb


##########################################################################
#Get O, NB, SCF energy, MO coefficients, Orbital energies ################
##########################################################################

def getFort(mol):
  # mol=sys.argv[1]
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
    # baf = qcb.QCBinAr(file=f"{mol}.baf")
    # MOCoef=baf.matlist["PBC ALPHA ORBITALS"].array
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
  # #Fock
  # if(ipbc):
  #   # PBC calculation
  #   with open(f"{mol}_txts/fock.txt","r") as reader:
  #     text=[]
  #     for line in reader:
  #       text.append(line.split())
  #   # Remove parentheses
  #   for i in range(len(text)):
  #     for j in range(len(text[i])):
  #       text[i][j] = text[i][j].replace("[","")
  #       text[i][j] = text[i][j].replace("]","")
  #   # Remove empty slots
  #   for i in range(len(text)):
  #     text[i][:] = [x for x in text[i] if x]
  #   fock_r = []
  #   for i in range(len(text)):
  #     for j in range(len(text[i])):
  #       fock_r.append(float(text[i][j]))
  #   nmtpbc = ipbc[1]
  #   ntt = (NB*(NB+1))//2
  #   kp, l_list = fill_kl(ipbc)
  #   Nkp = len(kp)
  #   fock_r = np.array(fock_r).reshape((nmtpbc,ntt))
  #   fock_k_lt = fourier("Dir",ipbc,fock_r,False)
  #   FockA = basis_tran("Dir",True,False,"Herm",NB,Nkp,MOCoef,fock_k_lt)
  #   Fock = np.zeros((Nkp,NB*2,Nkp,NB*2),dtype=complex)
  #   for k in range(Nkp):
  #     # Fill out the alpha and beta blocks
  #     Fock[k,:O,k,:O] = FockA[k,:O,:O]
  #     Fock[k,O:2*O,k,O:2*O] = FockA[k,:O,:O]
  #     Fock[k,2*O:2*O+V,k,2*O:2*O+V] = FockA[k,O:,O:]
  #     Fock[k,2*O+V:,k,2*O+V:] = FockA[k,O:,O:]
  #   Fock = Fock.reshape((Nkp*NB*2,Nkp*NB*2))
  #   del fock_r, fock_k_lt, FockA
  #   # print(f" pbc fock {Nkp} {4*NB*NB}, {len(Fock)}: \n {Fock}")
  #   # exit()
  # else:
  #   # Molecular calculation
  #   OE=[]
  #   with open(f"{mol}_txts/orbE.txt","r") as reader:
  #     text=[]
  #     for line in reader:
  #       text.append(line.split())
  #   # Remove parentheses
  #   for i in range(len(text)):
  #     for j in range(len(text[i])):
  #       text[i][j] = text[i][j].replace("[","")
  #       text[i][j] = text[i][j].replace("]","")
  #   # Remove empty slots
  #   for i in range(len(text)):
  #     text[i][:] = [x for x in text[i] if x]
  #   for i in range(len(text)):
  #     for j in range(len(text[i])):
  #       OE.append(float(text[i][j]))
  #   OE=np.array(OE)
  #   FockD=np.zeros((NB*2))
  #   FockD[:O] = OE[:O]
  #   FockD[O:2*O] = OE[:O]
  #   FockD[2*O:2*O+V] = OE[O:NB]
  #   FockD[2*O+V:] = OE[O:NB] 
  #   Fock=np.zeros((NB*2,NB*2))
  #   Fock[:,:]=np.diag(FockD)
  # #
  # # Core Hamiltonian
  # if(ipbc):
  #   # PBC calculation
  #   with open(f"{mol}_txts/core.txt","r") as reader:
  #     text=[]
  #     for line in reader:
  #       text.append(line.split())
  #   # Remove parentheses
  #   for i in range(len(text)):
  #     for j in range(len(text[i])):
  #       text[i][j] = text[i][j].replace("[","")
  #       text[i][j] = text[i][j].replace("]","")
  #   # Remove empty slots
  #   for i in range(len(text)):
  #     text[i][:] = [x for x in text[i] if x]
  #   core_r = []
  #   for i in range(len(text)):
  #     for j in range(len(text[i])):
  #       core_r.append(float(text[i][j]))
  #   nmtpbc = ipbc[1]
  #   ntt = (NB*(NB+1))//2
  #   kp, l_list = fill_kl(ipbc)
  #   Nkp = len(kp)
  #   core_r = np.array(core_r).reshape((nmtpbc,ntt))
  #   core_k_lt = fourier("Dir",ipbc,core_r,False)
  #   CoreA = basis_tran("Dir",True,False,"Herm",NB,Nkp,MOCoef,core_k_lt)
  #   Core = np.zeros((Nkp,NB*2,NB*2),dtype=complex)
  #   for k in range(Nkp):
  #     # Fill out the alpha and beta blocks
  #     Core[k,:O,:O] = CoreA[k,:O,:O]
  #     Core[k,O:2*O,O:2*O] = CoreA[k,:O,:O]
  #     Core[k,2*O:2*O+V,2*O:2*O+V] = CoreA[k,O:,O:]
  #     Core[k,2*O+V:,2*O+V:] = CoreA[k,O:,O:]
  #   del core_r, core_k_lt, CoreA
  # else:
  #   # Molecular calculation
  #   corel=[]
  #   with open(f"{mol}_txts/core.txt","r") as reader:
  #     text=[]
  #     for line in reader:
  #       text.append(line.split())
  #   # Remove parentheses
  #   for i in range(len(text)):
  #     for j in range(len(text[i])):
  #       text[i][j] = text[i][j].replace("[","")
  #       text[i][j] = text[i][j].replace("]","")
  #   # Remove empty slots
  #   for i in range(len(text)):
  #     text[i][:] = [x for x in text[i] if x]
  #   for i in range(len(text)):
  #     for j in range(len(text[i])):
  #       corel.append(float(text[i][j]))
  #   corel=np.array(corel)
  #   # symmetrize and transform to MO basis
  #   coresq = np.zeros((NB,NB))
  #   coresq = square_m(NB,True,"Sym",corel,coresq)
  #   temp = np.einsum("in,nm->im", MOCoef, coresq, optimize=True)
  #   coresq = np.einsum("jm,im->ij", MOCoef, temp, optimize=True)
  #   # Fill out the alpha and beta blocks
  #   Core = np.zeros((2*NB,2*NB))
  #   Core[:O,:O] = coresq[:O,:O]
  #   Core[O:2*O,O:2*O] = coresq[:O,:O]
  #   Core[2*O:2*O+V,2*O:2*O+V] = coresq[O:,O:]
  #   Core[2*O+V:,2*O+V:] = coresq[O:,O:]
  #   del corel, coresq
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
#  return O, V, NB, scfE, Fock, MOCoef, ipbc, k_weights, Core
  return O, V, NB, scfE, MOCoef, ipbc, k_weights

########################################################
# Routine the read the Overlap.
########################################################
def getOvl(mol,O,V,NB,ipbc,basis,dk,MOCoef):
  # basis: "AO" or "MO"
  # dk: = F: regular MO(k) basis, = T: dS/dK in MO(k) basis
  with open(f"{mol}_txts/overlap.txt","r") as reader:
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
  Ovl_r = []
  for i in range(len(text)):
    for j in range(len(text[i])):
      Ovl_r.append(float(text[i][j]))
  if(ipbc):
    # PBC calculation
    nmtpbc = ipbc[1]
    ntt = (NB*(NB+1))//2
    Ovl_r = np.array(Ovl_r).reshape((nmtpbc,ntt))
    if(basis == "AO"):
      Ovl = np.copy(Ovl_r)
      del Ovl_r
    elif(basis == "MO"):
      kp, l_list = fill_kl(ipbc)
      Nkp = len(kp)
      Ovl_k_lt = fourier("Dir",ipbc,Ovl_r,dk)
      OvlA = basis_tran("Dir",True,False,"Herm",NB,Nkp,MOCoef,Ovl_k_lt)
      Ovl = np.zeros((Nkp,NB*2,Nkp,NB*2),dtype=complex)
      for k in range(Nkp):
        # Fill out the alpha and beta blocks
        # oa-oa
        Ovl[k,:O,k,:O] = OvlA[k,:O,:O]
        # ob-ob
        Ovl[k,O:2*O,k,O:2*O] = OvlA[k,:O,:O]
        # va-va
        Ovl[k,2*O:2*O+V,k,2*O:2*O+V] = OvlA[k,O:,O:]
        # vb-vb
        Ovl[k,2*O+V:,k,2*O+V:] = OvlA[k,O:,O:]
        # oa-va
        Ovl[k,:O,k,2*O:2*O+V] = OvlA[k,:O,O:]
        # va-oa
        Ovl[k,2*O:2*O+V,k,:O] = OvlA[k,O:,:O]
        # ob-vb
        Ovl[k,O:2*O,k,2*O+V:] = OvlA[k,:O,O:]
        # vb-ob
        Ovl[k,2*O+V:,k,O:2*O] = OvlA[k,O:,:O]
        # Ovl[k,:O,k,:O] = OvlA[k,:O,:O]
        # Ovl[k,O:2*O,k,O:2*O] = OvlA[k,:O,:O]
        # Ovl[k,2*O:2*O+V,k,2*O:2*O+V] = OvlA[k,O:,O:]
        # Ovl[k,2*O+V:,k,2*O+V:] = OvlA[k,O:,O:]
      del Ovl_r, Ovl_k_lt, OvlA
      Ovl = Ovl.reshape((Nkp*NB*2,Nkp*NB*2))
    else:
      print(f"Wrong basis option in getOvl: {basis}")
  else:
    # Molecular calculation
    Ovl_r = np.array(Ovl_r)
    if(basis == "AO"):
      Ovl = np.copy(Ovl_r)
      del Ovl_r
    elif(basis == "MO"):
      # symmetrize and transform to MO basis
      Ovlsq = np.zeros((NB,NB))
      Ovlsq = square_m(NB,True,"Sym",Ovl_r,Ovlsq)
      temp = np.einsum("in,nm->im", MOCoef, Ovlsq, optimize=True)
      Ovlsq = np.einsum("jm,im->ij", MOCoef, temp, optimize=True)
      # Fill out the alpha and beta blocks
      Ovl = np.zeros((2*NB,2*NB))
      Ovl[:O,:O] = Ovlsq[:O,:O]
      Ovl[O:2*O,O:2*O] = Ovlsq[:O,:O]
      Ovl[2*O:2*O+V,2*O:2*O+V] = Ovlsq[O:,O:]
      Ovl[2*O+V:,2*O+V:] = Ovlsq[O:,O:]
      del Ovl_r, Ovlsq
  return Ovl
  
########################################################
# Routine the read the Fock matrix
########################################################
def getFock(mol,O,V,NB,ipbc,basis,dk,MOCoef):
  # basis: "AO" or "MO"
  # dk: = F: regular MO(k) basis, = T: dS/dK in MO(k) basis
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
  Fock_r = []
  for i in range(len(text)):
    for j in range(len(text[i])):
      Fock_r.append(float(text[i][j]))
  if(ipbc):
    # PBC calculation
    nmtpbc = ipbc[1]
    ntt = (NB*(NB+1))//2
    Fock_r = np.array(Fock_r).reshape((nmtpbc,ntt))
    if(basis == "AO"):
      Fock = np.copy(Fock_r)
      del Fock_r
    elif(basis == "MO"):
      kp, l_list = fill_kl(ipbc)
      Nkp = len(kp)
      Fock_k_lt = fourier("Dir",ipbc,Fock_r,dk)
      FockA = basis_tran("Dir",True,False,"Herm",NB,Nkp,MOCoef,Fock_k_lt)
      Fock = np.zeros((Nkp,NB*2,Nkp,NB*2),dtype=complex)
      for k in range(Nkp):
        # Fill out the alpha and beta blocks
        # oa-oa
        Fock[k,:O,k,:O] = FockA[k,:O,:O]
        # ob-ob
        Fock[k,O:2*O,k,O:2*O] = FockA[k,:O,:O]
        # va-va
        Fock[k,2*O:2*O+V,k,2*O:2*O+V] = FockA[k,O:,O:]
        # vb-vb
        Fock[k,2*O+V:,k,2*O+V:] = FockA[k,O:,O:]
        # oa-va
        Fock[k,:O,k,2*O:2*O+V] = FockA[k,:O,O:]
        # va-oa
        Fock[k,2*O:2*O+V,k,:O] = FockA[k,O:,:O]
        # ob-vb
        Fock[k,O:2*O,k,2*O+V:] = FockA[k,:O,O:]
        # vb-ob
        Fock[k,2*O+V:,k,O:2*O] = FockA[k,O:,:O]
      del Fock_r, Fock_k_lt, FockA
      Fock = Fock.reshape((Nkp*NB*2,Nkp*NB*2))
    else:
      print(f"Wrong basis option in getFock: {basis}")
    # print(f"Fock matrices: \n {Fock}")
    # exit()
  else:
    # Molecular calculation
    Fock_r = np.array(Fock_r)
    if(basis == "AO"):
      Fock = np.copy(Fock_r)
      del Fock_r
    elif(basis == "MO"):
      # symmetrize and transform to MO basis
      Focksq = np.zeros((NB,NB))
      Focksq = square_m(NB,True,"Sym",Fock_r,Focksq)
      temp = np.einsum("in,nm->im", MOCoef, Focksq, optimize=True)
      Focksq = np.einsum("jm,im->ij", MOCoef, temp, optimize=True)
      # Fill out the alpha and beta blocks
      Fock = np.zeros((2*NB,2*NB))
      Fock[:O,:O] = Focksq[:O,:O]
      Fock[O:2*O,O:2*O] = Focksq[:O,:O]
      Fock[2*O:2*O+V,2*O:2*O+V] = Focksq[O:,O:]
      Fock[2*O+V:,2*O+V:] = Focksq[O:,O:]
      del Fock_r, Focksq
  return Fock
  
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
  AOInt=np.zeros((NBX, NBX, NBX, NBX))
  mol=sys.argv[1]
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
def conMO(molecule, scratch, O, V, NB, ipbc, MOCoef, AOInt):
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
    # twoEk = np.einsum('nam,hkgmbcd->nhkgabcd',np.conjugate(MOCoef),temp,optimize=True)
    # twoEk = np.transpose(twoEk,axes=(0,2,1,3,4,6,5,7))
    del temp
    start44=time.time()
    print(f"MO tranformation all, time: {start44-start4:.2f}s")
    # dbar = twoEk - np.transpose(twoEk,axes=(0,1,3,2,4,5,7,6))
    # dbar_diff = dbar + np.transpose(dbar,axes=(1,0,2,3,5,4,6,7))
    # dbar_prod = np.einsum('pqrsijkl,pqrsijkl->',dbar_diff,np.conjugate(dbar_diff),optimize=True)
    # print(f"dbar : {dbar_prod.real/(Nkp*Nkp*Nkp*Nkp)}")
    # # Symmetrizing integrals
    # twoEk /= 2
    # twoEk += np.transpose(twoEk,axes=(1,0,3,2,5,4,7,6))
    #
#     max2e = np.max(twoEk.imag)
#     twoEk_diff = twoEk - np.transpose(twoEk,axes=(1,0,3,2,5,4,7,6))
#     twoEk_diff = twoEk_diff.reshape(Nkp*NB*Nkp*NB*Nkp*NB*Nkp*NB)
#     twoEk_prod = np.einsum('p,p->',twoEk_diff,twoEk_diff,optimize=True)
# #    twoEk_prod = np.einsum('pqrsijkl,pqrsijkl->',twoEk_diff,twoEk_diff,optimize=True)
#     twoEk_sum = np.sum(abs(twoEk_diff.imag))
#     print(f"twoEk : {max2e} {twoEk_prod} {twoEk_sum}")
    
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
              # print(f"K points: <{n+1}-{k+1}|{h+1}-{g+1}>: {kn},{kk},{kh},{kg}")
              # for p in range(NB):
              #   for q in range(NB):
              #     for r in range(NB):
              #       for s in range(NB):
              #         print(f"{p+1},{q+1},{r+1},{s+1}: <pq|rs>={twoEk[n,k,h,g,p,q,r,s]} -- <qp|sr>={twoEk[k,n,g,h,q,p,s,r]}")
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
    ABCD = np.transpose(ABCD,axes=(0,4,1,5,2,6,3,7))
    ABCD = ABCD.reshape((V2k,V2k,V2k,V2k))
    np.save(f"{scratch}/{molecule}-ABCD",ABCD)
    del ABCD
    ABCD = []
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

#     NkpC = Nkp*Nkp*Nkp
#     IJAB_diff = IJAB - np.transpose(IJAB,axes=(1,0,3,2))
# #    IJAB_diff = IJAB + np.transpose(IJAB,axes=(1,0,2,3))
#     IJAB_prod = np.einsum('pqrs,pqrs->',IJAB_diff,np.conjugate(IJAB_diff),optimize=True)/NkpC
#     IJKL_diff = IJKL - np.transpose(IJKL,axes=(1,0,3,2))
# #    IJKL_diff = IJKL + np.transpose(IJKL,axes=(1,0,2,3))
#     IJKL_prod = np.einsum('pqrs,pqrs->',IJKL_diff,np.conjugate(IJKL_diff),optimize=True)/NkpC
#     # IJKA_diff = IJKA + np.transpose(IJKA,axes=(1,0,2,3))
#     # IJKA_prod = np.einsum('pqrs,pqrs->',IJKA_diff,np.conjugate(IJKA_diff),optimize=True)/NkpC
#     # IABJ_diff = IABJ + np.transpose(IABJ,axes=(1,0,2,3))
#     # IABJ_prod = np.einsum('pqrs,pqrs->',IABJ_diff,np.conjugate(IABJ_diff),optimize=True)/NkpC
#     # IABC_diff = IABC + np.transpose(IABC,axes=(1,0,2,3))
#     # IABC_prod = np.einsum('pqrs,pqrs->',IABC_diff,np.conjugate(IABC_diff),optimize=True)/NkpC
#     ABCD_diff = ABCD - np.transpose(ABCD,axes=(1,0,3,2))
# #    ABCD_diff = ABCD + np.transpose(ABCD,axes=(1,0,2,3))
#     ABCD_prod = np.einsum('pqrs,pqrs->',ABCD_diff,np.conjugate(ABCD_diff),optimize=True)/NkpC
#     print(f"IJAB : {IJAB_prod.real}")
#     print(f"IJKL : {IJKL_prod.real}")
#     # print(f"IJKA : {IJKA_prod.real}")
#     # print(f"IABJ : {IABJ_prod.real}")
#     # print(f"IABC : {IABC_prod.real}")
#     print(f"ABCD : {ABCD_prod.real}")

    # IJAB_diff = IJAB + np.transpose(IJAB,axes=(0,1,3,2))
    # IJAB_prod = np.einsum('pqrs,pqrs->',IJAB_diff,np.conjugate(IJAB_diff),optimize=True)/NkpC
    # IJKL_diff = IJKL + np.transpose(IJKL,axes=(0,1,3,2))
    # IJKL_prod = np.einsum('pqrs,pqrs->',IJKL_diff,np.conjugate(IJKL_diff),optimize=True)/NkpC
    # IJKA_diff = IJKA + np.transpose(IJKA,axes=(0,1,3,2))
    # IJKA_prod = np.einsum('pqrs,pqrs->',IJKA_diff,np.conjugate(IJKA_diff),optimize=True)/NkpC
    # IABJ_diff = IABJ + np.transpose(IABJ,axes=(1,0,2,3))
    # IABJ_prod = np.einsum('pqrs,pqrs->',IABJ_diff,np.conjugate(IABJ_diff),optimize=True)/NkpC
    # IABC_diff = IABC + np.transpose(IABC,axes=(0,1,3,2))
    # IABC_prod = np.einsum('pqrs,pqrs->',IABC_diff,np.conjugate(IABC_diff),optimize=True)/NkpC
    # ABCD_diff = ABCD + np.transpose(ABCD,axes=(0,1,3,2))
    # ABCD_prod = np.einsum('pqrs,pqrs->',ABCD_diff,np.conjugate(ABCD_diff),optimize=True)/NkpC
    # print(f"IJAB : {IJAB_prod.real}")
    # print(f"IJKL : {IJKL_prod.real}")
    # print(f"IJKA : {IJKA_prod.real}")
    # print(f"IABJ : {IABJ_prod.real}")
    # print(f"IABC : {IABC_prod.real}")
    # print(f"ABCD : {ABCD_prod.real}")
    # exit()
    
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
    tot_mem, avlb_mem = mem_check()
    print(f"MO tranformation, time: {finish-start:.2f}s AvlMem: {avlb_mem:.2f}GB") 
    #
    # Form double-bar integrals <12||12>. Spin blocks are stored as follows:
    # aaaa: Coulomb - Exchange
    # bbbb: Coulomb - Exchange
    # baba: Coulomb 
    # abab: Coulomb 
    # baab: - Exchange
    # abba: - Exchange
    #
    # start = finish
    # MO = np.zeros((2*NB,2*NB,2*NB,2*NB))
    # tot_mem, avlb_mem = mem_check()
    # print(f"Integral size0: {np.size(twoE)} {np.size(MO)} {MO.shape} {MO.dtype} AvlMem: {avlb_mem:.2f}GB")
    # MO[:NB,:NB,:NB,:NB] = np.copy(twoE)
    # tot_mem, avlb_mem = mem_check()
    # print(f"Integral size1: {np.size(twoE)} {np.size(MO)} {MO.shape} {MO.dtype} AvlMem: {avlb_mem:.2f}GB {tot_mem:.2f}GB")
    # MO[:NB,:NB,:NB,:NB] -= np.einsum('pqsr->pqrs',twoE,optimize=True)
    # tot_mem, avlb_mem = mem_check()
    # print(f"Integral size2: {np.size(twoE)} {np.size(MO)} {MO.shape} {MO.dtype} AvlMem: {avlb_mem:.2f}GB")
    # MO[NB:,NB:,NB:,NB:] = np.copy(MO[:NB,:NB,:NB,:NB])
    # tot_mem, avlb_mem = mem_check()
    # print(f"Integral size3: {np.size(twoE)} {np.size(MO)} {MO.shape} {MO.dtype} AvlMem: {avlb_mem:.2f}GB")
    # MO[NB:,:NB,NB:,:NB] = np.copy(twoE)
    # tot_mem, avlb_mem = mem_check()
    # print(f"Integral size4: {np.size(twoE)} {np.size(MO)} {MO.shape} {MO.dtype} AvlMem: {avlb_mem:.2f}GB")
    # MO[:NB,NB:,:NB,NB:] = np.copy(MO[NB:,:NB,NB:,:NB])
    # tot_mem, avlb_mem = mem_check()
    # print(f"Integral size5: {np.size(twoE)} {np.size(MO)} {MO.shape} {MO.dtype} AvlMem: {avlb_mem:.2f}GB")
    # MO[NB:,:NB,:NB,NB:] = -np.einsum('pqsr->pqrs',twoE,optimize=True)
    # tot_mem, avlb_mem = mem_check()
    # print(f"Integral size6: {np.size(twoE)} {np.size(MO)} {MO.shape} {MO.dtype} AvlMem: {avlb_mem:.2f}GB")
    # MO[:NB,NB:,NB:,:NB] = np.copy(MO[NB:,:NB,:NB,NB:])
    # tot_mem, avlb_mem = mem_check()
    # print(f"Integral size7: {np.size(twoE)} {np.size(MO)} {MO.shape} {MO.dtype} AvlMem: {avlb_mem:.2f}GB")
    # # del twoE
    # finish = time.time()
    # tot_mem, avlb_mem = mem_check()
    # print(f"Double bar formation, time: {finish-start:.2f}s AvlMem: {avlb_mem:.2f}GB")
    #
    # ABCD
    start = finish
    ABCD = np.zeros((V2,V2,V2,V2))
    ABCD[:V,:V,:V,:V] = np.copy(twoE[O:NB,O:NB,O:NB,O:NB])
    ABCD[:V,:V,:V,:V] -= np.transpose(twoE[O:NB,O:NB,O:NB,O:NB],axes=(0,1,3,2))
    ABCD[V:,V:,V:,V:] = np.copy(ABCD[:V,:V,:V,:V])
    ABCD[V:,:V,V:,:V] = np.copy(twoE[O:NB,O:NB,O:NB,O:NB])
    ABCD[:V,V:,:V,V:] = np.copy(ABCD[V:,:V,V:,:V])
    ABCD[V:,:V,:V,V:] = -np.transpose(twoE[O:NB,O:NB,O:NB,O:NB],axes=(0,1,3,2))
    ABCD[:V,V:,V:,:V] = np.copy(ABCD[V:,:V,:V,V:])
    # ABCD[:V,:V,:V,:V] = np.copy(MO[O:NB,O:NB,O:NB,O:NB])
    # ABCD[V:,V:,V:,V:] = np.copy(MO[O+NB:2*NB,O+NB:2*NB,O+NB:2*NB,O+NB:2*NB])
    # ABCD[V:,:V,V:,:V] = np.copy(MO[O+NB:2*NB,O:NB,O+NB:2*NB,O:NB])
    # ABCD[:V,V:,:V,V:] = np.copy(MO[O:NB,O+NB:2*NB,O:NB,O+NB:2*NB])
    # ABCD[V:,:V,:V,V:] = np.copy(MO[O+NB:2*NB,O:NB,O:NB,O+NB:2*NB])
    # ABCD[:V,V:,V:,:V] = np.copy(MO[O:NB,O+NB:2*NB,O+NB:2*NB,O:NB])
    np.save(f"{scratch}/{molecule}-ABCD",ABCD)
    del ABCD
    ABCD = []
    finish = time.time()
    tot_mem, avlb_mem = mem_check()
    print(f"ABCD, time: {finish-start:.2f}s AvlMem: {avlb_mem:.2f}GB") 
    #
    # IJAB
    start = finish
    IJAB = np.zeros((O2,O2,V2,V2))
    IJAB[:O,:O,:V,:V] = np.copy(twoE[:O,:O,O:NB,O:NB])
    IJAB[:O,:O,:V,:V] -= np.transpose(twoE[:O,:O,O:NB,O:NB],axes=(0,1,3,2))
    IJAB[O:,O:,V:,V:] = np.copy(IJAB[:O,:O,:V,:V])
    IJAB[O:,:O,V:,:V] = np.copy(twoE[:O,:O,O:NB,O:NB])
    IJAB[:O,O:,:V,V:] = np.copy(IJAB[O:,:O,V:,:V])    
    IJAB[O:,:O,:V,V:] = -np.transpose(twoE[:O,:O,O:NB,O:NB],axes=(0,1,3,2))
    IJAB[:O,O:,V:,:V] = np.copy(IJAB[O:,:O,:V,V:])
    # IJAB[:O,:O,:V,:V] = np.copy(MO[:O,:O,O:NB,O:NB])
    # IJAB[O:,O:,V:,V:] = np.copy(MO[NB:O+NB,NB:O+NB,O+NB:2*NB,O+NB:2*NB])
    # IJAB[O:,:O,V:,:V] = np.copy(MO[NB:O+NB,:O,O+NB:2*NB,O:NB])
    # IJAB[:O,O:,:V,V:] = np.copy(MO[:O,NB:O+NB,O:NB,O+NB:2*NB])
    # IJAB[O:,:O,:V,V:] = np.copy(MO[NB:O+NB,:O,O:NB,O+NB:2*NB])
    # IJAB[:O,O:,V:,:V] = np.copy(MO[:O,NB:O+NB,O+NB:2*NB,O:NB])
    finish = time.time()
    tot_mem, avlb_mem = mem_check()
    print(f"IJAB, time: {finish-start:.2f}s AvlMem: {avlb_mem:.2f}GB") 
    #
    # IJKL
    start = finish
    IJKL = np.zeros((O2,O2,O2,O2))
    IJKL[:O,:O,:O,:O] = np.copy(twoE[:O,:O,:O,:O])
    IJKL[:O,:O,:O,:O] -= np.transpose(twoE[:O,:O,:O,:O],axes=(0,1,3,2))
    IJKL[O:,O:,O:,O:] = np.copy(IJKL[:O,:O,:O,:O])
    IJKL[O:,:O,O:,:O] = np.copy(twoE[:O,:O,:O,:O])
    IJKL[:O,O:,:O,O:] = np.copy(IJKL[O:,:O,O:,:O])
    IJKL[O:,:O,:O,O:] = -np.transpose(twoE[:O,:O,:O,:O],axes=(0,1,3,2))
    IJKL[:O,O:,O:,:O] = np.copy(IJKL[O:,:O,:O,O:])
    # IJKL[:O,:O,:O,:O] = np.copy(MO[:O,:O,:O,:O])
    # IJKL[O:,O:,O:,O:] = np.copy(MO[NB:O+NB,NB:O+NB,NB:O+NB,NB:O+NB])
    # IJKL[O:,:O,O:,:O] = np.copy(MO[NB:O+NB,:O,NB:O+NB,:O])
    # IJKL[:O,O:,:O,O:] = np.copy(MO[:O,NB:O+NB,:O,NB:O+NB])
    # IJKL[O:,:O,:O,O:] = np.copy(MO[NB:O+NB,:O,:O,NB:O+NB])
    # IJKL[:O,O:,O:,:O] = np.copy(MO[:O,NB:O+NB,NB:O+NB,:O])
    finish = time.time()
    tot_mem, avlb_mem = mem_check()
    print(f"IJKL, time: {finish-start:.2f}s AvlMem: {avlb_mem:.2f}GB") 
    #
    # IJKA
    start = finish
    IJKA = np.zeros((O2,O2,O2,V2))
    IJKA[:O,:O,:O,:V] = np.copy(twoE[:O,:O,:O,O:NB])
    IJKA[:O,:O,:O,:V] -= np.transpose(twoE[:O,:O,O:NB,:O],axes=(0,1,3,2))
    IJKA[O:,O:,O:,V:] = np.copy(IJKA[:O,:O,:O,:V])
    IJKA[O:,:O,O:,:V] = np.copy(twoE[:O,:O,:O,O:NB])
    IJKA[:O,O:,:O,V:] = np.copy(IJKA[O:,:O,O:,:V])
    IJKA[O:,:O,:O,V:] = -np.transpose(twoE[:O,:O,O:NB,:O],axes=(0,1,3,2))
    IJKA[:O,O:,O:,:V] = np.copy(IJKA[O:,:O,:O,V:])
    # IJKA[:O,:O,:O,:V] = np.copy(MO[:O,:O,:O,O:NB])
    # IJKA[O:,O:,O:,V:] = np.copy(MO[NB:O+NB,NB:O+NB,NB:O+NB,O+NB:2*NB])
    # IJKA[O:,:O,O:,:V] = np.copy(MO[NB:O+NB,:O,NB:O+NB,O:NB])
    # IJKA[:O,O:,:O,V:] = np.copy(MO[:O,NB:O+NB,:O,O+NB:2*NB])
    # IJKA[O:,:O,:O,V:] = np.copy(MO[NB:O+NB,:O,:O,O+NB:2*NB])
    # IJKA[:O,O:,O:,:V] = np.copy(MO[:O,NB:O+NB,NB:O+NB,O:NB])
    finish = time.time()
    tot_mem, avlb_mem = mem_check()
    print(f"IJKA, time: {finish-start:.2f}s AvlMem: {avlb_mem:.2f}GB") 
    #
    # IABJ
    start = finish
    IABJ = np.zeros((O2,V2,V2,O2))
    IABJ[:O,:V,:V,:O] = np.copy(twoE[:O,O:NB,O:NB,:O])
    IABJ[:O,:V,:V,:O] -= np.transpose(twoE[:O,O:NB,:O,O:NB],axes=(0,1,3,2))
    IABJ[O:,V:,V:,O:] = np.copy(IABJ[:O,:V,:V,:O])
    IABJ[O:,:V,V:,:O] = np.copy(twoE[:O,O:NB,O:NB,:O])
    IABJ[:O,V:,:V,O:] = np.copy(IABJ[O:,:V,V:,:O])
    IABJ[O:,:V,:V,O:] = -np.transpose(twoE[:O,O:NB,:O,O:NB],axes=(0,1,3,2))
    IABJ[:O,V:,V:,:O] = np.copy(IABJ[O:,:V,:V,O:])
    # IABJ[:O,:V,:V,:O] = np.copy(MO[:O,O:NB,O:NB,:O])
    # IABJ[O:,V:,V:,O:] = np.copy(MO[NB:O+NB,O+NB:2*NB,O+NB:2*NB,NB:O+NB])
    # IABJ[O:,:V,V:,:O] = np.copy(MO[NB:O+NB,O:NB,O+NB:2*NB,:O])
    # IABJ[:O,V:,:V,O:] = np.copy(MO[:O,O+NB:2*NB,O:NB,NB:O+NB])
    # IABJ[O:,:V,:V,O:] = np.copy(MO[NB:O+NB,O:NB,O:NB,NB:O+NB])
    # IABJ[:O,V:,V:,:O] = np.copy(MO[:O,O+NB:2*NB,O+NB:2*NB,:O])
    finish = time.time()
    print(f"IABJ, time: {finish-start:.2f}s AvlMem: {avlb_mem:.2f}GB") 
    #
    # IABC
    start = finish
    IABC = np.zeros((O2,V2,V2,V2))
    IABC[:O,:V,:V,:V] = np.copy(twoE[:O,O:NB,O:NB,O:NB])
    IABC[:O,:V,:V,:V] -= np.transpose(twoE[:O,O:NB,O:NB,O:NB],axes=(0,1,3,2))
    IABC[O:,V:,V:,V:] = np.copy(IABC[:O,:V,:V,:V])
    IABC[O:,:V,V:,:V] = np.copy(twoE[:O,O:NB,O:NB,O:NB])
    IABC[:O,V:,:V,V:] = np.copy(IABC[O:,:V,V:,:V])
    IABC[O:,:V,:V,V:] = -np.transpose(twoE[:O,O:NB,O:NB,O:NB],axes=(0,1,3,2))
    IABC[:O,V:,V:,:V] = np.copy(IABC[O:,:V,:V,V:])
    # IABC[:O,:V,:V,:V] = np.copy(MO[:O,O:NB,O:NB,O:NB])
    # IABC[O:,V:,V:,V:] = np.copy(MO[NB:O+NB,O+NB:2*NB,O+NB:2*NB,O+NB:2*NB])
    # IABC[O:,:V,V:,:V] = np.copy(MO[NB:O+NB,O:NB,O+NB:2*NB,O:NB])
    # IABC[:O,V:,:V,V:] = np.copy(MO[:O,O+NB:2*NB,O:NB,O+NB:2*NB])
    # IABC[O:,:V,:V,V:] = np.copy(MO[NB:O+NB,O:NB,O:NB,O+NB:2*NB])
    # IABC[:O,V:,V:,:V] = np.copy(MO[:O,O+NB:2*NB,O+NB:2*NB,O:NB])
    finish = time.time()
    tot_mem, avlb_mem = mem_check()
    print(f"IABC, time: {finish-start:.2f}s AvlMem: {avlb_mem:.2f}GB ") 
    # del MO
    del twoE
    tot_mem, avlb_mem = mem_check()
    print(f"MO release, time: {finish-start:.2f}s AvlMem: {avlb_mem:.2f}GB") 
    # exit()

    # IJAB_diff = IJAB + np.transpose(IJAB,axes=(1,0,2,3))
    # IJAB_prod = np.einsum('pqrs,pqrs->',IJAB_diff,np.conjugate(IJAB_diff),optimize=True)
    # IJKL_diff = IJKL + np.transpose(IJKL,axes=(1,0,2,3))
    # IJKL_prod = np.einsum('pqrs,pqrs->',IJKL_diff,np.conjugate(IJKL_diff),optimize=True)
    # IJKA_diff = IJKA + np.transpose(IJKA,axes=(1,0,2,3))
    # IJKA_prod = np.einsum('pqrs,pqrs->',IJKA_diff,np.conjugate(IJKA_diff),optimize=True)
    # # IABJ_diff = IABJ - np.transpose(IABJ,axes=(1,0,2,3))
    # # IABJ_prod = np.einsum('pqrs,pqrs->',IABJ_diff,np.conjugate(IABJ_diff),optimize=True)
    # # IABC_diff = IABC - np.transpose(IABC,axes=(1,0,2,3))
    # # IABC_prod = np.einsum('pqrs,pqrs->',IABC_diff,np.conjugate(IABC_diff),optimize=True)
    # ABCD_diff = ABCD + np.transpose(ABCD,axes=(1,0,2,3))
    # ABCD_prod = np.einsum('pqrs,pqrs->',ABCD_diff,np.conjugate(ABCD_diff),optimize=True)
    # print(f"IJAB : {IJAB_prod.real}")
    # print(f"IJKL : {IJKL_prod.real}")
    # print(f"IJKA : {IJKA_prod.real}")
    # # print(f"IABJ : {IABJ_prod.real}")
    # # print(f"IABC : {IABC_prod.real}")
    # print(f"ABCD : {ABCD_prod.real}")
    # IJAB_diff = IJAB + np.transpose(IJAB,axes=(0,1,3,2))
    # IJAB_prod = np.einsum('pqrs,pqrs->',IJAB_diff,np.conjugate(IJAB_diff),optimize=True)
    # IJKL_diff = IJKL + np.transpose(IJKL,axes=(0,1,3,2))
    # IJKL_prod = np.einsum('pqrs,pqrs->',IJKL_diff,np.conjugate(IJKL_diff),optimize=True)
    # # IJKA_diff = IJKA + np.transpose(IJKA,axes=(0,1,3,2))
    # # IJKA_prod = np.einsum('pqrs,pqrs->',IJKA_diff,np.conjugate(IJKA_diff),optimize=True)
    # # IABJ_diff = IABJ - np.transpose(IABJ,axes=(1,0,2,3))
    # # IABJ_prod = np.einsum('pqrs,pqrs->',IABJ_diff,np.conjugate(IABJ_diff),optimize=True)
    # IABC_diff = IABC + np.transpose(IABC,axes=(0,1,3,2))
    # IABC_prod = np.einsum('pqrs,pqrs->',IABC_diff,np.conjugate(IABC_diff),optimize=True)
    # ABCD_diff = ABCD + np.transpose(ABCD,axes=(0,1,3,2))
    # ABCD_prod = np.einsum('pqrs,pqrs->',ABCD_diff,np.conjugate(ABCD_diff),optimize=True)
    # print(f"IJAB : {IJAB_prod.real}")
    # print(f"IJKL : {IJKL_prod.real}")
    # # print(f"IJKA : {IJKA_prod.real}")
    # # print(f"IABJ : {IABJ_prod.real}")
    # print(f"IABC : {IABC_prod.real}")
    # print(f"ABCD : {ABCD_prod.real}")
    # exit()
    
  return IJKL, ABCD, IABC, IJAB, IJKA, IABJ

#########################################################
# Get perturbation integrals and return them in MO basis
#########################################################
def getPert(O, V, NB, ipbc, MOCoef, Fock, pert_type, mol):
  # print(f"{mol}_txts/dipole_r.txt and {pert_type}")
  NBX = NB
  O2 = 2*O
  V2 = 2*V
  O2k = O2
  V2k = V2
  ntt = NB*(NB+1)//2
  nttx = ntt
  if(ipbc):
    kp, l_list = fill_kl(ipbc)
    Nkp = len(kp)
    nmtpbc = ipbc[1]
    NBX = NB*nmtpbc
    O2k = O2*Nkp
    V2k = V2*Nkp
    nttx = ntt*nmtpbc
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
      AOPert = np.zeros((3*nttx))
      for i in range(len(text)):
        for j in range(len(text[i])):
          AOPert[ind] = float(text[i][j])
          ind += 1
      NP = 3
      AOPert = AOPert.reshape(NP,nttx)
      print(f"AOPert read {ind} {NP} {nttx}")
    else:
      print(f" No electric dipole integrals found\n")
      exit()
  else:
    print(f" Perturbation ",pert_type," is not available")
    exit()
  # print (f"AOPert\n",AOPert)
  if(ipbc):
    # PBC case
    #
    # Read translation vector
    if(f"{mol}_txts/tv.txt"):
      with open(f"{mol}_txts/tv.txt","r") as reader:
        text=[]
        for line in reader:
          text.append(line.split())
      # print(f"TV text: {text}")
      # # Remove parentheses and empty spaces
      # for j in range(len(text[0])):
      #   text[0][j] = text[0][j].replace("[","")
      #   text[0][j] = text[0][j].replace("]","")
      #   text[0][j] = text[0][j].replace(" ","")
      # print(f"TV text: {text}")
      tv = np.zeros((3))
      tv[0] = float(text[0][0])
      tv[1] = float(text[0][1])
      tv[2] = float(text[0][2])
      # Convert to Bohr
      bohr_radius = physical_constants["Bohr radius"][0]
      tv = np.array(tv)*angstrom / bohr_radius
      print(f"TV: {tv.shape} \n {tv}")
      # exit()
    else:
      print(f" Translation vector is not available")
      exit()
    #
    # dF/dk = F'
    FockDk = getFock(mol,O,V,NB,ipbc,"MO",True,MOCoef)
    #
    # dS/dk = S'
    OvlDk = getOvl(mol,O,V,NB,ipbc,"MO",True,MOCoef)
    #
    # Form i(U + 1/2S')
    OrbE = np.diag(Fock.real)
    NB2k = NB*2*Nkp
    if(len(OrbE)!=NB2k):
      print(f"Mismatch in the number of orbital energies: {NB2k} != {len(OrbE)}")
      exit()
    DE = DEk(1,NB2k,OrbE)
    UMat = FockDk - np.einsum('ij,j->ij',OvlDk,OrbE,optimize=True)
    UMat /= -DE
    UMat += 0.5*OvlDk
    # The diagonal of this matrix is 0
    # Sdiag = -np.diag(OvlDk)*1j
    # np.fill_diagonal(UMat,Sdiag)
    np.fill_diagonal(UMat,0)
    UMat = UMat*1j
    # # print matrices out
    # OvlDk = OvlDk.reshape((Nkp,NB*2,Nkp,NB*2))
    # FockDk = FockDk.reshape((Nkp,NB*2,Nkp,NB*2))
    # UMat = UMat.reshape((Nkp,NB*2,Nkp,NB*2))
    # for k in range(Nkp):
    #   # print(f"dF/dk for k = {k} \n {FockDk[k,:,k,:]}")
    #   # print(f"dS/dk for k = {k} \n {OvlDk[k,:,k,:]}")
    #   print(f"Utilde real for k = {k} \n {UMat[k,:,k,:].real}")
    #   print(f"Utilde imag for k = {k} \n {UMat[k,:,k,:].imag}")
    # exit()
    del FockDk, OvlDk, OrbE, DE
    # Now form the perturbation matrices in MO(k) basis
    X_ij = np.zeros((NP,Nkp,Nkp,O2,O2),dtype=complex)
    X_ia = np.zeros((NP,Nkp,Nkp,O2,V2),dtype=complex)
    X_ab = np.zeros((NP,Nkp,Nkp,V2,V2),dtype=complex)
    print(f"AOPert {NP} {nmtpbc} {ntt} {AOPert.shape}")
    AOPert = AOPert.reshape((NP,nmtpbc,ntt))
    for n in range (NP):
      # for ncell in range (nmtpbc):
      #   print(f"{AOPert[n,ncell,:]}")
      Pert_k_lt = fourier("Dir",ipbc,AOPert[n,:,:],False)
      PertA = basis_tran("Dir",True,False,"Herm",NB,Nkp,MOCoef,Pert_k_lt)
      Pert = np.zeros((Nkp,NB*2,Nkp,NB*2),dtype=complex)
      for k in range(Nkp):
        # Fill out the alpha and beta blocks
        # oa-oa
        Pert[k,:O,k,:O] = PertA[k,:O,:O]
        # ob-ob
        Pert[k,O:2*O,k,O:2*O] = PertA[k,:O,:O]
        # va-va
        Pert[k,2*O:2*O+V,k,2*O:2*O+V] = PertA[k,O:,O:]
        # vb-vb
        Pert[k,2*O+V:,k,2*O+V:] = PertA[k,O:,O:]
        # oa-va
        Pert[k,:O,k,2*O:2*O+V] = PertA[k,:O,O:]
        # va-oa
        Pert[k,2*O:2*O+V,k,:O] = PertA[k,O:,:O]
        # ob-vb
        Pert[k,O:2*O,k,2*O+V:] = PertA[k,:O,O:]
        # vb-ob
        Pert[k,2*O+V:,k,O:2*O] = PertA[k,O:,:O]
        # Pert[k,:O,:O] = PertA[k,:O,:O]
        # Pert[k,O:2*O,O:2*O] = PertA[k,:O,:O]
        # Pert[k,2*O:2*O+V,2*O:2*O+V] = PertA[k,O:,O:]
        # Pert[k,2*O+V:,2*O+V:] = PertA[k,O:,O:]
      Pert = Pert.reshape((Nkp*NB*2,Nkp*NB*2))
      # Add UMat contribution
      # I'm not sure about the overall sign here.
      Pert -= UMat*tv[n]
      Pert = Pert.reshape((Nkp,NB*2,Nkp,NB*2))
      Pert = np.transpose(Pert,axes=(0,2,1,3))
      for k in range(Nkp):
        dipprod = np.einsum('ij,ij->',Pert[k,k,:,:],np.conjugate(Pert[k,k,:,:]),optimize=True)
        print(f"Pert for Cart {n+1} for k={k}: {dipprod}")
      for k in range(Nkp):
        X_ij[n,k,k,:,:] = Pert[k,k,:O2,:O2]
        X_ia[n,k,k,:,:] = Pert[k,k,:O2,O2:]
        X_ab[n,k,k,:,:] = Pert[k,k,O2:,O2:]
      # print matrices out
    #   for k in range(Nkp):
    #     print(f"X_ij real for n={n+1} k={k+1} \n {X_ij[n,k,k,:,:].real}")
    #     print(f"X_ij imag for n={n+1} k={k+1} \n {X_ij[n,k,k,:,:].imag}")
    #     print(f"X_ia real for n={n+1} k={k+1} \n {X_ia[n,k,k,:,:].real}")
    #     print(f"X_ia imag for n={n+1} k={k+1} \n {X_ia[n,k,k,:,:].imag}")
    #     print(f"X_ab real for n={n+1} k={k+1} \n {X_ab[n,k,k,:,:].real}")
    #     print(f"X_ab imag for n={n+1} k={k+1} \n {X_ab[n,k,k,:,:].imag}")
    # exit()
    del AOPert, Pert_k_lt, PertA, Pert
    X_ij = np.transpose(X_ij,axes=(0,1,3,2,4))
    print(f"X_ij")
    for k in range(Nkp):
      for h in range(Nkp):
        for i in range(O2):
          for a in range (O2):
            if(abs(X_ij[0,k,i,h,a].real) > 1e-12 or abs(X_ij[0,k,i,h,a].imag) > 1e-12):
              print(f"{k+1},{i+1},{h+1},{a+1} {X_ij[0,k,i,h,a]:.6e}")
    X_ij = X_ij.reshape((NP,O2k,O2k))
    X_ia = np.transpose(X_ia,axes=(0,1,3,2,4))
    print(f"X_ia")
    for k in range(Nkp):
      for h in range(Nkp):
        for i in range(O2):
          for a in range (V2):
            if(abs(X_ia[0,k,i,h,a].real) > 1e-12 or abs(X_ia[0,k,i,h,a].imag) > 1e-12):
              print(f"{k+1},{i+1},{h+1},{a+1} {X_ia[0,k,i,h,a]:.6e}")
    X_ia = X_ia.reshape((NP,O2k,V2k))
    X_ab = np.transpose(X_ab,axes=(0,1,3,2,4))
    print(f"X_ab")
    for k in range(Nkp):
      for h in range(Nkp):
        for i in range(V2):
          for a in range (V2):
            if(abs(X_ab[0,k,i,h,a].real) > 1e-12 or abs(X_ab[0,k,i,h,a].imag) > 1e-12):
              print(f"{k+1},{i+1},{h+1},{a+1} {X_ab[0,k,i,h,a]:.6e}")
    X_ab = X_ab.reshape((NP,V2k,V2k))
    # X_ij = -X_ij
    # X_ia = -X_ia
    # X_ab = -X_ab
    X_ij = np.conjugate(X_ij)
    X_ia = np.conjugate(X_ia)
    X_ab = np.conjugate(X_ab)
  else:
    # Molecular case
    PertSQ  = np.zeros((3, NB, NB))
    for n in range (NP):
      PertSQ[n,:,:] = square_m(NB,True,"Sym",AOPert[n,:],PertSQ[n,:,:])
    temp = np.einsum('im,kml,jl->kij',MOCoef,PertSQ,MOCoef,optimize=True)
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
    del temp, AOPert, PertSQ
    print(f"X_ij")
    for i in range(O2):
      for a in range (O2):
        if(abs(X_ij[0,i,a]) > 1e-12):
          print(f"{i+1},{a+1} {X_ij[0,i,a]:.6e}")
    print(f"X_ia")
    for i in range(O2):
      for a in range (V2):
        if(abs(X_ia[0,i,a]) > 1e-12):
          print(f"{i+1},{a+1} {X_ia[0,i,a]:.6e}")
    print(f"X_ab")
    for i in range(V2):
      for a in range (V2):
        if(abs(X_ab[0,i,a]) > 1e-12):
          print(f"{i+1},{a+1} {X_ab[0,i,a]:.6e}")
  # with open(f"{mol}.txt","a") as writer:
  #   writer.write(f"MOPert\n {MOPert}\n")
  # print (f"MOPert\n",MOPert)
  return NP, X_ij, X_ia, X_ab

