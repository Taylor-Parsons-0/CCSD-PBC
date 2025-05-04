import numpy as np
import os
import sys
import re
import time
# from read import getFort
# 
# ##########################################################################
# # Compute E and intermediates for CCSD equations
# ##########################################################################
# 
# #Initialize arrays
# #Define molecule
# if len(sys.argv)==2:
#   molecule=sys.argv[1]
# else:
#   print("MISSING MOLECULE NAME AS FIRST ARG")
#   exit()
# O, V, NB, scfE, Fock, MOCoef, ipbc, k_weights, Core=getFort(molecule)

##########################################################################
# Compute energy denominators
##########################################################################
def denom(T, O2, V2, kp, Fock, W):
  if T==1:
    NB2 = O2+V2
    if(kp):
      # PBC
      Nkp = len(kp)
      O2k = O2*Nkp
      V2k = V2*Nkp
      kp2 = kp
    else:
      # Molecular
      Nkp = 1
      O2k = O2
      V2k = V2
      kp2 = np.zeros((1))
    pi2 = round(2*np.pi,10)
    D1 = np.ones((Nkp,O2,Nkp,V2),dtype=Fock.dtype)
    D2 = np.ones((Nkp,O2,Nkp,O2,Nkp,V2,Nkp,V2),dtype=Fock.dtype)
    Fock = Fock.reshape((Nkp,NB2,Nkp,NB2))
#    print(f"shape D2 {D2.shape}")
    # D1 denominator
    NksumS = 0
    for n in range(Nkp):
      for k in range(Nkp):
        kn = kp2[n]
        kk = kp2[k]
        ktot = round(kn-kk,10)
        if(abs(ktot) < 1e-8 or abs(ktot%pi2) < 1e-8): 
          NksumS += 1
          for a in range(V2):
            for i in range(O2):
              D1[n,i,k,a]=Fock[n,i,n,i]-Fock[k,a+O2,k,a+O2]
    D1 -= W
    D1 = D1.reshape((O2k,V2k))
    if(NksumS != Nkp):
      print(f"Issue with k-point count for singles denominator: {NksumS} != {Nkp} ")
      exit()
    # D2 denominator
    NksumD = 0
    for n in range(Nkp):
      for k in range(Nkp):
        for h in range(Nkp):
          for g in range(Nkp):
            # IJAB          
            kn = kp2[n]
            kh = kp2[h]
            kk = kp2[k]
            kg = kp2[g]
            ktot = round(kn-kh+kk-kg,10)
            if(abs(ktot) < 1e-8 or abs(ktot%pi2) < 1e-8): 
              NksumD += 1
              for i in range(O2):
                deni = Fock[n,i,n,i] - W
                for j in range(O2):
                  denj = deni + Fock[k,j,k,j]
                  for a in range(V2):
                    dena = denj - Fock[h,a+O2,h,a+O2]
                    for b in range(V2):
                      D2[n,i,k,j,h,a,g,b] = dena - Fock[g,b+O2,g,b+O2]
    if(NksumD != Nkp*Nkp*Nkp):
      print(f"Issue with k-point count for singles denominator: {NksumD} != {Nkp*Nkp*Nkp} ")
      exit()
    D2 -= W
    D2 = D2.reshape((O2k,O2k,V2k,V2k))
    Fock = Fock.reshape((Nkp*NB2,Nkp*NB2))
    return D1, D2

##########################################################################
# Wrapper routine for iterative solution of CCSD amplitude equations
##########################################################################
def AmpIt(AmpType,molecule,O,V,Nkp,MaxIt,ThrE,ThrA,scfE,Fock,IJKL,ABCD,
          IABC,IJAB,IABJ,IJKA,tau,W_efam,W_iemn,W_mbej,W_mnij,W_abef,F_ae,
          F_mi,F_me,rhs1,rhs2,D1,D2,t1,t2,l1,l2,tx1,tx2,ipbc):
  E_Corr2 = 0
  N = 0
  not_conver = True
  # Setup DIIS arrays
  st1 = []
  st2 = []
  e_DIIS = []
  MaxD = 6
  RepD = 5
  DoDIIS = "F"
  B_mat = np.zeros((MaxD,MaxD),dtype=Fock.dtype)
  # Start loop
  start0=time.time()
  while not_conver and N< MaxIt:
    start = time.time()
    N +=1
    E_Corr1 = E_Corr2
    if(AmpType == "T"):
      # Ground state T amplitudes
      if(N==1):
        # Initialize DIIS amplitudes with guess
        st1.append(list(t1.flatten()))
        st2.append(list(t2.flatten()))
      # Calculate intermediates
      st_time = time.time()
      tau_tilde = tau_tildeEq(1, Nkp, t1, t2)
      fi_time=time.time()
      print(f"Tau tilde: {fi_time-st_time:.2f}s") 
      st_time = fi_time
      tau = tauEq(1, Nkp, t1, t2)
      fi_time=time.time()
      print(f"Tau: {fi_time-st_time:.2f}s") 
      st_time = fi_time
      F_ae,F_mi,F_me,W_mnij,W_abef,W_mbej = T_interm(1,O,V,Nkp,Fock,t1,t2,
                                                     IJKL,ABCD,IABC,IJAB,
                                                     IABJ,IJKA,tau_tilde,
                                                     tau)
      fi_time=time.time()
      print(f"Intermediates: {fi_time-st_time:.2f}s") 
      st_time = fi_time
      # Amplitude iteration
      t1_f = t1Eq(1,O,Nkp,Fock,t1,t2,IABC,IJKA,IABJ,F_ae,F_mi,F_me,D1)
      # t1_f = np.zeros((O*2,V*2),dtype=Fock.dtype)
      # if (ipbc):
      #   kp, l_list = fill_kl(ipbc)
      #   Nkp = len(kp)
      #   oo = O*2//Nkp
      #   vv = V*2//Nkp
      #   F_ae = F_ae.reshape((Nkp,vv,Nkp,vv))
      #   F_ae = np.transpose(F_ae,axes=(0,2,1,3))
      #   # print(f"Fae: \n {F_ae}")
      #   print(f"F_ae: \n ")
      #   for h in range(Nkp):
      #     for k in range(Nkp):
      #       if(k==h):
      #         amptr = np.einsum('ij,ij->',np.conjugate(F_ae[h,k,:,:]),F_ae[h,k,:,:],optimize=True)
      #         if(abs(amptr.real) > 1e-10 or abs(amptr.imag) > 1e-10):
      #           print(f"K indexes {h} {k}: {amptr} ")
      #   F_ae = np.transpose(F_ae,axes=(0,2,1,3))
      #   F_ae = F_ae.reshape((Nkp*vv,Nkp*vv))
      # print(f"Fme: \n {F_me}")
      #exit()
      t2_f = t2Eq(1,Nkp,t1,t2,IABC,IJAB,IJKA,IABJ,tau,F_ae,F_mi,F_me,
                  W_mnij,W_abef,W_mbej,D2)
      # t2_f = np.conjugate(IJAB)/D2.real
      fi_time=time.time()
      print(f"T2: {fi_time-st_time:.2f}s") 
      st_time = fi_time
      # if (N>2 and ipbc):
      #   t2_f = t2_f.reshape((Nkp,oo,Nkp,oo,Nkp,vv,Nkp,vv))
      #   t2_f = np.transpose(t2_f,axes=(0,2,4,6,1,3,5,7))
      #   print(f"Iter: {N} t2_f: \n ")
      #   pi2 = round(2*np.pi,10)
      #   for n in range(Nkp):
      #     for k in range(Nkp):
      #       for h in range(Nkp):
      #         for g in range(Nkp):
      #           kn = kp[n]
      #           kh = kp[h]
      #           kk = kp[k]
      #           kg = kp[g]
      #           ktot = round(kn-kh+kk-kg,10)
      #           if(abs(ktot) > 1e-8 and abs(ktot%pi2) > 1e-8): 
      #             amptr = np.einsum('ijab,ijab->',np.conjugate(t2_f[n,k,h,g,:,:,:,:]),t2_f[n,k,h,g,:,:,:,:],optimize=True)
      #             if(abs(amptr.real) > 1e-10 or abs(amptr.imag) > 1e-10):
      #               print(f"K values {kn} {kh} {kk} {kg} {ktot} {abs(ktot % pi2)}")
      #               print(f"K indexes {n} {h} {k} {g}: {amptr} ")
      # if(N>3): exit()
      tau = tauEq(1, Nkp, t1_f, t2_f)
      # t1_prod = np.einsum('ia,ia->',np.conjugate(t1_f),t1_f,optimize=True)/Nkp
      # t2_prod = np.einsum('ijab,ijab->',np.conjugate(t2_f),t2_f,optimize=True)/(Nkp*Nkp*Nkp)
      # tau_prod = np.einsum('ijab,ijab->',np.conjugate(tau),tau,optimize=True)/(Nkp*Nkp*Nkp)
      # print(f"Amp prod: {t1_prod.real} {t2_prod.real} {tau_prod.real}") 
      fi_time=time.time()
      print(f"Tau again: {fi_time-st_time:.2f}s") 
      st_time = fi_time
      # Evaluate convergence
      not_conver,E_Corr2,t1,t2 = AmpConv(AmpType,O,Nkp,t1,t2,t1_f,t2_f,tau,
                                         Fock,D1,IJAB,ThrE,ThrA,E_Corr1)
      fi_time=time.time()
      print(f"Energy: {fi_time-st_time:.2f}s") 
      st_time = fi_time
      del t1_f, t2_f
      t1, t2, DoDIIS = DIIS(O,V,N,MaxD,ThrA,RepD,t1,t2,B_mat,st1,st2,
                            e_DIIS,DoDIIS)
      a1 = t1
      a2 = t2
    elif (AmpType == "L"):
      # Ground state Lambda (or Z) amplitudes
      if(N==1):
        # Initialize DIIS amplitudes with guess
        st1.append(list(l1.flatten()))
        st2.append(list(l2.flatten()))
      # Calculate intermediates
      G_ae, G_mi = L_Interm(1,Nkp,t2,l2)
      # Amplitude iteration
      l1_f = l1Eq(1,Nkp,t1,l1,l2,IJAB,IABC,IJKA,W_efam,W_iemn,W_mbej,F_ae,
                  F_mi,F_me,G_ae,G_mi,D1)
      # l2_f = l2
      # l1_f = l1
      l2_f = l2Eq(1,Nkp,t1,l1,l2,IABC,IJAB,IJKA,F_ae,F_mi,F_me,G_ae,G_mi,
                  W_mnij,W_abef,W_mbej,D2)
      tau_tilde = tauEq(1, Nkp, l1_f, l2_f)
      # E_Corr2 = E_CCSD(O, Fock, l1_f, IJAB, tau_tilde)
      # Evaluate convergence
      not_conver, E_Corr2, l1, l2 = AmpConv(AmpType,O,Nkp,l1,l2,l1_f,l2_f,
                                            tau_tilde,Fock,D1,IJAB,ThrE,ThrA,
                                            E_Corr1)
      del l1_f, l2_f, G_ae, G_mi 
      l1, l2, DoDIIS = DIIS(O,V,N,MaxD,ThrA,RepD,l1,l2,B_mat,st1,st2,
                            e_DIIS,DoDIIS)
      a1 = l1
      a2 = l2
    # elif (AmpType == "Tx"):
    #   # Perturbed T amplitudes
    #   if(N==1):
    #     # Initialize DIIS amplitudes with guess
    #     st1.append(list(tx1.flatten()))
    #     st2.append(list(tx2.flatten()))
    #   # Calculate intermediates
    #   G_ae, G_mi = L_Interm(1,Nkp,IJAB,tx2)
    #   # Amplitude iteration
    #   tx1_f = tx1Eq(1,Nkp,tx1,tx2,t1,IABC,IJKA,W_mbej,F_ae,F_mi,F_me,G_ae,G_mi,D1)
    #   tx1_f -= rhs1/D1
    #   tx2_f = tx2Eq(1,Nkp,tx1,tx2,t1,t2,IABC,IJAB,IJKA,F_ae,F_mi,F_me,G_ae,G_mi,
    #                 W_mnij,W_abef,W_efam,W_iemn,W_mbej,D2)
    #   tx2_f -= rhs2/D2
    #   # Evaluate convergence
    #   not_conver, E_Corr2, tx1, tx2 = AmpConv(AmpType,O,Nkp,tx1,tx2,tx1_f,tx2_f,tau,
    #                                           Fock,rhs1,rhs2,ThrE,ThrA,E_Corr1)
    #   del tx1_f, tx2_f, G_ae, G_mi 
    #   tx1, tx2, DoDIIS = DIIS(O,V,N,MaxD,ThrA,RepD,tx1,tx2,B_mat,st1,st2,
    #                           e_DIIS,DoDIIS)
    #   a1 = tx1
    #   a2 = tx2
    else :
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f"Amplitude type {AmpType} is not implemented. ")
      exit()
    # for i in range(2*O):
    #   for a in range(2*V):
    #     if(abs(a1[i,a])> 1.e-5):
    #       with open(f"{molecule}.txt","a") as writer:
    #         writer.write(f"T({i},{a}) = {a1[i,a]:.6f}\n")
        
    textA = f"Iter. {N}: DIIS = {DoDIIS}, DE({AmpType}) {E_Corr2:.10f}, E({AmpType}-CCSD): {scfE.real+E_Corr2:.10f}"
    
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"{textA}, Time: {time.time()-start:.2f}s\n")
  if(not_conver):
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"{AmpType} amplitude equations convergence failure\n")
    exit()
  else:
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"{AmpType} amplitude equations converged in {time.time()-start0:.2f}s\n\n")
  del st1, st2, e_DIIS, B_mat
  return a1, a2

##########################################################################
# Evaluate convergence criteria and update amplitudes for amplitude
# iterations
##########################################################################
def AmpConv(AmpType,O,Nkp,a1,a2,a1_f,a2_f,tau,Fock,I1Int,I2Int,ThrE,ThrA,
            E_Corr1):
  DiffA1 = abs(np.max(abs(a1_f.real)-abs(a1.real)))
  DiffA2 = abs(np.max(abs(a2_f.real)-abs(a2.real)))
  a1RMSE = (np.sum(((a1_f.real-a1.real)**(2))/(np.size(a1))))**(1/2)
  a2RMSE = (np.sum(((a2_f.real-a2.real)**(2))/(np.size(a2))))**(1/2)
  a1 = np.copy(a1_f)
  a2 = np.copy(a2_f)
  NkpC = Nkp*Nkp*Nkp
  if(AmpType == "T"):
#  if(AmpType == "T" or AmpType == "L"):
    # Here I2int should be the IJAB integrals
    E_Corr2 = E_CCSD(O,Nkp,Fock,a1,I2Int,tau)
  elif(AmpType == "L"):
    # Here I2int should be the IJAB integrals
    E_Corr2 = E_CCSD(O,Nkp,np.conjugate(Fock),a1,np.conjugate(I2Int),tau)
  elif (AmpType == "Tx"):
    # Here I1/2int should be the right hand side perturbations
    E_Corr2 = -0.25*np.einsum('ijab,ijab->',I2Int,a2,optimize=True)/NkpC
    E_Corr2 -= np.einsum('ia,ia->',I1Int,a1,optimize=True)/Nkp
  # E_Corr2 = E_Corr2
  E_Corr2 = E_Corr2.real
  DiffE = abs(E_Corr2-E_Corr1)
  print(f"Diffs {DiffE} {DiffA1} {DiffA2} {a1RMSE} {a2RMSE}")
  not_conver = DiffE> ThrE or DiffA1> ThrA*10 or DiffA2> ThrA*10 or a1RMSE> ThrA or a2RMSE> ThrA
  E_Corr1 = E_Corr2
  return not_conver, E_Corr2, a1, a2

##########################################################################
# DIIS Extrapolation
##########################################################################
def DIIS(O,V,Iter,MaxD,Thr,RepD,amp1,amp2,B,st1,st2,e_DIIS,DoDIIS):
  # Iter: current iteration
  # MaxD: size of the extrapolation space + 1 (for the constraint)
  # Thr: threshold on error to activate DIIS step
  # RepD: perform extrapolation every RepD iterations
  # amp1/2: amplitudes to extrapolate
  # st1/2: saved amplitudes from previous iterations
  # e_DIIS: save errors between iterations
  # B: DIIS matrix
  amp_type = amp1.dtype
  if(amp_type != amp2.dtype):
    print(f"Amplitude type mismatch: a1={amp_type} vs a2={amp2.dtype}")
    exit()
  ThrD = Thr/100
  st1.append(list(amp1.flatten()))
  st2.append(list(amp2.flatten()))
  if len(st1)!= len(st2):
    print(f"String length mismatch in DIIS: {len(st1)}, {len(st2)}\n")
    exit()
  ev = list(np.array(st1[len(st1) - 1]) - np.array(st1[len(st1) - 2])) + list(np.array(st2[len(st2) - 1]) - np.array(st2[len(st2) - 2]))
  e_DIIS.append(ev)
  if len(st1) > MaxD:
    # Remove the oldest information 
    del st1[0]
    del st2[0]
    del e_DIIS[0]
  e_DIIS = np.array(e_DIIS)
  DoDIIS = "F"
  if len(st1)==MaxD and (Iter%RepD==0):
    B[:MaxD-1,:MaxD-1] += np.einsum('ik,jk->ij',np.conjugate(e_DIIS),e_DIIS,optimize=True)
    B[MaxD-1,:] = 1
    B[:,MaxD-1] = 1
    B[MaxD-1,MaxD-1] = 0
#    print(f"B matrix:\n {B}")
    rhs = np.zeros(MaxD)
    rhs[MaxD-1] = 1
    ETest = np.max(abs(B[:MaxD-1,:MaxD-1]))
    if ETest >= ThrD:
      csol = np.linalg.solve(B,rhs)
      csum = np.sum(csol[:MaxD-1])
      if(abs(csum-1)>ThrD):
        print(f"Issue with coefficients in DIIS: sum_C = {csum}\n")
        exit()
      t1d = np.zeros((len(st1[0])),dtype=amp_type)
      t2d = np.zeros((len(st2[0])),dtype=amp_type)
      for p in range(MaxD-1):
        t1d += np.array(st1[p+1]) * csol[p]
        t2d += np.array(st2[p+1]) * csol[p]
      amp1 = np.reshape(t1d,((2*O),(2*V)))
      amp2 = np.reshape(t2d,((2*O),(2*O),(2*V),(2*V)))
      del t1d, t2d
      DoDIIS = "T"
  return amp1, amp2, DoDIIS

##########################################################################
# tau_tilde intermediate for CCSD T equations
##########################################################################
def tau_tildeEq(T, Nkp, t1, t2):
  if T==1:
    tau_tilde = np.copy(t2)
    tau_tilde += 0.5*np.einsum('ia,jb->ijab',t1,t1,optimize=True)*Nkp 
    tau_tilde -= 0.5*np.einsum('ib,ja->ijab',t1,t1,optimize=True)*Nkp
  return tau_tilde

##########################################################################
# tau intermediate for CCSD T equations
##########################################################################
def tauEq(T, Nkp, t1, t2):
  if T==1:
    tau = np.copy(t2)
    tau += np.einsum('ia,jb->ijab',t1,t1,optimize=True)*Nkp
    tau -= np.einsum('ib,ja->ijab',t1,t1,optimize=True)*Nkp
  return tau

##########################################################################
# F and W intermediates for CCSD T equations
##########################################################################
def T_interm(T, O, V, Nkp, Fock, t1, t2, IJKL, ABCD, IABC, IJAB, IABJ, IJKA,
            tau_tilde, tau):
  # O,V are assumed to be multiplied by Nkp in a PBC calculation
  O2=2*O
  V2=2*V
  NkpS = Nkp*Nkp
  if T==1:
    # F_ae
    F_ae = np.zeros((V2, V2),dtype=Fock.dtype)
    F_ae += (1 - np.eye(V2)) * Fock[O2:, O2:] #Add flag, function to set diagonal elements to zero
    F_ae -= 0.5 * np.einsum('me,ma->ae', Fock[:O2, O2:], t1, optimize=True)
    F_ae += np.einsum('mf,mafe->ae', t1, IABC, optimize=True)/Nkp
    F_ae -= 0.5 * np.einsum('mnaf,mnef->ae',tau_tilde,IJAB,optimize=True)/NkpS
    # F_mi
    F_mi = np.zeros((O2, O2),dtype=Fock.dtype)
    F_mi += (1 - np.eye(O2)) * Fock[:O2, :O2]
    F_mi += 0.5 * np.einsum('ie,me->mi', t1, Fock[:O2, O2:], optimize=True)
    F_mi += np.einsum('ne,mnie->mi', t1, IJKA, optimize=True)/Nkp
    F_mi += 0.5 * np.einsum('inef,mnef->mi', tau_tilde, IJAB, optimize=True)/NkpS
    # F_me
    F_me = np.zeros((O2, V2),dtype=Fock.dtype)
    F_me = np.copy(Fock[:O2, O2:])
    F_me += np.einsum('nf,mnef->me', t1, IJAB, optimize=True)/Nkp
    # W_mnij
    W_mnij = np.copy(IJKL)
    W_mnij += np.einsum('je,mnie->mnij', t1, IJKA, optimize=True)
    W_mnij -= np.einsum('ie,mnje->mnij', t1, IJKA, optimize=True)
    W_mnij += 0.5 * np.einsum('mnef,ijef->mnij', IJAB, tau, optimize=True)/Nkp
    # W_abef
    W_abef = np.copy(ABCD)
    W_abef += np.einsum('mb,maef->abef',t1,IABC,optimize=True)
    W_abef -= np.einsum('ma,mbef->abef',t1,IABC,optimize=True)
    # W_mbej
    W_mbej = np.copy(IABJ)
    W_mbej += np.einsum('jf,mbef->mbej', t1, IABC, optimize=True)
    W_mbej += np.einsum('nb,mnje->mbej', t1, IJKA, optimize=True)
    W_mbej -= 0.5 * np.einsum('jnfb,mnef->mbej', t2, IJAB, optimize=True)/Nkp
    W_mbej -= np.einsum('jf,nb,mnef->mbej', t1, t1, IJAB, optimize=True)/Nkp
  return F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej

#########################################################################
# CCSD T1 amplitude equation
#########################################################################
def t1Eq(T,O,Nkp,Fock,t1,t2,IABC,IJKA,IABJ,F_ae,F_mi,F_me,D1):
  if T==1:
    O2=2*O
    NkpS = Nkp*Nkp
    t1_f = np.copy(Fock[:O2, O2:])  
    t1_f += np.einsum('ie,ae->ia', t1, F_ae, optimize=True)
    t1_f -= np.einsum('ma,mi->ia', t1, F_mi, optimize=True)
    t1_f += np.einsum('imae,me->ia', t2, F_me, optimize=True)/Nkp
    t1_f -= 0.5 * np.einsum('imef,maef->ia',t2,IABC,optimize=True)/NkpS
    t1_f += 0.5 * np.einsum('mnae,nmie->ia',t2,IJKA,optimize=True)/NkpS
    t1_f += np.einsum('nf,nafi->ia', t1, IABJ,optimize=True)/Nkp
    t1_f /= D1
  return t1_f

#########################################################################
# CCSD T2 amplitude equation
#########################################################################
def t2Eq(T,Nkp,t1,t2,IABC,IJAB,IJKA,IABJ,tau,F_ae,F_mi,F_me,W_mnij,W_abef,
         W_mbej,D2):
  if T==1:
    NkpS = Nkp*Nkp
    # Constant term
    t2_f = np.copy(np.conjugate(IJAB))
    # P(ab) terms
    X1 = F_ae - 0.5*np.einsum('mb,me->be',t1,F_me,optimize=True)
    X2 = np.einsum('ijae,be->ijab',t2,X1,optimize=True)
    X2 -= np.einsum('ma,ijmb->ijab',t1,np.conjugate(IJKA),optimize=True)
    t2_f += X2 - np.transpose(X2,axes=(0,1,3,2))
    del X1, X2
    # P(ij) terms
    X1 = F_mi + 0.5*np.einsum('je,me->mj',t1,F_me,optimize=True)
    X2 = -np.einsum('imab,mj->ijab',t2,X1,optimize=True)
    X2 -= np.einsum('ie,jeab->ijab',t1,np.conjugate(IABC),optimize=True)
    t2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
    del X1, X2
    # P(ij,ab) terms
#    X1 = -np.einsum('ie,mbej->mbij',t1,np.conjugate(IABJ),optimize=True)
    X1 = -np.einsum('ie,mbej->mbij',t1,IABJ,optimize=True)
    X2 = np.einsum('imae,mbej->ijab',t2,W_mbej,optimize=True)/Nkp
#    X2 += np.einsum('ma,mbij->ijab',t1,np.conjugate(X1),optimize=True)
    X2 += np.einsum('ma,mbij->ijab',t1,X1,optimize=True)
    t2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
    t2_f -= np.transpose(X2,axes=(0,1,3,2))
    t2_f += np.transpose(X2,axes=(1,0,3,2))
    del X1, X2
    # tau terms
    t2_f += 0.5*np.einsum('ijef,abef->ijab',tau,W_abef,optimize=True)/Nkp
    t2_f += 0.5*np.einsum('mnab,mnij->ijab',tau,W_mnij,optimize=True)/Nkp
    # Divide by energy denominator
    t2_f /= D2    
  return t2_f

#########################################################################
# CCSD energy
#########################################################################
def E_CCSD(O,Nkp,Fock,t1,IJAB,tau):
  O2 = 2*O
  NkpC = Nkp*Nkp*Nkp
  E_Corr2_1 = np.einsum('ia,ia->', t1, np.conjugate(Fock[:O2, O2:]),optimize=True)/Nkp
  E_Corr2_2 = 0.25 * np.einsum('ijab,ijab->', tau,IJAB,optimize=True)/NkpC
  E_Corr2 = E_Corr2_1.real + E_Corr2_2.real
  print(f"E(corr) {E_Corr2_1} {E_Corr2_2} Mol: {E_Corr2_2/21}")
  return E_Corr2

#########################################################################
# Define constant intermediates for CCSD Lambda and response equations
#########################################################################
def Const_Interm(T,Nkp,t1,t2,tau,IJAB,IABJ,IJKA,IABC,F_ae,F_mi,F_me,W_mnij,
                 W_abef,W_mbej):
  if T==1:
    # Remember that the contraction for Lambda is over the opposite
    # one or two indices (same for W_mnij)
    F_ae -= 0.5*np.einsum('ma,me->ae',t1,F_me,optimize=True)
    # The sign of this terms is wrong in the paper
    F_mi += 0.5*np.einsum('me,ie->mi',F_me,t1,optimize=True)
    # Here we are forming the tilde-W_abef intermediate as in the
    # paper, at the cost of doing a o2v4 contraction once. The
    # tilde-W_nmij is already as in the paper, as we already doubled
    # the IJAB contribution for the t2 equations.
    W_abef += 0.5*np.einsum('mnab,mnef->abef',tau,IJAB,optimize=True)/Nkp
    W_mbej += 0.5*np.einsum('nmfe,jnbf->mbej',IJAB,t2,optimize=True)/Nkp
    # These intermediates are new
    W_efam = np.einsum('mnef,na->efam',t2,F_me,optimize=True)
    W_efam -= np.transpose(np.conjugate(IABC),axes=(2,3,1,0)) 
#    W_efam -= np.transpose(IABC,axes=(2,3,1,0)) 
    W_efam += np.einsum('efag,mg->efam',W_abef,t1,optimize=True)
    # This is the opposite of what's in Gauss' paper
    # W_efam -= 0.5*np.einsum('noef,noma->efam',tau,np.conjugate(IJKA),optimize=True)/Nkp
    W_efam -= 0.5*np.einsum('noef,noma->efam',tau,IJKA,optimize=True)/Nkp
    W_iemn = -np.einsum('mnef,if->iemn',t2,F_me,optimize=True)
    W_iemn += np.transpose(np.conjugate(IJKA),axes=(2,3,0,1)) 
#    W_iemn += np.transpose(IJKA,axes=(2,3,0,1)) 
    W_iemn -= np.einsum('iomn,oe->iemn',W_mnij,t1,optimize=True)
    W_iemn += 0.5*np.einsum('iefg,mnfg->iemn',IABC,tau,optimize=True)/Nkp
    # Create a temp intermediates
    WW_mbej = -np.einsum('mnef,njbf->mbej',IJAB,t2,optimize=True)/Nkp
    WW_mbej += IABJ
    X1 = - np.einsum('ne,nfam->efam',t1,WW_mbej,optimize=True)
    X1 += np.einsum('nega,mnfg->efam',IABC,t2,optimize=True)/Nkp
    X2 = X1 - np.transpose(X1,axes=(1,0,2,3))
    W_efam += X2
    del X1,X2
    X1 = np.einsum('mf,iefn->iemn',t1,WW_mbej,optimize=True)
    X1 += np.einsum('iomf,noef->iemn',IJKA,t2,optimize=True)/Nkp
    X2 = X1 - np.transpose(X1,axes=(0,1,3,2))
    W_iemn += X2
    del X1,X2,WW_mbej
    # Done and return
  return F_ae, F_mi, W_abef, W_mbej, W_efam, W_iemn

#########################################################################
# Define changing intermediates for CCSD Lambda equations
#########################################################################
def L_Interm(T, Nkp, t2, l2):
  if T==1:
    NkpS = Nkp*Nkp
    G_ae = -0.5*np.einsum('mnaf,mnef->ae',l2,t2,optimize=True)/NkpS
    G_mi = 0.5*np.einsum('mnef,inef->mi',t2,l2,optimize=True)/NkpS
  return G_ae, G_mi

#########################################################################
# CCSD Lambda1 amplitude equation
#########################################################################
def l1Eq(T,Nkp,t1,l1,l2,IJAB,IABC,IJKA,W_efam,W_iemn,W_mbej,F_ae,F_mi,
         F_me,G_ae,G_mi,D1):
  if T==1:
    NkpS = Nkp*Nkp
    l1_f = np.copy(F_me)  
    l1_f += np.einsum('ie,ea->ia',l1,F_ae,optimize=True)
    l1_f -= np.einsum('im,ma->ia',F_mi,l1,optimize=True)
    l1_f += np.einsum('me,ieam->ia',l1,W_mbej,optimize=True)/Nkp
    l1_f += 0.5*np.einsum('imef,efam->ia',l2,W_efam,optimize=True)/NkpS
    l1_f -= 0.5*np.einsum('iemn,mnae->ia',W_iemn,l2,optimize=True)/NkpS
    l1_f += np.einsum('ef,iefa->ia',G_ae,IABC,optimize=True)/Nkp
    l1_f += np.einsum('mn,imna->ia',G_mi,IJKA,optimize=True)/Nkp
    # I'm not sure why mi->im inversion makes a difference. Maybe it's
    # a limitation in the k-space sampling?
    # l1_f -= np.einsum('mn,mina->ia',G_mi,IJKA,optimize=True)/Nkp
    X1 = np.einsum('mf,fe->me',t1,G_ae,optimize=True)
    X1 -= np.einsum('mn,ne->me',G_mi,t1,optimize=True)
    l1_f += np.einsum('me,imae->ia',X1,IJAB,optimize=True)/Nkp
    del X1
    l1_f /= D1
  return l1_f

#########################################################################
# CCSD Lambda2 amplitude equation
#########################################################################
def l2Eq(T,Nkp,t1,l1,l2,IABC,IJAB,IJKA,F_ae,F_mi,F_me,G_ae,G_mi,W_mnij,
         W_abef,W_mbej,D2):
  if T==1:
#    l2_f = np.copy(np.conjugate(IJAB))
    l2_f = np.copy(IJAB)
    l2_f += 0.5*np.einsum('ijef,efab->ijab',l2,W_abef,optimize=True)/Nkp
    l2_f += 0.5*np.einsum('ijmn,mnab->ijab',W_mnij,l2,optimize=True)/Nkp
    # l2_f += 0.5*np.einsum('ijef,efab->ijab',l2,np.conjugate(W_abef),optimize=True)/Nkp
    # l2_f += 0.5*np.einsum('ijmn,mnab->ijab',np.conjugate(W_mnij),l2,optimize=True)/Nkp
    # P(ab) terms
    X1 = G_ae - np.einsum('mb,me->be',l1,t1,optimize=True)
    X2 = np.einsum('ijae,be->ijab',IJAB,X1,optimize=True)
    X2 -= np.einsum('ma,ijmb->ijab',l1,IJKA,optimize=True)
#    X2 += np.einsum('ijae,eb->ijab',l2,np.conjugate(F_ae),optimize=True) 
    X2 += np.einsum('ijae,eb->ijab',l2,F_ae,optimize=True) 
    l2_f += X2 - np.transpose(X2,axes=(0,1,3,2))
    del X1, X2
    # P(ij) terms
    X1 = G_mi + np.einsum('me,je->mj',t1,l1,optimize=True)
    X2 = np.einsum('imab,mj->ijab',IJAB,X1,optimize=True)
    X2 += np.einsum('ie,jeab->ijab',l1,IABC,optimize=True)
    X2 += np.einsum('imab,jm->ijab',l2,F_mi,optimize=True) 
    l2_f += np.transpose(X2,axes=(1,0,2,3)) - X2 
    del X1, X2
    # P(ij,ab) terms
    X2 = np.einsum('imae,jebm->ijab',l2,W_mbej,optimize=True)/Nkp
#    X2 += np.einsum('ia,jb->ijab',l1,F_me,optimize=True)
    X2 += np.einsum('ia,jb->ijab',l1,F_me,optimize=True)*Nkp
    l2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
    l2_f -= np.transpose(X2,axes=(0,1,3,2))
    l2_f += np.transpose(X2,axes=(1,0,3,2))
    del X2
    # Divide by energy denominator
    l2_f /= D2    
  return l2_f

#########################################################################
# Form constant terms based on 1e perturbation X:
# <S|e^{-T}Xe^{T}|0> and <D|e^{-T}Xe^{T}|0>.
#########################################################################
def pert_rhs(T, t1, t2, X_ij, X_ia, X_ab):
  if T==1:
  # # X is supposed to be in MO basis and already divided in oo, ov, and vv blocks
  #   X_ij, X_ia, X_ab = getpert(Coeff,pert_type,i_pert)
    # Singles
    rhs1 = np.copy(X_ia) 
    rhs1 += np.einsum('kc,ikac->ia',X_ia,t2,optimize=True)
    rhs1 -= np.einsum('kc,ic,ka->ia',X_ia,t1,t1,optimize=True)
    rhs1 += np.einsum('ic,ac->ia',t1,X_ab,optimize=True)
    rhs1 -= np.einsum('ki,ka->ia',X_ij,t1,optimize=True)
    # Doubles
    # P(ij) terms: -P(ij) t(kjab)(X(ik)+X(kc)t(ic))
    X1 = np.copy(X_ij) + np.einsum('ic,kc->ik',t1,X_ia,optimize=True)
    X2 = -np.einsum('ik,kjab->ijab',X1,t2,optimize=True)
    rhs2 = X2 - np.transpose(X2,axes=(1,0,2,3))
    # P(ab) terms: P(ab) t(ijac)(X(cb)-X(kc)t(kb))
    X1 = np.copy(X_ab) - np.einsum('kc,kb->cb',X_ia,t1,optimize=True)
    X2 = np.einsum('ijac,cb->ijab',t2,X1,optimize=True)
    rhs2 += X2 - np.transpose(X2,axes=(0,1,3,2))
    del X1, X2
  return rhs1, rhs2

#########################################################################
# CCSD Tx1 (or EOM R1) amplitude equation
#########################################################################
def tx1Eq(T,Nkp,tx1,tx2,t1,IABC,IJKA,W_mbej,F_ae,F_mi,F_me,G_ae,G_mi,D1):
  # Constant term needs to be added outside (as it's not in the EOM eqs.)
  # It requires getting G_ae, G_mi = L_Interm(T, Nkp, IJAB, tx2)
  if T==1:
    NkpS = Nkp*Nkp
    # tx1_f = np.copy(F_me) #this one needs to be checked! 
    tx1_f = np.einsum('ie,ae->ia',tx1,F_ae,optimize=True)
    tx1_f -= np.einsum('mi,ma->ia',F_mi,tx1,optimize=True)
    tx1_f += np.einsum('me,maei->ia',tx1,W_mbej,optimize=True)/Nkp
    tx1_f += np.einsum('imae,me->ia', tx2, F_me, optimize=True)/Nkp
    tx1_f -= 0.5 * np.einsum('imef,maef->ia', tx2, IABC,optimize=True)/NkpS
    tx1_f += 0.5 * np.einsum('nmea,nmie->ia', tx2, IJKA,optimize=True)/NkpS
    tx1_f += np.einsum('ib,ab->ia',t1,G_ae,optimize=True)
    tx1_f -= np.einsum('ji,ja->ia',G_mi,t1,optimize=True)
    tx1_f /= D1
  return tx1_f

#########################################################################
# CCSD Tx2 (or EOM R2) amplitude equation
#########################################################################
def tx2Eq(T,Nkp,tx1,tx2,t1,t2,IABC,IJAB,IJKA,F_ae,F_mi,F_me,G_ae,G_mi,
          W_mnij,W_abef,W_efam,W_iemn,W_mbej,D2):
  # Constant term needs to be added outside (as it's not in the EOM eqs.)
  # It requires getting G_ae, G_mi = L_Interm(T, Nkp, IJAB, tx2)
  if T==1:
    NkpS = Nkp*Nkp
    #tx2_f = np.copy(IJAB) #this one needs to be checked! 
    tx2_f = 0.5*np.einsum('ijef,abef->ijab',tx2,W_abef,optimize=True)/Nkp
    tx2_f += 0.5*np.einsum('mnij,mnab->ijab',W_mnij,tx2,optimize=True)/Nkp
    # P(ij) terms
    X0 = np.einsum('kc,kmcd->md',tx1,IJAB,optimize=True)/Nkp
    X1 = G_mi + np.einsum('md,jd->mj',X0,t1,optimize=True)
    X1 += np.einsum('kc,mkjc->mj',tx1,IJKA,optimize=True)/Nkp
    X2 = -np.einsum('imab,mj->ijab',t2,X1,optimize=True)
    X2 += np.einsum('ic,abcj->ijab',tx1,W_efam,optimize=True)
    X2 -= np.einsum('imab,mj->ijab',tx2,F_mi,optimize=True)
    tx2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
    del X1,X2
    # P(ab) terms
    X1 = G_ae - np.einsum('mb,md->bd',t1,X0,optimize=True)
    X1 += np.einsum('kc,kbcd->bd',tx1,IABC,optimize=True)/Nkp
    X2 = np.einsum('ijae,be->ijab',t2,X1,optimize=True)
    X2 -= np.einsum('ka,kbij->ijab',tx1,W_iemn,optimize=True)
    X2 += np.einsum('ijae,be->ijab',tx2,F_ae,optimize=True)
    tx2_f += X2 - np.transpose(X2,axes=(0,1,3,2))
    del X0,X1,X2
    # P(ij,ab) terms
    X2 = np.einsum('imae,mbej->ijab',tx2,W_mbej,optimize=True)/Nkp
    #+ np.einsum('ia,jb->ijab',tx1,F_me,optimize=True) # not sure about this one because it's disconnected
    tx2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
    tx2_f -= np.transpose(X2,axes=(0,1,3,2))
    tx2_f += np.transpose(X2,axes=(1,0,3,2))
    del X2
    # Divide by energy denominator
    tx2_f /= D2    
  return tx2_f
# 
# #########################################################################
# # CCSD Xi amplitudes for LR and EOM gradients
# #########################################################################
# def Xi(T, tx1, tx2, l1, l2, t1, IABC, IJAB, IJKA, F_ae, F_mi, F_me,
#        W_mbej, D2):
#   # L can be the ground or excited state Lambda amplitudes
#   # Tx can be the LR Tx or the EOM R amplitudes
#   if T==1:
#     # Term 1
#     # Xi1 : R1*Lg*H
#     #       -R(jb)*(Lg(ja)*X(ib) + Lg(ib)*X(ja))
#     #             _                  _                 _
#     #       <0|L[[H,O1],R]|0> = <0|L(HR)conn|1> - <0|L(HR)disc|1>
#     # P(ab) terms
#     X2 = np.einsum('ijae,eb->ijab',l2,F_ae,optimize=True) 
#     X1 = X2 - np.transpose(X2,axes=(0,1,3,2))
#     del X2
#     # P(ij) terms
#     X2 = np.einsum('imab,jm->ijab',l2,F_mi,optimize=True) 
#     X1 += np.transpose(X2,axes=(1,0,2,3)) - X2 
#     del X2
#     # P(ij,ab)-like terms
#     X2 = np.einsum('imae,jebm->ijab',l2,W_mbej,optimize=True)
#     X2 += np.einsum('ia,jb->ijab',l1,F_me,optimize=True)
#     X1 += X2 + np.transpose(X2,axes=(1,0,3,2))
#     del X2
#     X1 -= l2*D2
#     Xi1 = -np.einsum('ijab,jb->ia',X1,tx1,optimize=True)
#     del X1
#     # Term 2
#     # Xi1 : -Lg(ijdb)t(md)R(kjcb)<mk||ac>
#     X1 = np.einsum('kjcb,mkac->jmab',tx2,IJAB,optimize=True)
#     X2 = np.einsum('ijdb,md->ijmb',l2,t1,optimize=True)
#     Xi1 -= np.einsum('ijmb,jmab->ia',X2,X1,optimize=True)
#     del X1, X2
#     # Term 3
#     # Xi1 : Lg(jibd)R(jkbc)<kd||ca>
#     #     : Lg(jmba)R(jkbc)[<ki||mc>+t(md)<ki||dc>]
#     # Xi2 : P(ij,ab)Lg(kica)[R(kmcd)-R(mc)t(kd)-t(mc)R(kd)]<mj||db>
#     X1 = np.einsum('jibd,jkbc->ikdc',l2,tx2,optimize=True)
#     Xi1 += np.einsum('ikdc,kdca->ia',X1,IABC,optimize=True)
#     X2 = IJKA + np.einsum('md,kidc->kimc',t1,IJAB,optimize=True)
#     Xi1 += np.einsum('mkac,kimc->ia',X1,X2,optimize=True)
#     del X2
#     X2 = np.einsum('kica,mc->kima',l2,tx1,optimize=True)
#     X1 -= np.einsum('kima,kd->imad',X2,t1,optimize=True)
#     X2 = np.einsum('kica,mc->kima',l2,t1,optimize=True)
#     X1 -= np.einsum('kima,kd->imad',X2,tx1,optimize=True)
#     del X2
#     X2 = np.einsum('imad,mjdb->ijab',X1,IJAB,optimize=True)
#     Xi2 = X2 - np.transpose(X2,axes=(0,1,3,2))
#     Xi2 -= np.transpose(X2,axes=(1,0,2,3))
#     Xi2 += np.transpose(X2,axes=(1,0,3,2))
#     del X1, X2
#     # Term 4
#     # Xi1 :  1/2R(jkbc)Lg(jkbd)<di||ca>
#     #     :  1/2R(jmbd)Lg(jmba)t(kc)<ik||cd>
#     #     : -1/2R(jmbd)Lg(jmbc)t(kc)<ik||ad>
#     # Xi2 : -1/2P(ab)R(kmcd)Lg(kmbd)<ij||ac>
#     X1 = 0.5*np.einsum('jkbc,jkbd->cd',tx2,l2,optimize=True)
#     Xi1 -= np.einsum('cd,idca->ia',X1,IABC,optimize=True)
#     X2 = np.einsum('kc,ikcd->id',t1,IJAB,optimize=True)
#     Xi1 += np.einsum('da,id->ia',X1,X2,optimize=True)
#     del X2
#     X2 = np.einsum('dc,kc->kd',X1,t1,optimize=True)
#     Xi1 -= np.einsum('kd,ikad->ia',X2,IJAB,optimize=True)
#     del X2
#     X2 = -np.einsum('cb,ijac->ijab',X1,IJAB,optimize=True)
#     Xi2 += X2 - np.transpose(X2,axes=(0,1,3,2))
#     del X1, X2
#     # Term 5
#     # Xi1 : -1/2Lg(id)R(kmcd)<km||ca>
#     # Xi2 : -1/2P(ab)Lg(ijad)R(kmcd)<km||cb>
#     #     :    -P(ab)Lg(ijad)R(md)t(kc)<km||cb>
#     #     :    -P(ab)Lg(ijad)t(md)R(kc)<km||cb>
#     #     :     P(ab)Lg(ijad)R(kc)<kd||cb>
#     X1 = -0.5*np.einsum('kmcd,kmca->da',tx2,IJAB,optimize=True)
#     Xi1 += np.einsum('id,da->ia',l1,X1,optimize=True)
#     X2 = np.einsum('kc,kmcb->mb',t1,IJAB,optimize=True)
#     X1 -= np.einsum('md,mb->db',tx1,X2,optimize=True)
#     del X2
#     X2 = np.einsum('kc,kmcb->mb',tx1,IJAB,optimize=True)
#     X1 -= np.einsum('md,mb->db',t1,X2,optimize=True) 
#     del X2
#     X1 += np.einsum('kc,kdcb->db',tx1,IABC,optimize=True) 
#     X2 = np.einsum('ijad,db->ijab',l2,X1,optimize=True)
#     Xi2 += X2 - np.transpose(X2,axes=(0,1,3,2))
#     del X1, X2
#     # Term 6
#     # Xi1 :  1/2Lg(jkbc)R(jmbc)<im||ka>
#     #     : -1/2Lg(jibc)R(jmbc)t(kd)<mk||ad>
#     #     : -1/2Lg(jkbc)R(jmbc)t(kd)<im||ad>
#     #     :     Lg(jb)R(jmbd)<im||ad>
#     # Xi2 : -1/2P(ij)Lg(jmcd)R(kmcd)<ik||ab>
#     X1 = 0.5*np.einsum('jkbc,jmbc->km',l2,tx2,optimize=True)
#     Xi1 += np.einsum('km,imka->ia',X1,IJKA,optimize=True)
#     X2 = np.einsum('ik,jkab->ijab',X1,IJAB,optimize=True)
#     Xi2 += X2 - np.transpose(X2,axes=(1,0,2,3))
#     del X2
#     X2 = np.einsum('kd,mkad->ma',t1,IJAB,optimize=True)
#     Xi1 -= np.einsum('im,ma->ia',X1,X2,optimize=True)
#     del X2
#     X2 = -np.einsum('km,kd->md',X1,t1,optimize=True)
#     X2 += np.einsum('jb,jmbd->md',l1,tx2,optimize=True)
#     Xi1 += np.einsum('md,imad->ia',X2,IJAB,optimize=True)
#     del X1, X2
#     # Term 7
#     # Xi1 : -1/2Lg(ka)R(kmcd)<im||cd>
#     # Xi2 : -1/2P(ij)Lg(kjab)R(kmcd)<im||cd>
#     #     :    -P(ij)Lg(kjab)R(kc)t(md)<im||cd>
#     #     :    -P(ij)Lg(kjab)t(kc)R(md)<im||cd>
#     #     :    -P(ij)Lg(kjab)R(mc)<im||kc>
#     X1 = -0.5*np.einsum('imcd,kmcd->ik',IJAB,tx2,optimize=True)
#     Xi1 += np.einsum('ik,ka->ia',X1,l1,optimize=True)
#     X2 = np.einsum('md,imcd->ic',t1,IJAB,optimize=True)
#     X1 -= np.einsum('ic,kc->ik',X2,tx1,optimize=True)
#     del X2
#     X2 = np.einsum('md,imcd->ic',tx1,IJAB,optimize=True)
#     X1 -= np.einsum('ic,kc->ik',X2,t1,optimize=True)
#     del X2
#     X1 -= np.einsum('imkc,mc->ik',IJKA,tx1,optimize=True)
#     X2 = np.einsum('ik,kjab->ijab',X1,l2,optimize=True)
#     Xi2 += X2 - np.transpose(X2,axes=(1,0,2,3))
#     del X1, X2
#     # Term 8
#     # Xi1 : -1/4Lg(jkac)R(jkbd)<ic||bd>
#     #     :  1/4Lg(jkac)t(mc)R(jkbd)<im||bd>
#     # Xi2 :  1/4Lg(kmab)R(kmcd)<ij||cd>
#     #     :     Lg(kmab)R(kc)t(md)<ij||cd>
#     #     :     Lg(kmab)R(kc)<ij||cm>
#     X1 = 0.25*np.einsum('ijcd,kmcd->ijkm',IJAB,tx2,optimize=True)
#     X2 = np.einsum('imjk,mc->icjk',X1,t1,optimize=True)
#     X2 -= 0.25*np.einsum('icbd,jkbd->icjk',IABC,tx2,optimize=True)
#     Xi1 += np.einsum('icjk,jkac->ia',X2,l2,optimize=True)
#     del X2
#     X2 = IJKA + np.einsum('ijdc,md->ijmc',IJAB,t1,optimize=True)
#     X1 -= np.einsum('ijmc,kc->ijkm',X2,tx1,optimize=True)
#     Xi2 += np.einsum('ijkm,kmab->ijab',X1,l2,optimize=True)
#     del X1, X2
#     # Term 9
#     # Xi1 : 1/4Lg(kicd)R(jmcd)<jm||ka>
#     #     : 1/4Lg(kicd)R(jmcd)t(kb)<jm||ba>
#     # Xi2 : 1/4Lg(ijcd)R(kmcd)<km||ab>
#     #     :    Lg(ijcd)R(kc)t(md)<km||ab>
#     X1 = 0.25*np.einsum('ijcd,kmcd->ijkm',l2,tx2,optimize=True)
#     X2 = IJKA + np.einsum('jmba,kb->jmka',IJAB,t1,optimize=True)
#     Xi1 += np.einsum('kijm,jmka->ia',X1,X2,optimize=True)
#     del X2
#     X2 = np.einsum('ijcd,kc->ijkd',l2,tx1,optimize=True)
#     X1 += np.einsum('ijkd,md->ijkm',X2,t1,optimize=True)
#     Xi2 += np.einsum('ijkm,kmab->ijab',X1,IJAB,optimize=True)
#     del X1, X2
#     # Term 10
#     # Xi2 :  P(ij,ab)Lg(ia)R(kc)<kj||cb>
#     #     :    -P(ab)Lg(ka)R(kc)<ij||cb>
#     #     :    -P(ij)Lg(ic)R(kc)<kj||ab>
#     X1 = np.einsum('kc,kjcb->jb',tx1,IJAB,optimize=True)
#     X2 = np.einsum('ia,jb->ijab',l1,X1,optimize=True)
#     Xi2 += X2 - np.transpose(X2,axes=(0,1,3,2))
#     Xi2 -= np.transpose(X2,axes=(1,0,2,3))
#     Xi2 += np.transpose(X2,axes=(1,0,3,2))
#     del X1, X2
#     X1 = np.einsum('ka,kc->ac',l1,tx1,optimize=True)
#     X2 = -np.einsum('ac,ijcb->ijab',X1,IJAB,optimize=True)
#     Xi2 += X2 - np.transpose(X2,axes=(0,1,3,2))
#     del X1, X2
#     X1 = np.einsum('ic,kc->ik',l1,tx1,optimize=True)
#     X2 = -np.einsum('ik,kjab->ijab',X1,IJAB,optimize=True)
#     Xi2 += X2 - np.transpose(X2,axes=(1,0,2,3))
#     del X1, X2
#     # Term 11
#     # Xi2 :  P(ij,ab)Lg(ikac)R(mc)<jm||kb>
#     #     : -P(ij,ab)Lg(ikac)R(kd)<jc||db>
#     X1 = np.einsum('mc,jmkb->jbkc',tx1,IJKA,optimize=True)
#     X1 -= np.einsum('kd,jcdb->jbkc',tx1,IABC,optimize=True)
#     X2 = np.einsum('ikac,jbkc->ijab',l2,X1,optimize=True)
#     Xi2 += X2 - np.transpose(X2,axes=(0,1,3,2))
#     Xi2 -= np.transpose(X2,axes=(1,0,2,3))
#     Xi2 += np.transpose(X2,axes=(1,0,3,2))
#     del X1, X2
#     # Term 12
#     # Xi2 : -Lg(ijcd)R(kc)<kd||ab>
#     X1 = np.einsum('ijcd,kc->ijkd',l2,tx1,optimize=True)
#     Xi2 -= np.einsum('ijkd,kdab->ijab',X1,IABC,optimize=True)
#   return Xi1, Xi2
# 
# #########################################################################
# # CCSD rho1 transition density for LR and EOM 
# #########################################################################
# def TrDen1(T, O2, NB2, tx1, tx2, l1, l2, t1, t2):
#   # For now, implement only the LR term: <0|(1+Lg)[e^{-T}{p^{+}q}e^{T},X^{B}]|0>
#   # Density is returned in MO basis
#   if T==1:
#     rho1 = np.zeros((NB2,NB2))
#     # AI block
#     # There are no contributions for the LR function
#     #
#     # IJ block
#     # - R(ic)L(jc) -1/2Rx(ikcd)Lg(jkcd)
#     X1 = -np.einsum('ic,jc->ij',tx1,l1,optimize=True)
#     X1 -= 0.5*np.einsum('ikcd,jkcd->ij',tx2,l2,optimize=True)
#     rho1[:O2,:O2] = np.copy(X1)
#     # AB block
#     # L(ka)R(kb) +1/2Lg(kmca)Rx(kmcb)
#     X2 = np.einsum('ka,kb->ab',l1,tx1,optimize=True)
#     X2 += 0.5*np.einsum('kmca,kmcb->ab',l2,tx2,optimize=True)
#     rho1[O2:,O2:] = np.copy(X2)
#     # IA block
#     # Rx(ia) -t(ka)[Rx(ic)Lg(kc) + 1/2Rx(imcd)Lg(kmcd)]
#     # -t(ic)[Rx(ka)Lg(ck) + 1/2Rx(kmad)Lg(cdkm)]
#     rho1[:O2,O2:] = np.copy(tx1)
#     rho1[:O2,O2:] += np.einsum('ik,ka->ia',X1,t1,optimize=True)
#     rho1[:O2,O2:] -= np.einsum('ic,ca->ia',t1,X2,optimize=True)
#     del X1, X2
#     # + Rx(ikac)Lg(ck)
#     rho1[:O2,O2:] += np.einsum('ikac,kc->ia',tx2,l1,optimize=True)
#     # -1/2t(kmad)Rx(ic)Lg(kmcd)
#     X2 = 0.5*np.einsum('kmca,kmcd->ad',l2,t2,optimize=True)
#     rho1[:O2,O2:] -= np.einsum('id,ad->ia',tx1,X2,optimize=True)
#     del X2
#     # -1/2 t(imcd)Rx(ka)Lg(kmcd)
#     X1 = 0.5*np.einsum('imcd,kmcd->ik',t2,l2,optimize=True)
#     rho1[:O2,O2:] -= np.einsum('ik,ka->ia',X1,tx1,optimize=True)
#   return rho1

#########################################################################
# Function to put a linearized matrix into lower triangular form and
# then square it
#########################################################################
def square_m(NDim,Lin,MType,Mat,MatSq):
  #
  # NDim : leading dimension of the matrix
  # Lin: T = Mat is linearized and needs to be reshaped into square.
  #      F = Mat is already in square form but stored lower/upper triangular
  # MType: Sym  = Square Mat in symmetrical form
  #        ASym = Square Mat in anti-symmetrical form
  #        Herm = Square Mat in Hermitian form
  #        AHer = Square Mat in anti-Hermitian form
  if (Lin):
    off = 0
    for N in range(NDim):
      MatSq[N,:N+1] = np.copy(Mat[off:off+N+1])
      off += N+1
  if(MType == "Sym"):
    MatSq = MatSq + MatSq.T
  elif(MType == "ASym"):
    MatSq = MatSq - MatSq.T
  elif(MType == "Herm"):
    MatSq = MatSq + np.conjugate(MatSq).T
  elif(MType == "AHerm"):
    MatSq = MatSq - np.conjugate(MatSq).T
  else:
    print(f"Wrong matrix type in square_m: {MType}")
    exit()
  np.fill_diagonal(MatSq,np.diag(MatSq)/2)
  return MatSq

#########################################################################
# Function to form the auxiliary arrays for the Fourier Transform
#########################################################################
def fill_kl(ipbc):
  # ipbc: integer array containing PBC info
  # kp: output array with k-point values in [-pi,0] range
  # l_list: output integer array with index over repeated cells: [0,+1,-1,+2,-2,...
  nmtpbc = ipbc[1]
  nrecip = ipbc[9]
  #Build l_list
  l_list = [0]
  for i in range(1,nmtpbc,2):
    l_list.append(-(i//2 + 1))
    l_list.append(i//2 + 1)
#  print(f"L list {l_list}")
  #Build kp
  kp = []
  if nrecip == 1:
    kp = [0]
  elif nrecip % 2 == 0:
    for k in range(1, nrecip, 2):
      kp.append((np.pi * (k - nrecip) ) / nrecip)
    for k in range(1, nrecip, 2):
      kp.append((np.pi * (nrecip - k) ) / nrecip)
  elif nrecip % 2 != 0 and nrecip != 1:
    tmpn = np.ceil(nrecip/2) - 1
    for k in range(int(tmpn + 1)):
      kp.append((np.pi * (k - tmpn) ) / tmpn )
    for k in range(1,int(tmpn)):
      kp.append((np.pi * (tmpn - k) ) / tmpn )
  # if nrecip == 1:
  #   kp = [0]
  # elif nrecip % 2 == 0:
  #   for k in range(1, nrecip, 2):
  #     kp.append((np.pi * (k - nrecip) ) / nrecip)
  # elif nrecip % 2 != 0 and nrecip != 1:
  #   tmpn = np.ceil(nrecip/2) - 1
  #   for k in range(int(tmpn + 1)):
  #     kp.append((np.pi * (k - tmpn) ) / tmpn )
  # print(f"K list {len(kp)} {kp}")
  # Nkp = len(kp)
  # ind = 0
  # pi2 = 2*np.pi
  # for n in range(Nkp):
  #   for h in range(Nkp):
  #     for k in range(Nkp):
  #       for g in range(Nkp):
  #         kn = kp[n]
  #         kh = kp[h]
  #         kk = kp[k]
  #         kg = kp[g]
  #         ktot = kn-kh+kk-kg
  #         # if(n==0 and h ==1 and k ==3 and g == 2): print(f"0 1 3 2 K values {kn} {kh} {kk} {kg} {ktot} {abs(ktot % pi2)}")
  #         if(abs(ktot) < 1e-10 or abs(ktot % pi2) < 1e-10):
  #           ind +=1
  #           print(f"K values {kn} {kh} {kk} {kg} {ktot} {abs(ktot % pi2)}")
  #           print(f"K indexes {n} {h} {k} {g} ")
  # print(f"Total = {ind}")
  # kp = np.array(kp)
  # for n in range(Nkp):
  #   for h in range(Nkp):
  #     for k in range(Nkp):
  #       g = n+k-h
  #       if(g > (Nkp-1)):
  #         g = Nkp-1-g
  #       elif( g < 0 ):
  #         g = -(Nkp-1)+g
  #       kn = kp[n]
  #       kh = kp[h]
  #       kk = kp[k]
  #       kg = kp[g]
  #       # kg = round(kn+kk-kh,15)
  #       # if(kg>0): kg = -kg
  #       # mkg = abs(kg//(np.pi))
  #       # if(mkg!=0): kg /=mkg
  #       # g = kp.index(kg)
  #       print(f"K indexes n={n} h={h} k={k} g=n+k-h={g} ")
  #       print(f"K values {kn} {kh} {kk} {kg}")
  # exit()
  return kp, l_list

#########################################################################
# Function to perform the Fourier tranform of a 2-index array
#########################################################################
def fourier(FT,ipbc,MatIn):
  # FT: "Dir" = R -> k
  #     "Inv" = k -> R
  # ipbc: integer array containing PBC info
  # MatIn : input array (real for Dir/complex for Inv)
  # MatOut : output array (complex for Dir/real for Inv)
  kp, l_list = fill_kl(ipbc)
  co = np.einsum('k,l', kp, l_list, optimize=True)
  cof = np.cos(co) + 1j*np.sin(co)
  if(FT == "Dir"):
    MatOut = np.einsum('kl,ln->kn', cof, MatIn, optimize=True)
  elif(FT == "Inv"):
    MatOut = np.einsum('kl,kn->ln', cof, MatIn, optimize=True, dtype=real)
    print(f"Inverse FT needs to be tested")
    exit()
  else:
    print(f"Wrong call to fourier: {FT}")
    exit()
  return MatOut

#########################################################################
# Function for AO(k)<->MO(k) tranformation for a 2-index array
#########################################################################
def basis_tran(Opt,LinIn,LinOut,MType,NDim,Nkp,mocoef,MatIn):
  # Opt: "Dir" = AO(k)->MO(k)
  #      "Inv" = MO(k)->AO(k)
  # LinIn: T = MatIn is linearized and needs to be reshaped into square.
  #        F = MatIn is already in square form but stored lower/upper triangular
  # LinOut: T = MatOut is returned linearized 
  #         F = MatOut is returned square 
  # MType: Sym  = Square Mat in symmetrical form
  #        ASym = Square Mat in anti-symmetrical form
  #        Herm = Square Mat in Hermitian form
  #        AHer = Square Mat in anti-Hermitian form
  # NDim: leading dimension
  # Nkp: number og k points
  # mocoef: array to MO(k) coefficients
  # MatIn: input array [Nkp,:]
  # MatOut: output array
  # All arrays are expected to be complex
  if(Opt=="Dir"):
    if(LinIn):
      mat_k = np.zeros((Nkp,NDim,NDim),dtype=complex)
      for k in range(Nkp):
        mat_k[k,:,:] = square_m(NDim,True,MType,MatIn[k,:],mat_k[k,:,:])
      temp = np.einsum("kin,knm->kim", np.conjugate(mocoef), mat_k, optimize=True)
      MatOut = np.einsum("kjm,kim->kij", mocoef, temp, optimize=True)
    else:
      MatOut = np.einsum("kjm,kim->kij", mocoef, temp, optimize=True)
    if(LinOut):
      print(f"This LinOut:{LinOut} is not implemented yet in basis_tran")
      exit()
  elif(Opt=="Inv"):
    print(f"This Opt:{Opt} is not implemented yet in basis_tran")
    exit()
  else:
    print(f"Wrong Opt in basis_tran: {Opt}")
    exit()
  return MatOut
