#!/opt/intel/intelpython2/bin/python
import os
import sys
import time
import pandas as pd
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve, svds
from math import ceil

from scipy.constants import electron_mass as me
from scipy.constants import Planck as hplanck
from scipy.constants import epsilon_0
from scipy.constants import eV
from scipy.constants import e
from scipy.constants import pi
from scipy.constants import nano
from scipy.constants import micro
from scipy.constants import atto
from numpy.linalg import norm

from ctypes import *
mkl_root = os.popen("echo $LD_LIBRARY_PATH | tr ':' '\n' | grep mkl").read().rstrip()
if not mkl_root:
    mkl_root = "/opt/intel/mkl/lib/intel64"

mkl  = cdll.LoadLibrary(mkl_root+"/libmkl_rt.so")
N_mkl = mkl.mkl_get_max_threads()
print "Number of threads: ", N_mkl

extlib  = cdll.LoadLibrary("../lib/solver_dss.so")
solve_dss  = extlib.zsolve_dss_sy

if len(sys.argv)==0:
    print("Usage: pyrthon kirchhoff.py R_cut L_box logF_step {N_patch patch_id}\nWhere:")
    print("\tR_cut - cutoff distance to include electrostatic interactions")
    print("\tL_box - periodic box size (in micometers)")
    print("\tlogF_step - step on log-frequency scale (resolution)")
    print("\tN_patch - number of subdivisions along frequency scale (optional, choose >1 to split while processing in parallel threads)")
    print("\tpatch_id - index of the subdivision along frequency scale processed by this thread")

# Command line arguments:
d_range = float(sys.argv[1])
solver_type = int(1)
L_box   = float(sys.argv[2])*micro
log_dF = float(sys.argv[3])

if len(sys.argv)>4:
    Nptc = int(sys.argv[4])
    piter = int(sys.argv[5])
else:
    Nptc = int(1)
    piter = int(0)

R1=1
C1=0.0
R2=1e12
C2=1e-15

L_cnt   = 1.5*micro
R_cnt   = 4.75*nano
phi_avg = 3.5*eV
epsilon = 2.9 # PC
delta_min = 0.34*nano
delta_tun = 1.1*nano
s0 = 6.7*10**7  # S/m

#delta_eff = 50.0*nano
delta_eff = d_range*nano
S_min = np.pi*R_cnt*R_cnt
S_max = L_cnt*2*R_cnt

logF_max0 = 14
logF_min0 = -2

Nfreq = int((logF_max0-logF_min0)/log_dF)+1
patch = ceil(Nfreq*1.0/Nptc)
logF_minI = logF_min0 + piter*patch*log_dF
logF_maxI = logF_minI + patch*log_dF

# Scaling parameters:
A = 1.0/np.sqrt(2*me*phi_avg)*(hplanck/e)**2
B = 4*pi/hplanck*np.sqrt(2*me*phi_avg)
R0 = A*np.exp(B*delta_min)*delta_min/S_min
C0 = epsilon_0*epsilon*S_min/delta_min
w0 = 1.0/(R0*C0)
print "frequency scale: ",w0
print "impedance scale: ",R0,C0

def admittance_self_scl(la,a):
    # conductance of a CNT segment
    # la - ratio of segment length to radius
    # s0 - conductivity of CNT
    s = s0*pi*a/la
    return s*R0

def admittance_imag_scl(d,S):
    # capacitance according to de Vivo
    # d - distance between CNT surfaces
    # S - surface of contact 
    C = (S/S_min)/(d/delta_min) 
    if (C<0.005):
       C = 0.0
    return C

def admittance_real_scl(d,S):
    # resistance according to de Vivo
    # d - distance between CNT surfaces
    # S - surface of contact
    if (d<=delta_tun):
        R = np.exp(B*(d-delta_min))*(d/delta_min)/(S/S_min)
        G = (1/R)
        #print "Tunneling"
    else:
        G = 0
    return G

# Load lists of segment lengths and radii, scale radii:
LDATA = pd.read_table("segment.La.dat",skiprows=1,sep='\s+',header=None,index_col=False,usecols=[0,1,2,3],names=["IA","JA","LA","RA"])
i0 = np.array(LDATA.IA)
j0 = np.array(LDATA.JA)
L0  = np.array(LDATA.LA)
A0  = np.array(LDATA.RA)
print "Segment data read"
LDATA = None

# Load lists of contact distances and surface areas, correct distance:
RDATA = pd.read_table("internal.DS.dat",skiprows=1,sep='\s+',header=None,index_col=False,usecols=[0,1,2,3],names=["IA","JA","DA","SA"])
ia = np.array(RDATA.IA)
ja = np.array(RDATA.JA)
DA  = np.array(RDATA.DA)
SA  = np.array(RDATA.SA)
print "Internal contacts read"
RDATA = None

ib,DB,SB = np.loadtxt("source.DS.dat",skiprows=1,usecols=(1,2,3),unpack=1)
print "Contacts to source read"
ic,DC,SC = np.loadtxt("sink.DS.dat",skiprows=1,usecols=(1,2,3),unpack=1)
print "Contacts to sink read"

A0 *= micro

DA *= micro; DA -=2*R_cnt 
DB *= micro; DB -=2*R_cnt
DC *= micro; DC -=2*R_cnt
DA = np.where(DA>delta_min,DA,delta_min*np.ones(len(DA)))
DB = np.where(DB>delta_min,DB,delta_min*np.ones(len(DB)))
DC = np.where(DC>delta_min,DC,delta_min*np.ones(len(DC)))

# correct units:
SA *= micro*micro
SB *= micro*micro
SC *= micro*micro
SA = np.where(SA>S_min,SA,S_min*np.ones(len(SA)))
SB = np.where(SB>S_min,SB,S_min*np.ones(len(SB)))
SC = np.where(SC>S_min,SC,S_min*np.ones(len(SC)))
SA = np.where(SA<=S_max,SA,S_max*np.ones(len(SA)))
SB = np.where(SB<=S_max,SB,S_max*np.ones(len(SB)))
SC = np.where(SC<=S_max,SC,S_max*np.ones(len(SC)))

# select only contacts within effective distance:
ia = np.extract(DA<delta_eff,ia)
ja = np.extract(DA<delta_eff,ja)
ib = np.extract(DB<delta_eff,ib)
ic = np.extract(DC<delta_eff,ic)

SA = np.extract(DA<delta_eff,SA)
SB = np.extract(DB<delta_eff,SB)
SC = np.extract(DC<delta_eff,SC)

DA = np.extract(DA<delta_eff,DA)
DB = np.extract(DB<delta_eff,DB)
DC = np.extract(DC<delta_eff,DC)

# Create admittance matrices
w_arr = 2*np.pi*np.array([10**(log_dF*x+logF_minI) for x in np.arange(patch) ])
I_arr = 1j*np.zeros(len(w_arr))
R_arr = np.zeros(len(w_arr))

Adm0_real = R0/R1*np.ones(len(L0))
Adm0_imag = C1/C0*np.ones(len(L0))
AdmA_real = R0/R2*np.ones(len(DA))
AdmA_imag = C2/C0*np.ones(len(DA))
AdmB_real = R0/R2*np.ones(len(DB))
AdmB_imag = C2/C0*np.ones(len(DB))
AdmC_real = R0/R2*np.ones(len(DC))
AdmC_imag = C2/C0*np.ones(len(DC))

for I in np.arange(len(L0)):
    Adm0_real[I] = admittance_self_scl(L0[I],A0[I])

for I in np.arange(len(DA)):
    AdmA_real[I] = admittance_real_scl(DA[I],SA[I])
    AdmA_imag[I] = admittance_imag_scl(DA[I],SA[I])

for I in np.arange(len(DB)):
    AdmB_real[I] = admittance_real_scl(DB[I],SB[I])
    AdmB_imag[I] = admittance_imag_scl(DB[I],SB[I])

for I in np.arange(len(DC)):
    AdmC_real[I] = admittance_real_scl(DC[I],SC[I])
    AdmC_imag[I] = admittance_imag_scl(DC[I],SC[I])

jb = np.zeros(len(ib))
jc = np.zeros(len(ic))
# Set N to largest node encountered:
N = int(j0[-1])+1
print "Number of CNTS: ",N
              
# pre-allocate arrays using index map:
T_mtx   =   (   sps.identity(N)
              - sps.csr_matrix( ( np.ones(len(ia)),(ia,ja)),shape=(N,N))
              - sps.csr_matrix( ( np.ones(len(i0)),(i0,j0)),shape=(N,N))   )
NNZ     =   T_mtx.nnz
Ia = (c_int*(N+1))(*T_mtx.indptr)
Ja = (c_int*(NNZ))(*T_mtx.indices)

# logging:
log_filename = "kirchhoff.{}.log".format(piter)
logf=open(log_filename,"w")
logf.write("# Frequency\tZ_real\t(-1)Z_imag\tConverged\tRMSD\tElapsed\n")
logf.close()

for FI in np.arange(len(w_arr)):
    w = w_arr[FI] 
    
    Adm0_arr    =   Adm0_real + w/w0*Adm0_imag*1.0j
    AdmA_arr    =   AdmA_real + w/w0*AdmA_imag*1.0j
    AdmB_arr    =   AdmB_real + w/w0*AdmB_imag*1.0j
    AdmC_arr    =   AdmC_real + w/w0*AdmC_imag*1.0j
           
    L_mtx       =   sps.csr_matrix( ( Adm0_arr,(i0,j0)),shape=(N,N));
    X_mtx       =   sps.csr_matrix( ( AdmA_arr,(ia,ja)),shape=(N,N)); 
    A_mtx       =   L_mtx + X_mtx
    B_mtx       =   sps.csr_matrix( ( AdmB_arr,(ib,jb) ), shape=(N,1) ); 
    C_mtx       =   sps.csr_matrix( ( AdmC_arr,(ic,jc) ), shape=(N,1) );  

    # Divide each row by sum of elements:
    Sdns = A_mtx.sum(axis=1)+A_mtx.sum(axis=0).transpose()+B_mtx+C_mtx
    S_mtx = sps.csr_matrix(np.where(np.abs(Sdns)<=1e-12,1.0+0.0j,Sdns))
    RHS = B_mtx.todense()
    LHS_upr = sps.identity(N).multiply(S_mtx)-A_mtx
    
    print "Solving matrix equation for frequency f={}".format(w/2/np.pi)
    tp1 = time.time()
    if   (solver_type == 0):
        # Python direct sparse solver:
        LHS_mtx = LHS_upr-A_mtx.transpose()
        V_arr = spsolve(LHS_mtx,RHS)
        error = 0
    if   (solver_type == 1):
        # MKL direct sparse solver:       
        Ar = (c_double*(NNZ))(*np.real(LHS_upr.data))
        Ai = (c_double*(NNZ))(*np.imag(LHS_upr.data))
        Br = (c_double*N)(*np.real(RHS))
        Bi = (c_double*N)(*np.imag(RHS))
        Cr = (c_double*N)()
        Ci = (c_double*N)()
        
        error   = solve_dss( N, NNZ, byref(Ia), byref(Ja), byref(Ar), byref(Ai), byref(Br), byref(Bi), byref(Cr), byref(Ci))
        Vr = np.array(np.fromiter(Cr,dtype=np.float64,count=N))
        Vi = np.array(np.fromiter(Ci,dtype=np.float64,count=N))
        V_arr =  Vr+1j*Vi
    
    tp2 = time.time()
    
    V_sps = sps.csr_matrix(V_arr).transpose()
    RMS = norm((LHS_upr*V_sps-A_mtx.transpose()*V_sps).todense()-RHS)
    I_arr[FI]  = np.dot((np.ones(N)-V_arr),np.asarray(RHS).ravel())
    
    print "At frequency f={} estimated conductance ({}) converged: {}, elapsed time: {}, RMS = {}".format(w/2/np.pi,I_arr[FI],error==0, (tp2-tp1), RMS)   
    
    logf=open(log_filename,"a")
    logf.write("{freq:>10.3e}\t{real:>10.3e}\t{imag:>10.3e}\t{flag:>d}\t{rmsd:>10.3e}\t{time:>10.3f}\n".format(  freq = w/2/np.pi,
                                                                                                            real = R0*L_box*(np.real(1.0/I_arr[FI])),
                                                                                                            imag = R0*L_box*(-1*np.imag(1.0/I_arr[FI])),
                                                                                                            flag = error==0,
                                                                                                            rmsd = RMS,
                                                                                                            time = (tp2-tp1)    ))
    logf.close()
    
    R_arr[FI] = RMS

z_arr = 1j*np.zeros(len(w_arr))
z_arr = R0*L_box*np.reciprocal(np.where(np.isfinite(I_arr),I_arr,np.inf))
np.savetxt("impedance_rRC.{}.dat".format(piter),np.c_[w_arr*1./2./np.pi,np.real(z_arr),-1*np.imag(z_arr),R_arr])

                                                            

    



