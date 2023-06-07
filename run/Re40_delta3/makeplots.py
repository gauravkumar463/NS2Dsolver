from cupy import *
import pandas as pd
from cupyx.scipy.fft import fftfreq, fft, ifft, fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io
from NavierStokes import NavierStokes

Re = 40 # Reynolds number
nu = 1/Re # Viscosity
T = 300 # Total simulation time
dt = 0.005 # Time step size
nt = int(T/dt) # Number of time steps for simulation
nout = int(1/dt) # Output after these many time steps
L = 2*pi # Domain size [L,L]
N = 2**7 # Number of grid points in one direction
dx = L/N
fext = True # external forcing switch
control = False
n = 4 # Parameter for Kolmogorov flow
epsilon1 = 0.07951133510957262 # control parameter
epsilon2 = 0
path = './data/'

E_lam = Re**2/(4*(n**4))*(L**2)
enstrophy_lam = Re**2/(2*(n**2))*(L**2)
NS = NavierStokes(L,N)

# for i in tqdm(range(0,nt,nout)):
# 	omegahat = load(path+'data_{}.npy'.format(i))
# 	u,v,omega,psi = NS.omg2vel(omegahat)
# 	NS.plotFields(u,'u',v,'v',omega,r'$\omega$',psi,r'$\psi$',\
# 			path+'../plots/plot_{}'.format(i))

stats = pd.read_csv(path+'statistics.csv')

fig = plt.figure(figsize=(4,4),dpi=300)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset']='cm'
plt.rcParams['axes.axisbelow'] = True
plt.rcParams.update({'font.size': 8})
ax1 = plt.subplot(311)
ax1.plot(stats['t'],stats['Energy']/E_lam,label='Energy',lw=1)
ax1.axhline(stats['Energy'][0]/E_lam,linestyle='dashed',lw=1, color='black')
ax1.set_ylabel(r'$E/E_{lam}$')
ax1.set_ylim(0,0.75)
plt.tick_params('x', labelbottom=False)

ax2 = plt.subplot(312, sharex=ax1)
ax2.plot(stats['t'],stats['Enstrophy']/enstrophy_lam,label='Enstrophy',lw=1)
ax2.axhline(stats['Enstrophy'][0]/enstrophy_lam,linestyle='dashed',lw=1, color='black')
ax2.set_ylabel(r'$D/D_{lam}$')
ax2.set_ylim(0,0.4)
plt.tick_params('x', labelbottom=False)

ax3 = plt.subplot(313, sharex=ax1)
ax3.plot(stats['t'],stats['Ldomegadt'],label=r'$||d\omega/dt||$',lw=1)
ax3.set_xlabel('time (s)')
ax3.set_ylabel(r'$||d\omega/dt||$')
ax3.set_ylim(1e-16,1e4)
ax3.set_yscale('log')

plt.subplots_adjust(left=0.15, bottom=0.1, top=0.95, right=0.95)
plt.savefig(path+'../plots/statistics.png')

# energy = 0.5*sum(psi*omega)*(dx**2)
# enstrophy = 0.5*sum(omega**2)*(dx**2)
# psi2 = 0.5*sum(psi**2)*(dx**2)

# F0 = (psi2*enstrophy)-(energy**2)
# epsilon1 = 3*energy/F0 # control parameter
# print(epsilon1)

# mat = scipy.io.loadmat(path+'data_20000.mat')
# omegahat = asarray(mat['omghat'])
# u,v,omega,psi = NS.omg2vel(omegahat)

# I = sum(psi*NS.g_omega(n,omegahat))*(dx**2)
# D = sum(psi*NS.diffusion(nu,omegahat))*(dx**2)
# P = sum(omega*NS.diffusion(nu,omegahat))*(dx**2)
# Q = sum(omega*NS.g_omega(n,omegahat))*(dx**2)
# F1 = sum(psi*NS.f_omega(epsilon1,epsilon2,omegahat))*(dx**2)
# F2 = sum(omega*NS.f_omega(epsilon1,epsilon2,omegahat))*(dx**2)
# print('I = ', I,\
#       '\n D = ', D,\
#       '\n P = ', P,\
#       '\n Q = ', Q,\
#       '\n F1 = ', F1,\
#       '\n F2 = ', F2)

# NS.plotFields(omega*(NS.f_omega(epsilon1,epsilon2,omegahat)+NS.g_omega(n,omegahat)+NS.diffusion(nu,omegahat)),\
#            omega*NS.g_omega(n,omegahat),\
#            omega,\
#            omega*NS.diffusion(nu,omegahat))