from cupy import *
import pandas as pd
from cupyx.scipy.fft import fftfreq, fft, ifft, fft2, ifft2, fftshift
from cupyx.scipy.sparse.linalg import LinearOperator, gmres
from cupy.linalg import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io
from NavierStokes import NavierStokes

def rhs(omegahat):
    rhs_ = NS.convection(omegahat)+NS.diffusion(nu,omegahat)
    if(fext==True):
        rhs_ = rhs_ + NS.g_omega(n,omegahat)
    rhs_[0,0]=0
    return rhs_

def NSd(v):
    return (rhs(omegahat+v.reshape(N,N))\
                    - rhs(omegahat)).flatten()



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
delta1 = 3
delta2 = 0
path = './data/'

NS = NavierStokes(L,N)

omegahat = load('./data/data_60000.npy')
save('./NKHsol/data_0.npy',omegahat)
u,v,omega,psi = NS.omg2vel(omegahat)
NS.plotFields(omega**2,r'$\omega^2$',psi*omega,r'$\psi\omega$',\
           omega,r'$\omega$',psi,r'$\psi$','./NKHsol/state_0.png')

A = LinearOperator((N**2,N**2), matvec=NSd)

omegahat_new = omegahat
for i in range(100):
    fval = rhs(omegahat_new)
    fnorm = norm(fval)
    if(fnorm < 1e-6):
        break
    domega, status = gmres(A, b = -fval.flatten(),\
                    x0 = ones(N**2)/N, restart=100, maxiter = 10)
    #print(status)
    s = 1.0
    smin=1/2**2
    while True:
        tomegahat_new = omegahat_new + s*domega.reshape(N,N)
        tfnorm = norm(rhs(tomegahat_new))
        if (tfnorm < (1-0.5*s)*fnorm or s<=smin):
            omegahat_new = tomegahat_new
            break
        s *= 0.5
    
    # if (tfnorm/fnorm > 0.9999):
    #     break
    save('./NKHsol/data_{}.npy'.format(i+1),omegahat_new)
    u,v,omega,psi = NS.omg2vel(omegahat_new)
    NS.plotFields(omega**2,r'$\omega^2$',psi*omega,r'$\psi\omega$',\
           omega,r'$\omega$',psi,r'$\psi$','./NKHsol/state_{}.png'.format(i+1))
    print('itr = ', i, ',', fnorm, '-->',  tfnorm, ',', s)