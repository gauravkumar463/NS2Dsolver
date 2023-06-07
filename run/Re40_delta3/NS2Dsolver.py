from cupy import *
import pandas as pd
from cupyx.scipy.fft import fftfreq, fft, ifft, fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io
from NavierStokes import NavierStokes

def rhs(omegahat):
    rhs_ = NS.convection(omegahat)+NS.diffusion(nu,omegahat)
    if(fext==True):
        rhs_ = rhs_ + NS.g_omega(n,omegahat)
    if(control==True):
        rhs_ = rhs_ + NS.f_omega(epsilon1,epsilon2,omegahat)
    rhs_[0,0]=0
    return rhs_

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
control = True
n = 4 # Parameter for Kolmogorov flow
delta1 = 3
delta2 = 0
path = './data/'

NS = NavierStokes(L,N)

E_lam = Re**2/(4*(n**4))*(L**2)
enstrophy_lam = Re**2/(2*(n**2))*(L**2)

energy = empty(nt+1,'double')
enstrophy = empty(nt+1,'double')
psi2 = empty(nt+1,'double')
Ldomegadt = empty(nt+1,'double')

#omega = cos(X[1,:,:])*cos(X[0,:,:]) # Kolmorogv initialization

### read a snapshot
# readData = pd.read_csv('./Re40/data_20000.csv')
# omega = pd.Series.to_numpy(readData['omega']).reshape(N,N)
# omegahat = fft2(omega)

### read from matlab
#mat = scipy.io.loadmat('./omghat_taylor_random.mat')
#omegahat = asarray(mat['omghat'])

omegahat = load('../Re40/data/data_20000.npy')

save(path+'data_0.npy',omegahat)

u0,v0,omega0,psi0 = NS.omg2vel(omegahat)
#NS.plotFields(u0,v0,omega0,psi0,path+'IC.png')

energy[0] = 0.5*sum(psi0*omega0)*(dx**2)
enstrophy[0] = 0.5*sum(omega0**2)*(dx**2)
psi2[0] = 0.5*sum(psi0**2)*(dx**2)
Ldomegadt[0] = 0

F0 = (psi2[0]*enstrophy[0])-(energy[0]**2)
epsilon1 = delta1*energy[0]/F0 # control parameter
epsilon2 = delta2*enstrophy[0]/F0
print(epsilon1,epsilon2)

for t in tqdm(range(1,nt+1)):
    omegahat,domegahatdt = NS.integrate(rhs,dt,omegahat)
    
    u,v,omega,psi = NS.omg2vel(omegahat)
    
    if(t%nout == 0):
        save(path+'data_{}.npy'.format(t),omegahat)
        
    energy[t] = 0.5*sum(psi*omega)*(dx**2)
    enstrophy[t] = 0.5*sum(omega**2)*(dx**2)
    psi2[t] = 0.5*sum(psi**2)*(dx**2)
    
    Ldomegadt[t] = sum(abs(ifft2(domegahatdt))**2)*(dx**2)
    #epsilon1 = 3*0.001*Ldomegadt[t]*energy[0]/F0
    
    if(isnan(energy[t])==True or isnan(enstrophy[t])==True):
        break
        
pd.DataFrame(data=concatenate((linspace(0,T,nt+1),\
                    energy,enstrophy,psi2,Ldomegadt)).get()\
                .reshape(5,nt+1).T,columns=['t','Energy','Enstrophy',\
                    'psi2','Ldomegadt'])\
            .to_csv(path+"statistics.csv",index=False)

#NS.plotFields(u,v,omega,psi,path+'finalSolution.png')

# df = pd.DataFrame(data=concatenate((xx.flatten(),\
#                                             yy.flatten(),\
#                                             omega.flatten(),\
#                                             psi.flatten())\
#                                           ).reshape(4,N**2).T,\
#                   columns=['X','Y','omega','psi'])
#         df.to_csv(path+"data_{}.csv".format(t),index=False)