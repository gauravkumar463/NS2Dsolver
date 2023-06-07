from cupy import *
from cupyx.scipy.fft import fftfreq, fft, ifft, fft2, ifft2, fftshift
import matplotlib.pyplot as plt
import nmmn.plots

class NavierStokes:
    
    def __init__(self, L, N):
        self.L = L
        self.N = N
        self.dx = L/N

        ### Create grid
        X = mgrid[:N, :N].astype(double)*self.dx
        self.xx=X[0,:,:].T
        self.yy=X[1,:,:].T
        del X

        ### Create spectral grid variables
        self.Kx = repeat(concatenate((linspace(0,N//2,N//2+1),linspace(-N//2+1,-1,N//2-1)))\
                    .reshape(N,1).T,N,axis=0)/L*2*pi
        self.Ky = self.Kx.T
        self.Kx2Ky2 = (self.Kx**2+self.Ky**2)
        self.Kx2Ky2[0,0]=1e100

    ### Navier-Stokes solution functions

    #### __Pad__ and __chop__ functions are used for dealiasing the advection term $u\frac{\partial \omega}{\partial x}+v\frac{\partial \omega}{\partial y}$ while calculating RHS of the vorticity equation

    def pad(self,f,n):
        fp=zeros((n,n),'cdouble')
        n = f.shape[0]
        fp[:n//2+1,:n//2+1]=f[:n//2+1,:n//2+1]
        fp[1-n//2:,:n//2+1]=f[1-n//2:,:n//2+1]
        fp[:n//2+1,1-n//2:]=f[:n//2+1,1-n//2:]
        fp[1-n//2:,1-n//2:]=f[1-n//2:,1-n//2:]
        return fp

    def chop(self,fp,n):
        f=empty((n,n),'cdouble')
        f[:n//2+1,:n//2+1]=fp[:n//2+1,:n//2+1]
        f[1-n//2:,:n//2+1]=fp[1-n//2:,:n//2+1]
        f[:n//2+1,1-n//2:]=fp[:n//2+1:,1-n//2:]
        f[1-n//2:,1-n//2:]=fp[1-n//2:,1-n//2:]
        return f

    #### __omg2vel__ function calculates variables $u, v, \omega, \psi$ from $\hat{\omega}$ using the following equations:
    ##### $$ \nabla^2\psi = \omega \implies \hat{\psi} = \frac{\omega}{k_x^2+k_y^2} $$
    ##### $$ u = \frac{\partial \psi}{\partial y} \implies \hat{u} = ik_y\hat{\psi}$$
    ##### $$ v = -\frac{\partial \psi}{\partial x} \implies \hat{v} = -ik_x\hat{\psi}$$

    def omg2vel(self,omegahat):

        psihat = omegahat/self.Kx2Ky2
        psihat[0,0] = 0
        uhat = 1j*self.Ky*psihat
        vhat = -1j*self.Kx*psihat
        return  real(ifft2(uhat)),\
                real(ifft2(vhat)),\
                real(ifft2(omegahat)),\
                real(ifft2(psihat))

    #### __convection__ function calculates the advection term in the vorticity equation: 
    #### $$u\frac{\partial \omega}{\partial x}+v\frac{\partial \omega}{\partial y}$$

    def convection(self,omegahat):
        Npad = int(self.N*1.5)
        psihat = omegahat/self.Kx2Ky2
        psihat[0,0] = 0
        u_ = real(ifft2(self.pad(1j*self.Ky*psihat,Npad)))
        v_ = real(ifft2(self.pad(-1j*self.Kx*psihat,Npad)))
        dOmegadx = real(ifft2(self.pad(1j*self.Kx*omegahat,Npad)))
        dOmegady = real(ifft2(self.pad(1j*self.Ky*omegahat,Npad)))
        return self.chop(fft2((-u_*dOmegadx - v_*dOmegady)),self.N)*2.25

    #### __diffusion__ function calculates the diffusion term in the vorticity equation: 
    #### $$-\nu \nabla^2\omega \implies -\nu(k_x^2+k_y^2)\hat{\omega}$$

    def diffusion(self,nu,omegahat):
        return -nu*(self.Kx**2+self.Ky**2)*omegahat

    #### __g_omega__ function calculates the external forcing in the vorticity equation.
    ##### For Kolmogorov flow: $g_\omega = -n\cos(ny)$.

    def g_omega(self,n,omegahat):
        return fft2((-n*cos(n*self.yy)))

    #### control function 

    # Returns first and second terms of f_omega
    def f_omega(self,epsilon1,epsilon2,omegahat):

        psihat_ = omegahat/self.Kx2Ky2
        psihat_[0,0] = 0

        psi_ = real(ifft2(psihat_))
        omega_ = real(ifft2(omegahat))

        energy_ = sum(psi_*omega_)*(self.dx**2)
        enstrophy_ = sum(omega_**2)*(self.dx**2)
        psi2_ = sum(psi_**2)*(self.dx**2)

        return -0.5*epsilon1*(enstrophy_*psihat_ - energy_*omegahat)\
                -0.5*epsilon2*(psi2_*omegahat - energy_*psihat_)


    #### time integration using RK4 method

    def integrate(self,rhs,dt,omegahat):
        K1 = rhs(omegahat)
        K2 = rhs(omegahat+0.5*dt*K1)
        K3 = rhs(omegahat+0.5*dt*K2)
        K4 = rhs(omegahat+dt*K3)
        tomegahat = omegahat + dt*(K1+2*(K2+K3)+K4)/6

        return [tomegahat, (tomegahat-omegahat)/dt]
    
    ### Plotting functions
    
    #### Function to plot contours of $\omega$, $\psi$, $u$ and $v$
    
    def plotFields(self,u,uname,v,vname,omega,omeganame,psi,psiname,filename):
    
        colormap=nmmn.plots.parulacmap()
        flevels = 64
        clevels = 13
        fig = plt.figure(figsize=(4,5),dpi=300)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset']='cm'
        plt.rcParams['axes.axisbelow'] = True
        plt.rcParams.update({'font.size': 8})
        
        ax1 = plt.subplot(221)
        omgPlot = ax1.contourf(self.xx.get(),self.yy.get(),omega.get(),\
                                cmap=colormap,levels=flevels)
        ax1.contour(self.xx.get(),self.yy.get(),omega.get(),\
                            levels=clevels,linewidths=0.2,colors='black')
        ax1.set_aspect('equal')
        ax1.set_xticks([0,pi,2*pi],['0',r'$\pi$',r'$2\pi$'])
        ax1.set_yticks([0,pi,2*pi],['0',r'$\pi$',r'$2\pi$'])
        ax1.set_xlabel(r'$x$',labelpad=-1)
        ax1.set_ylabel(r'$y$',labelpad=-1)
        cbar = plt.colorbar(omgPlot,label=omeganame,location='top',shrink=0.95)
        cbar.ax.locator_params(nbins=5)
        
        ax2 = plt.subplot(222)
        psiPlot = ax2.contourf(self.xx.get(),self.yy.get(),psi.get(),\
                                cmap=colormap,levels=flevels)
        ax2.contour(self.xx.get(),self.yy.get(),psi.get(),\
                            levels=clevels,linewidths=0.2,colors='black')
        ax2.set_aspect('equal')
        ax2.set_xticks([0,pi,2*pi],['0',r'$\pi$',r'$2\pi$'])
        ax2.set_yticks([0,pi,2*pi],['0',r'$\pi$',r'$2\pi$'])
        ax2.set_xlabel(r'$x$',labelpad=-1)
        ax2.set_ylabel(r'$y$',labelpad=-1)
        cbar = plt.colorbar(psiPlot,label=psiname,location='top',shrink=0.95)
        cbar.ax.locator_params(nbins=5)
        
        ax3 = plt.subplot(223)
        uPlot = ax3.contourf(self.xx.get(),self.yy.get(),u.get(),\
                                cmap=colormap,levels=flevels)
        ax3.contour(self.xx.get(),self.yy.get(),u.get(),\
                            levels=clevels,linewidths=0.2,colors='black')
        ax3.set_aspect('equal')
        ax3.set_xticks([0,pi,2*pi],['0',r'$\pi$',r'$2\pi$'])
        ax3.set_yticks([0,pi,2*pi],['0',r'$\pi$',r'$2\pi$'])
        ax3.set_xlabel(r'$x$',labelpad=-1)
        ax3.set_ylabel(r'$y$',labelpad=-1)
        cbar = plt.colorbar(uPlot,label=uname,location='top',shrink=0.95)
        cbar.ax.locator_params(nbins=5)
        
        ax4 = plt.subplot(224)
        vPlot = ax4.contourf(self.xx.get(),self.yy.get(),v.get(),\
                                cmap=colormap,levels=flevels)
        ax4.contour(self.xx.get(),self.yy.get(),v.get(),\
                            levels=clevels,linewidths=0.2,colors='black')
        ax4.set_aspect('equal')
        ax4.set_xticks([0,pi,2*pi],['0',r'$\pi$',r'$2\pi$'])
        ax4.set_yticks([0,pi,2*pi],['0',r'$\pi$',r'$2\pi$'])
        ax4.set_xlabel(r'$x$',labelpad=-1)
        ax4.set_ylabel(r'$y$',labelpad=-1)
        cbar = plt.colorbar(vPlot,label=vname,location='top',shrink=0.95)
        cbar.ax.locator_params(nbins=5)
        
        plt.subplots_adjust(left=0.075, bottom=0.075, right=0.975, top=0.95)
        plt.savefig(filename)
        plt.close()