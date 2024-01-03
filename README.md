The file __NavierStokes.py__ contains the class defintion to hold mesh information and other stuff that are useful for solving the Navier-Stokes equation. 

__omg2vel__ function calculates variables $u, v, \omega, \psi$ from $\hat{\omega}$ using the following equations:

``` math 
\nabla^2\psi = \omega \implies \hat{\psi} = \frac{\omega}{k_x^2+k_y^2} \\
u = \frac{\partial \psi}{\partial y} \implies \hat{u} = ik_y\hat{\psi} \\
v = -\frac{\partial \psi}{\partial x} \implies \hat{v} = -ik_x\hat{\psi}
```

__convection__ function calculates the advection term in the vorticity equation: 

$$u\frac{\partial \omega}{\partial x}+v\frac{\partial \omega}{\partial y}$$

__Pad__ and __chop__ functions are used for dealiasing the advection term $u\frac{\partial \omega}{\partial x}+v\frac{\partial \omega}{\partial y}$ while calculating RHS of the vorticity equation

__diffusion__ function calculates the diffusion term in the vorticity equation: 

$$-\nu \nabla^2\omega \implies -\nu(k_x^2+k_y^2)\hat{\omega}$$

__g_omega__ function calculates the external forcing in the vorticity equation.
For Kolmogorov flow: $g_\omega = -n\cos(ny)$.

RHS terms of the Navier-Stokes equations are assembled in another python script __NS2Dsolver.py__ and time integrated to simulate the flow

The __makeplots.py__ script is used for plotting the flow fields from the simulation

The __NKH.py__ script solves for the steady state solution of Navier-Stoke equation (or its variant) using Newton-Krylob-Hookstep method. 
