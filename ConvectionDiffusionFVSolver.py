import numpy as np
from fipy import *

def ConvDiffFV(peclet,convCoeff,valueRight,valueLeft,
               nx,dx,dt,steps):
    
    convCoeff =(convCoeff,)
    mesh = Grid1D(nx=nx, dx=dx)
    
    # initial and boundary conditions
    phi = CellVariable(name="solution variable", 
                       mesh=mesh,
                       value=0.)

    phi.faceGrad.constrain(valueRight, where=mesh.facesRight)
    phi.constrain(valueLeft, where=mesh.facesLeft)

    x = mesh.cellCenters[0]
    cout_mod = 0 * np.ndarray(steps+1)

    # differential equation
    eqI = (TransientTerm() - DiffusionTerm(coeff=1/peclet) + 
       ExponentialConvectionTerm(coeff=convCoeff) == 0)

    # solving the equation
    for step in range(steps):
            eqI.solve(var=phi, dt=dt)
            cout_mod[step+1] = phi._value[-1]
            
    return phi._value, cout_mod
        
