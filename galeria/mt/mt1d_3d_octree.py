#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:58:00 2020

@author: isadora
"""

# Libraries for MT 1D
import math
import cmath
import matplotlib.pyplot as plt
import numpy as np

# Libraries for MT 3D
import SimPEG as simpeg
from SimPEG.EM import NSEM_octree as NSEM
from scipy.constants import mu_0
try:
    from pymatsolver import Pardiso as Solver
except:
    from SimPEG import Solver

#from scipy.constants import mu_0, epsilon_0 as eps_0
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

#------------------------------------------------------------------------------
# Resposta analítica MT 1D
#------------------------------------------------------------------------------
mu = 4*math.pi*1E-7; #Magnetic Permeability (H/m)
resistivities = [1,100,1] #[300, 2500, 0.8, 3000, 2500];
thicknesses = [100,100]#[200, 400, 40, 500];
n = len(resistivities);

frequencies = [0.0001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,10000];
nfreq=len(frequencies);
rhoapp=np.zeros(nfreq);
phsapp=np.zeros(nfreq);
ifreq=0
print('freq\tares\t\t\tphase');

def mt_1d_analytic(n,frequencies,resistivities,thicknesses):
    mu = 4*math.pi*1E-7; #Magnetic Permeability (H/m)
    ifreq=0
    for frequency in frequencies:   
        w =  2*math.pi*frequency;       
        impedances = list(range(n));
        #compute basement impedance
        impedances[n-1] = cmath.sqrt(w*mu*resistivities[n-1]*1j);
            
        for j in range(n-2,-1,-1):
                resistivity = resistivities[j];
                thickness = thicknesses[j];
  
                # 3. Compute apparent resistivity from top layer impedance
                #Step 2. Iterate from bottom layer to top(not the basement) 
                # Step 2.1 Calculate the intrinsic impedance of current layer
                dj = cmath.sqrt((w * mu * (1.0/resistivity))*1j);
                wj = dj * resistivity;
                # Step 2.2 Calculate Exponential factor from intrinsic impedance
                ej = cmath.exp(-2*thickness*dj);                       
                # Step 2.3 Calculate reflection coeficient using current layer
                #          intrinsic impedance and the below layer impedance
                belowImpedance = impedances[j + 1];
                rj = (wj - belowImpedance)/(wj + belowImpedance);
                re = rj*ej; 
                Zj = wj * ((1 - re)/(1 + re));
                impedances[j] = Zj;    

        # Step 3. Compute apparent resistivity from top layer impedance
        Z = impedances[0];
        absZ = abs(Z);
        apparentResistivity = (absZ * absZ)/(mu * w);
        phase = math.atan2(Z.imag, Z.real);
        rhoapp[ifreq]=apparentResistivity;
        phsapp[ifreq]=phase;
        ifreq=ifreq+1;
        print(frequency, '\t', apparentResistivity, '\t', phase)
    return rhoapp, phsapp


#------------------------------------------------------------------------------
# Executar function analítica MT 1D
#------------------------------------------------------------------------------
rhoapp, phsapp=mt_1d_analytic(n,frequencies,resistivities,thicknesses)


#------------------------------------------------------------------------------
# Plot only analytical result
#------------------------------------------------------------------------------
fig,ax = plt.subplots(num=1,clear=True) 
#ax.plot(frequencies,rhoapp,frequencies,data[:,1],'--')
ax.plot(frequencies,rhoapp)
ax.legend(('MT 1D Analytic'))
ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('Apparent Resistivity (Rho.m)')
ax.set_xscale('log')
ax.set_yscale('linear')
ax.invert_xaxis()
ax.grid()  

fig,ax2 = plt.subplots(num=2,clear=True) 
#ax.plot(frequencies,rhoapp,frequencies,data[:,1],'--')
ax2.plot(frequencies,phsapp)
ax2.legend(('MT 1D analytic'))
ax2.set_xlabel('frequency (Hz)')
ax2.set_ylabel('Apparent Phase')
ax2.set_xscale('log')
ax2.set_yscale('linear')
ax2.invert_xaxis()
ax2.grid()  

#------------------------------------------------------------------------------
# Executar MT 3D na malha octree para o mesmo modelo canônico do analítico
#------------------------------------------------------------------------------

nFreq = 2
#freqs = np.logspace(-3, 3, nFreq)
freqs = np.array([0.0001,0.01,0.1,10,100,10000]);
freqs = np.array([0.1,10]);
# Definir malha e modelo

dx = 50    # tamanho minimo da celula em x
dy = 50    # tamanho minimo da celula em y
dz = 50     # tamanho minimo da celula em z

x_length = 5000     # tamanho do dominio em x
y_length = 5000     # tamanho do dominio em y
z_length = 16000    # tamanho do dominio em z

# Compute number of base mesh cells required in x and y
nbcx = 2**int(np.round(np.log(x_length/dx)/np.log(2.)))
nbcy = 2**int(np.round(np.log(y_length/dy)/np.log(2.)))
nbcz = 2**int(np.round(np.log(z_length/dz)/np.log(2.)))

# Define the base mesh
hx = [(dx, nbcx)]
hy = [(dy, nbcy)]
hz = [(dz, nbcz)]
M = TreeMesh([hx, hy, hz], x0='CCC')

# # Refine surface topography
#[xx, yy] = np.meshgrid(M.vectorNx, M.vectorNy)
#[xx, yy,zz] = np.meshgrid([-5000,5000], [-5000,5000],[-100,100])
#zz = 0.*(xx**2 + yy**2)  - 1000.
##zz = np.zeros([300,300])
#pts = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]
#M = refine_tree_xyz(
#     M, pts, octree_levels=[1, 1], method='surface', finalize=False
#                   )

# Refine box
xp, yp, zp = np.meshgrid([-600., 600.], [-1000., 1000.], [200., -3000.])
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]

M = refine_tree_xyz(
    M, xyz, octree_levels=[1,0], method='box', finalize=False)  

## Refine surface no alvo
#xp, yp, zp = np.meshgrid([-5000., 5000.], [-5000., 5000.], [-1000., -2000.])
#xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
#
#M = refine_tree_xyz(
#    M, xyz, octree_levels=[1,0], method='surface', finalize=False)  

#Refine rest of the grid
def refine(cell):
    if np.sqrt(((np.r_[cell.center]-0.5)**2).sum()) < 0.4:
        return 1
    return 1

M.refine(refine)
M.finalize()

conds = [0.01,1]   # [heterogeneidade,background]
sig = simpeg.Utils.ModelBuilder.defineBlock(
    M.gridCC, [-500, -500, -200], [500, 500, -100], conds)

#sig[M.gridCC[:,2] > -1000] = 3.3    # água    
sig[M.gridCC[:,2] > 0] = 1e-12      # ar

sigBG = np.zeros(M.nC) + conds[1]
#sigBG[M.gridCC[:, 2] > -1000] = 3.3   
sigBG[M.gridCC[:, 2] > 0] = 1e-12   

# MESH 1D (para modelo de background)
mesh1d = simpeg.Mesh.TensorMesh([M.hz], np.array([M.x0[2]]))
sigBG1d = np.zeros(mesh1d.nC) + conds[1]
#sigBG1d[mesh1d.gridCC > -1000] = 3.3
sigBG1d[mesh1d.gridCC > 0] = 1e-12
   
fig,axes = plt.subplots(num=3,clear=True)
M.plotSlice(np.log(sig), grid=True, normal='y',ax=axes)
plt.show()


#------------------------------------------------------------------------------
# Fim modelo e malha
#------------------------------------------------------------------------------





#------------------------------------------------------------------------------
# Iniciar modelagem
#------------------------------------------------------------------------------

rx_x = np.array([0.])
rx_y = np.array([0.])
rx_z = np.array([-1000.])
rx_loc = np.hstack((simpeg.Utils.mkvc(rx_x, 2), simpeg.Utils.mkvc(rx_y, 2),
                    np.zeros((np.prod(rx_x.shape), 1))))
#rx_loc=np.array([0., 0., 0.])

rxList = []
for rx_orientation in ['xx', 'xy', 'yx', 'yy']:
    rxList.append(NSEM.Rx.Point_impedance3D(rx_loc, rx_orientation, 'real'))
    rxList.append(NSEM.Rx.Point_impedance3D(rx_loc, rx_orientation, 'imag'))
for rx_orientation in ['zx', 'zy']:
    rxList.append(NSEM.Rx.Point_tipper3D(rx_loc, rx_orientation, 'real'))
    rxList.append(NSEM.Rx.Point_tipper3D(rx_loc, rx_orientation, 'imag'))

# Source list,
srcList = []
for freq in freqs:
    #srcList.append(NSEM.Src.Planewave_xy_1Dprimary(rxList, freq))
    srcList.append(NSEM.Src.Planewave_xy_1Dprimary(rxList, freq, sigBG1d, sigBG))
# Make the survey
survey = NSEM.Survey(srcList)

# Set the problem
problem = NSEM.Problem3D_ePrimSec(M, sigma=sig, sigmaPrimary=sigBG)
problem.pair(survey)
problem.Solver = Solver

# Calculate the data
fields = problem.fields()    # returns secondary field

#------------------------------------------------------------------------------
# Colher os campos
#------------------------------------------------------------------------------

grid_field_px = np.empty((M.nE,nFreq),dtype=complex)
grid_field_py = np.empty((M.nE,nFreq),dtype=complex)
for i in range(nFreq):
    grid_field_px[:,i] = np.transpose(fields._getField('e_pxSolution', i))
    grid_field_py[:,i] = np.transpose(fields._getField('e_pySolution', i))


# campos E e H calculado em todas as arestas d malha
e_px_full  = fields._e_px(grid_field_px, srcList)
e_py_full  = fields._e_py(grid_field_py, srcList)
h_px_full  = fields._b_px(grid_field_px, srcList)/mu_0
h_py_full  = fields._b_py(grid_field_py, srcList)/mu_0

ex_px_field = e_px_full[0:np.size(M.gridEx,0),:]
ex_py_field = e_py_full[0:np.size(M.gridEx,0),:]
# hx_px_field = h_px_full[0:np.size(M.gridEx,0),:]
# hx_py_field = h_py_full[0:np.size(M.gridEx,0),:]
# hx_px_field = h_px_full[0:np.size(M.gridFx,0),:]
# hx_py_field = h_py_full[0:np.size(M.gridFx,0),:]

# interpolar o campo h nas arestas
Pbx = M.getInterpolationMat(M.gridEx, 'Fx')
hx_px_field = Pbx*h_px_full
hx_py_field = Pbx*h_py_full

ey_px_field = e_px_full[np.size(M.gridEx,0):np.size(M.gridEx,0)+np.size(M.gridEy,0),:]
ey_py_field = e_py_full[np.size(M.gridEx,0):np.size(M.gridEx,0)+np.size(M.gridEy,0),:]
hy_px_field = h_px_full[np.size(M.gridEx,0):np.size(M.gridEx,0)+np.size(M.gridEy,0),:]
hy_py_field = h_py_full[np.size(M.gridEx,0):np.size(M.gridEx,0)+np.size(M.gridEy,0),:]



#---




indx = np.logical_and( abs(M.gridEx[:,2]) < 1e-6, abs(M.gridEx[:,1]) < 1e-6)
indy = np.logical_and( abs(M.gridEy[:,2]) < 1e-4, abs(M.gridEy[:,0]) < 1e-4)

ex_px = ex_px_field[indx]
ex_py = ex_py_field[indx]
hx_px = hx_px_field[indx]
hx_py = hx_py_field[indx]

ey_px = ey_px_field[indy]
ey_py = ey_py_field[indy]
hy_px = hy_px_field[indy]
hy_py = hy_py_field[indy]

#x = M.getTensor('Ex')[0]
x = M.gridEx[indx,0]

ix = 11    # indice da posição x de onde vai medir -> vetor x
Zij = ex_px/hx_py
rho_app = 1/(2*np.pi*freqs*mu_0) * abs(Zij[ix,:])**2
phs_app     = np.arctan2(Zij[ix,:].imag, Zij[ix,:].real)
#phs     = np.arctan2(Zij[ix,:].imag, Zij[ix,:].real)*(180./np.pi)
#phs = math.atan2(Zij[ix,:].imag, Zij[ix,:].real)



fig,ax = plt.subplots(num=4,clear=True) 
ax.plot(frequencies,rhoapp,freqs,rho_app,'--')
#ax.plot(freqs,rho_app)
ax.legend(('MT 1D Analytic', 'MT 3D Numeric'))
ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('Apparent Resistivity (Rho.m)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.invert_xaxis()
ax.grid()  

fig,ax2 = plt.subplots(num=5,clear=True) 
ax2.plot(frequencies,phsapp,freqs,phs_app,'--')
#ax2.plot(frequencies,phsapp)
ax2.legend(('MT 1D analytic', 'MT 3D Numeric'))
ax2.set_xlabel('frequency (Hz)')
ax2.set_ylabel('Apparent Phase')
ax2.set_xscale('log')
ax2.set_yscale('linear')
ax2.invert_xaxis()
ax2.grid()






































