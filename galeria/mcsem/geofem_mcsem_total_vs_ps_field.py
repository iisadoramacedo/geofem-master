#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 13:25:08 2020

@author: isadora
"""
import geofem.emg3d as emg3d
import discretize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('ggplot')

#-----------------------------------------------------------------------------
# Plot: total field and primary + secundary field for the canonical model
#-----------------------------------------------------------------------------

src = [0, 0, -950, 0, 0]    # x-dir. source at the origin, 50 m above seafloor
freq = 0.01
off = np.arange(-1000,1025,25)  # Offsets
rec = [off, off*0, -1000]   # In-line receivers on the seafloor
res = [1e10, 0.3, 1]        # Resistivities (Hz): [air, seawater, background]

# Eixo X
ctx=500    # tamanho inicial da célula
npadx=12   # quantidade de células para esquerda e direita do 0
incx=1.1   # incremento de aumento das células a partir de 0
nx=[(ctx,npadx,-incx),(ctx,npadx,incx)]            

# Eixo Y
cty=500    # tamanho inicial da célula
npady=12   # quantidade de células para esquerda e direita do 0
incy=1.1   # incremento de aumento das células a partir de 0
ny=[(cty,npady,-incy),(cty,npady,incy)]            

# Eixo Z: do topo (ar) para base (substrato)

#--> parte positiva: parte imediatamente acima de 0 (ar)
ct2=50    # tamanho inicial da célula
npad2=7  # quantidade de célula
inc2=1.9  # incremento de aumento das células

#--> parte central negativa: parte negativa logo abaixo de 0 (mais rasa)
ct0=20    # tamanho inicial da célula
npad0=120 # quantidade de células a partir de 0
inc0=1.0  # incremento = 1: células do mesmo tamanho

#--> parte negativa: parte negativa abaixo da parte central (mais profunda)
ct1   = 20   # tamanho da célula
npad1 = 20   # quantidade de células a partir da parte central
inc1  = 1.3  # incremento de aumento do tamanho da célula

nz = [(ct1,npad1,-inc1),(ct0,npad0,inc0),(ct2,npad2,inc2)]   
    
# Origem do sistema de coordenadas para o simpeg no eixo z: base do substrato
ztop=0
for i in range(1,npad1+1):
    ztop += ct1*inc1**(i)
ztop += ct0*npad0

# Criar malha do tipo tensor mesh
# INPUTS
# [nx, ny, nz]: parâmetros que descrevem o refinamento em cada direção
# x0: origem do sistema de coordenadas
grid=discretize.TensorMesh([nx,nx,nz], x0=['C', 'C', -ztop])


# Create model
# ------------

# Layered_background
res_x = np.ones(grid.nC)*res[0]            # Air resistivity
res_x[grid.gridCC[:, 2] < 0] = res[1]      # Water resistivity
res_x[grid.gridCC[:, 2] < -1000] = res[2]  # Background resistivity

res_y = res_x.copy()
res_z = res_x.copy()

# Background model - 1D
model_pf = emg3d.models.Model(grid, res_x.copy())

# Include the target: anomalia
res_target = 100.
target_x = np.array([-500, 500])
target_y = target_x
target_z = -1000 + np.array([-200, -100])

target_inds = (
    (grid.gridCC[:, 0] >= target_x[0]) & (grid.gridCC[:, 0] <= target_x[1]) &
    (grid.gridCC[:, 1] >= target_y[0]) & (grid.gridCC[:, 1] <= target_y[1]) &
    (grid.gridCC[:, 2] >= target_z[0]) & (grid.gridCC[:, 2] <= target_z[1])
              )

res_x[target_inds] = res_target
res_y[target_inds] = res_target
res_z[target_inds] = res_target

# # Include the target
# xx = (grid.gridCC[:, 0] >= 0) & (grid.gridCC[:, 0] <= 6000)
# yy = abs(grid.gridCC[:, 1]) <= 500
# zz = (grid.gridCC[:, 2] > -2500)*(grid.gridCC[:, 2] < -2000)

# res_x[xx*yy*zz] = 100.  # Target resistivity

# Create target model
model = emg3d.models.Model(grid, res_x)

# Plot a slice
grid.plot_3d_slicer(
        model.res_x, zslice=-1150, clim=[0.3, 500],
        xlim=(-4000, 4000), ylim=(-4000, 4000), zlim=(-2000, 500),
        pcolorOpts={'norm': LogNorm()}
)


#------------------------------------------------------------------------------
# Calcular o campo elétrico total
# -----------------------------------------------------------------------------

modparams = {
        'verb': -1, 'sslsolver': True,
        'semicoarsening': True, 'linerelaxation': True
}

sfield_tf = emg3d.fields.get_source_field(grid, src, freq, strength=0)
em3_tf = emg3d.solve(grid, model, sfield_tf, **modparams)

#------------------------------------------------------------------------------
# Calcular campo elétrico primário no modelo de background model_pf
# -----------------------------------------------------------------------------

sfield_pf = emg3d.fields.get_source_field(grid, src, freq, strength=0)
em3_pf = emg3d.solve(grid, model_pf, sfield_pf, **modparams)

#------------------------------------------------------------------------------
# Calcular campo elétrico secundário só com a anomalia do modelo
#------------------------------------------------------------------------------

# Calcular o delta sigma: model - model_background 
# Get the difference of conductivity as volume-average values
dsigma = grid.vol.reshape(grid.vnC, order='F')*(1/model.res_x-1/model_pf.res_x)

# Cálculo do campo primário com emg3d. 
# This could be done with a 1D modeller such as empymod instead.
fx = em3_pf.fx.copy()
fy = em3_pf.fy.copy()
fz = em3_pf.fz.copy()

# Average delta sigma to the corresponding edges
fx[:, 1:-1, 1:-1] *= 0.25*(dsigma[:, :-1, :-1] + dsigma[:, 1:, :-1] +
                           dsigma[:, :-1, 1:] + dsigma[:, 1:, 1:])
fy[1:-1, :, 1:-1] *= 0.25*(dsigma[:-1, :, :-1] + dsigma[1:, :, :-1] +
                           dsigma[:-1, :, 1:] + dsigma[1:, :, 1:])
fz[1:-1, 1:-1, :] *= 0.25*(dsigma[:-1, :-1, :] + dsigma[1:, :-1, :] +
                           dsigma[:-1, 1:, :] + dsigma[1:, 1:, :])

# Create field instance iwu dsigma E
sfield_sf = sfield_pf.smu0*emg3d.fields.Field(fx, fy, fz, freq=freq)
sfield_sf.ensure_pec

# Plot da fonte secundária - anomalia/target/alvo
# Our secondary source is the entire target, the scatterer. Here we look at the
# :math:`E_x` secondary source field. But note that the secondary source has
# all three components :math:`E_x`, :math:`E_y`, and :math:`E_z`, even though
# our primary source was purely :math:`x`-directed. (Change ``fx`` to ``fy`` or
# ``fz`` in the command below, and simultaneously ``Ex`` to ``Ey`` or ``Ez``,
# to show the other source fields.)

# grid.plot_3d_slicer(
#         sfield_sf.fx.ravel('F'), view='abs', vType='Ex',
#         zslice=-1150, clim=[1e-17, 1e-9],
#         xlim=(-4000, 4000), ylim=(-4000, 4000), zlim=(-2000, 500),
#         pcolorOpts={'norm': LogNorm()}
#                    )

# Calcular fonte secundária

em3_sf = emg3d.solve(grid, model, sfield_sf, **modparams)

#------------------------------------------------------------------------------
# Plot dos resultados
#------------------------------------------------------------------------------

# E = E^p + E^s
em3_ps = em3_pf + em3_sf

# Resposta na posição dos receptores
rectuple = (rec[0], rec[1], rec[2])
em3_pf_rec = emg3d.fields.get_receiver(grid, em3_pf.fx, rectuple)
em3_tf_rec = emg3d.fields.get_receiver(grid, em3_tf.fx, rectuple)
em3_sf_rec = emg3d.fields.get_receiver(grid, em3_sf.fx, rectuple)
em3_ps_rec = emg3d.fields.get_receiver(grid, em3_ps.fx, rectuple)

plt.figure(figsize=(9, 5))

ax1 = plt.subplot(121)
plt.title('|Real part|')
plt.plot(off/1e3, abs(em3_pf_rec.real), 'k',
         label='Primary Field (1D Background)')
plt.plot(off/1e3, abs(em3_sf_rec.real), '.4', ls='--',
         label='Secondary Field (Scatterer)')
plt.plot(off/1e3, abs(em3_ps_rec.real))
plt.plot(off[::2]/1e3, abs(em3_tf_rec[::2].real), '.')
plt.plot(off/1e3, abs(em3_ps_rec.real-em3_tf_rec.real))
plt.xlabel('Offset (km)')
plt.ylabel('$E_x$ (V/m)')
plt.yscale('log')
plt.legend()

ax2 = plt.subplot(122, sharey=ax1)
plt.title('|Imaginary part|')
plt.plot(off/1e3, abs(em3_pf_rec.imag), 'k')
plt.plot(off/1e3, abs(em3_sf_rec.imag), '.4', ls='--')
plt.plot(off/1e3, abs(em3_ps_rec.imag), label='P/S Field')
plt.plot(off[::2]/1e3, abs(em3_tf_rec[::2].imag), '.', label='Total Field')
plt.plot(off/1e3, abs(em3_ps_rec.imag-em3_tf_rec.imag),
         label=r'$\Delta$|P/S-Total|')
plt.xlabel('Offset (km)')
plt.ylabel('$E_x$ (V/m)')
plt.yscale('log')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.legend()

plt.tight_layout()
plt.show()

