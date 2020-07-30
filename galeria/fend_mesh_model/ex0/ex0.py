#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:36:23 2020

Frontend: Construção de malha (tensor e octree) e modelo simplificados


@author: isadora
"""

import argparse
from geofem.frontend import fend
from geofem.frontend import load_in_file as load

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.close('all')


# Gerando dicionário do mesh&model pelo spyder 
out=load.inputfiles('ex0.in') # parâmetros de entrada mesh model
print('Leitura dos parâmetros de entrada mesh&model: ok!')

# Gerando dicionário do mesh&model pelo terminal!
#parser = argparse.ArgumentParser()
#parser.add_argument("inp", help="input file",type=str)
#ARG = parser.parse_args()
#out=load.inputfiles(ARG.inp)
#print(out['box'])


# opt ---> tensor ou octree
# default: tensor mesh
Me,cond=fend.MT3D(out,opt='octree') # run octree mesh
print('Criação do objeto de mesh&model: ok!')

print("\n the mesh has {} cells".format(Me))
print("\n the mesh has {} cells".format(Me.nC))


#plot matplotlib
fig,a1 = plt.subplots(num=1,clear=True)
fig.canvas.set_window_title('Slice Y')
Me.plotSlice(np.log(cond), grid=True, normal='y',ax=a1)

fig,a2 = plt.subplots(num=2,clear=True)
fig.canvas.set_window_title('Slice X')
Me.plotSlice(np.log(cond), grid=True, normal='x',ax=a2)
