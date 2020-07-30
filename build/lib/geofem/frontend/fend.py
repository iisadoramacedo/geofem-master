#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:03:20 2020

@author: akel
controi modelo e malha MT3D(input,opt='string')
input --> dicionário contendo informações da simulação,modelo geologico e meshgrid.
          veja a função loadmigeo para maiores informações.
          
opt ---> Variavel string que define o tipo de malha usada 'tensor' ou 'octree'
         se não declarado opt o código roda com a malha tensor.
         
         
         

def utilizadas 
         
         easymodelbox    Cria uma estrutura 3D com uma condutividade definida
         easymodellayer  Cria camadas planas 
         layermesh       Realiza mesh nas camadas
         boxmesh         Realiza mesh nas estruturas do tipo box
               
"""

from geofem import SimPEG as simpeg
import numpy as np
from discretize.utils import mkvc, refine_tree_xyz

def MT3D(input_var,**kwargs):
    
    op=kwargs.get('opt')

    dx=input_var['dxdydz'][0]
    dy=input_var['dxdydz'][1]
    dz=input_var['dxdydz'][2]
    x_length = input_var['x']    # tamanho do dominio em x
    y_length = input_var['y']    # tamanho do dominio em y
    z_length = input_var['z']    # tamanho do dominio em z
    
#Definções de mesh   
#    # Compute number of base mesh cells required in x and y
    nbcx = 2**int(np.round(np.log(x_length/dx)/np.log(2.)))
    nbcy = 2**int(np.round(np.log(y_length/dy)/np.log(2.)))
    nbcz = 2**int(np.round(np.log(z_length/dz)/np.log(2.)))

    hx = [(dx, nbcx)]
    hy = [(dy, nbcy)]
    hz = [(dz, nbcz)]
     
    M=simpeg.Mesh.TensorMesh([hx, hy, hz], x0='CCC')   
    if op == 'tensor':
        M=simpeg.Mesh.TensorMesh([hx, hy, hz], x0='CCC')
        pass
    
    if op == 'octree':
       M = simpeg.Mesh.TreeMesh([hx, hy, hz], x0='CCC')
       layermesh(M,input_var['layer'])
       boxmesh(M,input_var['box'])     
       M.finalize()
    
#Contrução do modelo   
    sig=np.zeros(M.nC) + 1e-12 # define 
    
    #inclusão de camadas, se for o caso    
    if 'layer' in input_var:
        easymodellayer(M,sig,input_var['layer'],input_var['cond'])
        
        pass
        
    sigBG = sig
  
   #inclusão de estruturas , se for o caso    
    if 'box' in input_var:
        easymodelbox(M,sigBG,input_var['box'])
        pass
    
    
    #To work
    #add  simulação MT3D
    #add  plot com resultados da simulação
    
    return M,sig


"""
Funções(Def) usadas: easymodelbox,easymodellayer(),layermesh,boxmesh
  
"""


def easymodelbox(M,S,B):
    n_box=len(B)/7
    for i in range(0,int(n_box),1):
        x=B[0+int(i*7)]
        Lx=B[1+int(i*7)]
        y=B[2+int(i*7)]
        Ly=B[3+int(i*7)]
        z=B[4+int(i*7)]
        Lz=B[5+int(i*7)]
        aim_cond=B[6+int(i*7)]
        S[(M.gridCC[:,0]  < x+Lx) & (M.gridCC[:,0]  > x) & (M.gridCC[:,1]  < y+Ly) & (M.gridCC[:,1]  > y) & (M.gridCC[:,2]  < z+Lz) & (M.gridCC[:,2]  > z) ]  =  aim_cond
    return

#função para criar as camadas
def easymodellayer(M,S,camada,cond):
    S[M.gridCC[:,2] >= 0] = 1e-12  #cond. ar
    c=0
#    print('Model camada',camada)
#    print('model cond',cond)
    for i in range(0,len(camada),1):
        c=0+c
#        print('c-->',c)
        S[(M.gridCC[:,2]  < -c) & (M.gridCC[:,2]  >= -camada[i])]=cond[i]
        c=camada[i]
        
    #define limite do grid igual a camada anterior
    
#    S[(M.gridCC[:,2]  < -c) & (M.gridCC[:,2]  >= -M.gridCC[-1,2])]=cond[i]
    print()
    S[(M.gridCC[:,2]  < -c) ]=cond[i+1]

    return

def layermesh(M,camada):
    
    xp, yp, zp = np.meshgrid( [-np.sum(M.hx)/2, np.sum(M.hx)/2],[-np.sum(M.hy)/2,np.sum(M.hy)/2], [-0-1*M.hz[0],-0+1*M.hz[0]])
    xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)] 
    M = refine_tree_xyz(
    M, xyz, octree_levels=[1,1,1], method='box', finalize=False
    )
    for i in range(0,len(camada),1):
        xp, yp, zp = np.meshgrid( [-np.sum(M.hx)/2, np.sum(M.hx)/2],[-np.sum(M.hy)/2,np.sum(M.hy)/2], [-camada[i]-1*M.hz[0],-camada[i]+1*M.hz[0]])
        xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)] 
        M = refine_tree_xyz(M, xyz, octree_levels=[1,1,1], method='box', finalize=False)
    return

def boxmesh(M,box):
    n_box=len(box)/7
    for i in range(0,int(n_box),1):
        x1=box[0+int(i*7)]
        x2=x1+box[1+int(i*7)]
        y1=box[2+int(i*7)]
        y2=y1+box[3+int(i*7)]
        z1=box[4+int(i*7)]
        z2=z1+box[5+int(i*7)]
    #plano1 XY-ztop
        xp, yp, zp = np.meshgrid( [x1, x2],[y1,y2], [z1-M.hz[0],z1+M.hz[0]])
        xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)]  
        M = refine_tree_xyz(
                M, xyz, octree_levels=[1,1,1], method='box', finalize=False
                )
    #plano2 XY-zboton
        xp, yp, zp = np.meshgrid( [x1,x2],[y1,y2], [z2-M.hz[0],z2+M.hz[0]])
        xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)] 
        M = refine_tree_xyz(
                M, xyz, octree_levels=[1,1,1], method='box', finalize=False
                )
    #plano3 XZ-yleft
        xp, yp, zp = np.meshgrid( [x1-2*M.hx[0],x1+2*M.hx[0]],[y1,y2], [z2,z1])
        xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)] 
        M = refine_tree_xyz(
                M, xyz, octree_levels=[1,1,1], method='box', finalize=False
                )
    #plano4 XZ-yrigth
        xp, yp, zp = np.meshgrid( [x2-2*M.hx[0],x2+2*M.hx[0]],[y1,y2], [z2,z1])
        xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)]  # mkvc creates vectors
        M = refine_tree_xyz(
                M, xyz, octree_levels=[1,1,1], method='box', finalize=False
                )
    #plano5 YZ-Xrigth
        xp, yp, zp = np.meshgrid( [x1,x2],[y1-2*M.hy[0],y1+2*M.hy[0]], [z2,z1])
        xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)]  # mkvc creates vectors
        M = refine_tree_xyz(
                M, xyz, octree_levels=[1,1,1], method='box', finalize=False
                )
    #plano5 YZ-Xrigth
        xp, yp, zp = np.meshgrid( [x1,x2],[y2-2*M.hy[0],y2+2*M.hy[0]], [z2,z1])
        xyz = np.c_[mkvc(xp), mkvc(yp),mkvc(zp)]  # mkvc creates vectors
        M = refine_tree_xyz(
                M, xyz, octree_levels=[1,1,1], method='box', finalize=False
                )
    return
