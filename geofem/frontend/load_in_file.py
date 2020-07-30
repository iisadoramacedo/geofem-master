#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 19:56:12 2020

@author: akel
Leitura dos arquivos de input 
input --> arquivo com as instruções do modelo geologico e caracteristicas
          da simulação. Todas as instruções são identificadas com um label
          seguidos ":" com a informação. Por exemplo, para o método geofísico,
          temos duas opções MT3D e MCSEM. Assim a instrução no arquivo fica.
          metodo: MT3D
"""

import numpy as np
import time

import sys 

def inputfiles(filename):
    """ Leitura arquivo input"""
    out={'info input migeo': time.ctime()} #informação da hora da leitura
    arq=open(filename,"rt")
    count=0
    while True:
        linha=arq.readline() 
        if not linha:
            break
        linha=linha[:-1]
        try:
            st,temp=linha.split(":",1)
        except ValueError:
            break
        if st[0]==str('#'):
            pass
        elif st.lower()==str('metodo'):
            M=temp.split(",")
            out['metodo']=str(M).lower()
         #info_domain  
        elif st.lower()==str('freq'):
            f=np.array(temp.split(","),dtype=float)
            freq=np.logspace(np.log10(f[0]),np.log10(f[1]),int(f[2]))
            out['freq']= freq
        elif st.lower()==str('x'):
            X=np.array(temp.split(","),dtype=float)
            out['x']=X
        elif st.lower()==str('y'):
            Y=np.array(temp.split(","),dtype=float)
            out['y']=Y
        elif st.lower()==str('z'):   
            Z=np.array(temp.split(","),dtype=float)
            out['z']=Z
        #cellsize
        elif st.lower()==str('dxdydz'):
            dx,dy,dz=np.array(temp.split(","),dtype=float)
            out['dxdydz']=dx,dy,dz
        elif st.lower()==str('layer'):
            layer=np.array(temp.split(","),dtype=float)
            out['layer']=layer
        elif st.lower()==str('cond'):
            cond=np.array(temp.split(","),dtype=float)
            out['cond']=cond
        elif st.lower()==str('box'):
            count+=1
            if count==1:
                box=np.array(temp.split(","),dtype=float)
            else:
                tmp_box=np.array(temp.split(","),dtype=float)
                box = np.concatenate((box,tmp_box))
            out['box']=box
           
        #infomeshgrid

        #infomodelMT1D    
        elif st.lower()==str('thi'):
            thi=np.array(temp.split(","),dtype=float)
            out['thi']=thi
        elif st.lower()==str('res'):
            res=np.array(temp.split(","),dtype=float)
            out['res']=res
        else:
            nome = None
            print("Label",st, "invalido")
            sys.exit()
    arq.close()
    return out




# if __name__ == "__main__":
#     import sys
#     print(input_1(str(sys.argv[1])))
