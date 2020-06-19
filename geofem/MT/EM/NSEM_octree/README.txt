Para ter provisoriamente o MT rodando com tensor mesh e octree mesh
no mesmo ambiente conda, seguir os seguintes passos:

1) adicionar a pasta NSEM_octree no diretório
   /anaconda3/envs/SEU_AMBIENTE_SIMPEG/lib/python3.7/site-packages/SimPEG/EM

2) abrir o script __init__.py do diretório
   /anaconda3/envs/SEU_AMBIENTE_SIMPEG/lib/python3.7/site-packages/SimPEG/EM
   e adicionar uma linha com o comando:
   from . import NSEM_octree

3) Nos scripts que utilizarem malha tipo octree utilizar o comando
   from SimPEG.EM import NSEM_octree as NSEM

4) Nos scripts que utilizarem malha tipo tensor utilizar o comando normal:
   from SimPEG.EM import NSEM
