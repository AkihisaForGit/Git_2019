import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFHasAllMoleculeParams, MMFFOptimizeMolecule
from rdkit.Chem.Draw import IPythonConsole
import psi4

import datetime
import time

time
#計算時間を見てみる

# ハードウェア側の設定（計算に用いるCPUのスレッド数とメモリ設定）
psi4.set_num_threads(nthread=3)
psi4.set_memory("3GB")


## SMILESをxyz形式に変換
def smi2xyz(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv2())
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer(-1)
    
    xyz = '0 1'
    for atom, (x,y,z) in zip(mol.GetAtoms(), conf.GetPositions()):
        xyz += '\n'
        xyz += '{}\t{}\t{}\t{}'.format(atom.GetSymbol(), x, y, z)
        
    return xyz
    
# 入力する分子（dinotefuran）

smiles = 'CNC(=N[N+](=O)[O-])NCC1CCOC1'


psi4.set_output_file('dinotefuran.txt')
dinotefuran = psi4.geometry(smi2xyz(smiles))
_, wfn_dtf = psi4.optimize('hf/sto-3g', molecule=dinotefuran, return_wfn=True)
rdkit_dinotefuran = Chem.AddHs(Chem.MolFromSmiles(smiles))
## 双極子モーメントの計算

psi4,oeprop(wfn_dtf, 'DIPOLE')
dipole_x, dipole_y, dipole_z = psi4.variable('SCF DIPOLE X'), psi4.variable('SCF DIPOLE Y'), psi4.variable('SCF DIPOLE Z') 
dipole_moment = np.sqrt(dipole_x ** 2 + dipole_y ** 2 + dipole_z ** 2)

print(round(dipole_moment,3),'D')
