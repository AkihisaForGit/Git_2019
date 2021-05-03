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

# 入力する分子（thiametoxam）

smiles = 'CN1COCN(C1=N[N+](=O)[O-])CC2=CN=C(S2)Cl'

# ファイル名を決める
t = datetime.datetime.fromtimestamp(time.time())
psi4.set_output_file("{}_{}{}{}_{}{}.log".format(smiles,
                                              t.year,
                                              t.month,
                                              t.day,
                                              t.hour,
                                              t.minute))

# SMILES から三次元構造を発生させて、粗3D構造最適化
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
params = ETKDGv3()
params.randomSeed = 1
EmbedMolecule(mol, params)

# MMFF（Merck Molecular Force Field） で構造最適化する
MMFFOptimizeMolecule(mol)
#UFF（Universal Force Field）普遍力場で構造最適化したい場合は
#UFFOptimizeMolecule(mol)

conf = mol.GetConformer()


# Psi4 に入力可能な形式に変換する。
# 電荷とスピン多重度を設定（下は、電荷０、スピン多重度1)
mol_input = "0 1"

#各々の原子の座標をXYZフォーマットで記述
for atom in mol.GetAtoms():
    mol_input += "\n " + atom.GetSymbol() + " " + str(conf.GetAtomPosition(atom.GetIdx()).x)\
    + " " + str(conf.GetAtomPosition(atom.GetIdx()).y)\
    + " " + str(conf.GetAtomPosition(atom.GetIdx()).z)

molecule = psi4.geometry(mol_input)

# 計算手法（汎関数）、基底関数を設定
level = "b3lyp/6-31G*"

# 計算手法（汎関数）、基底関数の例
#theory = ['hf', 'b3lyp']
#basis_set = ['sto-3g', '3-21G', '6-31G(d)', '6-31+G(d,p)', '6-311++G(2d,p)']

# 構造最適化計算を実行
energy, wave_function = psi4.optimize(level, molecule=molecule, return_wfn=True)

