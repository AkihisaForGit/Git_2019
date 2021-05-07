from rdkit import Chem
from rdkit.Chem import Descriptors
import psi4
import datetime
import time

smiles = 'CNC(=N[N+](=O)[O-])NCC1CCOC1'
Descriptors.TPSA(Chem.MolFromSmiles(smiles)) #dinotefuran

t = datetime.datetime.fromtimestamp(time.time())
psi4.set_output_file("{}_{}{}{}_{}{}.log".format(smiles,
                                              t.year,
                                              t.month,
                                              t.day,
                                              t.hour,
                                              t.minute))
