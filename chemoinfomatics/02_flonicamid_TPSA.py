from rdkit.Chem import Descriptors
Descriptors.TPSA(Chem.MolFromSmiles('C1=CN=CC(=C1C(F)(F)F)C(=O)NCC#N'))