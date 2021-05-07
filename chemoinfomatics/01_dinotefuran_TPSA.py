from rdkit.Chem import Descriptors
Descriptors.TPSA(Chem.MolFromSmiles('CNC(=N[N+](=O)[O-])NCC1CCOC1')) #dinotefuran