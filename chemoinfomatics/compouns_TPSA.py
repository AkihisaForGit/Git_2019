Descriptors.TPSA(Chem.MolFromSmiles('CNC(=N[N+](=O)[O-])NCC1CCOC1')) # 01_Dinotefuran 
Descriptors.TPSA(Chem.MolFromSmiles('C1=CN=CC(=C1C(F)(F)F)C(=O)NCC#N')) # 02_Flonicamid
Descriptors.TPSA(Chem.MolFromSmiles('CNC(=N[N+](=O)[O-])NCC1=CN=C(S1)Cl')) # 03_Clothianidin
Descriptors.TPSA(Chem.MolFromSmiles('C1CSC(=NC#N)N1CC2=CN=C(C=C2)Cl')) # 04_Thiacloprid
Descriptors.TPSA(Chem.MolFromSmiles('CCN(CC1=CN=C(C=C1)Cl)C(=C[N+](=O)[O-])NC')) # 05_Nitempyram
Descriptors.TPSA(Chem.MolFromSmiles('CN1COCN(C1=N[N+](=O)[O-])CC2=CN=C(S2)Cl')) # 06_Thiametoxam
Descriptors.TPSA(Chem.MolFromSmiles('C1CN(C(=N[N+](=O)[O-])N1)CC2=CN=C(C=C2)Cl')) # 08_Imidacloprid
Descriptors.TPSA(Chem.MolFromSmiles('CC(=NC#N)N(C)CC1=CN=C(C=C1)Cl')) # 10_Acetamiprid

Descriptors.MolLogP(Chem.MolFromSmiles('CNC(=N[N+](=O)[O-])NCC1CCOC1')) # 01
Descriptors.MolLogP(Chem.MolFromSmiles('C1=CN=CC(=C1C(F)(F)F)C(=O)NCC#N')) # 02
Descriptors.MolLogP(Chem.MolFromSmiles('CNC(=N[N+](=O)[O-])NCC1=CN=C(S1)Cl')) # 03
Descriptors.MolLogP(Chem.MolFromSmiles('C1CSC(=NC#N)N1CC2=CN=C(C=C2)Cl')) # 04
Descriptors.MolLogP(Chem.MolFromSmiles('CCN(CC1=CN=C(C=C1)Cl)C(=C[N+](=O)[O-])NC')) # 05
Descriptors.MolLogP(Chem.MolFromSmiles('CN1COCN(C1=N[N+](=O)[O-])CC2=CN=C(S2)Cl')) # 06
Descriptors.MolLogP(Chem.MolFromSmiles('C1CN(C(=N[N+](=O)[O-])N1)CC2=CN=C(C=C2)Cl')) # 08
Descriptors.MolLogP(Chem.MolFromSmiles('CC(=NC#N)N(C)CC1=CN=C(C=C1)Cl')) # 10