import psi4
import datetime
import time

t = datetime.datetime.fromtimestamp(time.time())

psi4.set_num_threads(nthread=2)
psi4.set_memory('4GB')
psi4.set_output_file('{}{}{}_{}{}.log'.format(t.year,
                                              t.month,
                                              t.day,
                                              t.hour,
                                              t.minute)

### XYZ
h2O_xyz = psi4.geometry('''
0 1
O 0 0　-0.11
H 0 -1.4　1.2
H 0 1.2 1.2
''')
### Z-matrix
h2O_zmat = psi4.geometry('''
0 1
O
H 1 0.96
H 1 0.96 2 104.5''')
