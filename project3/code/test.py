import numpy as np

Nx = 100
Nt = Nx
t1 = np.ones(Nt)*0.1
t2 = np.ones(Nt)*0.5
x = np.linspace(0, 1, Nx)

X = np.vstack([t1, x]).T
print(X)