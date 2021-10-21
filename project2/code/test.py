import numpy as np
eta = "np.linspace(0,1,10)"
command = "eta_val = " + eta
print(command)
exec(command)
print(eta_val)
# eta = "2"
# eta_val = exec(eta)
# print(eta_val)
a = 1
print(np.shape(a) == ())