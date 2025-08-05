import numpy as np
import scipy.sparse as sp

def Y_matrix(size):
    Y = np.zeros((size, size + 1))
    for i in range(size):
        Y[i, i] = -1
        Y[i, i + 1] = 1
    return Y




p = 3
m = 3
Npsi_x = p
Npsi_y = m-1

Nx = (2*p-1)*m
Ny = 2*(m-1)*p
N = Nx + Ny

Npsi = Npsi_x*Npsi_y

print(f"Npsi = {Npsi}, Nx = {Nx}, Ny = {Ny}, N = {N}")

id = sp.eye(Npsi_x-1)
W = sp.csr_matrix([1,1])
A = sp.kron(id, W)
X = sp.block_diag((A, 1), format='csr')
Y = Y_matrix(Npsi_y)



Xsi_x =  sp.kron(Y, X)
# print(Xsi_x.shape)
# print(Xsi_x.toarray())



id = sp.eye(Npsi_y)
Y_y = sp.csr_matrix(Y_matrix(Npsi_x-1))
Z = sp.csr_matrix(([-1], ([0], [Y_y.shape[1] - 1])), shape=(1, Y_y.shape[1]))
Y_y = sp.vstack([Y_y, Z])
Xsi_y = sp.kron(id, sp.hstack((Y_y, Y_y)))


print(Xsi_y.shape)
print(Xsi_y.toarray())