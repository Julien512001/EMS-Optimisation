import numpy as np
import scipy.sparse as sp



p = 3
m = 3

Npsi_x = p
Npsi_z = m-1





def Y_matrix(size):
    Y = np.zeros((size, size + 1))
    for i in range(size):
        Y[i, i] = -1
        Y[i, i + 1] = 1
    return Y


id = sp.eye(Npsi_x-1)
W = sp.csr_matrix([1,1])
A = sp.kron(id, W)
X = sp.block_diag((A, 1), format='csr')
Y = sp.csr_matrix(Y_matrix(Npsi_z))
Xsi_x = sp.kron(Y, X)


print(id.toarray())
print(X.toarray())
print(Y.toarray())



