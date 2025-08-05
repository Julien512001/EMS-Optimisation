import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import scipy.sparse as sp
from test import *




print(Xsi_x.toarray())
# --------------------------------------------------------


MMF_x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
MMF_y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


MMF = np.hstack([MMF_x, MMF_y]).reshape(-1, 1)

# --------------------------------------------

Rx = np.eye(15)
Ry = np.eye(12)*2

R = block_diag(Rx, Ry)

# ------------------------------------------------

print(f"R:{R.shape}, Xsi:{Xsi.shape}")


A = Xsi @ R @ np.transpose(Xsi)
F = Xsi @ MMF

print(A.shape)
print(F.shape)


psi = np.linalg.solve(A, F)


phi = np.transpose(Xsi)@psi

print(phi.shape)


phi_x = phi[:Nx]
phi_y = phi[Nx:]


B_x = np.reshape(phi_x, (m, P))
B_y = np.reshape(phi_y, (M, p))



B_x_new = np.zeros((m, p))
B_y_new = np.zeros((m, p))

j = 0
for i in range(B_x_new.shape[1]):
    if (j == 0):
        B_x_new[:, i] = B_x[:, j]
        j += 1
    else:
        B_x_new[:, i] = (B_x[:, j] + B_x[:, j+1])/2.0
        j += 2

j = 0
for i in range(B_y_new.shape[0]):
    if (j == 0):
        B_y_new[i, :] = B_y[j, :]
        j += 1
    elif (j == B_y.shape[0] - 1):
        B_y_new[i, :] = B_y[j,:]
    else:
        B_y_new[i, :] = (B_y[j, :] + B_y[j+1, :])/2.0
        j += 2



# plt.figure(figsize=(8, 6)) 
# plt.imshow(B_x_new, cmap="coolwarm", aspect="auto")
# plt.colorbar()
# plt.title("Heatmap de la matrice B")
# plt.xlabel("Colonnes")
# plt.ylabel("Lignes")

# plt.figure(figsize=(8, 6)) 
# plt.imshow(B_y_new, cmap="coolwarm", aspect="auto")
# plt.colorbar()
# plt.title("Heatmap de la matrice B")
# plt.xlabel("Colonnes")
# plt.ylabel("Lignes")

# plt.figure(figsize=(8, 6)) 
# plt.imshow(np.sqrt(B_x_new**2 + B_y_new**2), cmap="coolwarm", aspect="auto")
# plt.colorbar()
# plt.title("Heatmap de la matrice B")
# plt.xlabel("Colonnes")
# plt.ylabel("Lignes")

# plt.show()
