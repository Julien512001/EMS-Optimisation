import numpy as np
import matplotlib.pyplot as plt




I = np.array([10,20,40,60,80,100])
RMSE_x = np.array([1.49222791051655,
                   1.4818753286425745,
                   1.6952362409346369,
                   2.1046710315230257,
                   2.1264604819465553,
                   2.34628758689614
                   ])
RMSE_z = np.array([0.2864070598421038,
                   0.3308702899034335,
                   0.4221401004270957,
                   0.9813212239742017,
                   0.970673731807534,
                   0.7900501145523862,
                   ])

couleur_Bx_MEC = "#CD5C5C"      # Rouge clair/gris rosé
couleur_Bz_MEC = "#7BAFD4"      # Bleu doux


plt.figure()
plt.title("Erreur quadratique moyenne exprimée en pourcentage")
plt.plot(I, RMSE_x, label='$RMSE_x$',c=couleur_Bx_MEC)
plt.plot(I, RMSE_z, label='$RMSE_z$',c=couleur_Bz_MEC)

plt.scatter(I, RMSE_x, marker='x', color='black')  
plt.scatter(I, RMSE_z, marker='x', color='black') 

plt.xlabel("Courant I (A)")
plt.ylabel("RMSE (%)")
plt.legend()
plt.grid(True)
plt.savefig("Validation/Compare/RMSE.pdf")
plt.show()
