import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def pareto_front(erreurs, temps):
    # Convertir en tableau numpy
    data = np.array(list(zip(erreurs, temps)))
    # Trier par erreur croissante
    data = data[data[:,0].argsort()]
    
    # Construire le front (points non dominés)
    pareto = [data[0]]
    for point in data[1:]:
        if point[1] < pareto[-1][1]:  # Temps plus bas
            pareto.append(point)
    
    return np.array(pareto)





Fref = 80.14

df = pd.read_csv("Validation/Maillage/data.csv",header=None,sep=";")

t = df.iloc[:,0].to_numpy()
F = df.iloc[:,1].to_numpy()

print(t, F)

error = np.zeros_like(F)
for i in range(len(F)):
    error[i] = 100-100*np.min([F[i],Fref])/np.max([F[i],Fref])


pareto = pareto_front(error, t)

plt.scatter(error, t, color='gray', label='Points')
plt.plot(pareto[:,0], pareto[:,1], 'r-o', label='Front de Pareto')
plt.xlabel('Erreur (%)')
plt.ylabel('Temps (s)')
plt.title('Front de Pareto - Erreur vs Temps')
plt.axhline(5)
plt.axvline(5)
plt.legend()
plt.grid(True)
plt.show()




