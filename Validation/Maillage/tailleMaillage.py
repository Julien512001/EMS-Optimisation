import numpy as np
import matplotlib.pyplot as plt



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


time = np.array([189.42,
                 8.4936,
                 6.99,
                 5.8738,
                 5.562,
                 3.936,
                 1.65666,
                 0.9561652,
                 0.5826563])

time_log = np.log10(time)
print(time_log)

I0 = np.array([33.4533,
               33.452684723,
               33.44855538955784,
               33.441255767955305,
               33.438355964245034,
               33.47024370509139,
               33.53,
               33.4935987,
               33.554])

Fz0_test = np.array([79.9579,
                     79.9591,
                     79.97244,
                     79.99589378635848,
                     79.992,
                     79.85839,
                     79.7476969,
                     79.67342,
                     79.5511])
Fz0 = np.array([93136.71,
                93136.59,
                93136.186,
                93135.3993,
                93134.01,
                93141.0665,
                93156.2139,
                93147.5,
                93161.9858])


x = np.arange(1,len(Fz0)+1,1)

err_Fz0 = np.zeros_like(Fz0)
for i in range(len(err_Fz0)):
    err_Fz0[i] = 100-100*np.min([Fz0[i],Fz0[0]])/np.max([Fz0[i],Fz0[0]])

err_Fz0_test = np.zeros_like(Fz0_test)
for i in range(len(err_Fz0_test)):
    err_Fz0_test[i] = 100-100*np.min([Fz0_test[i],Fz0_test[0]])/np.max([Fz0_test[i],Fz0_test[0]])


# pareto = pareto_front(err_Fz0, time)


# plt.scatter(err_Fz0, time, color='gray', label='Points')
# plt.plot(pareto[:,0], pareto[:,1], 'r-o', label='Front de Pareto')
# plt.xlabel('Erreur (%)')
# plt.ylabel('Temps (s)')
# plt.title('Front de Pareto - Erreur vs Temps')
# plt.legend()
# plt.grid(True)


plt.figure()
plt.plot(x, I0, label="I0")
plt.scatter(x,I0, marker="x")
plt.grid()




fig, ax1 = plt.subplots()

ax1.plot(x, err_Fz0, color="blue", label="Erreur 1", zorder=4)
ax1.scatter(x, err_Fz0, marker="x", color="blue", zorder=4)
ax1.plot(x, err_Fz0_test, color="red", label="Erreur 2", zorder=4)
ax1.scatter(x, err_Fz0_test, marker="x", color="red", zorder=4)
ax1.set_ylabel("Erreur (%)")
ax1.set_xlabel("Rafinement du maillage")
ax1.tick_params(axis='y')
ax1.legend(loc="center right", bbox_to_anchor=(1.0, 0.4))

ax2 = ax1.twinx()
ax2.bar(x, time, width=0.5, alpha=0.3, color="blue", zorder=3, label="Temps")
ax2.set_ylabel("Temps (s)")
ax2.set_yscale("log")
ax2.tick_params(axis='y')
ax2.legend(loc="center right", bbox_to_anchor=(1.0, 0.55))
ax2.set_xticks([1,2,3,4,5,6,7,8,9])


ax1.grid(zorder=0)
plt.title("Influence de la taille des éléments du maillage")

plt.show()

