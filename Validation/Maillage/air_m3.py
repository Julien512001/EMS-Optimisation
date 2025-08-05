import numpy as np
import matplotlib.pyplot as plt


# TEST
# Sweep sur la quantité d'air hm_air
# l_air_p1 = 2.5*l_core
# N_p = 55
# N_m = hm_air*100 en int
# Sur la suspension classique avec maillage fin
# Air-gap = 15e-3



air1 = np.arange(20, 1000, 50)
air2 = np.arange(20, 1000, 50)

# k = 20

Fz0_1 = np.array([34173.71759338, 34136.02220519, 34110.10371299, 34093.6574619,
              34082.97277667, 34075.72459783, 34070.61614808, 34066.91542898,
              34064.18638159, 34062.15199465, 34060.62584036, 34059.47684072,
              34058.61006656, 34057.95548396, 34057.46085561, 34057.08697393,
              34056.80430867, 34056.59057761, 34056.42895076, 34056.30671062])*1e-3

T1 = np.array([4.8029073, 5.1434905, 5.5458187, 5.8355181, 6.265158,
              6.1292359, 6.0573625, 6.2409767, 5.9536214, 7.1703568,
              6.797502, 6.6539224, 6.996504, 6.9720589, 7.0611535,
              7.4256512, 7.9038297, 9.5175649, 7.9172781, 7.8719009])

Bmax_1 = np.array([1.49841892, 1.50543828, 1.5090986, 1.51110657, 1.51231079,
              1.51309602, 1.51364118, 1.5140354, 1.51432737, 1.51454644,
              1.51471187, 1.51483716, 1.51493216, 1.5150042, 1.51505883,
              1.51510023, 1.5151316, 1.51515536, 1.51517336, 1.51518699])


# k = 80

Fz0_2 = np.array([244848.50269493, 252050.19026756, 256672.22276446, 259910.30086619,
              262296.40097238, 264095.7533201, 265465.32364654, 266510.54536924,
              267308.06310056, 267915.81540293, 268378.26796472, 268729.67045701,
              268996.37506318, 269198.60380315, 269351.82969958, 269467.86075997,
              269555.68896727, 269622.149424, 269672.43041824, 269710.46612445])*1e-3

T2 = np.array([16.4072818, 21.3892176, 17.49757, 20.1078272, 20.1922577,
              16.471969, 19.9572648, 19.8250942, 19.8683432, 20.3483336,
              27.836235, 25.8782129, 25.7023171, 26.6988164, 27.4626572,
              26.8595131, 25.9080431, 27.2360744, 27.5559335, 29.8331147])

Bmax_2 = np.array([5.25095561, 5.28318006, 5.29983831, 5.30895023, 5.31441249,
              5.31797556, 5.32045043, 5.32224068, 5.32356679, 5.32456179,
              5.32531318, 5.32588225, 5.32631371, 5.32664089, 5.32688894,
              5.32707695, 5.3272194, 5.3273273, 5.32740902, 5.32747091])









err1 = np.zeros_like(Fz0_1)
for i in range(len(err1)):
    err1[i] = 100-100*np.min([Fz0_1[i], Fz0_1[-1]])/np.max([Fz0_1[i], Fz0_1[-1]])

err2 = np.zeros_like(Fz0_2)
for i in range(len(err2)):
    err2[i] = 100-100*np.min([Fz0_2[i], Fz0_2[-1]])/np.max([Fz0_2[i], Fz0_2[-1]])

print(err2, err1)

# plt.figure()
# plt.plot(air, Fz0_1)
# plt.scatter(air, Fz0_1, marker="x")
# plt.plot(air, Fz0_2)
# plt.scatter(air, Fz0_2, marker="x")
# plt.grid()


fig, ax1 = plt.subplots()

ax1.plot(air1, Fz0_1, color="blue", label="Force 1", zorder=4)
ax1.scatter(air1, Fz0_1, marker="x", color="blue", zorder=4)
ax1.plot(air2, Fz0_2, color="red", label="Force 2", zorder=4)
ax1.scatter(air2, Fz0_2, marker="x", color="red", zorder=4)
ax1.set_ylabel("Force (kN)")
ax1.set_xlabel("$h_{air} [mm]$")
ax1.tick_params(axis='y')
ax1.legend(loc="center right", bbox_to_anchor=(1.0, 0.4))

ax2 = ax1.twinx()
ax2.bar(air1, err1, width=10, alpha=0.3, color="blue", zorder=3, label="erreur 1")
ax2.bar(air2, err2, width=10, alpha=0.3, color="red", zorder=3, label="erreur 2")
ax2.set_ylabel("Erreur (%)", color="red")
ax2.tick_params(axis='y', labelcolor="red")
ax2.legend(loc="center right", bbox_to_anchor=(1.0, 0.55))

ax2.set_ylim(0, max(err2) * 1.2)

ax1.grid(zorder=0)
plt.title("Impact de la saturation et des dimensions du domaine")
plt.savefig("Validation/Maillage/h_air.pdf")
plt.show()



