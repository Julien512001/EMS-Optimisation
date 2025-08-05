import numpy as np
import matplotlib.pyplot as plt

air = np.arange(10, 550, 50)
air = np.arange(20, 550, 50)


Fz0_1 = np.array([85038.64080829, 84980.09998718, 84949.4791002, 84934.80328979,
              84927.96461426, 84924.80617744, 84923.35032112, 84922.67825187,
              84922.36663253, 84922.22094513, 84922.15186258])

T1 = np.array([1.9654823, 2.0029064, 2.0719952, 2.2323389, 2.3843865,
              2.4168926, 2.9611122, 3.8360703, 3.354789, 3.2212823,
              3.5084969])

Bmax_1 = np.array([1.88108011, 1.89209497, 1.89797571, 1.90085366, 1.90221281,
              1.90284506, 1.90313756, 1.90327284, 1.90333564, 1.90336503,
              1.90337898])

Fz0_2 = np.array([370730.24068181, 382258.68902005, 387810.5431494,
              390538.79403156, 391846.25415763, 392460.2241718,
              392745.81936786, 392878.4075132, 392940.18238487,
              392969.23119793, 392983.12398565])

T2 = np.array([12.7742876, 10.734542, 10.8372462, 9.5932667,
              9.8453724, 9.5490366, 9.9345193, 9.8264483,
              10.088045, 10.0646806, 11.1827931])

Bmax_2 = np.array([5.52328814, 5.55526112, 5.5718924, 5.57995289,
              5.58375655, 5.58552075, 5.58633688, 5.58671434,
              5.58688956, 5.58697156, 5.58701048])


# plt.figure()
# plt.plot(air, Fz0_1)
# plt.scatter(air, Fz0_1, marker="x")
# plt.plot(air, Fz0_2)
# plt.scatter(air, Fz0_2, marker="x")
# plt.grid()


fig, ax1 = plt.subplots()

ax1.plot(air, Fz0_1, color="blue", label="Force", zorder=4)
ax1.scatter(air, Fz0_1, marker="x", color="blue", zorder=4)
ax1.plot(air, Fz0_2, color="red", label="Force", zorder=4)
ax1.scatter(air, Fz0_2, marker="x", color="red", zorder=4)
ax1.set_ylabel("Force [N]")
ax1.set_xlabel("$h_{air} [mm]$")
ax1.tick_params(axis='y')

# ax2 = ax1.twinx()
# ax2.bar(air, T1, width=5, alpha=0.3, color="blue", zorder=3)
# ax2.bar(air, T2, width=5, alpha=0.3, color="red", zorder=3)
# ax2.set_ylabel("Temps (s)", color="red")
# ax2.tick_params(axis='y', labelcolor="red")

# ax2.set_ylim(0, max(T2) * 1.2)

ax1.grid(zorder=0)
plt.title("Impact de la saturation et des dimensions du domaine")
plt.show()



