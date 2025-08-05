#
# numpy : polyfit
# Vincent Legat - 2018
# Ecole Polytechnique de Louvain
#

from numpy import *
import matplotlib 
from matplotlib import pyplot as plt

matplotlib.rcParams['toolbar'] = 'None'
plt.rcParams['figure.facecolor'] = 'silver'
plt.figure("Polynomial interpolation")

X = [ -55, -25,   5,  35,  65]
U = [3.25,3.20,3.02,3.32,3.10]
a = polyfit(X,U,4)

x = linspace(X[0],X[-1],100)
uh = polyval(a,x)

plt.plot(x,uh)
plt.plot(X,U,'or')
plt.show()

